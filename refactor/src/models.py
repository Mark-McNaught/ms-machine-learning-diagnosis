import torch
import torch.nn as nn
from torchvision import models

import numpy as np

#####################################################################################################
###################################### Model Component Modules ######################################
#####################################################################################################

class ChannelAttention(nn.Module):
    """ Channel Attention (like SE) """
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """ Spatial Attention"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        return self.sigmoid(out)


class CBAM(nn.Module):
    """ CBAM Block = Channel + Spatial attention """
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_channels, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out


class BasicBlockCBAMPre(nn.Module):
    """ CBAM applied after block convolutions, before residual addition """
    def __init__(self, block: nn.Module, channels: int):
        super().__init__()
        self.block = block
        self.cbam = CBAM(channels)

    def forward(self, x):
        identity = x

        out = self.block.conv1(x)
        out = self.block.bn1(out)
        out = self.block.relu(out)

        out = self.block.conv2(out)
        out = self.block.bn2(out)

        out = self.cbam(out)

        if self.block.downsample is not None:
            identity = self.block.downsample(x)

        out += identity
        out = self.block.relu(out)

        return out


class BasicBlockCBAMPost(nn.Module):
    """ CBAM applied after residual addition """
    def __init__(self, block: nn.Module, channels: int):
        super().__init__()
        self.block = block
        self.cbam = CBAM(channels)

    def forward(self, x):
        out = self.block(x)
        out = self.cbam(out)
        return out


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class BasicBlockSEPre(nn.Module):
    """ SE applied after block convolutions, before residual addition """
    def __init__(self, block: nn.Module, channels: int):
        super().__init__()
        self.block = block
        self.se = SEBlock(channels)

    def forward(self, x):
        identity = x

        out = self.block.conv1(x)
        out = self.block.bn1(out)
        out = self.block.relu(out)

        out = self.block.conv2(out)
        out = self.block.bn2(out)

        # Apply SE to the residual branch
        out = self.se(out)

        if self.block.downsample is not None:
            identity = self.block.downsample(x)

        out += identity
        out = self.block.relu(out)
        return out

#####################################################################################################
####################################### Model Architectures  ########################################
#####################################################################################################

class BaseResNet18(nn.Module):
    """ Base ResNet18 model with a custom classification head for binary use cases"""
    def __init__(self, num_classes=1):
        super().__init__()
        # Load pre-trained ResNet18 model
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Modify the final fully connected layer for binary classification
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)


class CBAMResNet18(nn.Module):
    """
    ResNet18 with flexible CBAM placement.

    cbam_location:
        - "end"        : single CBAM before classifier
        - "block_pre"  : CBAM inside each BasicBlock (before shortcut)
        - "block_post" : CBAM inside each BasicBlock (after shortcut)
    """
    def __init__(self, num_classes=1, cbam_location="end"):
        super().__init__()
        self.cbam_location = cbam_location
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        if cbam_location in ["block_pre", "block_post"]:
            wrapper = BasicBlockCBAMPre if cbam_location == "block_pre" else BasicBlockCBAMPost
            self.model.layer1 = self._wrap_layer(self.model.layer1, 64, wrapper)
            self.model.layer2 = self._wrap_layer(self.model.layer2, 128, wrapper)
            self.model.layer3 = self._wrap_layer(self.model.layer3, 256, wrapper)
            self.model.layer4 = self._wrap_layer(self.model.layer4, 512, wrapper)
        else:
            # CBAM layers are injected directly into the model structure
            # by replacing the global avgpool with a sequence of (CBAM -> AvgPool)
            self.model.avgpool = nn.Sequential(
                CBAM(512),
                nn.AdaptiveAvgPool2d((1, 1))
            )

        # Classification head
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def _wrap_layer(self, layer, channels, wrapper):
        return nn.Sequential(*[wrapper(block, channels) for block in layer])

    def forward(self, x, return_features=False):
            if return_features:
                # configuration for feature extraction
                x = self.model.conv1(x)
                x = self.model.bn1(x)
                x = self.model.relu(x)
                x = self.model.maxpool(x)
                x = self.model.layer1(x)
                x = self.model.layer2(x)
                x = self.model.layer3(x)
                x = self.model.layer4(x)
                x = self.model.avgpool(x)
                return torch.flatten(x, 1) # Returns 512-dim vector
            
            # Standard full pass for training/inference
            return self.model(x)

class SEResNet18(nn.Module):
    """
    ResNet18 with flexible SE placement.

    se_location:
        - "end"       : single SE block before classifier (after layer 4)
        - "block_pre" : SE inside each BasicBlock (before shortcut)
    """
    def __init__(self, num_classes=1, se_location="end"):
        super().__init__()
        self.se_location = se_location
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        if se_location == "block_pre":
            # Wrap every residual block in layers 1-4 with SE
            self.model.layer1 = self._wrap_layer(self.model.layer1, 64, BasicBlockSEPre)
            self.model.layer2 = self._wrap_layer(self.model.layer2, 128, BasicBlockSEPre)
            self.model.layer3 = self._wrap_layer(self.model.layer3, 256, BasicBlockSEPre)
            self.model.layer4 = self._wrap_layer(self.model.layer4, 512, BasicBlockSEPre)
        else:
            # "end" location: Inject SE into the global avgpool sequence
            self.model.avgpool = nn.Sequential(
                SEBlock(512),
                nn.AdaptiveAvgPool2d((1, 1))
            )

        # Custom classification head to match your pipeline
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def _wrap_layer(self, layer, channels, wrapper):
        return nn.Sequential(*[wrapper(block, channels) for block in layer])

    def forward(self, x, return_features=False):
        if return_features:
            # Manual forward pass to capture features for NCA/kNN
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)

            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)
            
            x = self.model.avgpool(x)
            return torch.flatten(x, 1) # Returns 512-dim vector

        return self.model(x)

#####################################################################################################
########################################### Model Factory ###########################################
#####################################################################################################

def get_model(architecture="base", device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """ Factory helper function to instantiate models by name """
    # Base ResNet18
    if architecture == "base":
        model = BaseResNet18()

    # ResNet18 + CBAM Variations
    elif architecture == "cbam_end":
        model = CBAMResNet18(cbam_location="end")
    elif architecture == "cbam_block_pre":
        model = CBAMResNet18(cbam_location="block_pre")
    elif architecture == "cbam_block_post":
        model = CBAMResNet18(cbam_location="block_post")  
    
    # ResNet18 + SE Variations
    elif architecture == "se_end":
        model= SEResNet18(se_location="end")
    elif architecture == "se_block_pre":
        model = SEResNet18(se_location="block_pre")
          
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    print(f"get_model()>>> \n", model)
    return model.to(device)
