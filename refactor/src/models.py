import torch
import torch.nn as nn
from torchvision import models

import numpy as np
import timm

#####################################################################################################
###################################### Model Component Modules ######################################
#####################################################################################################

# ── Classifier Head ───────────────────────────────────────────────────────────
#
# All models in this file use the same binary classification head system.
# The head type is selected via build_classifier_head(in_features, head=...).
#
# Supported values:
#   "mlp"    — two-layer MLP: Linear(in_features->128)->ReLU->Dropout(0.3)->Linear(128->1)  [default]
#   "linear" — single linear layer: Linear(in_features->1)
#
# Pass head= through get_model() to switch all models simultaneously:
#   get_model("cbam_end", head="linear")
#
# Both are compatible with BCEWithLogitsLoss + Adam + the two-phase protocol.
# No changes to trainer.py or criterion are required for either option.

VALID_HEADS = ("mlp", "linear")

def build_classifier_head(in_features: int, num_classes: int = 1, head: str = "mlp") -> nn.Sequential:
    """
    Builds the binary classification head.
    Head designs:
        "mlp" (default) — two-layer MLP with bottleneck and dropout:
            Linear(in_features -> 128) -> ReLU -> Dropout(0.3) -> Linear(128 -> 1)
        "linear" — single linear layer (linear probe):
            Linear(in_features -> 1)
    """
    if head not in VALID_HEADS:
        raise ValueError(f"Unknown head type '{head}'. Choose from: {VALID_HEADS}")

    if head == "linear":
        return nn.Sequential(
            nn.Linear(in_features, num_classes)
        )

    # "mlp" — default
    return nn.Sequential(
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, num_classes)
    )

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
    """
    Canonical SE block (Hu et al., 2018).
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
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

        out = self.se(out)

        if self.block.downsample is not None:
            identity = self.block.downsample(x)

        out += identity
        out = self.block.relu(out)

        return out


class ChannelAttentionSE(nn.Module):
    """
    SE-style channel attention (avg-pool only).
    Identical to SEBlock but without the sigmoid wrapper — used inside CBAMIsolated.
    """
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return self.sigmoid(y)


class CBAMIsolated(nn.Module):
    """
    CBAMIsolated for the spatial attention isolation experiment (SRQ1).

    Channel attention: SE-style (avg-pool only, Conv2d) — identical to SEBlock
    Spatial attention: standard CBAM spatial (avg+max, Conv2d 7×7)

    This variant isolates spatial attention's contribution by holding channel
    attention constant (matching SE exactly), so only spatial attention varies.
    """
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttentionSE(in_channels, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out


class BasicBlockCBAMIsolatedPre(nn.Module):
    """ CBAMIsolated applied after block convolutions, before residual addition """
    def __init__(self, block: nn.Module, channels: int):
        super().__init__()
        self.block = block
        self.cbam_iso = CBAMIsolated(channels)

    def forward(self, x):
        identity = x

        out = self.block.conv1(x)
        out = self.block.bn1(out)
        out = self.block.relu(out)

        out = self.block.conv2(out)
        out = self.block.bn2(out)

        out = self.cbam_iso(out)

        if self.block.downsample is not None:
            identity = self.block.downsample(x)

        out += identity
        out = self.block.relu(out)

        return out


#####################################################################################################
######################################## ResNet18 Model Classes #####################################
#####################################################################################################

class BaseResNet18(nn.Module):
    """Plain ResNet18 with custom classifier head."""
    def __init__(self, num_classes=1, head="mlp"):
        super().__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_features = self.model.fc.in_features
        self.model.fc = build_classifier_head(num_features, num_classes, head)

    def forward(self, x, return_features=False):
        if return_features:
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)
            x = self.model.avgpool(x)
            return torch.flatten(x, 1)
        return self.model(x)


class CBAMResNet18(nn.Module):
    """ResNet18 with flexible CBAM placement."""
    def __init__(self, num_classes=1, cbam_location="end", head="mlp"):
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
            self.model.avgpool = nn.Sequential(
                CBAM(512),
                nn.AdaptiveAvgPool2d((1, 1))
            )

        num_features = self.model.fc.in_features
        self.model.fc = build_classifier_head(num_features, num_classes, head)

    def _wrap_layer(self, layer, channels, wrapper):
        return nn.Sequential(*[wrapper(block, channels) for block in layer])

    def forward(self, x, return_features=False):
        if return_features:
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)
            x = self.model.avgpool(x)
            return torch.flatten(x, 1)
        return self.model(x)


class SEResNet18(nn.Module):
    """ResNet18 with flexible SE placement."""
    def __init__(self, num_classes=1, se_location="end", head="mlp"):
        super().__init__()
        self.se_location = se_location
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        if se_location == "block_pre":
            self.model.layer1 = self._wrap_layer(self.model.layer1, 64, BasicBlockSEPre)
            self.model.layer2 = self._wrap_layer(self.model.layer2, 128, BasicBlockSEPre)
            self.model.layer3 = self._wrap_layer(self.model.layer3, 256, BasicBlockSEPre)
            self.model.layer4 = self._wrap_layer(self.model.layer4, 512, BasicBlockSEPre)
        else:
            self.model.avgpool = nn.Sequential(
                SEBlock(512),
                nn.AdaptiveAvgPool2d((1, 1))
            )

        num_features = self.model.fc.in_features
        self.model.fc = build_classifier_head(num_features, num_classes, head)

    def _wrap_layer(self, layer, channels, wrapper):
        return nn.Sequential(*[wrapper(block, channels) for block in layer])

    def forward(self, x, return_features=False):
        if return_features:
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)
            x = self.model.avgpool(x)
            return torch.flatten(x, 1)
        return self.model(x)


class CBAMIsolatedResNet18(nn.Module):
    """ResNet18 with CBAMIsolated — spatial attention isolation experiment."""
    def __init__(self, num_classes=1, cbam_iso_location="end", head="mlp"):
        super().__init__()
        self.cbam_iso_location = cbam_iso_location
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        if cbam_iso_location == "block_pre":
            self.model.layer1 = self._wrap_layer(self.model.layer1, 64)
            self.model.layer2 = self._wrap_layer(self.model.layer2, 128)
            self.model.layer3 = self._wrap_layer(self.model.layer3, 256)
            self.model.layer4 = self._wrap_layer(self.model.layer4, 512)
        else:
            self.model.avgpool = nn.Sequential(
                CBAMIsolated(512),
                nn.AdaptiveAvgPool2d((1, 1))
            )

        num_features = self.model.fc.in_features
        self.model.fc = build_classifier_head(num_features, num_classes, head)

    def _wrap_layer(self, layer, channels):
        return nn.Sequential(*[BasicBlockCBAMIsolatedPre(block, channels) for block in layer])

    def forward(self, x, return_features=False):
        if return_features:
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)
            x = self.model.avgpool(x)
            return torch.flatten(x, 1)
        return self.model(x)


#####################################################################################################
########################################### ViT Model Classes #######################################
#####################################################################################################

# NEW: Vision Transformer models for SRQ4 (standalone ViT baseline)

class DeiTSmallBinary(nn.Module):
    """
    DeiT-Small/16 for binary MS classification.
    
    Architecture: 12 transformer layers, 384-dim, 6 attention heads
    Parameters: ~22M (1.9 x ResNet18)
    Pretrained: ImageNet-1K with distillation
    
    Use for: SRQ4 standalone ViT baseline (conservative comparison)
    """
    def __init__(self, num_classes=1, head="mlp"):
        super().__init__()
        if timm is None:
            raise ImportError("timm is required for ViT models. Install with: pip install timm")
        
        self.model = timm.create_model(
            'deit_small_patch16_224',
            pretrained=True,
            num_classes=0  # Remove default classifier
        )
        embed_dim = 384  # DeiT-Small feature dimension
        
        self.head = build_classifier_head(embed_dim, num_classes, head)
    
    def forward(self, x):
        features = self.model(x)
        return self.head(features)


class EfficientFormerBinary(nn.Module):
    """
    EfficientFormer-L1 for binary MS classification.
    
    Architecture: Hybrid MetaFormer with efficient token mixer
    Parameters: ~12M (1.03 x ResNet18) — closest parameter match
    Pretrained: ImageNet-1K
    
    Use for: SRQ4 standalone ViT baseline (fair parameter-matched comparison)
    """
    def __init__(self, num_classes=1, head="mlp"):
        super().__init__()
        if timm is None:
            raise ImportError("timm is required for ViT models. Install with: pip install timm")
        
        self.model = timm.create_model(
            'efficientformer_l1',
            pretrained=True,
            num_classes=0  # Remove default classifier
        )
        embed_dim = self.model.num_features  # 448 for L1
        
        self.head = build_classifier_head(embed_dim, num_classes, head)
    
    def forward(self, x):
        features = self.model(x)
        return self.head(features)


#####################################################################################################
########################################### Model Factory ###########################################
#####################################################################################################

def get_model(architecture="base", head="mlp", device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    Factory function to instantiate models by name.

    CNN Architecture keys (ResNet18-based):
        "base"                    — plain ResNet18
        "cbam_end"                — ResNet18 + single CBAM before avgpool
        "cbam_block_pre"          — ResNet18 + CBAM inside each block (pre-shortcut)
        "cbam_block_post"         — ResNet18 + CBAM inside each block (post-shortcut)
        "se_end"                  — ResNet18 + single SE before avgpool
        "se_block_pre"            — ResNet18 + SE inside each block (pre-shortcut)
        "cbam_isolated_end"       — ResNet18 + CBAMIsolated (SE channel + spatial) before avgpool
        "cbam_isolated_block_pre" — ResNet18 + CBAMIsolated inside each block
    
    ViT Architecture keys:
        "deit_small"              — DeiT-Small/16
        "efficientformer"         — EfficientFormer-L1
    
    Note: ViT models require timm library (pip install timm)
    """
    
    # ResNet18-based models
    if architecture == "base":
        model = BaseResNet18(head=head)

    elif architecture == "cbam_end":
        model = CBAMResNet18(cbam_location="end", head=head)
    elif architecture == "cbam_block_pre":
        model = CBAMResNet18(cbam_location="block_pre", head=head)
    elif architecture == "cbam_block_post":
        model = CBAMResNet18(cbam_location="block_post", head=head)

    elif architecture == "se_end":
        model = SEResNet18(se_location="end", head=head)
    elif architecture == "se_block_pre":
        model = SEResNet18(se_location="block_pre", head=head)

    elif architecture == "cbam_isolated_end":
        model = CBAMIsolatedResNet18(cbam_iso_location="end", head=head)
    elif architecture == "cbam_isolated_block_pre":
        model = CBAMIsolatedResNet18(cbam_iso_location="block_pre", head=head)
    
    # ViT-based models
    elif architecture == "deit_small":
        model = DeiTSmallBinary(head=head)
    elif architecture == "efficientformer":
        model = EfficientFormerBinary(head=head)

    else:
        raise ValueError(
            f"Unknown architecture: '{architecture}'. "
            f"Valid CNN options: base, cbam_end, cbam_block_pre, cbam_block_post, "
            f"se_end, se_block_pre, cbam_isolated_end, cbam_isolated_block_pre. "
            f"Valid ViT options: deit_small, efficientformer"
        )

    print(f"get_model()>>> architecture={architecture!r}  head={head!r}")
    return model.to(device)