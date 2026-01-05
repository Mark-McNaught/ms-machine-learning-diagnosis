import torch
import torch.nn as nn
from torchvision import models

class BaseResNet18(nn.Module):
    def __init__(self, num_classes=1, freeze_backbone=True):
        super().__init__()

        # Load pre-trained ResNet18 model
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Freeze pretrained layers if specified
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

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


def get_model(architecture="base", device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """ Model factory helper function to instantiate models by name."""
    if architecture == "base":
        model = BaseResNet18()
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    print(f"get_model()>>> \n", model)
    return model.to(device)
