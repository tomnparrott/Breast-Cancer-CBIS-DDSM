import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def make_resnet18_binary(pretrained: bool = True) -> nn.Module:
    # If True, use ImageNet weights
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)

    # Replace classifier head and output a single logit
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(model.fc.in_features, 1),
    )

    return model