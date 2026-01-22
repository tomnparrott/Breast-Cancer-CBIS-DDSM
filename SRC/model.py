import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

# Load ResNet, adapt for 1 channel input and binary output
def make_resnet18_binary(pretrained: bool = True) -> nn.Module:
    # If True, use ImageNet weights 
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    # Store the existing 3 channel conv layer
    old_conv = model.conv1
    # Replace with a 1 channel conv layer with the same settings
    model.conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=(old_conv.bias is not None),
    )
    # If pretrained, initialise the new conv weights by averaging RGB weights
    if pretrained and weights is not None:
        with torch.no_grad():
            model.conv1.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)

    # Replace classifier head and output a single logit
    model.fc = nn.Linear(model.fc.in_features, 1)

    return model
