import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

class RockfallDeepLab(nn.Module):
    def __init__(self, num_classes=6, pretrained=True):
        super().__init__()
        self.model = deeplabv3_mobilenet_v3_large(pretrained=pretrained)
        # Replace classifier head
        in_channels = self.model.classifier[4].in_channels
        self.model.classifier[4] = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        return self.model(x)
