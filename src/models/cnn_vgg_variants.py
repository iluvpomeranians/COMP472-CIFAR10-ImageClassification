# -----------------------------------------------------------------------------
# VGG11 Architecture Variants (Depth Experiments)
#
# Purpose:
#   Implements lighter and deeper variations of the VGG11 CNN to study how
#   network depth affects performance on CIFAR-10.
#
# Dependencies:
#   - Inherits training/evaluation logic from cnn_vgg11.VGG11
#   - Uses the same CIFAR-10 data loader and optimizer setup
#
# Variants:
#   1. VGG11_Lite  → removes final 512×512 convolution block
#   2. VGG11_Deep  → adds an extra 512×512 convolution block
#
# Expected Outcomes:
#   - Lite: trains faster, fewer params, may underfit
#   - Deep: slower training, potentially higher accuracy but more overfitting risk
# -----------------------------------------------------------------------------

import torch.nn as nn
from src.models.cnn_vgg11 import VGG11


# -------------------------------------------------------------------------
# 1: VGG11_Lite — fewer convolutional layers
# -------------------------------------------------------------------------
class VGG11_Lite(VGG11):
    """Simplified version of VGG11 with one less 512×512 convolution block."""
    def __init__(self):
        super(VGG11_Lite, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # ⚠ Removed one 512×512 block before final pooling
            nn.MaxPool2d(2, 2),
        )


# -------------------------------------------------------------------------
# 2: VGG11_Deep — extra convolutional layer (more depth)
# -------------------------------------------------------------------------
class VGG11_Deep(VGG11):
    """Extended version of VGG11 with an additional 512×512 convolution block."""
    def __init__(self):
        super(VGG11_Deep, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # ⬇️ Extra convolutional block (new depth)
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2),
        )
