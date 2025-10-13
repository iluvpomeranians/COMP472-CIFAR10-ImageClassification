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

import torch
import torch.nn as nn
from src.models.cnn_vgg11 import VGG11


# -------------------------------------------------------------------------
# 1: VGG11_Lite — fewer convolutional layers
# -------------------------------------------------------------------------
class VGG11_Lite(VGG11):
    def __init__(self):
        super(VGG11_Lite, self).__init__()

        # Replace feature extractor with a smaller stack (fewer layers)
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
            nn.MaxPool2d(2, 2)
        )

        # Recompute flatten dimension dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 32, 32)
            out = self.features(dummy_input)
            flatten_dim = out.numel() // out.shape[0]

        # Rebuild classifier to match new flatten size
        self.classifier = nn.Sequential(
            nn.Linear(flatten_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 10)
        )


class VGG11_Deep(VGG11):
    def __init__(self):
        super(VGG11_Deep, self).__init__()


        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 32, 32)
            out = self.features(dummy_input)
            flatten_dim = out.numel() // out.shape[0]

        self.classifier = nn.Sequential(
            nn.Linear(flatten_dim, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 10)
        )