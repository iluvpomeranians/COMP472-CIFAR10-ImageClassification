# -----------------------------------------------------------------------------
# VGG11 Convolutional Neural Network (Model A - from VGG paper, 2014)
#
# Architecture Summary:
#   Conv(3, 64, 3, 1, 1)  → BatchNorm(64)  → ReLU  → MaxPool(2, 2)
#   Conv(64, 128, 3, 1, 1) → BatchNorm(128) → ReLU → MaxPool(2, 2)
#   Conv(128, 256, 3, 1, 1) → BatchNorm(256) → ReLU
#   Conv(256, 256, 3, 1, 1) → BatchNorm(256) → ReLU → MaxPool(2, 2)
#   Conv(256, 512, 3, 1, 1) → BatchNorm(512) → ReLU
#   Conv(512, 512, 3, 1, 1) → BatchNorm(512) → ReLU → MaxPool(2, 2)
#   Conv(512, 512, 3, 1, 1) → BatchNorm(512) → ReLU
#   Conv(512, 512, 3, 1, 1) → BatchNorm(512) → ReLU → MaxPool(2, 2)
#
# Classifier (fully connected):
#   Linear(512 → 4096) → ReLU → Dropout(0.5)
#   Linear(4096 → 4096) → ReLU → Dropout(0.5)
#   Linear(4096 → 10)
#
# Training Setup:
#   - Loss: CrossEntropyLoss
#   - Optimizer: SGD (momentum = 0.9)
#   - Input: CIFAR-10 32×32 RGB images
#
# Concept:
#   The convolutional layers act as stacked "feature extractors" that learn
#   spatial patterns (edges → shapes → objects). The dense layers then map
#   these learned features to class probabilities for prediction.
# -----------------------------------------------------------------------------

import os
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from src.data_pipeline.data_loader import load_cifar10_vgg

class VGG11(nn.Module):
    def __init__(self):
        super(VGG11, self).__init__()
        self.momentum = 0.9
        self.num_classes = 10

        # VG11 architecture
        # Feature extractor (convolutional backbone)
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

            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        # Classifier (fully-connected head)
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(4096, self.num_classes)
        )

    # Defines the how the input tensor flows through the network
    # (image → conv layers → flatten → dense layers → logits)
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch to 1D tensor
        x = self.classifier(x) # produces logits (e.g. output scores for each class)
        return x

    def cross_entropy_loss(self, outputs, labels):
        loss = nn.CrossEntropyLoss()
        result = loss(outputs, labels)
        return result

    @staticmethod
    def vgg_train(model, device="cuda", epochs=10):
        print(f"Training on device: {device}")
        model.to(device)
        train_loader, _ = load_cifar10_vgg()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)

            for batch_idx, (imgs, lbls) in loop:
                imgs, lbls = imgs.to(device), lbls.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, lbls)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
                loop.set_postfix(loss=loss.item())

            avg_loss = running_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f}")

        return model

    @staticmethod
    @staticmethod
    def vgg_evaluate(model, device="cuda"):
        model.to(device)
        model.eval()
        _, test_loader = load_cifar10_vgg()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for imgs, lbls in test_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                outputs = model(imgs)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(lbls.cpu().numpy())

        return model, all_labels, all_preds


    @staticmethod
    def save_model(model, path="./models/trained/vgg11_cifar10.pth"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(model.state_dict(), path)
        print(f"Model saved to {path}")

    def vgg_test():
        pass