# feature_extract.py
# ------------------------------------------------------------
# Step 4: Use pre-trained ResNet-18 to extract 512-D feature
# vectors from CIFAR-10 images.
# ------------------------------------------------------------

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
import numpy as np
from .data_loader import load_cifar10
import os
os.makedirs("./data/features", exist_ok=True)

# ------------------------------------------------------------
# Helper: extract 512-D features from each image batch
# ------------------------------------------------------------
def extract_features(model, dataloader, device):
    features, labels = [], []

    model.eval()  # make sure we’re in inference mode
    with torch.no_grad():
        for imgs, lbls in dataloader:
            imgs = imgs.to(device)
            outputs = model(imgs)          # shape: (B, 512, 1, 1)
            outputs = outputs.squeeze()    # → (B, 512)
            features.append(outputs.cpu().numpy())
            labels.append(lbls.numpy())

    # merge all batches
    X = np.concatenate(features, axis=0)
    y = np.concatenate(labels, axis=0)
    return X, y


# ------------------------------------------------------------
# Main routine
# ------------------------------------------------------------
def main():
    # 1. Load CIFAR-10 subsets
    print("Loading CIFAR-10 subsets...")
    train_ds, test_ds = load_cifar10()
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False)

    # 2. Load pretrained ResNet-18 and remove final layer
    print("Loading pretrained ResNet-18...")
    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Freeze parameters (we only want to extract features)
    for param in resnet.parameters():
        param.requires_grad = False

    # Remove the classification layer (fc)
    feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

    # 3. Set device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor.to(device)
    print(f"Using device: {device}")

    # 4. Extract train/test features
    print("Extracting training features...")
    X_train, y_train = extract_features(feature_extractor, train_loader, device)
    print("Extracting test features...")
    X_test, y_test   = extract_features(feature_extractor, test_loader, device)

    # 5. Save to disk
    np.savez("./data/features/features_cifar10_resnet18.npz",
             X_train=X_train, y_train=y_train,
             X_test=X_test,  y_test=y_test)
    print("Saved extracted features → features_cifar10_resnet18.npz")
    print(f"Train feature shape: {X_train.shape}")
    print(f"Test  feature shape: {X_test.shape}")


if __name__ == "__main__":
    main()
