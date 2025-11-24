# -----------------------------------------------------------------------
# Multi-Layer Perceptron (MLP) with Variants:
#   - MLP_Base   (512 → 512)
#   - MLP_Shallow (512 only)
#   - MLP_Deep   (512 → 512 → 512)
# -----------------------------------------------------------------------
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import os


# ================================================================
# FACTORY: Create different MLP variants
# ================================================================
def create_mlp(model_type="base"):
    if model_type.lower() == "shallow":
        return MLP_Shallow()
    elif model_type.lower() == "deep":
        return MLP_Deep()
    else:
        return MLP_Base()


# ================================================================
# BASE CLASS (shared training, testing, evaluation)
# ================================================================
class MLP_BaseClass(nn.Module):

    @staticmethod
    def load_50npz():
        data = np.load("./data/features/features_cifar10_resnet18_pca50.npz")
        X_train, y_train = data["X_train"], data["y_train"]
        X_test,  y_test  = data["X_test"],  data["y_test"]
        print(f"Loaded PCA-reduced features: X_train={X_train.shape}, X_test={X_test.shape}")
        return X_train, y_train, X_test, y_test

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        return self.features(x)

    # ------------------------- TRAIN -------------------------
    def mlp_training(self, device="cuda", epoch_num=20):
        print(f"[C] Training {self.model_name} with {device}...")

        self.to(device)

        X_train, y_train, X_test, y_test = self.load_50npz()
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)

        train_load = DataLoader(TensorDataset(X_train, y_train),
                                batch_size=128, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=0.01, momentum=0.9)

        for epoch in range(epoch_num):
            self.train()
            running_loss = 0

            for batch_idx, (imgs, lbls) in enumerate(
                    tqdm(train_load, desc=f"Training-{self.model_name}", leave=False)):
                imgs, lbls = imgs.to(device), lbls.to(device)

                optimizer.zero_grad()
                loss = criterion(self(imgs), lbls)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            tqdm.write(f"{self.model_name} Epoch [{epoch+1}/{epoch_num}] "
                       f"Avg Loss: {running_loss / len(train_load):.4f}")

    def mlp_evaluate(self, device="cuda"):
        self.to(device)
        self.eval()

        _, _, X_test, y_test = self.load_50npz()
        X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test = torch.tensor(y_test, dtype=torch.long)

        test_load = DataLoader(TensorDataset(X_test, y_test), batch_size=128)

        preds, labels = [], []

        with torch.no_grad():
            for imgs, lbls in test_load:
                imgs = imgs.to(device)
                out = self(imgs)
                _, pred = torch.max(out, 1)
                preds.extend(pred.cpu().numpy())
                labels.extend(lbls.numpy())

        return np.array(labels), np.array(preds)


    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f"Saved {self.model_name} → {path}")


# ================================================================
# 1) BASE MODEL (512 → 512)
# ================================================================
class MLP_Base(MLP_BaseClass):
    def __init__(self):
        super().__init__()
        self.model_name = "MLP_Base"
        self.features = nn.Sequential(
            nn.Linear(50, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 10)
        )


# ================================================================
# 2) SHALLOW MODEL (512 only)
# ================================================================
class MLP_Shallow(MLP_BaseClass):
    def __init__(self):
        super().__init__()
        self.model_name = "MLP_Shallow"
        self.features = nn.Sequential(
            nn.Linear(50, 512), nn.ReLU(),
            nn.Linear(512, 10)
        )


# ================================================================
# 3) DEEP MODEL (512 → 512 → 512)
# ================================================================
class MLP_Deep(MLP_BaseClass):
    def __init__(self):
        super().__init__()
        self.model_name = "MLP_Deep"
        self.features = nn.Sequential(
            nn.Linear(50, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 10)
        )


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = create_mlp("base")
    model.mlp_training(device=device, epoch_num=20)
    y_test, y_pred = model.mlp_evaluate(device=device)
    model.save_model("./src/models/trained/mlp_base.pth")
