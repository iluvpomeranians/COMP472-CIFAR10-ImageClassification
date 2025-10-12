# pca_reduction.py
# ------------------------------------------------------------
# Reduce 512-D features to 50-D using PCA (scikit-learn)
# ------------------------------------------------------------

import numpy as np
from sklearn.decomposition import PCA
import os

def main():
    # 1. Load previously extracted features
    data = np.load("./data/features/features_cifar10_resnet18.npz")
    X_train, y_train = data["X_train"], data["y_train"]
    X_test,  y_test  = data["X_test"],  data["y_test"]

    print(f"Loaded features: X_train={X_train.shape}, X_test={X_test.shape}")

    # 2. Fit PCA on training set
    pca = PCA(n_components=50)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca  = pca.transform(X_test)

    print(f"Reduced feature shapes: X_train_pca={X_train_pca.shape}, X_test_pca={X_test_pca.shape}")

    # 3. Save reduced features
    os.makedirs("./data/features", exist_ok=True)
    np.savez("./data/features/features_cifar10_resnet18_pca50.npz",
             X_train=X_train_pca, y_train=y_train,
             X_test=X_test_pca,   y_test=y_test)

    print("Saved PCA-reduced features → features_cifar10_resnet18_pca50.npz")

if __name__ == "__main__":
    main()
