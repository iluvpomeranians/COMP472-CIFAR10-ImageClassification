# run_data_pipeline.py
# ------------------------------------------------------------
# Full data preprocessing pipeline:
#  1. Load & verify CIFAR-10 subset
#  2. Extract 512-D features using ResNet-18
#  3. Reduce to 50-D with PCA
# ------------------------------------------------------------

import os
import pickle
import subprocess
from torch.utils.data import DataLoader
from .data_loader import load_cifar10

# ------------------------------------------------------------
# Utility: quick inspection of CIFAR-10 raw data
# ------------------------------------------------------------
def inspect_raw_cifar10():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    root = os.path.join(project_root, "data/raw")

    batch_path = os.path.join(root, "cifar-10-batches-py", "data_batch_1")
    if os.path.exists(batch_path):
        with open(batch_path, "rb") as f:
            batch = pickle.load(f, encoding="bytes")
        print("  ➜ CIFAR-10 data keys:", list(batch.keys()))
        print("  ➜ Data shape:", batch[b"data"].shape)  # (10000, 3072)
        print("  ➜ Label samples:", batch[b"labels"][:10])
    else:
        print("Error: Raw CIFAR-10 data not found.")


# ------------------------------------------------------------
# Helper to run a step as subprocess (sequentially)
# ------------------------------------------------------------
def run_script(module_name):
    print(f"\n=== Running {module_name} ===")
    subprocess.run(["python", "-m", module_name], check=True)
    print(f"Finished {module_name}")


# ------------------------------------------------------------
# Main orchestrator
# ------------------------------------------------------------
def main():
    print("INFO: Starting full data pipeline...\n")

    # STEP 1 — Load CIFAR-10 subsets
    print("=== Step 1: Loading CIFAR-10 ===")
    train_ds, test_ds = load_cifar10()
    print(f"  ➜ Train subset size: {len(train_ds)}")
    print(f"  ➜ Test subset size: {len(test_ds)}")

    # Peek at one example (for verification)
    img, label = train_ds[2001]
    print("  ➜ Sample image shape:", img.shape)
    print("  ➜ Sample label:", label)

    # Verify raw data
    inspect_raw_cifar10()

    base_dir = os.path.dirname(__file__)

    # STEP 2 — Extract 512-D ResNet-18 features
    run_script("src.data_pipeline.feature_extractor")

    # STEP 3 — Reduce to 50-D using PCA
    run_script("src.data_pipeline.pca_reduction")

    print("\nData pipeline completed successfully!")
    print("Output saved to ./data/features/")
    print("   - features_cifar10_resnet18.npz")
    print("   - features_cifar10_resnet18_pca50.npz")


if __name__ == "__main__":
    main()
