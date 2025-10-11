from data_loader import load_cifar10
import pickle, os
from torch.utils.data import DataLoader

# This is just to inspect our databatches in ./data/cifar-10-batches-py folder
def inspect_raw_cifar10(root="./data"):
    batch_path = os.path.join(root, "cifar-10-batches-py", "data_batch_1")
    if os.path.exists(batch_path):
        with open(batch_path, "rb") as f:
            batch = pickle.load(f, encoding="bytes")
        print("Keys:", batch.keys())
        print("Data shape:", batch[b"data"].shape)
        print("Labels sample:", batch[b"labels"][:10])
    else:
        print("Raw CIFAR-10 data not found.")

# So this runs data_loader.py and then inspects the raw data
# We can run this file to see if data loading works fine
def main():
    print("=== Running CIFAR-10 data loader ===")
    train_ds, test_ds = load_cifar10()
    print("Train subset size:", len(train_ds))
    print("Test subset size:", len(test_ds))

    img, label = train_ds[0]
    print("Sample image shape:", img.shape)
    print("Sample label:", label)

    inspect_raw_cifar10()

if __name__ == "__main__":
    main()
