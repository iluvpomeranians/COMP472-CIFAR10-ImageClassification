# COMP472
# Project: CIFAR-10 Image Classification with ResNet Features and Naive Bayes
# Authors: David Martinez (29556869) Tomas() ZJ ()

from data_loader import load_cifar10
from torch.utils.data import DataLoader
from run_data_loader import main as run_data_loader_main

def main():

    run_data_loader_main()  # This will run the data loading and inspection
    print("=== Finished Loading  ===")

    # later steps will go here:
    # 1. Extract features (ResNet)
    # 2. Reduce with PCA
    # 3. Train Naive Bayes
    # 4. Evaluate

if __name__ == "__main__":
    main()
