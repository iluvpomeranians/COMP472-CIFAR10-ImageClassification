# COMP472
# Project: CIFAR-10 Image Classification with ResNet Features and Naive Bayes
# Authors: David Martinez (29556869) Thomas Kamil Brochet (40121143) ZJ ()

from src.data_pipeline.data_loader import load_cifar10
from torch.utils.data import DataLoader
from src.data_pipeline.run_data_pipeline import main as run_data_pipeline_main
import torch

def check_cuda():
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU.")
        print("GPU in use:", torch.cuda.get_device_name(0))
    else:
        print("CUDA is not available. Using CPU.")

def main():

    check_cuda()
    run_data_pipeline_main()
    print("=== Finished Loading  ===")

    # later steps will go here:
    # 1. Train Naive Bayes
    # 2. Decision tree
    # 3. MLP
    # 4. VGG

if __name__ == "__main__":
    main()
