COMP472 Fall 2025 – CIFAR-10 Classification (Naive Bayes to CNN)
Concordia University — Artificial Intelligence Project

This repository contains our team’s implementation for the COMP472 project, focusing on image classification using the CIFAR-10 dataset.
We begin with Naive Bayes (Step 3) and will later expand to Decision Trees, MLP, and CNNs.

Setup & Installation
Prerequisites

Python 3.11+
pip (latest version)

Installation Steps

Upgrade pip and install dependencies:

python -m pip install --upgrade pip
pip install -r requirements.txt


Verify setup by running the data loader:

python run_data_loader.py


This will automatically download the CIFAR-10 dataset into the ./data/ directory.
If successful, you’ll see:

Train subset size: 5000
Test subset size: 1000

📁 Repository Structure
comp472_project/
│
├── data_loader.py          # Loads & subsets CIFAR-10 (500 train + 100 test per class)
├── feature_extract.py      # ResNet-18 feature extraction + PCA reduction
├── naive_bayes.py          # Manual + Scikit-Learn Gaussian Naive Bayes models
├── utils.py                # Helper functions (accuracy, confusion matrix, plotting)
│
├── run_data_loader.py      # Runs & inspects the data loader independently
├── main.py                 # Main pipeline (load → extract → train → evaluate)
│
├── requirements.txt        # Dependencies for all modules
├── .gitignore              # Files and folders excluded from version control
└── README.md               # This documentation


Future files to be added:

decision_tree.py → Step 4 (Gini-based classifier)

mlp.py → Step 5 (3-layer PyTorch MLP)

cnn_vgg11.py → Step 6 (CNN training directly on images)

models/ → Folder to store trained .pth or .pkl models


⚙️ Running the Full Pipeline

Once all modules are implemented, the main entry point will be:

python main.py


This will execute the entire flow:

--Load CIFAR-10 subset (data_loader.py)

--Extract 512-D features using ResNet-18 (feature_extract.py)

--Reduce to 50-D using PCA

--Train and evaluate Naive Bayes models (naive_bayes.py)

--Print metrics and confusion matrices (utils.py)

👥 Team Workflow

Install requirements (only needed once).

Run python main.py to regenerate data and models locally — datasets are not versioned.

Never commit /data/ or /__pycache__/.
