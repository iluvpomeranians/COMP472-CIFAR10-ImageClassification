# COMP472 Fall 2025 – CIFAR-10 Image Classification
### Concordia University — Artificial Intelligence Project

# Contributions
David Martinez – System design, Naive Bayes modules, VGG modules
Thomas Kamil Brochet – Decision Tree modules
Zhan Jun Cheung – MLP modules
All members jointly reviewed results and prepared final documentation.

Student IDs:
David Martinez – 29556869
Thomas Kamil Brochet – 40121143
Zhan Jun Cheung – 40212301

---

This repository contains our team’s implementation for the COMP472 Machine Learning Project, focused on CIFAR-10 image classification using a progression of models:

- Gaussian Naive Bayes
- Decision Tree
- MLP (optional)
- CNNs (VGG11, VGG11-Lite, VGG11-Deep)

We perform feature extraction using ResNet-18, dimensionality reduction using PCA, and evaluation using standardized metrics.

---

# Quick Start (TA-Friendly)

The fastest and recommended way to run our entire project is through Google Colab:

Colab Notebook:
https://colab.research.google.com/drive/1Pm4H2EBUWQVK2KxcXSQV1DRKIROY9sWq?usp=sharing

No installation required.
Runs on GPU.
No dependency issues.

The Colab notebook:
- Automatically clones the repo
- Installs lightweight dependencies
- Uses Colab’s built-in PyTorch
- Runs main.py end-to-end in one click

---

# Local Installation (Optional)

If you prefer running the project locally:

## Prerequisites
- Python 3.10
- pip

## Setup
```
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run the full pipeline
```
python main.py
```

This will:
1. Load CIFAR-10
2. Extract ResNet-18 features (512-D)
3. Apply PCA (50-D)
4. Train and evaluate all models
5. Output metrics and confusion matrices

---

# Repository Structure

```
COMP472-CIFAR10-ImageClassification/
│
├── data/                                 # Auto-created dataset/features folder
│
├── src/
│   ├── data_pipeline/
│   │   ├── data_loader.py
│   │   ├── feature_extractor.py
│   │   ├── pca_reduction.py
│   │   ├── run_data_pipeline.py
│   │   └── __init__.py
│   │
│   ├── models/
│   │   ├── naive_bayes.py
│   │   ├── decision_tree.py
│   │   ├── cnn_vgg11.py
│   │   ├── cnn_vgg_variants.py
│   │   └── __init__.py
│   │
│   ├── utils/
│   │   ├── metrics.py
│   │   └── __init__.py
│   │
│   └── __init__.py
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Pipeline Overview

Running `main.py` performs all steps automatically:

1. Dataset loading
2. ResNet-18 feature extraction
3. PCA to 50 dimensions
4. Train models: Naive Bayes, Decision Tree, CNN (VGG11 + Lite + Deep)
5. Evaluation: confusion matrices, accuracy, precision, recall, F1
6. Optional: save trained models

---

# Team Workflow

- Do not commit `data/` or `__pycache__/`
- All teammates run preprocessing locally or via Colab
- Each model is modular and testable independently
- The Colab notebook serves as the canonical test environment

---

# Notes for the Marker / TA

The Colab notebook is the recommended and official way to run the project.

If testing locally, please use Python 3.10 for compatibility.

---

# Contact

For any issues running the code, please contact the project team.