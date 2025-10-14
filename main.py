# COMP472
# Project: CIFAR-10 Image Classification with ResNet Features and Naive Bayes
# Authors: David Martinez (29556869) Thomas Kamil Brochet (40121143) ZJ ()

import os
import torch
from src.data_pipeline.run_data_pipeline import main as run_data_pipeline_main
from src.utils.metrics import Metrics
from src.models.naive_bayes import GaussianNaiveBayes
from src.models.cnn_vgg11 import VGG11
from src.models.cnn_vgg_variants import VGG11_Lite, VGG11_Deep


def check_cuda():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available. Using CPU.")
    return device


def main():
    #---------------------#
    # 0. Setup
    #---------------------#
    print("=== Starting Setup ===")
    device = check_cuda()

    # Check if processed CIFAR-10 feature file already exists
    npz_path = "./data/features/features_cifar10_resnet18_pca50.npz"
    if os.path.exists(npz_path):
        print(f"[✓] Found cached feature file: {npz_path}")
    else:
        print("[⟳] Feature file not found — running data pipeline...")
        run_data_pipeline_main()

    print("=== Finished Setup ===\n")

    #---------------------#
    # 1. Naive Bayes
    #---------------------#
    print("=== Starting Naive Bayes Evaluation ===")
    GaussModel = GaussianNaiveBayes()
    X_train, y_train, X_test, y_test = GaussModel.load_50npz()
    mean_c, variance_c, priors = GaussModel.fit(X_train, y_train)
    y_pred = GaussModel.predict_gaussian_bayes_v2(X_test, mean_c, variance_c, priors)

    classifiers = Metrics.extract_classes()
    nb_cm = Metrics.confusion_matrix(y_test, y_pred, num_classes=10)
    Metrics.tabulate_confusion_matrix(nb_cm, "Gauss Naive Bayes Confusion Matrix", classifiers)
    Metrics.export_confusion_matrix(nb_cm, "naive_bayes_confusion_matrix", classifiers)
    gauss_metrics = Metrics.evaluate_model(y_test, y_pred, "Gauss Naive Bayes")

    _, y_pred_scikit = GaussianNaiveBayes.scikit_learn_gaussian_nb(X_train, y_train, X_test)
    scikit_metrics = Metrics.evaluate_model(y_test, y_pred_scikit, "Scikit-learn GaussianNB")

    Metrics.compare_models(gauss_metrics, scikit_metrics)

    #---------------------#
    # 2. Decision Tree
    #---------------------#
    print("\n=== Decision Tree Classifier ===")
    # TODO: Implement Decision Tree Classifier using Scikit-learn
    # - Use features from load_50npz()
    # - Train DecisionTreeClassifier()
    # - Evaluate accuracy, precision, recall, F1, confusion matrix
    # - Save metrics and append to comparison table

    #---------------------#
    # 3. MLP Classifier
    #---------------------#
    print("\n=== Multi-Layer Perceptron (MLP) ===")
    # TODO: Implement MLP (using either PyTorch or Scikit-learn MLPClassifier)
    # - Input size matches ResNet features (512)
    # - Hidden layers: e.g. 256 → 128 → 10
    # - Use ReLU activations, CrossEntropyLoss, SGD/Adam
    # - Save model, evaluate, and append metrics

    #---------------------#
    # 4. CNN Models (VGG11 + Variants)
    #---------------------#
    classifiers = Metrics.extract_classes()

    # --- VGG11 ---
    print("\n=== VGG11 ===")
    vgg11_path = "./src/models/trained/vgg11_cifar10.pth"
    Vgg11Model = VGG11()

    if os.path.exists(vgg11_path):
        print(f"[✓] Found pretrained VGG11: {vgg11_path}")
        state_dict = torch.load(vgg11_path, map_location=device)
        Vgg11Model.load_state_dict(state_dict)
    else:
        print("[⟳] Training VGG11 from scratch...")
        VGG11.vgg_train(Vgg11Model, device, epochs=10)
        VGG11.save_model(Vgg11Model, vgg11_path)

    Vgg11Model, y_test_vgg, y_pred_vgg = VGG11.vgg_evaluate(Vgg11Model, device)
    vgg11_metrics = Metrics.evaluate_model(y_test_vgg, y_pred_vgg, "VGG11 CNN")
    vgg_cm = Metrics.confusion_matrix(y_test_vgg, y_pred_vgg, num_classes=10)
    Metrics.export_confusion_matrix(vgg_cm, "vgg11_confusion_matrix", classifiers)

    # --- VGG11-Lite ---
    print("\n=== VGG11-Lite ===")
    vgg11_lite_path = "./src/models/trained/vgg11_lite_cifar10.pth"
    VggLiteModel = VGG11_Lite()

    if os.path.exists(vgg11_lite_path):
        print(f"[✓] Found pretrained VGG11-Lite: {vgg11_lite_path}")
        VggLiteModel.load_state_dict(torch.load(vgg11_lite_path, map_location=device))
    else:
        print("[⟳] Training VGG11-Lite from scratch...")
        VGG11.vgg_train(VggLiteModel, device, epochs=10)
        VGG11.save_model(VggLiteModel, vgg11_lite_path)

    VggLiteModel, y_test_lite, y_pred_lite = VGG11.vgg_evaluate(VggLiteModel, device)
    vgg11_lite_metrics = Metrics.evaluate_model(y_test_lite, y_pred_lite, "VGG11-Lite CNN")

    # --- VGG11-Deep ---
    print("\n=== VGG11-Deep ===")
    vgg11_deep_path = "./src/models/trained/vgg11_deep_cifar10.pth"
    VggDeepModel = VGG11_Deep()

    if os.path.exists(vgg11_deep_path):
        print(f"[✓] Found pretrained VGG11-Deep: {vgg11_deep_path}")
        VggDeepModel.load_state_dict(torch.load(vgg11_deep_path, map_location=device))
    else:
        print("[⟳] Training VGG11-Deep from scratch...")
        VGG11.vgg_train(VggDeepModel, device, epochs=10)
        VGG11.save_model(VggDeepModel, vgg11_deep_path)

    VggDeepModel, y_test_deep, y_pred_deep = VGG11.vgg_evaluate(VggDeepModel, device)
    vgg11_deep_metrics = Metrics.evaluate_model(y_test_deep, y_pred_deep, "VGG11-Deep CNN")

    #---------------------#
    # 5. Final Comparison
    #---------------------#
    print("\n=== Final Model Comparison ===")
    Metrics.compare_models(
        gauss_metrics,
        scikit_metrics,
        vgg11_metrics,
        vgg11_lite_metrics,
        vgg11_deep_metrics
    )


if __name__ == "__main__":
    main()
