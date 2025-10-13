# COMP472
# Project: CIFAR-10 Image Classification with ResNet Features and Naive Bayes
# Authors: David Martinez (29556869) Thomas Kamil Brochet (40121143) ZJ ()

import torch
from torch.utils.data import DataLoader
from src.data_pipeline.data_loader import load_cifar10
from src.data_pipeline.run_data_pipeline import main as run_data_pipeline_main
from src.utils.metrics import Metrics
from src.models.naive_bayes import GaussianNaiveBayes
from src.models.cnn_vgg11 import VGG11

def check_cuda():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU.")
        print("GPU in use:", torch.cuda.get_device_name(0))
    else:
        print("CUDA is not available. Using CPU.")

    return device

def main():
    #---------------------#
    # 0. Setup
    #---------------------#
    print("=== Starting Loading ===")
    device = check_cuda()
    run_data_pipeline_main()
    print("=== Finished Loading  ===")

    #---------------------#
    # 1. Naive Bayes
    #---------------------#
    GaussModel = GaussianNaiveBayes()
    X_train, y_train, X_test, y_test = GaussModel.load_50npz()
    mean_c, variance_c, priors = GaussModel.fit(X_train, y_train)
    y_pred = GaussModel.predict_gaussian_bayes_v2(X_test, mean_c, variance_c, priors)

    classifiers = Metrics.extract_classes()
    naive_gauss_cm = Metrics.confusion_matrix(y_test, y_pred, num_classes=10)
    Metrics.tabulate_confusion_matrix(naive_gauss_cm, matrix_name="Gauss Naive Bayes Confusion Matrix", class_labels=classifiers)
    Metrics.export_confusion_matrix(naive_gauss_cm, filename_prefix="naive_bayes_confusion_matrix", class_labels=classifiers)
    model1 = Metrics.evaluate_model(y_test, y_pred, model_name="Gauss Naive Bayes")

    _, y_pred_scikit = GaussianNaiveBayes.scikit_learn_gaussian_nb(X_train, y_train, X_test)
    model2 = Metrics.evaluate_model(y_test, y_pred_scikit, model_name="Scikit-learn GaussianNB")

    Metrics.compare_models(model1, model2)

    #---------------------#
    # 2. Decision tree
    #---------------------#
    #TODO

    #---------------------#
    # 3. MLP
    #---------------------#
    #TODO

    #---------------------#
    # 4. CNN-VGG11
    #---------------------#
    print("\n=== Starting VGG11 Training ===")
    Vgg11Model = VGG11()
    VGG11.vgg_train(Vgg11Model, device, epochs=10)
    Vgg11Model, y_test_vgg, y_pred_vgg = VGG11.vgg_evaluate(Vgg11Model, device)

    # Metrics + Confusion Matrix (just like Naive Bayes)
    classifiers = Metrics.extract_classes()
    vgg_cm = Metrics.confusion_matrix(y_test_vgg, y_pred_vgg, num_classes=10)
    Metrics.tabulate_confusion_matrix(vgg_cm, matrix_name="VGG11 Confusion Matrix", class_labels=classifiers)
    Metrics.export_confusion_matrix(vgg_cm, filename_prefix="vgg11_confusion_matrix", class_labels=classifiers)

    model3 = Metrics.evaluate_model(y_test_vgg, y_pred_vgg, model_name="VGG11 CNN")
    Metrics.compare_models(model1, model2, model3)

    VGG11.save_model(Vgg11Model)
    print("=== Finished VGG11 Training ===")

if __name__ == "__main__":
    main()
