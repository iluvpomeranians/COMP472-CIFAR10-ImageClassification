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
from src.models.cnn_vgg_variants import VGG11_Lite, VGG11_Deep

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
    gauss_metrics = Metrics.evaluate_model(y_test, y_pred, model_name="Gauss Naive Bayes")

    _, y_pred_scikit = GaussianNaiveBayes.scikit_learn_gaussian_nb(X_train, y_train, X_test)
    scikit_metrics = Metrics.evaluate_model(y_test, y_pred_scikit, model_name="Scikit-learn GaussianNB")
    scikit_cm = Metrics.confusion_matrix(y_test, y_pred_scikit, num_classes=10)
    Metrics.tabulate_confusion_matrix(scikit_cm, matrix_name="Scikit-learn GaussianNB Confusion Matrix", class_labels=classifiers)
    Metrics.export_confusion_matrix(scikit_cm, filename_prefix="scikit_gaussian_nb_confusion_matrix", class_labels=classifiers)

    Metrics.compare_models(gauss_metrics, scikit_metrics)

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

    classifiers = Metrics.extract_classes()
    vgg_cm = Metrics.confusion_matrix(y_test_vgg, y_pred_vgg, num_classes=10)
    Metrics.tabulate_confusion_matrix(vgg_cm, matrix_name="VGG11 Confusion Matrix", class_labels=classifiers)
    Metrics.export_confusion_matrix(vgg_cm, filename_prefix="vgg11_confusion_matrix", class_labels=classifiers)

    vgg11_metrics = Metrics.evaluate_model(y_test_vgg, y_pred_vgg, model_name="VGG11 CNN")
    Metrics.compare_models(gauss_metrics, scikit_metrics, vgg11_metrics)

    VGG11.save_model(Vgg11Model)

    # --- Train VGG11-Lite ---
    print("\n=== Starting VGG11-Lite Training ===")
    VggLiteModel = VGG11_Lite()
    VGG11.vgg_train(VggLiteModel, device, epochs=10)
    VggLiteModel, y_test_lite, y_pred_lite = VGG11.vgg_evaluate(VggLiteModel, device)
    vgglite_cm = Metrics.confusion_matrix(y_test_lite, y_pred_lite, num_classes=10)
    Metrics.tabulate_confusion_matrix(vgglite_cm, matrix_name="VGG11-Lite Confusion Matrix", class_labels=classifiers)
    Metrics.export_confusion_matrix(vgglite_cm, filename_prefix="vgg11_lite_confusion_matrix", class_labels=classifiers)
    vgg11_lite_metrics = Metrics.evaluate_model(y_test_lite, y_pred_lite, model_name="VGG11-Lite CNN")
    VGG11.save_model(VggLiteModel, "./models/trained/vgg11_lite_cifar10.pth")

    # --- Train VGG11-Deep ---
    print("\n=== Starting VGG11-Deep Training ===")
    VggDeepModel = VGG11_Deep()
    VGG11.vgg_train(VggDeepModel, device, epochs=10)
    VggDeepModel, y_test_deep, y_pred_deep = VGG11.vgg_evaluate(VggDeepModel, device)
    vgg11_deep_metrics = Metrics.evaluate_model(y_test_deep, y_pred_deep, model_name="VGG11-Deep CNN")
    vggdeep_cm = Metrics.confusion_matrix(y_test_deep, y_pred_deep, num_classes=10)
    Metrics.tabulate_confusion_matrix(vggdeep_cm, matrix_name="VGG11-Deep Confusion Matrix", class_labels=classifiers)
    Metrics.export_confusion_matrix(vggdeep_cm, filename_prefix="vgg11_deep_confusion_matrix", class_labels=classifiers)
    VGG11.save_model(VggDeepModel, "./models/trained/vgg11_deep_cifar10.pth")

    print("=== Finished VGG11 Training ===")
    # --- Compare all ---
    Metrics.compare_models(gauss_metrics, scikit_metrics, vgg11_metrics, vgg11_lite_metrics, vgg11_deep_metrics)



if __name__ == "__main__":
    main()
