# COMP472
# Project: CIFAR-10 Image Classification with ResNet Features and Naive Bayes
# Authors: David Martinez (29556869) Thomas Kamil Brochet (40121143) ZJ ()

from src.data_pipeline.data_loader import load_cifar10
from torch.utils.data import DataLoader
from src.data_pipeline.run_data_pipeline import main as run_data_pipeline_main
import torch
from src.utils.metrics import Metrics
from src.models.naive_bayes import GaussianNaiveBayes

def check_cuda():
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU.")
        print("GPU in use:", torch.cuda.get_device_name(0))
    else:
        print("CUDA is not available. Using CPU.")

def main():

    # check_cuda()
    # run_data_pipeline_main()
    print("=== Finished Loading  ===")

    # later steps will go here:
    # 1. Naive Bayes
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

    # 2. Decision tree
    # 3. MLP
    # 4. VGG

if __name__ == "__main__":
    main()
