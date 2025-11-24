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
from src.models.decision_tree import DTree
from src.models.mlp import mlp


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

    max_depth = 50
    min_samples_split = 2
    min_samples_leaf = 5
    max_features = None
    random_state = 50

    print("\n=== Decision Tree Classifier ===")
    DtreeModel = DTree()
    x_train, Y_train, x_test,Y_test =DtreeModel.load_50npz()
    tree,n_classes= DtreeModel.train_decision_tree_gini( x_train, Y_train,max_depth=max_depth,min_samples_split=min_samples_split,min_impurity_decrease=0.0)
    y_pred = DtreeModel.predict(tree,X_test)
    acc=DtreeModel.accuracy(y_test, y_pred)
    print(f"\n Accuracy(decision tree with Gini, max_depth={max_depth}): ",acc)
    print("\n[PIPELINE] Training scikit-learn DecisionTreeClassifier...")
    sk_clf = DtreeModel.train_sklearn_decision_tree(x_train, Y_train,max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,max_features=max_features,criterion='gini',random_state=random_state)
    y_pred_sklearn = sk_clf.predict(x_test)
    acc_sklearn = DtreeModel.accuracy(Y_test, y_pred_sklearn)
    print(f"[PIPELINE] Scikit-learn tree Accuracy: {acc_sklearn:.4f}")
    classifiers = Metrics.extract_classes()
    Dtree_cm = Metrics.confusion_matrix(y_test, y_pred, n_classes)
    Metrics.tabulate_confusion_matrix(Dtree_cm, "Decision tree Confusion Matrix", classifiers)
    Metrics.export_confusion_matrix(Dtree_cm, "decision_tree_confusion_matrix", classifiers)
    Dtree_metrics = Metrics.evaluate_model(y_test, y_pred, "Decision Tree")
    scikit_metrics = Metrics.evaluate_model(y_test, y_pred_sklearn, "Scikit-learn Decision Tree")
    Metrics.compare_models(Dtree_metrics, scikit_metrics)

    #---------------------#
    # 3. MLP Classifier
    #---------------------#
    print("\n=== Multi-Layer Perceptron (MLP) ===")
    mlp_path = "./src/models/trained/mlp_cifar10.pth"
    MLPModel = mlp()

    if os.path.exists(mlp_path):
        print(f"[✓] Found pretrained MLP: {mlp_path}")
        state_dict = torch.load(mlp_path, map_location=device)
        MLPModel.load_state_dict(state_dict)
    else:
        print("[⟳] Training MLP from scratch...")
        MLPModel.mlp_training(device=device, epoch_num=20)
        MLPModel.save_model(mlp_path)

    # Evaluate
    y_test_mlp, y_pred_mlp = MLPModel.mlp_evaluate(device=device)
    mlp_metrics = Metrics.evaluate_model(y_test_mlp, y_pred_mlp, "MLP Classifier")
    mlp_cm = Metrics.confusion_matrix(y_test_mlp, y_pred_mlp, num_classes=10)
    Metrics.tabulate_confusion_matrix(mlp_cm, "MLP Confusion Matrix", classifiers)
    Metrics.export_confusion_matrix(mlp_cm, "mlp_confusion_matrix", classifiers)


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

    # --- VGG11-Lite (Kernel Size Experiments) ---
    print("\n=== VGG11-Lite: Kernel Size Experiments ===")
    kernel_sizes = [2, 3, 5, 7]
    vgg_kernel_metrics = []

    for k in kernel_sizes:
        print(f"\n--- VGG11-Lite (kernel={k}×{k}) ---")
        model_path = f"./src/models/trained/vgg11_lite_k{k}_cifar10.pth"
        VggLiteK = VGG11_Lite(kernel_size=k)

        # Check for pretrained weights
        if os.path.exists(model_path):
            print(f"[✓] Found pretrained model: {model_path}")
            VggLiteK.load_state_dict(torch.load(model_path, map_location=device))
        else:
            print(f"[⟳] Training VGG11-Lite (kernel={k}) from scratch...")
            VGG11.vgg_train(VggLiteK, device, epochs=10)
            VGG11.save_model(VggLiteK, model_path)

        # Evaluate
        VggLiteK, y_test_k, y_pred_k = VGG11.vgg_evaluate(VggLiteK, device)
        metrics_k = Metrics.evaluate_model(y_test_k, y_pred_k, f"VGG11-Lite (kernel={k}) CNN")
        cm_k = Metrics.confusion_matrix(y_test_k, y_pred_k, num_classes=10)

        # Display and export confusion matrix
        Metrics.tabulate_confusion_matrix(cm_k, f"VGG11-Lite (kernel={k}) CNN", classifiers)
        Metrics.export_confusion_matrix(cm_k, f"vgg11_lite_k{k}_confusion_matrix", classifiers)

        vgg_kernel_metrics.append(metrics_k)



    #---------------------#
    # 5. Final Comparison
    #---------------------#
    print("\n=== Final Model Comparison ===")
    Metrics.compare_models(
        gauss_metrics,
        Dtree_metrics,
        mlp_metrics,
        vgg11_metrics,
        vgg11_lite_metrics,
        vgg11_deep_metrics,
        *vgg_kernel_metrics
    )

if __name__ == "__main__":
    main()
