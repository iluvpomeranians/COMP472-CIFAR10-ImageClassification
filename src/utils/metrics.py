# -----------------------------------------------------------------------------
# Confusion Matrix (Evaluation Metric)
#
# Matrix Layout:
#   Rows    → Actual classes
#   Columns → Predicted classes
#
# Each cell M[i, j] represents the number of samples whose
# true label was class i but were predicted as class j.
#
# Example:
#   M[3, 3] = 420  → 420 “class 3” images correctly predicted as class 3
#   M[3, 7] = 50   → 50 “class 3” images incorrectly predicted as class 7
#
# General Structure (for 10 classes):
# |                     | Predicted: 0 | Predicted: 1 | ... | Predicted: 9 |
# | ------------------- | ------------ | ------------ | --- | ------------ |
# | Actual: 0           |     M[0,0]   |     M[0,1]   | ... |     M[0,9]   |
# | Actual: 1           |     M[1,0]   |     M[1,1]   | ... |     M[1,9]   |
# | ...                 |     ...      |     ...      | ... |     ...      |
# | Actual: 9           |     M[9,0]   |     M[9,1]   | ... |     M[9,9]   |
#
# Interpretation:
#   • Strong diagonal values → good classification accuracy per class
#   • Large off-diagonal values → frequent misclassifications
#   • Columns with consistently high totals → model bias toward that class
#
# Purpose:
#   - Visualize how well each class is predicted
#   - Identify which classes are often confused
#   - Evaluate per-class precision, recall, and F1 in later metrics
# -----------------------------------------------------------------------------

import numpy as np
import pandas as pd
import os
import pickle
from tabulate import tabulate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.models.naive_bayes import GaussianNaiveBayes as GaussModel

class Metrics:

    #Grid NxN - Predicted vs Actual
    @staticmethod
    def confusion_matrix(y_true, y_pred, num_classes):
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for true, pred in zip(y_true, y_pred):
            cm[true][pred] += 1
        return cm

    @staticmethod
    def evaluate_model(y_true, y_pred, model_name="Model"):
        accuracy  = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro')
        recall    = recall_score(y_true, y_pred, average='macro')
        f1        = f1_score(y_true, y_pred, average='macro')

        table = [[model_name, accuracy, precision, recall, f1]]
        print(f"\n{model_name} Evaluation Metrics:")
        print(tabulate(table,
                    headers=["Model", "Accuracy", "Precision", "Recall", "F1-Score"],
                    floatfmt=".4f",
                    tablefmt="fancy_grid"))
        return {
            "model": model_name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    @staticmethod
    def compare_models(*models):

        table = [
            [m["model"], m["accuracy"], m["precision"], m["recall"], m["f1"]]
            for m in models
            if m is not None
        ]

        print("\nModel Comparison:")
        print(
            tabulate(
                table,
                headers=["Model", "Accuracy", "Precision", "Recall", "F1-Score"],
                floatfmt=".4f",
                tablefmt="fancy_grid",
            )
        )


    @staticmethod
    def print_confusion_matrix(cm):
        print("Confusion Matrix:")
        print("Predicted →", end="")
        for i in range(cm.shape[1]):
            print(f"\t{i}", end="")
        print()
        for i, row in enumerate(cm):
            print(f"Actual {i} |", end="")
            for val in row:
                print(f"\t{val}", end="")
            print()

    @staticmethod
    def tabulate_confusion_matrix(cm, matrix_name="Confusion Matrix", class_labels=None):
        headers = [f"Pred {i}\n({class_labels[i]})" for i in range(cm.shape[1])]
        index = [f"Act {i}\n({class_labels[i]})" for i in range(cm.shape[0])]
        table = np.column_stack((index, cm))
        print(f"\n{matrix_name}:")
        print(tabulate(table, headers=[" "] + headers, tablefmt="fancy_grid"))

    @staticmethod
    def export_confusion_matrix(cm, filename_prefix="confusion_matrix", class_labels=None):
        df = pd.DataFrame(cm,
                          index=[f"Act {i}\n({class_labels[i]})" for i in range(cm.shape[0])],
                          columns=[f"Pred {i}\n({class_labels[i]})" for i in range(cm.shape[1])])

        csv_path = f"src/utils/results/{filename_prefix}.csv"
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(csv_path, index=True)
        #print(f"Saved CSV to {csv_path}")

        # LaTeX table for report
        tex_path = f"src/utils/results/{filename_prefix}.tex"
        os.makedirs(os.path.dirname(tex_path), exist_ok=True)
        with open(tex_path, "w") as f:
            f.write(df.to_latex(index=True,
                                 float_format="%.0f",
                                 bold_rows=True,
                                 caption=filename_prefix,
                                 label=filename_prefix.replace(" ", "_").lower(),
                                 ))
        #print(f"Saved LaTeX table to {tex_path}")

    @staticmethod
    def extract_classes():
        meta_path = os.path.join("data", "raw", "cifar-10-batches-py", "batches.meta")

        with open(meta_path, "rb") as f:
            meta_dict = pickle.load(f, encoding="latin1")

        class_names = meta_dict["label_names"]
        return class_names