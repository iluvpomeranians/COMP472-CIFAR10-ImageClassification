#TODO: Naive Bayes model

# | Equation Term                      | What it corresponds to in your project               |                                                                |
# | ---------------------------------- | ---------------------------------------------------- | -------------------------------------------------------------- |
# | ( y )                              | One of the 10 CIFAR-10 classes (from `y_train`)      |                                                                |
# | ( x_i )                            | One of the 50 PCA features (from a row in `X_train`) |                                                                |
# | ( P(y) )                           | Class frequency (counts of `y_train` values)         |                                                                |
# | ( P(x_i                            | y) )                                                 | Gaussian distribution (mean/variance) of feature i for class y |
# | ( P(y                              | x) )                                                 | Posterior probability used for prediction                      |
# | `GaussianNB.fit(X_train, y_train)` | Internally estimates all those means and variances   |                                                                |
# | `GaussianNB.predict(X_test)`       | Applies the formula for each test vector             |                                                                |

# “If this were a cat, how likely would it be to see these 50 feature values?”
# “If it were a dog, how likely?”
# …and so on for all 10 classes.

import numpy as np
import torch
from torch.utils.data import DataLoader

def load_50npz():
    data = np.load("./data/features/features_cifar10_resnet18_pca50.npz")
    X_train, y_train = data["X_train"], data["y_train"]
    X_test,  y_test  = data["X_test"],  data["y_test"]
    print (f"Loaded PCA-reduced features: X_train={X_train.shape}, X_test={X_test.shape}")
    return X_train, y_train, X_test, y_test

def P_xi_given_y(x_i, mean, variance):
    # Gaussian probability density function
    coeff = 1.0 / np.sqrt(2.0 * np.pi * variance)
    exponent = np.exp(-((x_i - mean) ** 2) / (2 * variance))
    return coeff * exponent

def gaussian_naive_bayes(X_train, y_train):
    X_c = None
    mean_c = None
    variance_c = None
    probability_y = {}
    total_log_probability = {}
    num_classes = 10
    print(f"X_train: {X_train.shape}  y_train: {y_train.shape}")

    for c in range(num_classes):
        X_c = X_train[y_train == c]
        mean_c = np.mean(X_c, axis=0)
        variance_c = np.var(X_c, axis=0)
        num_imgs = X_c.shape[0]
        num_features = X_train.shape[1]
        probability_y[c] = np.sum(y_train == c) / len(y_train)

        for img_index in range(num_imgs):
            print(f"Class {c}, Image {img_index}: {X_c[img_index]}")
            sum_log_gaussian_density = 0.0

            for feature_index in range(num_features):
                curr_mean = mean_c[feature_index]
                curr_variance = variance_c[feature_index]

                gaussian_probability_density = P_xi_given_y(X_c[img_index][feature_index], curr_mean, curr_variance)
                # print(f"Feature {feature_index}: mean={curr_mean}, variance={curr_variance}")
                # print(f"P(x_i|y) = {gaussian_probability_density, curr_mean, curr_variance}")
                sum_log_gaussian_density += np.log(gaussian_probability_density + 1e-9)  # Avoid log(0)
            total_log_probability[c] = np.log(probability_y[c]) + sum_log_gaussian_density
            print(f"Total probability for class {c}: { np.exp(total_log_probability[c])}")
    return

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_50npz()
    gaussian_naive_bayes(X_train, y_train)