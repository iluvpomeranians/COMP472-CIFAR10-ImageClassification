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

means = {}
variances = {}
priors = {}
def train_gaussian_bayes(X_train, y_train):
    num_classes = 10
    for c in range(num_classes):
        X_c = X_train[y_train == c]
        means[c] = np.mean(X_c, axis=0)
        variances[c] = np.var(X_c, axis=0)
        priors[c] = np.sum(y_train == c) / len(y_train)
    return means, variances, priors

#TODO: need to fix and simplify predict function
def predict_gaussian_bayes(X_test, y_test, means, variances, priors):
    print(f"X_test: {X_test.shape}  y_test: {y_test.shape}")
    num_imgs = X_test.shape[0]
    num_features = X_test.shape[1]
    num_classes = 10
    current_log_probability = np.zeros((num_imgs, num_classes))
    y_predicitons = np.zeros(num_imgs)

    for c in range(num_classes):
        mean_c = means[c]
        variance_c = variances[c]
        prior_c = np.log(priors[c])

        for img_index in range(num_imgs):
            # print(f"Class {c}, Image {img_index}: {X_test[img_index]}")
            sum_log_gaussian_density = 0.0

            for feature_index in range(num_features):
                curr_mean = mean_c[feature_index]
                curr_variance = variance_c[feature_index]

                gaussian_probability_density = P_xi_given_y(X_test[img_index][feature_index], curr_mean, curr_variance)
                # print(f"Feature {feature_index}: mean={curr_mean}, variance={curr_variance}")
                # print(f"P(x_i|y) = {gaussian_probability_density, curr_mean, curr_variance}")
                sum_log_gaussian_density += np.log(gaussian_probability_density + 1e-9)

            current_log_probability[img_index][c] = prior_c + sum_log_gaussian_density
            # print(f"Total probability for class {c}: { np.exp(current_log_probability[img_index][c])}")
            if c == (num_classes - 1):
                y_prediction = np.argmax(current_log_probability[img_index])
                current_probability = np.exp(current_log_probability[img_index][y_prediction])
                print(f"Predicted class for image {img_index}: {y_prediction} with probability {current_probability}")
                y_predicitons[img_index] = y_prediction

    return y_predicitons

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_50npz()
    mean_c, variance_c, priors = train_gaussian_bayes(X_train, y_train)
    y_predictions = predict_gaussian_bayes(X_test, y_test, mean_c, variance_c, priors)

    acc = np.mean(y_predictions == y_test)
    print(f"Accuracy: {acc*100:.2f}%")