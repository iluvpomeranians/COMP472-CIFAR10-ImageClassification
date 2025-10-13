# -----------------------------------------------------------------------------
# Gaussian (Normal) Probability Density Function:
#
#           1
# P(x_i|y) = ----------------------------- * exp( - (x_i - μ_y)^2 / (2 * σ_y^2) )
#             √(2 * π * σ_y^2)
#
# where:
#   x_i   = value of feature i for a given sample
#   μ_y   = mean of feature i for class y
#   σ_y^2 = variance of feature i for class y
#
# In Naive Bayes:
#   - Each feature i is assumed independent given the class y
#   - The total class likelihood is the product of all P(x_i|y)
#   - To avoid underflow, we use log-probabilities:
#         log P(y|x) = log P(y) + Σ_i log P(x_i|y)
# -----------------------------------------------------------------------------
#
# “If this were a cat, how likely would it be to see these 50 feature values?”
# “If it were a dog, how likely?”
# …and so on for all 10 classes.

import numpy as np
from sklearn.naive_bayes import GaussianNB

class GaussianNaiveBayes:
    def __init__(self):
        self.means = {}
        self.variances = {}
        self.priors = {}
        self.num_classes = 10

    def _gaussian_pdf(self, x_i, mean, variance):
        # Gaussian probability density function
        coeff = 1.0 / np.sqrt(2.0 * np.pi * variance)
        exponent = np.exp(-((x_i - mean) ** 2) / (2 * variance))
        return coeff * exponent

    def load_50npz(self):
        data = np.load("./data/features/features_cifar10_resnet18_pca50.npz")
        X_train, y_train = data["X_train"], data["y_train"]
        X_test,  y_test  = data["X_test"],  data["y_test"]
        print (f"Loaded PCA-reduced features: X_train={X_train.shape}, X_test={X_test.shape}")
        return X_train, y_train, X_test, y_test


    def fit(self, X_train, y_train):
        num_classes = 10
        for c in range(num_classes):
            X_c = X_train[y_train == c]
            self.means[c] = np.mean(X_c, axis=0)
            self.variances[c] = np.var(X_c, axis=0)
            self.priors[c] = np.sum(y_train == c) / len(y_train)
        return self.means, self.variances, self.priors

    # First version with detailed logging
    def predict_gaussian_bayes_v1(self, X_test, means, variances, priors):
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

                    gaussian_probability_density = self._gaussian_pdf(X_test[img_index][feature_index], curr_mean, curr_variance)
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

    # Version 2:
    # This is a cleaner version of the predict function above
    # It does the same thing but without the nested loops
    def predict_gaussian_bayes_v2(self, X_test, means, variances, priors):
        num_classes = len(means)
        num_imgs, num_features = X_test.shape
        log_probs = np.zeros((num_imgs, num_classes))

        for c in range(num_classes):
            mean_c = means[c]
            var_c = variances[c]
            prior_c = np.log(priors[c])

            gaussian = self._gaussian_pdf(X_test, mean_c, var_c)

            log_likelihoods = np.sum(np.log(gaussian + 1e-9), axis=1)
            log_probs[:, c] = prior_c + log_likelihoods

        y_pred = np.argmax(log_probs, axis=1)
        return y_pred

    def scikit_learn_gaussian_nb(X_train, y_train, X_test):
        # model = GaussianNB()
        # model.fit(X_train, y_train)
        # return model.predict(X_test)
        pass

    #TODO:
    # 0) Repeat with Scikit-learn's GaussianNB and compare results.
    # 2) Summarize your findings in a table detailing the metrics accuracy, precision, recall, and F1- measure.
    #    The table must have separate rows for the four models and their variants.

if __name__ == "__main__":
    model = GaussianNaiveBayes()
    X_train, y_train, X_test, y_test = GaussianNaiveBayes.load_50npz()
    mean_c, variance_c, priors = GaussianNaiveBayes.fit(model, X_train, y_train)
    y_predictions = GaussianNaiveBayes.predict_gaussian_bayes_v2(model, X_test, mean_c, variance_c, priors)

    acc = np.mean(y_predictions == y_test)
    print(f"Accuracy: {acc*100:.2f}%")
