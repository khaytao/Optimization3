import numpy as np

import augmented_lagrangian_svm_methods


class AugmentedLagrangianSVM:

    def __init__(self):
        self.Q = None
        self.w = None
        self.w0 = None
        self.lamda = None

    def fit(self, X, y, lamda_0=None, mu_0=1, p0=1, beta=1.1, p_max=100, num_iter=100, tolerance=1e-5,
            armijo_sigma=0.3, armijo_beta=0.1, armijo_a0=1,
            C=0.07):
        if lamda_0 is None:
            np.random.seed(0)
            #lamda_0 = np.random.rand(*y.shape) * C  # initialize randomly if no better guess
            lamda_0 = np.ones_like(y) * C / 2
        # internal functions implemented with X[num_features, num samples]. Transposing to use the same API as sklearn
        X = X.T
        y = y.T
        self.lamda = augmented_lagrangian_svm_methods.augmented_lagrangian_method(X, y, lamda_0, mu_0, p0, C, beta,
                                                                                  p_max, num_iter, tolerance,
                                                                                  armijo_sigma, armijo_beta, armijo_a0)
        self.w = X @ (self.lamda * y)

        w_0_possibilities = 1 / y - X.T @ self.w
        self.w0 = np.max(w_0_possibilities)

    def predict(self, X):
        return np.sign(X @ self.w + self.w0)
