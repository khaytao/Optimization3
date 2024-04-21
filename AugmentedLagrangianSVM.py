import numpy as np
from augmented_lagrangian_implementations import solve_augmented_lagrangian


class AugmentedLagrangianSVM:
    """
    Implementations of our algorithms with sklearn classifiers API
    """

    def __init__(self):
        self.Q = None
        self.w = None
        self.w0 = None
        self.lamda = None

    def fit(self, X, y, lamda_0=None, mu_0=1, p0=1, beta=1.1, p_max=10000, num_iter=100, tolerance=1e-5,
            armijo_sigma=0.3, armijo_beta=0.1, armijo_a0=10,
            C=0.07):


        X = np.array(X)
        y = np.array(y)
        if lamda_0 is None:
            lamda_0 = np.zeros_like(y) # initialize with zeros if no better guess

        # internal functions implemented with X[num_features, num samples]. Transposing to use the same API as sklearn which is X[num samples, num_features]
        X = X.T
        y = y.T

        self.lamda = solve_augmented_lagrangian(X, y, lamda_0, mu_0, p0, C, beta, p_max, num_iter, tolerance, armijo_sigma, armijo_beta, armijo_a0)
        #self.w = X @ (self.lamda * y)
        self.w = np.sum(self.lamda[:, None] * y[:, None] * X.T, axis=0)

        w_0_possibilities = y - X.T @ self.w
        self.w0 = np.mean(w_0_possibilities)

    def predict(self, X):
        return np.sign(X @ self.w + self.w0)
