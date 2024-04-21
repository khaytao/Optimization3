from augmented_lagrangian_svm_methods import get_Q, projected_gradient_descent, get_projection, ArmijoRule
import numpy as np


class Lagrangian:

    def __init__(self, X, y, mu, p):
        self.y = y
        self.mu = mu
        self.p = p
        self.Q = get_Q(X, y)

    def evaluate(self, lamda):
        return 0.5 * lamda.T @ self.Q @ lamda - np.sum(lamda) + self.mu * (self.y @ lamda) + 0.5 * self.p * (
                self.y @ lamda) ** 2

    def gradient(self, lamda):
        return self.Q @ lamda - np.ones_like(lamda) + self.mu * self.y + self.p * self.y.T @ lamda * self.y


def solve_augmented_lagrangian(X, y, lamda_0, mu_0, p0, C, beta, p_max, num_iter, tolerance, armijo_sigma, armijo_beta,
                               armijo_a0):
    lamda_k = lamda_0
    mu_k = mu_0
    p_k = p0

    # lamda_k = np.zeros_like(y)
    # mu_k = 0
    # p_k = 1

    for k in range(num_iter):

        #print("debug log", f"augmented lagrangian iteration {k}",
        #      f"Constructing new Augmented lagrangian with mu={mu_k}, p={p_k}")
        # Build new lagrangian
        L = Lagrangian(X, y, mu_k, p_k)

        # update lambda
        lamda_k = projected_gradient_descent(L.evaluate, L.gradient, lamda_k, 0, C, armijo_sigma, armijo_beta,
                                             armijo_a0, num_iter=10, tolerance=0.01,
                                             alpha_tolerance=10 ** -5)

        # update mu
        mu_k = mu_k + p_k * lamda_k @ y

        # update p
        p_k = min(beta * p_k, p_max)

        if np.linalg.norm(L.gradient(lamda_k) + mu_k * y @ lamda_k) < tolerance: #KKT condition
            break


    return lamda_k


def projected_gradient_descent(f, df, x0, a, b, sigma, beta, alpha_0, num_iter=100, tolerance=0.01,
                               alpha_tolerance=10 ** -5):
    xk = x0
    p = get_projection(a, b)

    f_score = f(xk)
    for k in range(num_iter):
        #print("debug log", f"Projected Gradient descent iteration {k}")
        dk = -df(xk)
        alpha_k = ArmijoRule(f, xk, -dk, f(xk), dk, sigma, beta, alpha_0, True, p)

        if alpha_k < alpha_tolerance:
            break

        xk = p(xk + alpha_k * dk)
        new_score = f(xk)
        if new_score > f_score:
            break
        else:
            f_score = new_score
    return xk
