import random
from projected_gradient_descent import projected_gradient_descent
from tqdm import tqdm
import numpy as np


# todo delete
# def quadratic_form(lamda, X, y):
#     # # Calculate the outer product of lambda and y, and then multiply element-wise with the outer product of y itself
#     # L = np.outer(lamda, lamda) * np.outer(y, y)
#     # # Compute the matrix product of X and its transpose, and then perform element-wise multiplication with L
#     # K = X @ X.T * L
#     # # Finally, sum all the elements and multiply by 1/2
#     # return 0.5 * np.sum(K)

def quadratic_form(lamda, X, y, Q=None):
    if Q is None:
        Q = get_Q(X, y)

    return 0.5 * lamda.T @ Q @ lamda


def get_Q(X, y):
    Q = np.outer(y, y) * (X.T @ X)
    return Q


def dual_svm(lamda: np.array, X, y):
    return np.sum(lamda) - quadratic_form(lamda, X, y)


def get_f(Q):
    return lambda lamda: quadratic_form(lamda, None, None, Q) - np.sum(lamda)


def df(lamda, Q):
    return Q @ lamda - np.ones_like(lamda)


def get_df(Q):
    return lambda l: df(l, Q)


def get_h(y):
    return lambda lamda: lamda @ y


def get_dl(Q, y, mu, p):
    df = get_df(Q)

    def dl(lamda):
        return df(lamda) + mu * y + p * 2 * y @ y * lamda

    return dl


def get_augmented_lagrangian(Q, y, mu, p):
    f = get_f(Q)
    h = get_h(y)

    def L(lamda):
        return f(lamda) + mu * h(lamda) + 0.5 * p * h(lamda) ** 2

    return L


def augmented_lagrangian_method(X, y, lamda_0, mu_0, p0, C, beta, p_max, num_iter, tolerance, armijo_sigma, armijo_beta,
                                armijo_a0):
    # initialize augmented lagrangian and it's derivative.
    Q = get_Q(X, y)
    f = get_f(Q)
    h = get_h(y)


    lamda_k = lamda_0
    mu_k = mu_0
    p_k = p0

    if beta < 1:
        raise ValueError(f"parameter beta must be greater than 1. got {beta}")

    for k in tqdm(range(num_iter)):

        if f(lamda_k) < tolerance:
            return lamda_k
        # evaluate new solution
        L = get_augmented_lagrangian(Q, y, mu_k, p_k)
        dl = get_dl(Q, y, mu_k, p_k)
        lamda_k = projected_gradient_descent(L, dl, lamda_k, 0, C, armijo_sigma, armijo_beta,
                                             armijo_a0)  # todo think about convinient way to tune parameters

        # evaluate multiplier
        mu_k = p_k * h(lamda_k) + mu_k

        # calculate penalty parameter
        p_k = min(beta * p_k, p_max)

    return lamda_k


# test methods
def test_Q(Q, x, y, N=100, epsilon=10 ** (-4)):
    import random
    m, n = Q.shape

    for k in range(N):
        i = random.randint(0, m)
        j = random.randint(0, n)
        qij = Q[i, j]

        val = y[i] * y[j] * x[:, i] @ x[:, j]

        if abs(val - qij) > epsilon:
            print(abs(val - qij))
