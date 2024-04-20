import random
from projected_gradient_descent import projected_gradient_descent
from tqdm import tqdm
import numpy as np
from projected_gradient_descent import ArmijoRule, get_projection


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


# def dual_svm(lamda: np.array, X, y):
#     return np.sum(lamda) - quadratic_form(lamda, X, y)


def get_f(Q):
    # - because the original problem is maximization problem.
    return lambda lamda: quadratic_form(lamda, None, None, Q) - np.sum(lamda)


def df(lamda, Q):
    # - because the original problem is maximization problem.
    return Q @ lamda - np.ones_like(lamda)


def get_df(Q):
    return lambda l: df(l, Q)


def get_h(y):
    return lambda lamda: lamda @ y


def get_dl(Q, y, mu, p):
    df = get_df(Q)

    def dl(lamda):
        """we stack the derivative over mu under the derivative by lamda."""
        dlamda = df(lamda) + mu * y + p * (y @ y) * lamda
        return dlamda

    return dl


def get_augmented_lagrangian(Q, y, mu, p):
    f = get_f(Q)
    h = get_h(y)

    def L(lamda):
        return f(lamda) + mu * h(lamda) + 0.5 * p * h(lamda) ** 2

    return L


def approximate_gradient(x, f, delta=10 ** -5):
    n = len(x)
    dx = np.zeros_like(x)
    I = np.eye(n)
    for i in range(n):
        dx[i] = (f(x + delta * I[:, i]) - f(x)) / delta

    return dx


def mean_squere_error(x, y):
    return np.mean((x - y) ** 2)


def augmented_lagrangian_method(X, y, lamda_0, mu_0, p0, C, beta, p_max, num_iter, tolerance, armijo_sigma, armijo_beta,
                                armijo_a0):
    # initialize augmented lagrangian and it's derivative.
    Q = get_Q(X, y)
    f = get_f(Q)
    h = get_h(y)

    lamda_k = lamda_0
    mu_k = mu_0
    p_k = p0

    p = get_projection(0, C)

    if beta < 1:
        raise ValueError(f"parameter beta must be greater than 1. got {beta}")

    for k in tqdm(range(num_iter)):

        if f(lamda_k) < tolerance:
            return lamda_k
        # evaluate new solution
        L = get_augmented_lagrangian(Q, y, mu_k, p_k)
        dl = get_dl(Q, y, mu_k, p_k)
        # lamda_k = projected_gradient_descent(L, dl, lamda_k, 0, C, armijo_sigma, armijo_beta,
        # armijo_a0)  # todo think about convinient way to tune parameters
        decent_direction = -dl(lamda_k)
        # decent_direction = decent_direction/np.linalg.norm(decent_direction)
        alpha = ArmijoRule(L, lamda_k, -decent_direction, L(lamda_k), decent_direction, armijo_sigma, armijo_beta,
                           armijo_a0, Flag=True, projection=p)

        # debug, comparing gradient calculation to approximation
        grad_appx = approximate_gradient(lamda_k, L)

        print(f"approximation difference: {mean_squere_error(grad_appx, -decent_direction)}")

        # calculating the discrete derivative manually
        # lamda_k_old = np.array(lamda_k, copy=True)
        lamda_k = p(lamda_k + alpha * decent_direction)


        # f_x_delta = np.zeros(lamda_k.shape)
        # for i in range(f_x_delta.size):
        #     dx = np.array(lamda_k_old, copy=True)
        #     dx[i] = lamda_k[i]
        #     delta_x = (lamda_k[i] - lamda_k_old[i]) + 1e-6
        #     f_x_delta[i] = (L(dx) - L(lamda_k_old)) / delta_x
        # print(np.linalg.norm((f_x_delta + decent_direction)))
        #
        # dl_neq = dl(lamda_k)

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
