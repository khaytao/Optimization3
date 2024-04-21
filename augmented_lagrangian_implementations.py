import numpy as np


def K(x, y, kernel_type=None, d=2, gamma=1/2, kappa=2, c=-0.1):
    if kernel_type is None:
        return x @ y
    if kernel_type == "ploy_hom":
        return (x @ y) ** d
    if kernel_type == "gas_rad":
        return np.exp(-gamma * (np.linalg.norm(x-y, ord=2) ** 2))
    if kernel_type == "sig":
        if kappa <= 0 or c >= 0:
            raise "error, 'kappa' need to be negative and 'c' positive."
        return np.tanh(kappa * (x @ y) + c)

def get_Q(X, y):
    Q = np.outer(y, y) * (X.T @ X)
    return Q


def get_projection(a, b):
    if a <= b:

        def p(x):
            # duplicate array for safety. todo might be possible to implement in place
            y = np.array(x, copy=True)
            y[y < a] = a
            y[y > b] = b
            return y

    else:
        raise ValueError("lower limit larger than upper limit")

    return p


def ArmijoRule(
    f: callable,
    x_k: np.array,
    df_xk: np.array,
    f_xk: float,
    d_k: np.array,
    sigma,
    beta,
    alpha_0,
    Flag=False,
    projection: callable = None,
    kernel_type = None,
):
    """
    Perform a line search using Armijo rule and return the step size alpha.
    """
    # initialize parameters
    if Flag and callable(projection):
        p = projection
    else:
        p = lambda x: x

    # as a precaution, cast all arrays into np arrays to
    # support working with lists / scalars
    x_k = np.asarray(x_k).astype(float)
    df_xk = np.asarray(df_xk).astype(float)
    d_k = np.asarray(d_k).astype(float)

    get_x = lambda a: p(x_k + a * d_k)
    alpha = alpha_0
    x_alpha = get_x(alpha_0)

    def did_converge(x_alpha):
        return (f(x_alpha) - f_xk <= sigma *
                K(df_xk, (x_alpha - x_k), kernel_type=kernel_type))

    while not did_converge(x_alpha):
        alpha = beta * alpha
        x_alpha = get_x(alpha)

        # A practical threshold to avoid infinite loops
        if alpha < 1e-6 and alpha < np.linalg.norm((x_alpha - x_k), ord=2):
            break
    return alpha


class Lagrangian:

    def __init__(self, X, y, mu, p, kernel_type=None):
        self.y = y
        self.mu = mu
        self.p = p
        self.Q = get_Q(X, y)
        self.kernel_type = kernel_type

    def evaluate(self, lamda):
        return (
                0.5 * K(lamda.T, self.Q @ lamda, kernel_type=self.kernel_type)
                - np.sum(lamda)
                + self.mu * K(self.y, lamda, kernel_type=self.kernel_type)
                + 0.5 * self.p * K(self.y, lamda, kernel_type=self.kernel_type) ** 2
        )


    def gradient(self, lamda):
        return (
                self.Q @ lamda
                - np.ones_like(lamda)
                + self.mu * self.y
                + self.p * K(self.y.T, lamda, kernel_type=self.kernel_type) * self.y
        )


def projected_gradient_descent(
    f, df, x0, a, b, sigma, beta, alpha_0, num_iter=100,
        alpha_tolerance=10**-5, kernel_type=None):
    xk = x0
    p = get_projection(a, b)

    f_score = f(xk)
    for k in range(num_iter):
        dk = -df(xk)
        alpha_k = ArmijoRule(f, xk, -dk, f(xk), dk, sigma, beta, alpha_0, True, p, kernel_type=kernel_type)

        if alpha_k < alpha_tolerance:
            break

        xk = p(xk + alpha_k * dk)
        new_score = f(xk)
        if new_score > f_score:
            break
        else:
            f_score = new_score
    return xk


def solve_augmented_lagrangian(
    X,
    y,
    lamda_0,
    mu_0,
    p0,
    C,
    beta,
    p_max,
    num_iter,
    tolerance,
    armijo_sigma,
    armijo_beta,
    armijo_a0,
    kernel_type=None
):
    lamda_k = lamda_0
    mu_k = mu_0
    p_k = p0

    for k in range(num_iter):
        L = Lagrangian(X, y, mu_k, p_k)

        # update lambda
        lamda_k = projected_gradient_descent(
            L.evaluate,
            L.gradient,
            lamda_k,
            0,
            C,
            armijo_sigma,
            armijo_beta,
            armijo_a0,
            num_iter=10,
            alpha_tolerance=10**-5,
        )

        # update mu
        mu_k = mu_k + p_k * K(lamda_k, y, kernel_type=kernel_type)

        # update p
        p_k = min(beta * p_k, p_max)

        if (
            np.linalg.norm(L.gradient(lamda_k) + mu_k * K(y, lamda_k, kernel_type=kernel_type)) < tolerance
        ):  # KKT condition
            break

    return lamda_k
