import numpy as np


def get_projection(a, b):
    if a <= b:
        def p(x):
            # duplicate array for safety. todo might be possible to implement in place
            # y = x
            y = np.array(x, copy=True)
            y[y < a] = a
            y[y > b] = b
            return y
    else:
        raise ValueError("lower limit larger than upper limit")

    return p


# def ArmijoRule(f: callable, x_k: np.array, df_xk: np.array, f_xk: np.array, d_k: np.array, sigma, beta, alpha_0,
#                Flag=False, constraints_limits: tuple = None):
#     #initialize parameters
#     alpha = alpha_0
#     x_alpha = x_k + alpha * d_k
#
#     if Flag and len(constraints_limits) == 2:
#         p = get_projection(constraints_limits[0], constraints_limits[1])
#     else:
#         p = lambda x: x
#
#     while f(x_alpha) - f_xk <= sigma * df_xk @ (x_alpha - x_k):
#         x_alpha = p(x_k + alpha * d_k)
#
#     return x_alpha


def ArmijoRule(f: callable, x_k: np.array, df_xk: np.array, f_xk: float, d_k: np.array, sigma, beta, alpha_0,
               Flag=False, projection:callable = None):
    """
    Perform a line search using Armijo rule and return the step size alpha.

    :param f: a function Rn-> R
    :param x_k:
    :param df_xk:
    :param f_xk:
    :param d_k:
    :param sigma:
    :param beta:
    :param alpha_0:
    :param Flag:
    :param projection:
    :return:
    """
    # initialize parameters
    if Flag and callable(projection):
        p = projection
    else:
        p = lambda x: x

    # as a precaution, cast all arrays into np arrays to support working with lists / scalars
    x_k = np.asarray(x_k).astype(float)
    df_xk = np.asarray(df_xk).astype(float)
    d_k = np.asarray(d_k).astype(float)

    get_x = lambda a: p(x_k + a * d_k)
    alpha = alpha_0
    x_alpha = get_x(alpha_0)

    def did_converge(x_alpha):
        # print(f'RHS:{sigma * df_xk @  d_k * alpha}')
        # print(f'LHS:{f(x_alpha) - f_xk}')
        # return f(x_alpha) - f_xk <= sigma * df_xk @  d_k * alpha  # Equivalent to (x_alpha - x_k)
        return f(x_alpha) - f_xk <= sigma * alpha * df_xk @  (x_alpha - x_k)  # Equivalent to (x_alpha - x_k)

    while not did_converge(x_alpha):  # todo maybe add restriction on number of iterations
        alpha = beta * alpha
        x_alpha = get_x(alpha)
        # print(f"Amarijo: {alpha}")
        if alpha < 1e-8 and alpha < np.linalg.norm((x_alpha - x_k), ord=2):  # A practical threshold to avoid infinite loops
            print("we didn't converge")
            print(np.linalg.norm((x_alpha - x_k), ord=2))
            return alpha

    # print(f'RHS:{sigma * df_xk @  d_k * alpha}')
    # print(f'LHS:{f(x_alpha) - f_xk}')
    return alpha


def projected_gradient_descent(f, df, x0, a, b, sigma, beta, alpha_0, num_iter=100, tolerance=0.01):
    """

    :param f:
    :param df:
    :param x0:
    :param a:
    :param b:
    :param sigma:
    :param beta:
    :param alpha_0:
    :param num_iter:
    :param tolerance:
    :return:
    """
    xk = x0

    p = get_projection(a, b)
    for k in range(num_iter):
        fx = f(xk)
        if fx < tolerance:
            return xk

        # dk = - df(x0)
        dk = - df(xk)
        ak = ArmijoRule(f, xk, -dk, fx, dk, sigma, beta, alpha_0, True,  p)
        if ak < 1e-8:
            print(f'ak in so small (ak = {ak}) we are not moving. exiting the function in iteration {k}.')
            return xk
        xk = p(xk + ak * dk)
    return xk
