import matplotlib.pyplot as plt
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

armijo_calls = [0]


def ArmijoRule(f: callable, x_k: np.array, df_xk: np.array, f_xk: float, d_k: np.array, sigma, beta, alpha_0,
               Flag=False, projection: callable = None):
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

    xs = np.linspace(0, 1, 1000)
    yf = np.array([f(get_x(alpha)) for alpha in xs.tolist()])

    def did_converge(x_alpha):
        # print(f'RHS:{sigma * df_xk @  d_k * alpha}')
        # print(f'LHS:{f(x_alpha) - f_xk}')
        # return f(x_alpha) - f_xk <= sigma * df_xk @  d_k * alpha  # Equivalent to (x_alpha - x_k)
        return f(x_alpha) - f_xk <= sigma * df_xk @ (x_alpha - x_k)

    while not did_converge(x_alpha):  # todo maybe add restriction on number of iterations
        alpha = beta * alpha
        x_alpha = get_x(alpha)
        # print(f"Amarijo: {alpha}")
        if alpha < 1e-6 and alpha < np.linalg.norm((x_alpha - x_k),
                                                   ord=2):  # A practical threshold to avoid infinite loops

            break

    # print(f'RHS:{sigma * df_xk @  d_k * alpha}')
    # print(f'LHS:{f(x_alpha) - f_xk}')
    # x_alpha = get_x(alpha)
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(xs, yf)
    # plt.plot(alpha, f(x_alpha))
    # plt.savefig(f"outputs\\{armijo_calls[0]}.png")
    # armijo_calls[0] += 1
    print("debug log", f"Armijo Rule", f"alpha value is {alpha}", f"f(alpha) = {f(get_x(alpha))}",
          f"RHS value is {f_xk + sigma * df_xk @ (x_alpha - x_k)}")
    return alpha


def compute_projected_gradient(x, grad, feasible_region_projector):
    """
    More accurately compute the projected gradient at point x for a given gradient and feasible region projection function.
    """
    projected_x = feasible_region_projector(x)
    projected_step = feasible_region_projector(x - grad)
    projected_gradient = projected_step - projected_x
    return projected_gradient

def approximate_gradient(x, f, delta=10 ** -5):
    n = len(x)
    dx = np.zeros_like(x)
    I = np.eye(n)
    for i in range(n):
        dx[i] = (f(x + delta * I[:, i]) - f(x)) / delta

    return dx


def projected_gradient_descent(f, df, x0, a, b, sigma, beta, alpha_0, num_iter=100, tolerance=0.01,
                               alpha_tolerance=10 ** -5):
    """
    Performs the projected gradient descent algorithm to find a local minimum of a given function within specified bounds.

    Parameters:
        f (callable): The objective function to minimize.
        df (callable): The gradient of the objective function.
        x0 (ndarray): The starting point of the algorithm.
        a (float): The lower bound of the projection interval.
        b (float): The upper bound of the projection interval.
        sigma (float): The sufficient decrease constant in the Armijo condition (part of line search).
        beta (float): The factor by which the step size is multiplied in each iteration (should be between 0 and 1).
        alpha_0 (float): The initial step size.
        num_iter (int, optional): The maximum number of iterations to run. Default is 100.
        tolerance (float, optional): The tolerance for the stopping criterion based on the norm of the gradient. Default is 0.01.

    Returns:
        float or ndarray: The approximate local minimum found by the algorithm.

    Uses Armijo's rule for adaptive step size along with a projection onto the interval [a, b] to ensure that the
    updates remain within this interval. Logs debug information regarding the progression of the algorithm, including
    iteration number and the norm of the gradient.
    """
    xk = x0

    p = get_projection(a, b)
    it_number = []
    f_value = []
    projected_grad_values = []
    print("debug log", "entering Projected Gradient descent")
    for k in range(num_iter):
        it_number.append(k)
        fx = f(xk)
        f_value.append(fx)  # todo delete
        #dk = -df(xk)  # find descent direction
        dk = p(xk -df(xk)) - xk  # search direction projected onto feasible set

        # because this is a constrained problem, we;ll check the size of the projected gradient instead
        projected_grad = compute_projected_gradient(xk, df(xk), p)

        print("debug log", f"Projected Gradient descent iteration {k}",
              f"norm of projected gradient is {np.linalg.norm(projected_grad)}")
        projected_grad_values.append(np.linalg.norm(projected_grad))
        # Finding the optimal step size using Armijo's rule with projection
        ak = ArmijoRule(f, xk, -dk, fx, dk, sigma, beta, alpha_0, True, p)  # find step size

        diff = p(xk + ak * dk) - xk
        xk = p(xk + ak * dk)

        if np.linalg.norm(projected_grad) < tolerance or np.linalg.norm(diff) < alpha_tolerance:
            break

    # plt.figure()
    # plt.plot(it_number, f_value, label="f")
    # plt.plot(it_number, projected_grad_values, label="projected gradient")
    # plt.show()
    return xk
