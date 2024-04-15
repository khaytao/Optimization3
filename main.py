import numpy as np

def get_projection(a, b):

    if a < b:
        def p(x):
            if x < a:
                return a
            elif a < x < b:
                return x
            else:
                return b
    else:
        raise ValueError("lower limit larger than upper limit")

    return p

def ArmijoRule(f: callable, x_k:np.array, df_xk:np.array, f_xk: np.array, d_k:np.array, sigma, beta, alpha_0, Flag=False, constraints_limits: tuple=None):
    alpha = alpha_0
    x_alpha = x_k + alpha * d_k

    if Flag and len(constraints_limits) == 2:
        p = get_projection(constraints_limits[0], constraints_limits[1])
    else:
        p = lambda x: x

    while f(x_alpha) - f_xk <= sigma * df_xk @ (x_alpha - x_k):
        x_alpha = p(x_k + alpha * d_k)

    return x_alpha
