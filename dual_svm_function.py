import numpy as np


def kernel_linear(x1, x2):
    """Computes the linear kernel between two vectors."""
    return np.dot(x1, x2)


def compute_dual_objective(alpha, Y, K):
    """
    Computes the dual objective function of the SVM.

    Parameters:
        alpha (numpy.ndarray): The vector of Lagrange multipliers.
        Y (numpy.ndarray): The diagonal matrix of labels, implemented as a 1D array for simplicity.
        K (numpy.ndarray): The kernel matrix.

    Returns:
        float: The value of the dual objective function.
    """
    # Dual objective function: W(alpha) = sum(alpha) - 0.5 * alpha.T @ Y @ K @ Y @ alpha
    half_term = 0.5 * alpha @ (Y * (K @ (Y * alpha)))
    return np.sum(alpha) - half_term


def compute_gradient(alpha, Y, K):
    """
    Computes the gradient of the dual objective function.

    Parameters:
        alpha (numpy.ndarray): The vector of Lagrange multipliers.
        Y (numpy.ndarray): Labels as a 1D array.
        K (numpy.ndarray): The kernel matrix.

    Returns:
        numpy.ndarray: The gradient of the dual objective function.
    """
    return np.ones_like(alpha) - Y * (K @ (Y * alpha))

#
# # Example usage
# # Assuming you have your data points X and labels y
# X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])  # example feature matrix
# y = np.array([1, -1, 1, -1])  # example labels
#
# # Create the kernel matrix for a linear kernel
# K = np.array([[kernel_linear(xi, xj) for xj in X] for xi in X])
#
# # Label diagonal matrix as a 1D array for simplified multiplication
# Y = y
#
# # Initial Lagrange multipliers (alpha)
# alpha = np.random.random(size=y.shape)
#
# # Compute objective and gradient
# objective_value = compute_dual_objective(alpha, Y, K)
# gradient = compute_gradient(alpha, Y, K)
# print("Objective Value:", objective_value)
# print("Gradient:", gradient)
