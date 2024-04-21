import unittest
import numpy as np
from augmented_lagrangian_svm_methods import *
# Assuming the previously discussed functions are imported
# from your implementation file (e.g., from svm_dual import get_Q, get_f, get_df)


class TestSVMDualProblem(unittest.TestCase):
    def test_get_Q(self):
        """Test the computation of Q matrix."""
        X = np.array([[1, 2], [3, 4]])
        y = np.array([1, -1])
        expected_Q = np.array([[10, -14], [-14, 20]])  # because X.T @ X = [[10, 14], [14, 20]], outer(y, y) = [[1, -1], [-1, 1]]
        np.testing.assert_array_almost_equal(get_Q(X, y), expected_Q)

    def test_objective_function(self):
        """Test the objective function for known values."""
        Q = np.array([[4, 0], [0, 4]])
        lamda = np.array([1, 1])
        #expected_f_value = 0.5 * lamda.T @ Q @ lamda - np.sum(lamda))  # 0.5 * [1, 1] @ [[4, 0], [0, 4]] @ [1, 1] - 2

        expected_f_value = 2
        f = get_f(Q)
        self.assertAlmostEqual(f(lamda), expected_f_value)

    def test_gradient_function(self):
        """Test the gradient function for correctness."""
        Q = np.array([[4, 0], [0, 4]])
        lamda = np.array([1, 1])
        expected_gradient = np.array([3, 3])  #Q @ lamda - np.ones_like(lamda)  # [[4, 0], [0, 4]] @ [1, 1] - [1, 1] = [3,3]
        df = get_df(Q)
        np.testing.assert_array_almost_equal(df(lamda), expected_gradient)

    def test_constraints(self):
        """Ensure that the equality constraint is maintained."""
        y = np.array([1, -1, 1])
        lamda = np.array([0.5, 0.5, 0.5])
        h = get_h(y)

        self.assertAlmostEqual(h(lamda), 0.5)

    def test_augmented_lagrangian(self):
        X = np.array([[1, 2], [3, 4]])
        y = np.array([1, -1])
        Q = get_Q(X, y)
        lamda = np.array([1, 1])
        mu = 2
        p = 5
        L = get_augmented_lagrangian(Q, y, mu, p)
        self.assertAlmostEqual(L(lamda), 0.5)
# To run the tests
if __name__ == '__main__':
    unittest.main()
