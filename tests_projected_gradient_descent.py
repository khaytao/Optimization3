from projected_gradient_descent import *
import unittest
import numpy as np

def f(x):
    # Example: Quadratic function f(x) = x^T A x + b^T x + c
    A = np.array([[3, 2], [2, 6]])
    b = np.array([1, 1])
    c = 5
    return x.T @ A @ x + b.T @ x + c

def grad_f(x):
    # Gradient of the quadratic function above
    A = np.array([[3, 2], [2, 6]])
    b = np.array([1, 1])
    return 2 * A @ x + b

def g_prime(alpha, x, d):
    # Derivative of g(alpha)
    return grad_f(x + alpha * d).dot(d)

# Example vectors
x = np.array([1, 2])
d = np.array([2, 1])

x_k = np.zeros(shape=(2,))
df_xk = grad_f(x_k)
f_xk = f(x_k)
d_k = np.reshape(np.array([1, 1]), (2,))
sigma = 0.25
beta = 0.5
alpha_0 = 1

class TestArmijoRule(unittest.TestCase):
    def test_g_prime_at_zero(self):
        # Test derivative at alpha = 0
        x = np.array([1, 2])
        d = np.array([2, 1])
        alpha = 0
        expected_result = grad_f(x).dot(d)  # Expected based on the gradient at x
        self.assertAlmostEqual(g_prime(alpha, x, d), expected_result)

    def test_g_prime_linear(self):
        # Testing with a linear function where the gradient is constant
        def f_linear(x):
            return 3 * x[0] + 4 * x[1]

        def grad_f_linear(x):
            return np.array([3, 4])

        def g_prime_linear(alpha, x, d):
            return grad_f_linear(x + alpha * d).dot(d)

        x = np.array([0, 0])
        d = np.array([1, 1])
        alpha = 5
        expected_result = grad_f_linear(x).dot(d)  # Should be constant
        self.assertEqual(g_prime_linear(alpha, x, d), expected_result)

    def test_g_prime_quadratic(self):
        # Verify derivative calculation for known quadratic function
        x = np.array([1, 1])
        d = np.array([-1, 2])
        alpha = 1
        expected_result = grad_f(x + alpha * d).dot(d)
        self.assertAlmostEqual(g_prime(alpha, x, d), expected_result)

    def test_basic_decrease(self):
        # Test that the Armijo rule reduces alpha to satisfy the Armijo condition
        alpha = ArmijoRule(f, x_k, df_xk, f_xk, d_k, sigma, beta, alpha_0)
        # Check if the condition is met
        self.assertTrue(f(x_k + alpha * d_k) <= f_xk + sigma * alpha * df_xk @ d_k)

    def test_initial_alpha(self):
        # Test with initial alpha that already satisfies the condition
        initial_alpha = 0.1
        alpha = ArmijoRule(f, x_k, df_xk, f_xk, d_k, sigma, beta, initial_alpha)
        self.assertTrue(f(x_k + alpha * d_k) <= f_xk + sigma * alpha * df_xk @ d_k)

    def test_projection_used(self):
        # Test the function with a projection function
        Flag = True
        projection = get_projection(0, 5)
        alpha = ArmijoRule(f, x_k, df_xk, f_xk, d_k, sigma, beta, alpha_0, Flag, projection)
        self.assertTrue(f(projection(x_k + alpha * d_k)) <= f_xk + sigma * alpha * df_xk @ d_k)

    def test_small_beta_large_decrease(self):
        # Test that a smaller beta results in smaller step sizes
        small_beta = 0.1
        alpha_small_beta = ArmijoRule(f, x_k, df_xk, f_xk, d_k, sigma, small_beta, alpha_0)
        alpha_large_beta = ArmijoRule(f, x_k, df_xk, f_xk, d_k, sigma, beta, alpha_0)
        self.assertLess(alpha_small_beta, alpha_large_beta)

    def test_armijo1(self):
        def f(x):
            return x.T @ x

        def df(x):
            return 2 * x

        x_k = np.array([1.5])
        d_k = -df(x_k)
        p = get_projection(1, 3)
        alpha_hat = ArmijoRule(f, x_k, df(x_k), f(x_k), d_k, sigma=0.3, beta=0.2, alpha_0=1, Flag=True, projection=p)
        x_k2 = p(x_k + alpha_hat * d_k)
        print(alpha_hat)
        print(x_k2)
        print(f(p(x_k2)))
        self.assertAlmostEqual(f(x_k2), 1)

    def test_armijo2(self):
        def f(x):
            return x.T @ x

        def df(x):
            return 2 * x

        x_k = np.array([0.5])
        d_k = -df(x_k)
        p = get_projection(-1, 1)
        alpha_hat = ArmijoRule(f, x_k, df(x_k), f(x_k), d_k, sigma=0.006, beta=0.8, alpha_0=10, Flag=True, projection=p)
        x_k2 = p(x_k + alpha_hat * d_k)
        print(f'alpha hat is: {alpha_hat}')
        print(f'dk hat is: {d_k}')
        print(f'x_k2 hat is: {x_k2}')
        print(f'function value hat is: {f(x_k2)}')
        self.assertAlmostEqual(f(x_k2), 0)

    def test_compare_armijo(self):
        def step_size(beta, sigma, x, d, func):
            """
            Armijo's Rule
            """
            i = 0
            inequality_satisfied = True
            while inequality_satisfied:
                if func.eval(x + np.power(beta, i) * d) <= func.eval(x) + np.power(beta, i) * sigma * func.gradient(
                        x).dot(
                        d):
                    break

                i += 1

            return np.power(beta, i)

        class Quardatic:
            def __init__(self):
                pass
            def eval(self, x):
                return x.T @ x

            def gradient(self, x):
                return 2 * x
        f = Quardatic()
        x_k = np.array([0.5])
        d_k = -f.gradient(x_k)
        p = get_projection(-1, 1)
        alpha_hat = ArmijoRule(f.eval, x_k, -d_k, f.eval(x_k), d_k, sigma=0.006, beta=0.8, alpha_0=10, Flag=False, projection=p)
        alpha_hat2 = step_size(0.8, 0.03, x_k, d_k, f)
        x_k2 = p(x_k + alpha_hat * d_k)
        print(f'alpha hat is: {alpha_hat}')
        print(f'alpha hat is: {alpha_hat2}')

        self.assertAlmostEqual(alpha_hat, alpha_hat2)



# Run the tests
if __name__ == '__main__':
    unittest.main()
