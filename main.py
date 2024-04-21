import numpy as np
import numpy.random
from scipy.io import loadmat
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from AugmentedLagrangianSVM import AugmentedLagrangianSVM
from extract_features import ExtractFeatures
import scipy
import matplotlib.pyplot as plt


def plot_decision_boundary(X, y, w, w0):
    # Plotting the data points
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', marker='o', label='Class +1')
    plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], color='red', marker='x', label='Class -1')

    # Create a grid of points to evaluate the model
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 200), np.linspace(x2_min, x2_max, 200))

    # Compute the decision boundary based on the formula w^T x + w0 = 0
    # Rearrange terms to: x2 = -(w1/w2) * x1 - w0/w2
    Z = -(w[0] / w[1]) * xx1 - (w0 / w[1])

    # Plot decision boundary
    plt.contour(xx1, xx2, Z, levels=[0], colors='green', linestyles='--')
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(loc='upper left')
    plt.title('SVM Decision Boundary')
    plt.show()


def load_data(x_name, y_name):
    # Load the data from .mat files
    x = loadmat(x_name + ".mat")[x_name]
    y = loadmat(y_name)[y_name].ravel().astype(float)
    y[y == 9] = 1
    y[y == 0] = -1
    return x, y


X_train, y_train = load_data('xForTraining', 'labelsForTraining')
X_train = ExtractFeatures(X_train, 50)
X_test, y_test = load_data('xForTest', 'labelsForTest')
X_test = ExtractFeatures(X_test, 50)

# from sklearn.datasets import make_blobs
# X, Y = make_blobs(n_samples=200, centers=[(-1, -1), (1, 1)], cluster_std=0.5)
# Y[Y == 0] = -1 # to have +/- 1 labels
# X_train, X_test, y_train, y_test = train_test_split(X, Y)
# X_train, X_test = X_train.T, X_test.T

aug = AugmentedLagrangianSVM()
# Create an SVM classifier
clf = SVC(C=0.07, kernel='linear')  # You can change the kernel and other parameters as needed

# for model, iteration_num in zip([clf, aug], [1, 2]):
#     if iteration_num == 1:
#         print(f'Running algorithm clf.')
#     else:
#         print(f'Running algorithm aug.')
#     # Fit the SVM model
#     model.fit(X_train.T, y_train)
#
#     # Predict on the test set
#     y_pred = model.predict(X_test.T)
#
#     # Evaluate the model
#     accuracy = accuracy_score(y_test, y_pred)
#     print(f'Accuracy: {accuracy:.2f}')

from sklearn.datasets import make_blobs
X, Y = make_blobs(n_samples=200, centers=[(-1, -1), (1, 1)], cluster_std=0.5)
Y[Y == 0] = -1 # to have +/- 1 labels
X_train, X_test, y_train, y_test = train_test_split(X, Y)
X_train, X_test = X_train.T, X_test.T
aug.fit(X, Y)

plot_decision_boundary(X, Y, aug.w, aug.w0)
