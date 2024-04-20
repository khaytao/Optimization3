import numpy as np
import numpy.random
from scipy.io import loadmat
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from AugmentedLagrangianSVM import AugmentedLagrangianSVM
from extract_features import ExtractFeatures
import scipy


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
aug = AugmentedLagrangianSVM()
# Create an SVM classifier
clf = SVC(kernel='linear')  # You can change the kernel and other parameters as needed

for model, iteration_num in zip([clf, aug], [1, 2]):
    if iteration_num == 1:
        print(f'Running algorithm clf.')
    else:
        print(f'Running algorithm aug.')
    # Fit the SVM model
    model.fit(X_train.T, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test.T)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')
