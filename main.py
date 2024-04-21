from scipy.io import loadmat
from sklearn.metrics import accuracy_score
from AugmentedLagrangianSVM import AugmentedLagrangianSVM
from extract_features import ExtractFeatures


def load_data(x_name, y_name):
    # Load the data from .mat files
    x = loadmat(x_name + ".mat")[x_name]
    y = loadmat(y_name)[y_name].ravel().astype(float)
    y[y == 9] = 1
    y[y == 0] = -1
    return x, y


def question_16():
    # Loading data
    X_train, y_train = load_data("xForTraining", "labelsForTraining")
    X_train = ExtractFeatures(X_train, 50)
    X_test, y_test = load_data("xForTest", "labelsForTest")
    X_test = ExtractFeatures(X_test, 50)

    # Create our SVM classifier
    aug = AugmentedLagrangianSVM()
    aug.fit(X_train.T, y_train, num_iter=10)

    # Predict on the test set
    y_pred = aug.predict(X_test.T)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    score = y_test == y_pred
    print(
        f"The number of images that were misclassified is: {len(score[score == False])}"
    )

    return aug.w, aug.w0, aug.lamda


def question_17():
    # Loading data
    X_train, y_train = load_data("xForTraining", "labelsForTraining")
    X_train = ExtractFeatures(X_train, 50)
    X_test, y_test = load_data("xForTest", "labelsForTest")
    X_test = ExtractFeatures(X_test, 50)

    # Create our SVM classifier
    aug = AugmentedLagrangianSVM(kernel_type="gas_rad")
    aug.fit(X_train.T, y_train, num_iter=10)

    # Predict on the test set
    y_pred = aug.predict(X_test.T)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    score = y_test == y_pred
    print(
        f"The number of images that were misclassified is: {len(score[score == False])}"
    )

    return aug.w, aug.w0, aug.lamda
