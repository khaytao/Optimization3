import numpy as np
from scipy.io import loadmat
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the data from .mat files
x = loadmat('xForTest.mat')['xForTest']
y = loadmat('labelsForTest.mat')['labelsForTest'].ravel()

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Create an SVM classifier
clf = SVC(kernel='linear')  # You can change the kernel and other parameters as needed

# Fit the SVM model
clf.fit(X_train.T, y_train)

# Predict on the test set
y_pred = clf.predict(X_test.T)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
