# USAGE
# python classify.py

# import the necessary packages
from __future__ import print_function
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

# generate the XOR data
rng = np.random.default_rng(1)
tl = rng.uniform(size=(100, 2)) + np.array([-2.0, 2.0])
tr = rng.uniform(size=(100, 2)) + np.array([2.0, 2.0])
br = rng.uniform(size=(100, 2)) + np.array([2.0, -2.0])
bl = rng.uniform(size=(100, 2)) + np.array([-2.0, -2.0])
X = np.vstack([tl, tr, br, bl])
y = np.hstack([[1] * len(tl), [-1] * len(tr), [1] * len(br), [-1] * len(bl)])

# construct the training and testing split by taking 75% of the data for training
# and 25% for testing
(trainData, testData, trainLabels, testLabels) = train_test_split(X, y, test_size=0.25, random_state=42)

# train the linear SVM model, evaluate it, and show the results
print("[RESULTS] SVM w/ Linear Kernel")
model = SVC(kernel="linear")
model.fit(trainData, trainLabels)
print(classification_report(testLabels, model.predict(testData)))
print("")

# train the SVM + poly. kernel model, evaluate it, and show the results
print("[RESULTS] SVM w/ Polynomial Kernel")
model = SVC(kernel="poly", degree=3,coef0=1)
model.fit(trainData, trainLabels)
print(classification_report(testLabels, model.predict(testData)))
