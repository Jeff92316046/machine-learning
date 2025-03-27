# USAGE
# python sgd.py

# import the necessary packages
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import argparse

def sigmoid_activation(x):
    # compute the sigmoid activation value for a given input
    return 1.0 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    # compute the derivative of the sigmoid function
    return x * (1 - x)

def predict(x, w):
    # take the dot product between our features and weight matrix
    preds = sigmoid_activation(x.dot(w))
    # apply a step function to threshold the outputs to binary class labels
    preds[preds <= 0.5] = 0
    preds[preds > 0.5] = 1
    return preds

def next_batch(x_data, y, batch_size):
    for i in np.arange(0, x_data.shape[0], batch_size):
        yield (x_data[i:i + batch_size], y[i:i + batch_size])

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=int, default=100, help="# of epochs")
ap.add_argument("-a", "--alpha", type=float, default=0.01, help="learning rate")
ap.add_argument("-b", "--batch-size", type=int, default=32, help="size of SGD mini-batches")
args = vars(ap.parse_args())

# generate a 2-class classification problem
(X, y) = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=1.5, random_state=1)
y = y.reshape((y.shape[0], 1))
X = np.c_[X, np.ones((X.shape[0]))]  # add bias term

# split the data into training and testing sets
(trainX, testX, trainY, testY) = train_test_split(X, y, test_size=0.5, random_state=42)

# initialize weights and loss history
print("[INFO] training...")
rng = np.random.default_rng(1)
W = rng.standard_normal((X.shape[1], 1))
losses = []

# training loop
for epoch in np.arange(0, args["epochs"]):
    epochLoss = []
    for (batchX, batchY) in next_batch(trainX, trainY, args["batch_size"]):
        preds = sigmoid_activation(batchX.dot(W))  # Forward pass
        error = preds - batchY  # Compute error
        epochLoss.append(np.sum(error ** 2))
        d = error * sigmoid_deriv(preds)  # Compute gradient
        gradient = batchX.T.dot(d)  # Compute weight updates
        W += -args["alpha"] * gradient  # Update weights
    
    loss = np.average(epochLoss)
    losses.append(loss)
    if epoch == 0 or (epoch + 1) % 5 == 0:
        print(f"[INFO] epoch={epoch + 1}, loss={loss:.7f}")

# evaluate model
print("[INFO] evaluating...")
preds = predict(testX, W)
print(classification_report(testY, preds))

# plot data
plt.style.use("ggplot")
plt.figure()
plt.title("Data")
plt.scatter(testX[:, 0], testX[:, 1], marker="o", c=testY[:, 0], s=30)

# plot loss curve
plt.figure()
plt.plot(np.arange(0, args["epochs"]), losses)
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()