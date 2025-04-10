# neural_network.py
import numpy as np

class NeuralNetwork:
    def __init__(self, layers, alpha=0.1):
        self.W = []
        self.layers = layers
        self.alpha = alpha

        rng = np.random.default_rng(42)
        for i in range(0, len(layers) - 1):
            w = rng.standard_normal((layers[i] + 1, layers[i + 1] + (0 if i == len(layers) - 2 else 1)))
            self.W.append(w / np.sqrt(layers[i]))

    def __repr__(self):
        return "NeuralNetwork: {}".format("-".join(str(l) for l in self.layers))

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        return x * (1 - x)

    def fit(self, x_data, y, epochs=10000):
        x_data = np.c_[x_data, np.ones((x_data.shape[0]))]  # bias trick
        for _ in range(epochs):
            for (x, target) in zip(x_data, y):
                self.fit_partial(x, target)

    def fit_partial(self, x, y):
        A = [np.atleast_2d(x)]
        for layer in range(len(self.W)):
            net = A[layer].dot(self.W[layer])
            out = self.sigmoid(net)
            A.append(out)

        error = A[-1] - y
        D = [error * self.sigmoid_deriv(A[-1])]

        for layer in range(len(A) - 2, 0, -1):
            delta = D[-1].dot(self.W[layer].T) * self.sigmoid_deriv(A[layer])
            D.append(delta)

        D = D[::-1]
        for i in range(len(self.W)):
            self.W[i] -= self.alpha * A[i].T.dot(D[i])

    def predict(self, x_input):
        x_input = np.atleast_2d(x_input)
        x_input = np.c_[x_input, np.ones((x_input.shape[0]))]
        for layer in range(len(self.W)):
            x_input = self.sigmoid(x_input.dot(self.W[layer]))
        return x_input

def test_gate(gate_name, x_data, y):
    print(f"training {gate_name} gate")
    nn = NeuralNetwork([2, 2, 1])
    nn.fit(x_data, y, epochs=20000)
    print(f"testing {gate_name} gate")
    for (x, target) in zip(x_data, y):
        pred = nn.predict(x)[0][0]
        label = 1 if pred > 0.5 else 0
        print(f"Input: {x}, Predicted: {label}, Target: {int(target[0])}")

# 資料集
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# AND
y_and = np.array([[0], [0], [0], [1]])
test_gate("AND", X, y_and)

# OR
y_or = np.array([[0], [1], [1], [1]])
test_gate("OR", X, y_or)

# XOR
y_xor = np.array([[0], [1], [1], [0]])
test_gate("XOR", X, y_xor)
