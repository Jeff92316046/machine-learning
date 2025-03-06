from matplotlib import pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn import datasets

digits = datasets.load_digits()

(trainX, testX, trainY, testY) = train_test_split(digits.data, digits.target, test_size=0.25, random_state=42)

(trainX, valX, trainY, valY) = train_test_split(trainX, trainY, test_size=0.10, random_state=42)

k_values = range(1, 30, 2)
accuracies = []

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(trainX, trainY)

    score = model.score(valX, valY)
    print(f"k={k}, accuracy={score*100:.2f}%")
    accuracies.append(score)

best_k = k_values[accuracies.index(max(accuracies))]

final_model = KNeighborsClassifier(n_neighbors=best_k)
final_model.fit(trainX, trainY)

predictions = final_model.predict(testX)
print(classification_report(testY, predictions))
rng = np.random.default_rng(132423)
for i in rng.choice(len(testY), 5, replace=False):
    image = testX[i].reshape((8, 8))  
    prediction = final_model.predict([testX[i]])[0]

    plt.imshow(image, cmap="gray")
    plt.title(f"Predicted: {prediction}, Actual: {testY[i]}")
    plt.axis("off")
    plt.show()