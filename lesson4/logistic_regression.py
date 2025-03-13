import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.5)
X = lfw_people.data
y = lfw_people.target
images = lfw_people.images
target_names = lfw_people.target_names

X_train, X_test, y_train, y_test, images_train, images_test = train_test_split(
    X, y, images, test_size=0.25, random_state=42)

clf = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='auto')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("準確率：{:.2f}%".format(accuracy * 100))
print(classification_report(y_test, y_pred, target_names=target_names))

def plot_face(image, title=""):
    plt.imshow(image, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()

for i in range(5):
    idx = i
    true_name = target_names[y_test[idx]]
    pred_name = target_names[y_pred[idx]]
    title = f"true: {true_name} - predict: {pred_name}"
    plot_face(images_test[idx], title)