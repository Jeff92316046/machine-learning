# import the necessary packages
from keras.models import load_model
from pyimagesearch.preprocessing import imagetoarraypreprocessor
from pyimagesearch.preprocessing import SimplePreprocessor
from imutils import paths
import numpy as np
import argparse
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset")
ap.add_argument("-m", "--model", required=True,
                help="path to pre-trained model")
args = vars(ap.parse_args())

# initialize the class labels
classLabels = ["cat", "dog", "panda"]

# grab the paths to the images in our dataset
print("[INFO] sampling images...")
imagePaths = list(paths.list_images(args["dataset"]))
imagePaths = np.random.choice(imagePaths, size=(10,), replace=False)

# load the pre-trained network
print("[INFO] loading pre-trained network...")
model = load_model(args["model"])

# loop over the sample images
for imagePath in imagePaths:
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    orig = image.copy()
    cv2.imshow("orig",orig)
    sp = SimplePreprocessor(32, 32)
    iap = imagetoarraypreprocessor.ImageToArrayPreprocessor()
    image = iap.preprocess(sp.preprocess(image))
    image = image.astype("float") / 255.0
    image = image.reshape(1, 32, 32, 3)

    # make a prediction on the image
    print("[INFO] predicting...")
    preds = model.predict(image)

    # find the class label index with the largest corresponding
    # probability
    j = np.argmax(preds)
    label = classLabels[j]

    # draw the class label + probability on the output image
    proba = preds[0][j] * 100
    cv2.putText(orig, f"Label: {label} ({proba:.2f}%)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # display the image
    cv2.imshow("Image", orig)
    cv2.waitKey(0)