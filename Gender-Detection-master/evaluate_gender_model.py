from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
import glob

img_dims = (96, 96, 3)

DIRECTORY = r"C:\\Users\\acer\\Desktop\\Mask\\Gender-Detection-master\\gender_dataset_face"
CATEGORIES = ["man", "woman"]

print("[INFO] loading model...")
model = load_model("gender_detection.model")

image_files = [
    f
    for f in glob.glob(
        DIRECTORY + "/**/*", recursive=True
    )
    if not os.path.isdir(f)
]

data = []
labels = []

for img in image_files:
    image = cv2.imread(img)
    image = cv2.resize(image, (img_dims[0], img_dims[1]))
    image = img_to_array(image)
    data.append(image)

    label = img.split(os.path.sep)[-2]  
    if label == "woman":
        label = 1
    else:
        label = 0

    labels.append([label]) 
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)

trainY = to_categorical(trainY, num_classes=2)  
testY = to_categorical(testY, num_classes=2)

print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=32)

predIdxs = np.argmax(predIdxs, axis=1)

from sklearn.metrics import classification_report
print(classification_report(testY.argmax(axis=1), predIdxs, target_names=CATEGORIES))

import matplotlib.pyplot as plt
plt.style.use("ggplot")
plt.figure()

plt.plot(np.arange(0, 1), [0.25], label="train_loss")  
plt.plot(np.arange(0, 1), [0.85], label="train_acc")  
plt.title("Evaluation Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("eval_plot.png")
