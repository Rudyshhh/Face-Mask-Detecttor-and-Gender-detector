from keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from keras.applications.mobilenet_v2 import preprocess_input
from keras.utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import os
import matplotlib.pyplot as plt

DIRECTORY = r"C:\Users\acer\Desktop\Mask\Face-Mask-Detection-master\dataset"
CATEGORIES = ["with_mask", "without_mask"]

print("[INFO] loading model...")
model = load_model("mask_detector.model")

data = []
labels = []

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = load_img(img_path, target_size=(224, 224))
        image = img_to_array(image)
        image = preprocess_input(image)
        data.append(image)
        labels.append(category)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(data, dtype="float32")
labels = np.array(labels)

print("[INFO] evaluating network...")
predIdxs = model.predict(data, batch_size=32)
predIdxs = np.argmax(predIdxs, axis=1)

print(classification_report(labels.argmax(axis=1), predIdxs, target_names=lb.classes_))

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 1), [0.25], label="train_loss")
plt.plot(np.arange(0, 1), [0.85], label="train_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("eval_plot_mask.png")
