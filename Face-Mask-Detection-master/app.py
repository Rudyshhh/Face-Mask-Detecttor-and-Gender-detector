import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import imutils
import time
import os

# Load models
def load_models(face_detector_path, mask_detector_path):
    prototxtPath = os.path.sep.join([face_detector_path, "deploy.prototxt"])
    weightsPath = os.path.sep.join([face_detector_path, "res10_300x300_ssd_iter_140000.caffemodel"])
    net = cv2.dnn.readNet(prototxtPath, weightsPath)
    model = load_model(mask_detector_path)
    return net, model

# Detect and predict mask
def detect_and_predict_mask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locs = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)  # Expand dimensions for batch processing

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # Process multiple faces if detected
    preds = []
    if len(faces) > 0:
        faces = np.vstack(faces)  # Convert list to a 4D tensor
        preds = maskNet.predict(faces)  # Predict masks for each face

    return locs, preds

# Process image input
def process_image(image_path, faceNet, maskNet):
    image = cv2.imread(image_path)
    orig = image.copy()
    (h, w) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locs = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = image[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    if len(faces) > 0:
        faces = np.vstack(faces)  # Stack faces into a single batch
        preds = maskNet.predict(faces)

    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        cv2.putText(image, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

    return image

# Process video input
def process_video(video_path, faceNet, maskNet):
    vs = cv2.VideoCapture(video_path)
    while True:
        ret, frame = vs.read()
        if not ret:
            break

        frame = imutils.resize(frame, width=400)
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

        for box, pred in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        cv2.imshow("Video", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    vs.release()
    cv2.destroyAllWindows()

# Process webcam input
def process_webcam(faceNet, maskNet):
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

        for box, pred in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        cv2.imshow("Webcam", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    vs.stop()
    cv2.destroyAllWindows()

# UI functions
def select_image():
    image_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.jpg *.png")])
    if image_path:
        image = process_image(image_path, faceNet, maskNet)
        cv2.imshow("Mask Detection", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def select_video():
    video_path = filedialog.askopenfilename(title="Select a Video", filetypes=[("Video Files", "*.mp4 *.avi")])
    if video_path:
        process_video(video_path, faceNet, maskNet)

def start_webcam():
    process_webcam(faceNet, maskNet)

def on_select_mode(mode):
    if mode == "image":
        select_image()
    elif mode == "video":
        select_video()
    elif mode == "webcam":
        start_webcam()

# Initialize GUI
def create_ui():
    root = tk.Tk()
    root.title("Face Mask Detection")

    label = tk.Label(root, text="Select the Input Mode:")
    label.pack(pady=10)

    image_button = tk.Button(root, text="Select Image", command=lambda: on_select_mode("image"))
    image_button.pack(pady=5)

    video_button = tk.Button(root, text="Select Video", command=lambda: on_select_mode("video"))
    video_button.pack(pady=5)

    webcam_button = tk.Button(root, text="Use Webcam", command=lambda: on_select_mode("webcam"))
    webcam_button.pack(pady=5)

    root.mainloop()

# Load models
faceNet, maskNet = load_models("face_detector", "mask_detector.model")

# Run UI
create_ui()
