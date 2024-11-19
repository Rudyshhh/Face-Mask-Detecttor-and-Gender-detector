import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import filedialog
import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import imutils
import time
import os

def load_models():
    face_detector_path = "face_detector"
    mask_detector_path = "mask_detector.model"
    gender_model_path = "Gender-Detection-master/gender_detection.model"
    prototxtPath = os.path.sep.join([face_detector_path, "deploy.prototxt"])
    weightsPath = os.path.sep.join([face_detector_path, "res10_300x300_ssd_iter_140000.caffemodel"])
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
    maskNet = load_model(mask_detector_path)
    genderNet = load_model(gender_model_path)
    return faceNet, maskNet, genderNet

def detect_and_predict(frame, faceNet, model, task):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()
    faces, locs = [], []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224)) if task == "mask" else cv2.resize(face, (96, 96))
            face = img_to_array(face)
            face = preprocess_input(face) if task == "mask" else face / 255.0
            face = np.expand_dims(face, axis=0)
            faces.append(face)
            locs.append((startX, startY, endX, endY))
    preds = []
    if len(faces) > 0:
        faces = np.vstack(faces)
        preds = model.predict(faces)
    return locs, preds

def process_image(image_path, faceNet, model, task):
    image = cv2.imread(image_path)
    locs, preds = detect_and_predict(image, faceNet, model, task)
    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        if task == "mask":
            (mask, withoutMask) = pred
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
        else:
            classes = ["Man", "Woman"]
            idx = np.argmax(pred)
            label = f"{classes[idx]}: {pred[idx] * 100:.2f}%"
            color = (0, 255, 255) if classes[idx] == "Man" else (255, 0, 255)
        cv2.putText(image, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
    cv2.imshow(f"{task.capitalize()} Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_video(video_path, faceNet, model, task):
    vs = cv2.VideoCapture(video_path)
    while True:
        ret, frame = vs.read()
        if not ret:
            break
        frame = imutils.resize(frame, width=400)
        locs, preds = detect_and_predict(frame, faceNet, model, task)
        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            if task == "mask":
                (mask, withoutMask) = pred
                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
            else:
                classes = ["Man", "Woman"]
                idx = np.argmax(pred)
                label = f"{classes[idx]}: {pred[idx] * 100:.2f}%"
                color = (0, 255, 255) if classes[idx] == "Man" else (255, 0, 255)
            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        cv2.imshow(f"{task.capitalize()} Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vs.release()
    cv2.destroyAllWindows()

def process_webcam(faceNet, model, task):
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        locs, preds = detect_and_predict(frame, faceNet, model, task)
        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            if task == "mask":
                (mask, withoutMask) = pred
                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
            else:
                classes = ["Man", "Woman"]
                idx = np.argmax(pred)
                label = f"{classes[idx]}: {pred[idx] * 100:.2f}%"
                color = (0, 255, 255) if classes[idx] == "Man" else (255, 0, 255)
            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        cv2.imshow(f"{task.capitalize()} Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vs.stop()
    cv2.destroyAllWindows()

def select_task():
    def proceed(task):
        global current_task
        current_task = task
        root.destroy()
    root = ttk.Window(title="Detection Mode", themename="superhero")
    root.geometry("400x200")
    ttk.Label(root, text="Choose Detection Mode", font=("Helvetica", 16), bootstyle=PRIMARY).pack(pady=20)
    ttk.Button(root, text="Mask Detection", command=lambda: proceed("mask"), bootstyle=SUCCESS).pack(pady=10)
    ttk.Button(root, text="Gender Detection", command=lambda: proceed("gender"), bootstyle=INFO).pack(pady=10)
    root.mainloop()

def create_ui():
    def select_input_mode(mode):
        if mode == "image":
            image_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.jpg *.png")])
            if image_path:
                process_image(image_path, faceNet, model, current_task)
        elif mode == "video":
            video_path = filedialog.askopenfilename(title="Select a Video", filetypes=[("Video Files", "*.mp4 *.avi")])
            if video_path:
                process_video(video_path, faceNet, model, current_task)
        elif mode == "webcam":
            process_webcam(faceNet, model, current_task)
    root = ttk.Window(title="Detection App", themename="superhero")
    root.geometry("400x300")
    ttk.Label(root, text=f"{current_task.capitalize()} Detection", font=("Helvetica", 16), bootstyle=PRIMARY).pack(pady=20)
    ttk.Button(root, text="Image", command=lambda: select_input_mode("image"), bootstyle=SUCCESS).pack(pady=10)
    ttk.Button(root, text="Video", command=lambda: select_input_mode("video"), bootstyle=INFO).pack(pady=10)
    ttk.Button(root, text="Webcam", command=lambda: select_input_mode("webcam"), bootstyle=DANGER).pack(pady=10)
    root.mainloop()

faceNet, maskNet, genderNet = load_models()
current_task = None
select_task()
model = maskNet if current_task == "mask" else genderNet
create_ui()
