import customtkinter as ctk
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

current_task = None

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
        cv2.putText(image, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
        cv2.rectangle(image, (startX, startY), (endX, endY), color, 3)
    cv2.imshow(f"{task.capitalize()} Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_video(video_path, faceNet, model, task):
    vs = cv2.VideoCapture(video_path)
    while True:
        ret, frame = vs.read()
        if not ret:
            break
        frame = imutils.resize(frame, width=800)
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
            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 3)
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
        frame = imutils.resize(frame, width=800)
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
            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 3)
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
        create_ui()  

    root = ctk.CTk()  
    root.geometry("800x600")
    root.title("UniMask – Face Mask Detector and Gender Detector ")
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("green")
    ctk.CTkLabel(root, text="Select Detection Mode", font=("Helvetica", 32, "bold"), text_color="#E6C767").pack(pady=40)
    ctk.CTkButton(root, text="Mask Detection\nDetect if a person is wearing a mask", command=lambda: proceed("mask"), width=400, height=150, fg_color="#4C4B16", hover_color="#F87A53", text_color="#E6C767", font=("Helvetica", 20)).pack(pady=40)
    ctk.CTkButton(root, text="Gender Detection\nIdentify gender from image or video", command=lambda: proceed("gender"), width=400, height=150, fg_color="#898121", hover_color="#F87A53", text_color="#E6C767", font=("Helvetica", 20)).pack(pady=40)
    root.mainloop()

def create_ui():
    def select_input_mode(mode):
        if mode == "image":
            image_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
            if image_path:
                if current_task == "mask":
                    process_image(image_path, faceNet, maskNet, "mask")
                elif current_task == "gender":
                    process_image(image_path, faceNet, genderNet, "gender")
        elif mode == "video":
            video_path = filedialog.askopenfilename(title="Select a Video", filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
            if video_path:
                if current_task == "mask":
                    process_video(video_path, faceNet, maskNet, "mask")
                elif current_task == "gender":
                    process_video(video_path, faceNet, genderNet, "gender")
        elif mode == "webcam":
            if current_task == "mask":
                process_webcam(faceNet, maskNet, "mask")
            elif current_task == "gender":
                process_webcam(faceNet, genderNet, "gender")

    root = ctk.CTk()  
    root.geometry("800x600")
    root.title("Detection Mode")
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")

    ctk.CTkButton(root, text="Select Image\nUpload an image to analyze", command=lambda: select_input_mode("image"), width=400, height=80, fg_color="#4C4B16", hover_color="#F87A53", text_color="#E6C767", font=("Helvetica", 18)).pack(pady=25)
    ctk.CTkButton(root, text="Select Video\nUpload a video for analysis", command=lambda: select_input_mode("video"), width=400, height=80, fg_color="#898121", hover_color="#F87A53", text_color="#E6C767", font=("Helvetica", 18)).pack(pady=25)
    ctk.CTkButton(root, text="Use Webcam\nActivate webcam for real-time analysis", command=lambda: select_input_mode("webcam"), width=400, height=80, fg_color="#F87A53", hover_color="#E6C767", text_color="#4C4B16", font=("Helvetica", 18)).pack(pady=25)
    ctk.CTkButton(root, text="Switch Task", command=lambda: restart_task_selection(), width=400, height=50, fg_color="#5E5E5E", hover_color="#D6D6D6", text_color="#E6C767", font=("Helvetica", 16)).pack(pady=25)

    root.mainloop()

def restart_task_selection():
    global current_task
    current_task = None  
    select_task()  


faceNet, maskNet, genderNet = load_models()
select_task()
