import tkinter as tk
from tkinter import filedialog, messagebox
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import cvlib as cv

# Load the pre-trained gender detection model
model = load_model('Gender-Detection-master/gender_detection.model')

# Gender classes
classes = ['man', 'woman']

# Initialize output folder
output_folder = 'Output'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Function to process images
def process_image(image_path):
    # Read the image
    frame = cv2.imread(image_path)

    # Apply face detection
    face, confidence = cv.detect_face(frame)

    # Loop through detected faces
    for idx, f in enumerate(face):
        # Get corner points of face rectangle
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        # Draw rectangle over face
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # Crop the detected face region
        face_crop = np.copy(frame[startY:endY, startX:endX])

        # Skip small face regions
        if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
            continue

        # Preprocessing for gender detection model
        face_crop = cv2.resize(face_crop, (96, 96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        # Apply gender detection on the face
        conf = model.predict(face_crop)[0]

        # Get label with max accuracy
        idx = np.argmax(conf)
        label = classes[idx]
        label = "{}: {:.2f}%".format(label, conf[idx] * 100)

        Y = startY - 10 if startY - 10 > 10 else startY + 10

        # Write label and confidence above face rectangle
        cv2.putText(frame, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Save the processed image with gender detection label
    output_path = os.path.join(output_folder, os.path.basename(image_path))
    cv2.imwrite(output_path, frame)

    # Optionally, display the output image
    cv2.imshow("Gender Detection", frame)

    # Wait for user to press "Q" to close
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

# Function to process video
def process_video(video_path):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Get video properties
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for the output video

    # Output video file
    output_path = os.path.join(output_folder, 'output_video.mp4')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Loop through frames in the video
    while video.isOpened():
        status, frame = video.read()
        if not status:
            break

        # Apply face detection
        face, confidence = cv.detect_face(frame)

        # Loop through detected faces
        for idx, f in enumerate(face):
            # Get corner points of face rectangle
            (startX, startY) = f[0], f[1]
            (endX, endY) = f[2], f[3]

            # Draw rectangle over face
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

            # Crop the detected face region
            face_crop = np.copy(frame[startY:endY, startX:endX])

            # Skip small face regions
            if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
                continue

            # Preprocessing for gender detection model
            face_crop = cv2.resize(face_crop, (96, 96))
            face_crop = face_crop.astype("float") / 255.0
            face_crop = img_to_array(face_crop)
            face_crop = np.expand_dims(face_crop, axis=0)

            # Apply gender detection on the face
            conf = model.predict(face_crop)[0]

            # Get label with max accuracy
            idx = np.argmax(conf)
            label = classes[idx]
            label = "{}: {:.2f}%".format(label, conf[idx] * 100)

            Y = startY - 10 if startY - 10 > 10 else startY + 10

            # Write label and confidence above face rectangle
            cv2.putText(frame, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Write the processed frame to the output video
        out.write(frame)

        # Display the output frame
        cv2.imshow("Gender Detection", frame)

        # Press "Q" to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    out.release()
    cv2.destroyAllWindows()

# Function to use webcam for real-time detection
def start_webcam():
    webcam = cv2.VideoCapture(0)
    while webcam.isOpened():
        status, frame = webcam.read()

        # Apply face detection
        face, confidence = cv.detect_face(frame)

        # Loop through detected faces
        for idx, f in enumerate(face):
            # Get corner points of face rectangle
            (startX, startY) = f[0], f[1]
            (endX, endY) = f[2], f[3]

            # Draw rectangle over face
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

            # Crop the detected face region
            face_crop = np.copy(frame[startY:endY, startX:endX])

            # Skip small face regions
            if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
                continue

            # Preprocessing for gender detection model
            face_crop = cv2.resize(face_crop, (96, 96))
            face_crop = face_crop.astype("float") / 255.0
            face_crop = img_to_array(face_crop)
            face_crop = np.expand_dims(face_crop, axis=0)

            # Apply gender detection on the face
            conf = model.predict(face_crop)[0]

            # Get label with max accuracy
            idx = np.argmax(conf)
            label = classes[idx]
            label = "{}: {:.2f}%".format(label, conf[idx] * 100)

            Y = startY - 10 if startY - 10 > 10 else startY + 10

            # Write label and confidence above face rectangle
            cv2.putText(frame, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display the output frame
        cv2.imshow("Gender Detection", frame)

        # Press "Q" to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    webcam.release()
    cv2.destroyAllWindows()

# UI to select input type and run processing
def create_ui():
    def select_image():
        image_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
        if image_path:
            process_image(image_path)

    def select_video():
        video_path = filedialog.askopenfilename(title="Select a Video", filetypes=[("Video Files", "*.mp4 *.avi")])
        if video_path:
            process_video(video_path)

    def start_webcam_stream():
        start_webcam()

    root = tk.Tk()
    root.title("Gender Detection")

    label = tk.Label(root, text="Select Input Mode:")
    label.pack(pady=10)

    image_button = tk.Button(root, text="Select Image", command=select_image)
    image_button.pack(pady=5)

    video_button = tk.Button(root, text="Select Video", command=select_video)
    video_button.pack(pady=5)

    webcam_button = tk.Button(root, text="Use Webcam", command=start_webcam_stream)
    webcam_button.pack(pady=5)

    root.mainloop()

# Run the GUI
create_ui()
