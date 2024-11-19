from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import cvlib as cv

# Load the pre-trained gender detection model
model = load_model('Gender-Detection-master/gender_detection.model')

# Path to the folder containing images
image_folder = 'Mask'  # Replace with your image folder path
output_folder = 'Mask'  # Folder to save output images

# Gender classes
classes = ['man', 'woman']

# Check if output folder exists, if not, create it
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop through all image files in the folder
for image_name in os.listdir(image_folder):
    # Check if the file is an image (optional, can be modified based on image extensions)
    if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Read the image
        image_path = os.path.join(image_folder, image_name)
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

            # Preprocessing for the gender detection model
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
            cv2.putText(
                frame, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )

        # Save the processed image with gender detection label
        output_path = os.path.join(output_folder, image_name)
        cv2.imwrite(output_path, frame)

        # Optionally, display the output image
        cv2.imshow("Gender Detection", frame)

        # Press "Q" to stop (for each image, press Q to close)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            continue

# Close all OpenCV windows
cv2.destroyAllWindows()
