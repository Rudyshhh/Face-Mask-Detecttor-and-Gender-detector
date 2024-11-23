
---

# UniMask: Face Mask & Gender Detection Application

UniMask is a powerful tool that detects face masks and determines gender (man or woman) using deep learning techniques. The application offers an intuitive interface and supports input from images, videos, and real-time webcam feeds.

---

## Features

- **Mask Detection**: Identifies whether a person is wearing a face mask and provides the confidence level.
- **Gender Detection**: Classifies gender (man or woman) from detected faces.
- **Input Options**:
  - Image files (e.g., JPG, PNG)
  - Video files (e.g., MP4, AVI)
  - Real-time webcam feed
- **Sleek GUI**: Built with **CustomTkinter** for a modern and user-friendly experience.

---

## Requirements

### Python Version
- Python 3.7 or higher is recommended.

### Dependencies
Install the required packages using the following command:
```bash
pip install -r requirements.txt
```

#### Key Libraries
- **CustomTkinter**: For creating the graphical interface.
- **OpenCV**: For image and video processing.
- **TensorFlow/Keras**: For loading and running deep learning models.
- **Imutils**: For additional image processing utilities.
- **NumPy**: For numerical operations.

---

## Installation Guide

### Step 1: Clone the Repository
Clone the project repository to your local system:


### Step 2: Prepare Model Files
Ensure the following files are in the project directory:
1. **Face Detection Files**:
   - `face_detector/deploy.prototxt`
   - `face_detector/res10_300x300_ssd_iter_140000.caffemodel`
2. **Mask Detection Model**:
   - `mask_detector.model`
3. **Gender Detection Model**:
   - `Gender-Detection/gender_detection.model`

### Step 3: Install Dependencies
Install the required Python libraries:
```bash
pip install -r requirements.txt
```

### Step 4: Run the Application
Launch the application:
```bash
python app3.py
```

---

## Directory Structure

```
UniMask/
├── face_detector/
│   ├── deploy.prototxt
│   ├── res10_300x300_ssd_iter_140000.caffemodel
├── Gender-Detection-master/
│   ├── gender_detection.model
├── mask_detector.model
├── app3.py
├── requirements.txt
├── README.md
```

---

## Usage Instructions

### Step 1: Launch the Application
Run the command:
```bash
python app3.py
```
A graphical user interface (GUI) will appear.

### Step 2: Choose a Detection Task
Select either:
- **Mask Detection**: To detect if individuals in the input are wearing a mask.
- **Gender Detection**: To classify gender.

### Step 3: Select Input Mode
Choose from:
1. **Image**: Upload an image file for analysis.
2. **Video**: Upload a video file for analysis.
3. **Webcam**: Activate the webcam for real-time detection.

### Step 4: View Results
The application will display the processed image/video with bounding boxes, labels, and confidence percentages.

---

## How It Works

1. **Face Detection**:
   - Detects faces in the input using a model configured to identify facial regions.
2. **Mask Detection**:
   - Analyzes each detected face to determine if it is wearing a mask.
3. **Gender Detection**:
   - Classifies detected faces as "Man" or "Woman" based on the facial features.
4. **GUI**:
   - Provides a simple interface for task selection and input mode configuration.

---

## Example Use Cases

1. **Mask Detection**:
   - Monitor mask compliance in public areas.
2. **Gender Detection**:
   - Analyze demographics in images and videos for research purposes.

---

## Screenshots

### Main Menu
![Screenshot 2024-11-17 044736](https://github.com/user-attachments/assets/3d77090d-8d86-4094-950e-2ff21a001df0)

![Screenshot 2024-11-17 044709](https://github.com/user-attachments/assets/1c10a2c1-0431-4dbb-b81a-15170d42018c)


### Detection Results

![Screenshot 2024-11-17 044945](https://github.com/user-attachments/assets/81093515-d7b5-4a6c-8bb6-85f2fe54325a)

![Screenshot 2024-11-17 045247](https://github.com/user-attachments/assets/356ee68e-be8b-4d34-b657-66c0d8e949fb)

![Screenshot 2024-11-17 045214](https://github.com/user-attachments/assets/23375256-a31f-43b2-a90f-ff1c30d28b29)



---

## Troubleshooting

- **No Face Detected**:
  - Ensure the input image/video has clear and visible faces.
- **Low Performance**:
  - Use smaller images or videos to improve processing speed.
- **Model File Errors**:
  - Verify that all model files are in the correct directories.

---
