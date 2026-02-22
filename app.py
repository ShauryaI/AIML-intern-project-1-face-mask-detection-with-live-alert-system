# This is a running code with EXE creation. Version issues exists. Can be worked on it in future projects.
import sys
import types

# Mock the Module (Prevents PackageNotFoundError in EXE creation)
if 'tensorflow' not in sys.modules:
    # Create a dummy module to satisfy the metadata check
    mock_tf = types.ModuleType('tensorflow')
    mock_tf.__version__ = '2.16.1'
    sys.modules['tensorflow'] = mock_tf

# Mock the Metadata check
try:
    import importlib.metadata as metadata
except ImportError:
    import importlib_metadata as metadata

original_version = metadata.version
def patched_version(package):
    if package == 'tensorflow':
        return '2.16.1'
    return original_version(package)

metadata.version = patched_version

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=no info, 2=no warnings, 3=no errors
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import cv2
import numpy as np
import pygame
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# PATH HELPER (Crucial for .exe deployment)
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# Initialize Pygame Alert
pygame.mixer.init()
try:
    alert_sound = pygame.mixer.Sound("alert.wav")
except:
    alert_sound = None

# Load your trained model and OpenCV's built-in face detector
model = load_model("model-010.keras")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# DETECTION LOOP (Standard OpenCV)
vs = cv2.VideoCapture(0)

while True:
    ret, frame = vs.read()
    if not ret:
        break

    # Resize frame for faster processing on laptop
    frame = cv2.resize(frame, (600, 400))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        # Extract and preprocess the face
        face_img = frame[y:y+h, x:x+w]
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = cv2.resize(face_img, (128, 128)) # Must match your training size
        face_img = img_to_array(face_img)
        face_img = preprocess_input(face_img)
        face_img = np.expand_dims(face_img, axis=0)

        # Predict
        prediction = model.predict(face_img, verbose=0)[0][0]

        # Determine Label (0=Mask, 1=No Mask usually)
        label = "With Mask" if prediction < 0.5 else "NO MASK DETECTED"
        color = (0, 255, 0) if label == "With Mask" else (0, 0, 255)
        mask_violation = False if label == "With Mask" else True

        conf = (1 - prediction) if label == "With Mask" else prediction
        label_text = f"{label}: {conf * 100:.2f}%"

        if mask_violation:
            if alert_sound and not pygame.mixer.get_busy():
                alert_sound.play()
                print("[ALERT] No Mask Detected!")

        # Draw box and label
        cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    cv2.imshow("Face Mask Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty("Face Mask Detector", cv2.WND_PROP_VISIBLE) < 1:
        break

vs.release()
cv2.destroyAllWindows()