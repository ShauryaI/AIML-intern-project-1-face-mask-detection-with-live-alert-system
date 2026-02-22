import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import pygame

# Initialize Alert System
pygame.mixer.init()
alert_sound = pygame.mixer.Sound("alert.wav")

# Load your trained model
model_path = "model-010.keras"
if not os.path.exists(model_path):
    print(f"Error: {model_path} not found!")
else:
    model = load_model(model_path)

# Load OpenCV's built-in face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start Webcam
print("[INFO] Starting video stream... Press 'q' to quit.")
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

        # Format the text with confidence
        conf = (1 - prediction) if label == "With Mask" else prediction
        label_text = f"{label}: {conf * 100:.2f}%"

        if mask_violation:
            # Play beep sound if it's not already playing
            if alert_sound and not pygame.mixer.get_busy():
                alert_sound.play()
                print("[ALERT] No Mask Detected!")

        # Draw box and label
        cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    cv2.imshow("Face Mask Detector", frame)

    # Check if 'q' is pressed OR if the window "X" icon is clicked
    # WND_PROP_VISIBLE checks if the window is still open
    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty("Face Mask Detector", cv2.WND_PROP_VISIBLE) < 1:
        break

vs.release() # Cleanup: Calling vs.release() is critical; otherwise, your webcam light might stay on even after the window disappears.
cv2.destroyAllWindows()
# Sometimes the window "hangs" after closing. Adding a final cv2.waitKey(1) after destroyAllWindows() forces the operating system to finish the closing process immediately.