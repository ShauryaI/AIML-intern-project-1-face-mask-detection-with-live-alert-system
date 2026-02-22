import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=no info, 2=no warnings, 3=no errors
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import cv2
import numpy as np
import pygame
from flask import Flask, render_template, Response
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Initialize Flask and Pygame Alert
app = Flask(__name__)
pygame.mixer.init()
try:
    alert_sound = pygame.mixer.Sound("alert.wav")
except:
    alert_sound = None

# Load your trained model and OpenCV's built-in face detector
model = load_model("model-010.keras")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def gen_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # 1. Processing (Same as your detect script)
            frame = cv2.resize(frame, (600, 400))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            for (x, y, w, h) in faces:
                face_img = frame[y:y + h, x:x + w]
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                face_img = cv2.resize(face_img, (128, 128))
                face_img = img_to_array(face_img)
                face_img = preprocess_input(face_img)
                face_img = np.expand_dims(face_img, axis=0)

                # 2. Prediction
                prediction = model.predict(face_img, verbose=0)[0][0]
                label = "With Mask" if prediction < 0.5 else "NO MASK DETECTED"
                color = (0, 255, 0) if label == "With Mask" else (0, 0, 255)

                # 3. Alert Logic
                if label == "NO MASK DETECTED":
                    if alert_sound and not pygame.mixer.get_busy():
                        alert_sound.play()

                # 4. Drawing
                conf = (1 - prediction) if label == "With Mask" else prediction
                cv2.putText(frame, f"{label}: {conf * 100:.2f}%", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Encode the frame for the browser
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)