# Your main Flask/OpenCV code - the very first attempt to understand/create errors.
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Also hides general TF info messages

import cv2
import os
import numpy as np
from keras.utils import to_categorical
from keras.models import load_model
from flask import Flask, Response
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from matplotlib import pyplot as plt

data_path = 'dataset' # Path containing 'with_mask' and 'without_mask'
categories = os.listdir(data_path)
labels = [i for i in range(len(categories))]
label_dict = dict(zip(categories, labels))

data = []
target = []

# Preprocessing (Resize & Grayscale)
for category in categories:
    folder_path = os.path.join(data_path, category)
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (100, 100))
            data.append(resized)
            target.append(label_dict[category])
        except Exception as e:
            continue

data = np.array(data) / 255.0 # Normalization
data = np.reshape(data, (data.shape[0], 100, 100, 1))
target = to_categorical(np.array(target))

# Load the pre-trained base model (WITHOUT the top classification layer)
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# FREEZE the base model so we don't destroy the pre-learned patterns
base_model.trainable = False

# Create your custom head for Mask vs. No Mask and Train CNN Model
model = models.Sequential([
    base_model,
    layers.AveragePooling2D(pool_size=(7, 7)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5), # Combined with Dropout as we discussed!
    layers.Dense(2, activation="softmax") # 2 classes: Mask and No Mask
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
checkpoint = ModelCheckpoint('model-{epoch:03d}.keras', monitor='val_loss', save_best_only=True, mode='auto')
history = model.fit(data, target, epochs=8, validation_split=0.2, batch_size=64, callbacks=[checkpoint]) # Laptop freezes, so reduce epochs and add batch size, 94% reached in step 6.
# validation_split=0.2  # This creates the 'val_accuracy' data
model.save('model-008.keras') # Generated after training

# Plot Training vs Validation Accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot Training vs Validation Loss
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Step 5: Load Haar Cascade for face detection
model = load_model('model-008.keras')
face_clsfr = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

labels_dict = {0: 'MASK', 1: 'NO MASK'}
color_dict = {0: (0, 255, 0), 1: (0, 0, 255)}  # Green for Mask, Red for No Mask

def detect_mask(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Grayscale
    faces = face_clsfr.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = gray[y:y + h, x:x + w]
        resized = cv2.resize(face_img, (100, 100))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, 100, 100, 1))
        result = model.predict(reshaped)

        label = np.argmax(result, axis=1)[0]

        # Step 6: Logic to alert (Red Box + Text for No Mask)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color_dict[label], 2)
        cv2.putText(frame, labels_dict[label], (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return frame

app = Flask(__name__)
source = cv2.VideoCapture(0)

def gen_frames():
    while True:
        success, frame = source.read()
        if not success:
            break
        else:
            # Process the frame using our detection logic
            frame = detect_mask(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return "<h1>Face Mask Detection Stream</h1><img src='/video_feed'>"

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)