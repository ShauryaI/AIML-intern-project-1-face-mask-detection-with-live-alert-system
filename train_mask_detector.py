import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
# Clear any old models from RAM
tf.keras.backend.clear_session()

# Tell TensorFlow to use memory "on-demand"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, Input, MaxPooling2D, Flatten, Dense,Dropout
from keras.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Build the neural network

model = Sequential([
    Input(shape=(128, 128, 3)), # Matches our smaller target_size
    Conv2D(16, (3, 3), activation='relu'),  # Fewer filters (16 instead of 32)
    MaxPooling2D(2, 2),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(64, activation='relu'), # Smaller dense layer (64 instead of 128)
    Dropout(0.5), # Dropout to prevent the 20% gap we saw earlier in app.py, # 0.5 means 50% of neurons are randomly disabled during each training step
    Dense(1, activation='sigmoid') # for binary
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Summary to check the shapes

model.summary()

# Image Data Generation / Augmentation

TRAINING_DIR = "dataset"
# Setup the Generator with a 20% split
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, # Use this instead of 1.0/255
                                   rotation_range=40, # Added augmentation because you have enough data
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   validation_split=0.2,  # This is the key! 20% of total images of both folder
                                   fill_mode='nearest')

# Setup Training Data (80% of images)
train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                    batch_size=8,  # Smaller batch = Less RAM usage, BEST fix for memory errors
                                                    #batch_size=64,          # Increased for faster training if needed
                                                    target_size=(128, 128),
                                                    class_mode='binary', # Use 'categorical' if you have more than 2 classes
                                                    subset='training',  # Sets this as the training set
                                                    seed=123  # <--- ENSURES CONSISTENCY
                                                    )

# Setup Validation Data (The remaining 20%)

VALIDATION_DIR = TRAINING_DIR # Points to the SAME folder
validation_generator = train_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(128, 128),
    batch_size=8,
    class_mode='binary',
    subset='validation',    # Sets this as the testing set
    seed=123,  # <--- ENSURES CONSISTENCY
)

# Initialize a callback checkpoint to keep saving best model after each epoch while training

checkpoint = ModelCheckpoint('model2-{epoch:03d}.keras',monitor='val_loss',verbose=0,save_best_only=True,mode='auto')

# Train the model
history = model.fit(train_generator,
                              steps_per_epoch=len(train_generator), # Total training images / batch_size
                              epochs=10, # With 7000 images, 10-15 epochs is usually enough
                              validation_data=validation_generator, # to see your epoch results for both sets, this gives you val_loss and val_accuracy
                              validation_steps=len(validation_generator), # Total validation images / batch_size
                              callbacks=[checkpoint])

model.save('model-010.keras') # Generated after training

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