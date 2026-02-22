# AIML-intern-project-1-face-mask-detection-with-live-alert-system
Objective: Detect if people are wearing face masks in real-time using a webcam.

## Actions Performed ##
1. Set Up Virtual Environment to avoid library conflicts. 
2. Collect or use Kaggle dataset of masked/unmasked faces
   Folder - dataset/with_mask dataset/without_mask
3. Train CNN model using Keras
4. Integrate model with OpenCV video stream
5. Use Haar Cascades for face detection
6. Add logic to alert when no mask is detected
7. Deploy with Flask

## File/Folder Description ##
1. Dataset contains separate folder for masked/unmasked photos.
   - with_mask - 3725 images
   - without_mask - 3828 images
2. low_res_trash - low resolution images and shifted to this folder instead of deleting.
3. templates/index.html - browser file for webcam streaming.
4. accuracy_plot.png, loss_plot.png - Trained model result plotting 
5. alert.wav - sound to play for unmasked identification. Audio created via create_audio.py
6. data_cleaning_s1.py - to find corrupted or broken images and remove them.
7. data_cleaning_s2.py - to find duplicate images
8. data_cleaning_s3.py - to find low resolution images and remove them.
9. train_mask_detector.py - script to train the model.
10. detect_mask_webcam.py - script to detect mask/unmask by camera.
11. flask_app.py - script to detect face mask and deploy as flask app.
12. model2-001.keras and so on - files saved in middle of training with best results found.
13. model.010.keras - trained model.
14. requirements.txt - file mentioning project dependencies.

## Deliverables ##
1. Trained model: model-010.keras
2. real-time detection script: flask_app.py
3. short video demo - N/A as the repo is public
4. Project report: face-mask-detection-system.pdf