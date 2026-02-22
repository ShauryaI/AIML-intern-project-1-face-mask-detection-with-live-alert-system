# Find Corrupted or Broken Images and Remove them

import os
from PIL import Image

dataset_path = "dataset/"
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(('.jpg', '.jpeg', '.png')):
            try:
                img = Image.open(os.path.join(root, file))
                img.verify() # Check if the file is broken
            except (IOError, SyntaxError) as e:
                print(f'Bad file removed: {file}')
                os.remove(os.path.join(root, file))

# Result: None found