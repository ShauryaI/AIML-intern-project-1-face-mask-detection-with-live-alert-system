# Find Low Resolution Images and Remove them
import os
from PIL import Image

# Configuration
dataset_path = "dataset/"
min_size = 80  # Anything smaller than 80x80 is considered "noise"
trash_folder = "low_res_trash/"

if not os.path.exists(trash_folder):
    os.makedirs(trash_folder)

count = 0

# Scanning and Filtering
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(root, file)

            # Initialize variables to avoid errors
            width, height = 0, 0
            is_low_res = False

            try:
                # Open, get size, and IMMEDIATELY close
                with Image.open(file_path) as img:
                    width, height = img.size

                # Now the file is CLOSED. It is safe to move.
                # If either dimension is too small, it's "Noisy"
                if width < min_size or height < min_size:
                    is_low_res = True
                    print(f"Found Blur: {file} ({width}x{height})")

                    # Move to trash folder instead of permanent delete, below code was giving error - The process cannot access the file because it is being used by another process
                    # os.rename(file_path, os.path.join(trash_folder, file))
                    # count += 1
            except Exception as e:
                print(f"Skipping corrupted file {file}: {e}")

            # MOVE THE FILE OUTSIDE THE 'WITH' BLOCK
            if is_low_res:
                try:
                    destination = os.path.join(trash_folder, file)
                    # Use os.replace instead of os.rename for better Windows compatibility
                    os.replace(file_path, destination)
                    print(f"Moved Blur: {file} ({width}x{height})")
                    count += 1
                except OSError as e:
                    print(f"Error moving {file}: {e}. Try closing your File Explorer or Photo Viewer.")

print(f"\n[CLEANUP COMPLETE] Removed {count} low-resolution images.")

'''
Result:
Found Blur: without_mask_1527.jpg (64x64)
Moved Blur: without_mask_1527.jpg (64x64)
and the list continues
'''
