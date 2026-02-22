# Find Duplicate Images and Remove them
'''
Method 1 (Install Dependent but got version issue even after installing via terminal and code)
pip install difpy and python -m pip install difpy

# Code starts
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    from difpy import dif
except ImportError:
    print("Installing difpy...")
    install('difpy')
    from difpy import dif

# 1. Search for duplicates in your dataset folder
# 'delete=True' will automatically remove them,
# but I recommend 'delete=False' first to see what it finds!
search = dif("dataset/", delete=False, silent=False)

# 2. Print the results
print(f"Found {len(search.result)} duplicate images.")
'''

'''
Method 2 (this "No-Install" Duplicate Finder)
'''

import os
import hashlib

def find_duplicates(dataset_path):
    hashes = {}
    duplicates = []

    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            path = os.path.join(root, file)
            # Create a unique "fingerprint" for the image
            with open(path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()

            if file_hash in hashes:
                duplicates.append((path, hashes[file_hash]))
            else:
                hashes[file_hash] = path
    return duplicates


# Run it on your folder
dataset = "dataset/"
dupes = find_duplicates(dataset)

print(f"Found {len(dupes)} exact duplicates.")
for dupe, original in dupes[:10]:  # Show first 10
    print(f"DUPLICATE: {dupe}  |  ORIGINAL: {original}")
    # os.remove(dupe) # Uncomment this line to delete them automatically

'''
Result:
Found 306 exact duplicates.
DUPLICATE: dataset/without_mask\without_mask_1149.jpg  |  ORIGINAL: dataset/without_mask\without_mask_1148.jpg
and list continues
Keep them and fix it via seed parameter
'''
