import os
import shutil
import re
from collections import defaultdict

# Path to the folder with images and txts
data_folder = os.path.join("bear", "bear")

# Pattern to match filenames ending with two uppercase letters before extension
pattern = re.compile(r".*_(\w{2})\.(jpg|jpeg|png|txt)$", re.IGNORECASE)

# Dictionary to group file paths by the two-letter suffix
grouped_files = defaultdict(list)

# Walk through the data folder
for filename in os.listdir(data_folder):
    match = pattern.match(filename)
    if match:
        suffix = match.group(1).upper()
        full_path = os.path.join(data_folder, filename)
        grouped_files[suffix].append(full_path)

# Create folders and move files
for suffix, files in grouped_files.items():
    target_folder = os.path.join(os.getcwd(), suffix)
    os.makedirs(target_folder, exist_ok=True)
    for file_path in files:
        shutil.move(file_path, os.path.join(target_folder, os.path.basename(file_path)))

print("Files reorganized into folders based on suffix.")
