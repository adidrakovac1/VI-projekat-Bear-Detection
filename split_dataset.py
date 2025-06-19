import os
import random
import shutil

# === USER INPUT ===
SOURCE_FOLDER = input("Enter the name of the folder where all images and txts are located: ").strip()
OUTPUT_FOLDER = 'yolo_dataset'
SEED = 42

# === CHECK EXISTENCE ===
if not os.path.exists(SOURCE_FOLDER):
    print(f"❌ Folder '{SOURCE_FOLDER}' does not exist.")
    exit()

# === CREATE STRUCTURE ===
splits = ['train', 'val', 'test']
for split in splits:
    os.makedirs(os.path.join(OUTPUT_FOLDER, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_FOLDER, split, 'labels'), exist_ok=True)

# === GET IMAGE FILES ===
all_images = [f for f in os.listdir(SOURCE_FOLDER) if f.lower().endswith('.jpg')]
all_images.sort()

# === SHUFFLE ===
random.seed(SEED)
random.shuffle(all_images)

# === SPLIT ===
n = len(all_images)
train_end = int(0.8 * n)
val_end = int(0.9 * n)

split_files = {
    'train': all_images[:train_end],
    'val': all_images[train_end:val_end],
    'test': all_images[val_end:]
}

# === MOVE FILES & HANDLE LABELS ===
for split, files in split_files.items():
    for img_file in files:
        label_file = img_file.rsplit('.', 1)[0] + '.txt'
        src_img = os.path.join(SOURCE_FOLDER, img_file)
        src_lbl = os.path.join(SOURCE_FOLDER, label_file)

        dst_img = os.path.join(OUTPUT_FOLDER, split, 'images', img_file)
        dst_lbl = os.path.join(OUTPUT_FOLDER, split, 'labels', label_file)

        # Copy image
        shutil.copy2(src_img, dst_img)

        # Copy label or create empty label
        if os.path.exists(src_lbl):
            shutil.copy2(src_lbl, dst_lbl)
        else:
            # Create empty label file
            open(dst_lbl, 'w').close()

print("✅ Dataset split complete and organized into 'yolo_dataset/' folder.")
