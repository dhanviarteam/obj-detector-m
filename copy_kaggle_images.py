import os
import shutil

KAGGLE_PATH = r"C:\Users\admin\.cache\kagglehub\datasets\asdasdasasdas\garbage-classification\versions\2"
DEST_IMAGE_DIR = r"D:\Dhanvi\project\training\dataset\images"

os.makedirs(DEST_IMAGE_DIR, exist_ok=True)

copied = 0

for root, dirs, files in os.walk(KAGGLE_PATH):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            src = os.path.join(root, file)
            dst = os.path.join(DEST_IMAGE_DIR, file)

            # avoid overwrite
            if os.path.exists(dst):
                name, ext = os.path.splitext(file)
                dst = os.path.join(DEST_IMAGE_DIR, f"{name}_{copied}{ext}")

            shutil.copy(src, dst)
            copied += 1

print(f"✅ Copied {copied} images into dataset/images")
