import os
import shutil
import random
import yaml

# -----------------------
# CONFIGURATION
# -----------------------
LABEL_DIR = "dataset/labels"      # folder where your labeled .txt files are
IMAGE_DIR = "dataset/images"      # original images folder
OUTPUT_DIR = "dataset_ready"      # new folder for YOLO training
TRAIN_RATIO = 0.8                 # 80% train, 20% val

# -----------------------
# CREATE DIRECTORIES
# -----------------------
os.makedirs(os.path.join(OUTPUT_DIR, "images", "train"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "images", "val"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "labels", "train"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "labels", "val"), exist_ok=True)

# -----------------------
# SPLIT IMAGES
# -----------------------
all_images = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
random.shuffle(all_images)

#train_count = int(len(all_images) * TRAIN_RATIO)
train_count = max(1, int(len(all_images) * TRAIN_RATIO))

train_images = all_images[:train_count]
val_images = all_images[train_count:]

# -----------------------
# COPY FILES
# -----------------------
for img_list, subset in zip([train_images, val_images], ["train", "val"]):
    for img_file in img_list:
        # copy image
        shutil.copy(os.path.join(IMAGE_DIR, img_file),
                    os.path.join(OUTPUT_DIR, "images", subset, img_file))
        # copy label
        label_file = os.path.splitext(img_file)[0] + ".txt"
        src_label = os.path.join(LABEL_DIR, label_file)
        if os.path.exists(src_label):
            shutil.copy(src_label, os.path.join(OUTPUT_DIR, "labels", subset, label_file))

# -----------------------
# CREATE data.yaml
# -----------------------
# Load your classes
CLASS_FILE = "dataset/classes.json"
import json
with open(CLASS_FILE, "r") as f:
    classes = json.load(f)

data_yaml = {
    "train": "images/train",
    "val": "images/val",
    "nc": len(classes),
    "names": {i: name for i, name in enumerate(classes)}
}

with open(os.path.join(OUTPUT_DIR, "data.yaml"), "w") as f:
    yaml.dump(data_yaml, f)

print("Dataset prepared!")
print(f"Train images: {len(train_images)}, Val images: {len(val_images)}")
print(f"data.yaml created at {os.path.join(OUTPUT_DIR, 'data.yaml')}")
