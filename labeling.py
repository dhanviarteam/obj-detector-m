import cv2
from pathlib import Path

# =============================
# CONFIG
# =============================
IMAGE_PATH = "sample.jpg"
DATASET = Path("dataset")
VAL_EVERY = 5

# =============================
# AUTO CREATE DIRECTORIES
# =============================
for split in ["train", "val"]:
    (DATASET / "images" / split).mkdir(parents=True, exist_ok=True)
    (DATASET / "labels" / split).mkdir(parents=True, exist_ok=True)

# =============================
# LOAD / INIT CLASS MAP
# =============================
yaml_path = DATASET / "data.yaml"
class_map = {}

if yaml_path.exists():
    with open(yaml_path) as f:
        for line in f:
            if line.strip() and line.strip()[0].isdigit():
                cid, name = line.strip().split(":")
                class_map[name.strip()] = int(cid)

# =============================
# ACTIVE LABEL
# =============================
active_label = input("Enter active label name: ").strip()
if active_label not in class_map:
    class_map[active_label] = len(class_map)

# =============================
# LOAD IMAGE
# =============================
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError("Image not found")

h, w = img.shape[:2]
display = img.copy()

boxes = []
labels = []
drawing = False
current = []

# =============================
# MOUSE HANDLER
# =============================
def mouse(event, x, y, flags, param):
    global drawing, current

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        current = [x, y, x, y]

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        temp = display.copy()
        current[2], current[3] = x, y
        cv2.rectangle(temp, (current[0], current[1]), (x, y), (0,255,0), 2)
        cv2.imshow("Annotator", temp)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        current[2], current[3] = x, y

        boxes.append(current.copy())
        labels.append(active_label)

        cv2.rectangle(display, (current[0], current[1]),
                      (current[2], current[3]), (0,255,0), 2)
        cv2.putText(display, active_label,
                    (current[0], current[1]-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

# =============================
# UI
# =============================
cv2.namedWindow("Annotator", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Annotator", mouse)

print("Press 'N' to change label | 'S' to save | 'Q' to quit")

while True:
    cv2.imshow("Annotator", display)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("n"):
        active_label = input("New active label: ").strip()
        if active_label not in class_map:
            class_map[active_label] = len(class_map)

    elif key == ord("s"):
        break

    elif key == ord("q"):
        cv2.destroyAllWindows()
        exit()

cv2.destroyAllWindows()

# =============================
# AUTO SPLIT
# =============================
count = len(list((DATASET / "images" / "train").glob("*")))
split = "val" if (count + 1) % VAL_EVERY == 0 else "train"

# =============================
# SAVE IMAGE
# =============================
img_name = Path(IMAGE_PATH).name
cv2.imwrite(DATASET / "images" / split / img_name, img)

# =============================
# SAVE LABEL FILE
# =============================
label_file = DATASET / "labels" / split / f"{Path(img_name).stem}.txt"

with open(label_file, "w") as f:
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box
        xc = ((x1 + x2) / 2) / w
        yc = ((y1 + y2) / 2) / h
        bw = abs(x2 - x1) / w
        bh = abs(y2 - y1) / h
        f.write(f"{class_map[label]} {xc} {yc} {bw} {bh}\n")

# =============================
# WRITE data.yaml (SAFE)
# =============================
with open(yaml_path, "w") as f:
    f.write("path: dataset\n\n")
    f.write("train: images/train\n")
    f.write("val: images/val\n\n")
    f.write("names:\n")
    for name, cid in sorted(class_map.items(), key=lambda x: x[1]):
        f.write(f"  {cid}: {name}\n")

print("✔ Saved")
print("✔ Classes:", class_map)
