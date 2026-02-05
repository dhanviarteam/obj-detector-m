import cv2
import os
import json

# -----------------------
# CONFIGURATION
# -----------------------
IMAGE_DIR = "dataset/images"  # Put all your images here
LABEL_DIR = "dataset/labels"  # YOLO format labels
CLASS_FILE = "dataset/classes.json"  # Store all classes
WINDOW_NAME = "Labeling Tool"

# Create directories if not exist
os.makedirs(LABEL_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)
if not os.path.exists(CLASS_FILE):
    with open(CLASS_FILE, "w") as f:
        json.dump([], f)

# Load or initialize classes
with open(CLASS_FILE, "r") as f:
    classes = json.load(f)

# -----------------------
# GLOBAL VARIABLES
# -----------------------
drawing = False
ix, iy = -1, -1
boxes = []
current_class = 0
current_image_index = 0
image_list = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
if not image_list:
    print("No images found in dataset/images!")
    exit()

# -----------------------
# CALLBACK FUNCTION
# -----------------------
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, img_copy, boxes, current_class
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            temp = img_copy.copy()
            cv2.rectangle(temp, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow(WINDOW_NAME, temp)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        boxes.append((ix, iy, x, y, current_class))
        cv2.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)
        cv2.putText(img_copy, classes[current_class], (ix, iy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow(WINDOW_NAME, img_copy)

# -----------------------
# FUNCTIONS
# -----------------------
def save_labels(image_name):
    label_file = os.path.join(LABEL_DIR, os.path.splitext(image_name)[0] + ".txt")
    h, w = img.shape[:2]
    with open(label_file, "w") as f:
        for (x1, y1, x2, y2, cls) in boxes:
            # Convert to YOLO format
            x_center = ((x1 + x2) / 2) / w
            y_center = ((y1 + y2) / 2) / h
            width = abs(x2 - x1) / w
            height = abs(y2 - y1) / h
            f.write(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def add_new_class():
    global classes, current_class
    new_class = input("Enter new class name: ").strip()
    if new_class not in classes:
        classes.append(new_class)
        current_class = len(classes) - 1
        with open(CLASS_FILE, "w") as f:
            json.dump(classes, f)
    else:
        current_class = classes.index(new_class)
    print(f"Current class set to [{current_class}]: {classes[current_class]}")

# -----------------------
# MAIN LOOP
# -----------------------
while current_image_index < len(image_list):
    img_path = os.path.join(IMAGE_DIR, image_list[current_image_index])
    img = cv2.imread(img_path)
    img_copy = img.copy()
    boxes = []

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1000, 700)
    cv2.setMouseCallback(WINDOW_NAME, draw_rectangle)

    while True:
        cv2.imshow(WINDOW_NAME, img_copy)
        print(f"\nImage [{current_image_index + 1}/{len(image_list)}]: {image_list[current_image_index]}")
        print(f"Current class [{current_class}]: {classes[current_class] if classes else 'None'}")
        print("Press 'n' to next image, 'c' to change/create class, 'q' to quit")
        key = cv2.waitKey(0) & 0xFF

        if key == ord('n'):  # next image
            save_labels(image_list[current_image_index])
            current_image_index += 1
            break
        elif key == ord('c'):  # change/create class
            add_new_class()
        elif key == ord('q'):
            save_labels(image_list[current_image_index])
            print("Exiting...")
            cv2.destroyAllWindows()
            exit()
