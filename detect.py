from ultralytics import YOLO
import cv2

# 🔥 USE YOUR TRAINED MODEL
model = YOLO(
    "runs/detect/train4/weights/best.pt"
)
print(model.names)
model.val(data="dataset_ready/data.yaml", plots=True)

IMAGE_PATH = "dataset/images/detect-images/plastic130.jpg"

img = cv2.imread(IMAGE_PATH)
if img is None:
    raise ValueError("Image not found")

results = model.predict(
    source=IMAGE_PATH,
    imgsz=640,
    conf=0.1,
    iou=0.5,
    verbose=False
)

for r in results:
    if r.boxes is None:
        continue

    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = f"{model.names[cls_id]} {conf:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            label,
            (x1, max(y1 - 10, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

cv2.imshow("Custom YOLO Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
