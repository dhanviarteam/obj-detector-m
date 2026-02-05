from ultralytics import YOLO

def train():
    model = YOLO("yolov8_custom.yaml")

    model.train(
        data="dataset_ready/data.yaml",
        epochs=300,          # ✅ keep epochs
        imgsz=640,
        batch=8,             # 🔥 faster
        lr0=0.001,           # 🔥 correct LR
        warmup_epochs=3,
        patience=30,         # allows early stop if useless
        workers=8            # speedup data loading
    )

if __name__ == "__main__":
    train()
