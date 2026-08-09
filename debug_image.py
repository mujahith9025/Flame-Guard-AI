import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

def test_flame_detection():
    # Create a synthetic flame-like image or test on raw images
    print("=== TESTING MODEL CONFIDENCE SENSITIVITY ===")

    models_to_test = ["best.pt", "yolov8s.pt"]
    
    for m_path in models_to_test:
        if not Path(m_path).exists():
            continue
        print(f"\n--- Model: {m_path} ---")
        model = YOLO(m_path)
        print("Model Names:", model.names)

        # Test on raw train/val images
        raw_imgs = list(Path("data/raw/images/train").glob("*.jpg")) + list(Path("data/raw/images/val").glob("*.jpg"))
        for img_p in raw_imgs[:5]:
            img = cv2.imread(str(img_p))
            for conf in [0.25, 0.15, 0.05]:
                res = model.predict(source=img, conf=conf, verbose=False)
                boxes = res[0].boxes
                labels = [model.names[int(b.cls[0].item())] for b in boxes] if boxes else []
                scores = [round(float(b.conf[0].item()), 3) for b in boxes] if boxes else []
                if labels:
                    print(f"Image {img_p.name} (conf={conf}): {labels} with scores {scores}")

if __name__ == "__main__":
    test_flame_detection()
