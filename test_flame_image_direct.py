import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

def test_flame_image():
    model = YOLO("best.pt")
    print("Model classes:", model.names)

    # Test on raw train/val images
    train_imgs = list(Path("data/raw/images/train").glob("*.jpg")) + list(Path("data/raw/images/val").glob("*.jpg"))
    print(f"Testing direct predict vs track on {len(train_imgs)} images...")

    for img_p in train_imgs[:5]:
        img = cv2.imread(str(img_p))
        
        # Test predict with conf=0.10
        res_predict = model.predict(source=img, conf=0.10, verbose=False)
        boxes_p = res_predict[0].boxes
        labels_p = [model.names[int(b.cls[0].item())] for b in boxes_p] if boxes_p else []
        scores_p = [round(float(b.conf[0].item()), 2) for b in boxes_p] if boxes_p else []
        
        # Test track
        res_track = model.track(source=img, tracker="bytetrack.yaml", persist=False, conf=0.10, verbose=False)
        boxes_t = res_track[0].boxes
        labels_t = [model.names[int(b.cls[0].item())] for b in boxes_t] if boxes_t else []
        
        print(f"\nImage {img_p.name}:")
        print(f"  Predict(conf=0.10): {labels_p} scores={scores_p}")
        print(f"  Track(conf=0.10):   {labels_t}")

if __name__ == "__main__":
    test_flame_image()
