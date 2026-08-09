import cv2
from pathlib import Path
from ultralytics import YOLO
from src.inference.detector import FireSmokeDetector

def debug_detections():
    detector = FireSmokeDetector(model_path="best.pt", conf_threshold=0.15)
    print("Model class names:", detector.model.names)

    val_imgs = list(Path("data/raw/images/val").glob("*.jpg")) + list(Path("data/raw/images/train").glob("*.jpg"))
    
    print(f"Testing on {len(val_imgs)} images...")
    detection_counts = 0
    for img_path in val_imgs[:10]:
        img = cv2.imread(str(img_path))
        annotated, detections, _ = detector.process_frame(img, draw_fps=False)
        print(f"Image {img_path.name}: {len(detections)} detections -> {detections}")
        if len(detections) > 0:
            detection_counts += 1

    print(f"Total images with detections: {detection_counts}/{min(10, len(val_imgs))}")

if __name__ == "__main__":
    debug_detections()
