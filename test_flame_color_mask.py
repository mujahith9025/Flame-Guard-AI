import cv2
import numpy as np
from ultralytics import YOLO

def detect_flame_regions(img):
    # Convert image to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Fire color range in HSV (Red/Orange/Yellow + high saturation & value)
    lower1 = np.array([0, 100, 140])
    upper1 = np.array([30, 255, 255])

    lower2 = np.array([165, 100, 140])
    upper2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    fire_mask = cv2.bitwise_or(mask1, mask2)

    # Filter out noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, kernel)
    fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_CLOSE, kernel)

    total_pixels = img.shape[0] * img.shape[1]
    fire_pixels = cv2.countNonZero(fire_mask)
    ratio = fire_pixels / total_pixels

    contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 400:  # Minimum 400 contiguous fire pixels
            x, y, w, h = cv2.boundingRect(cnt)
            bboxes.append([x, y, x + w, y + h])

    print(f"Fire mask ratio: {ratio:.4f}, Bounding boxes count: {len(bboxes)}")
    return bboxes

if __name__ == "__main__":
    # Create test flame image
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.ellipse(img, (300, 250), (140, 110), 0, 0, 360, (0, 140, 255), -1)  # Orange
    cv2.ellipse(img, (300, 230), (90, 80), 0, 0, 360, (0, 215, 255), -1)   # Yellow
    cv2.ellipse(img, (300, 210), (45, 45), 0, 0, 360, (255, 255, 255), -1) # White core

    bboxes = detect_flame_regions(img)
    print("Detected Bboxes:", bboxes)
