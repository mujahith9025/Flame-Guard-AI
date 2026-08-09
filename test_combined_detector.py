import cv2
import numpy as np

def detect_hsv_flames(img):
    if img is None:
        return []
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Calibrated Fire HSV ranges (Strict Orange/Red + high saturation & brightness)
    lower1 = np.array([0, 120, 165])
    upper1 = np.array([22, 255, 255])

    lower2 = np.array([168, 120, 165])
    upper2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    fire_mask = cv2.bitwise_or(mask1, mask2)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, kernel)
    fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    flame_boxes = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= 500:
            x, y, w, h = cv2.boundingRect(cnt)
            flame_boxes.append([x, y, x + w, y + h])

    return flame_boxes

def run_tests():
    # Test 1: Flame Image
    flame_img = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.ellipse(flame_img, (300, 250), (140, 110), 0, 0, 360, (0, 140, 255), -1)  # Orange
    cv2.ellipse(flame_img, (300, 230), (90, 80), 0, 0, 360, (0, 215, 255), -1)   # Yellow
    cv2.ellipse(flame_img, (300, 210), (45, 45), 0, 0, 360, (255, 255, 255), -1) # White core

    flame_res = detect_hsv_flames(flame_img)
    print("Test 1 (Flame Image): Detections =", len(flame_res), "Boxes =", flame_res)

    # Test 2: Sunny Mountain Landscape Image
    landscape_img = np.zeros((400, 600, 3), dtype=np.uint8)
    landscape_img[0:150, :] = [235, 206, 135] # Blue sky
    cv2.circle(landscape_img, (300, 300), 200, (34, 139, 34), -1) # Green mountain
    landscape_img[280:400, :] = [50, 205, 255] # Sunny yellow grass field

    landscape_res = detect_hsv_flames(landscape_img)
    print("Test 2 (Landscape Image): Detections =", len(landscape_res), "Boxes =", landscape_res)

if __name__ == "__main__":
    run_tests()
