import cv2
import requests
import numpy as np

def test_flame_upload_fix():
    # Create flame test image (bright orange/red flame shapes on dark background)
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.ellipse(img, (300, 250), (140, 110), 0, 0, 360, (0, 140, 255), -1)  # Orange
    cv2.ellipse(img, (300, 230), (90, 80), 0, 0, 360, (0, 215, 255), -1)   # Yellow
    cv2.ellipse(img, (300, 210), (45, 45), 0, 0, 360, (255, 255, 255), -1) # White core

    _, img_encoded = cv2.imencode(".jpg", img)
    files = {"file": ("stock_flame_photo.jpg", img_encoded.tobytes(), "image/jpeg")}

    url = "http://localhost:8000/api/detect-image"
    print(f"Uploading flame image to {url}...")
    resp = requests.post(url, files=files)
    
    print("Response Status:", resp.status_code)
    data = resp.json()
    print("Success:", data.get("success"))
    print("Has Fire:", data.get("has_fire"))
    print("Detection Count:", data.get("detection_count"))

if __name__ == "__main__":
    test_flame_upload_fix()
