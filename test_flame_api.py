import cv2
import requests
import numpy as np

def test_upload_fire_image():
    # Create a flame test image (bright red/orange/yellow on dark background)
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    # Draw intense orange/red flame shapes
    cv2.ellipse(img, (300, 250), (120, 100), 0, 0, 360, (0, 140, 255), -1)  # Orange
    cv2.ellipse(img, (300, 240), (80, 70), 0, 0, 360, (0, 215, 255), -1)   # Yellow
    cv2.ellipse(img, (300, 220), (40, 40), 0, 0, 360, (255, 255, 255), -1) # Bright white center

    _, img_encoded = cv2.imencode(".jpg", img)
    files = {"file": ("fire_sample.jpg", img_encoded.tobytes(), "image/jpeg")}

    url = "http://localhost:8000/api/detect-image"
    print(f"Sending test image request to {url}...")
    resp = requests.post(url, files=files)
    
    print("Response Status Code:", resp.status_code)
    data = resp.json()
    print("API Response:", {
        "success": data.get("success"),
        "has_fire": data.get("has_fire"),
        "status_message": data.get("status_message"),
        "detection_count": data.get("detection_count"),
        "hazards": data.get("hazard_labels")
    })

if __name__ == "__main__":
    test_upload_fire_image()
