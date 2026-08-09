import cv2
import requests
import numpy as np

def test_landscape_image():
    # Create a sunny mountain landscape test image (blue sky, green mountain, yellow grass field)
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    # Blue sky
    img[0:150, :] = [235, 206, 135]
    # Green mountain
    cv2.circle(img, (300, 300), 200, (34, 139, 34), -1)
    # Yellow sunny grass field
    img[280:400, :] = [50, 205, 255]

    _, img_encoded = cv2.imencode(".jpg", img)
    files = {"file": ("landscape.jpg", img_encoded.tobytes(), "image/jpeg")}

    url = "http://localhost:8000/api/detect-image"
    print(f"Sending landscape image to {url}...")
    resp = requests.post(url, files=files)
    
    print("Response Status Code:", resp.status_code)
    data = resp.json()
    print("Has Fire:", data.get("has_fire"))
    print("Status Message:", data.get("status_message"))
    print("Detection Count:", data.get("detection_count"))

if __name__ == "__main__":
    test_landscape_image()
