# 🔥 Real-Time Fire and Smoke Detection System (YOLOv8)

An end-to-end computer vision application for real-time fire and smoke hazard detection built with **YOLOv8**, **OpenCV**, **Streamlit**, and automated alert notifications (**Email / Twilio**).

---

## 📁 Project Directory Structure

```
.
├── .env.example              # Template for API keys and secrets
├── .gitignore                # Git ignore patterns
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── config.yaml               # Model & system configuration
├── data/
│   ├── raw/                  # Raw dataset images and videos
│   └── processed/            # Processed dataset in YOLO annotation format
├── notebooks/                # Jupyter notebooks for data analysis & experimentation
├── src/
│   ├── __init__.py
│   ├── train/
│   │   ├── __init__.py
│   │   └── train.py          # YOLOv8 fine-tuning/training script
│   └── inference/
│       ├── __init__.py
│       └── detector.py       # OpenCV real-time object detection module
├── alerts/
│   ├── __init__.py
│   └── notifier.py           # Email (SMTP) & Twilio SMS notification engine
└── app/
    ├── __init__.py
    └── main.py               # Streamlit interactive dashboard
```

---

## 🚀 Quick Start Guide

### 1. Installation & Environment Setup

1. **Clone or Navigate to Project Directory**:
   ```bash
   cd d:/Projects
   ```

2. **Create & Activate Virtual Environment** (Optional but recommended):
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/macOS
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Secrets**:
   Copy `.env.example` to `.env` and fill in your notification credentials (optional):
   ```bash
   cp .env.example .env
   ```

---

## 🎯 Usage

### 1. Run Interactive Streamlit Dashboard
Launch the web interface for webcam monitoring, media uploads, and parameter tuning:
```bash
streamlit run app/main.py
```

### 2. Run Real-Time OpenCV Video / Webcam Detector
Run the detector directly on your camera feed:
```bash
python -m src.inference.detector
```

### 3. Fine-Tune / Train YOLOv8 Model
Train YOLOv8 on your custom fire & smoke dataset:
```bash
python -m src.train.train --data data/processed/data.yaml --epochs 50 --batch 16
```

---

## ⚙️ Configuration (`config.yaml`)

Key parameters can be customized in `config.yaml`:
- `model.weights`: Path to YOLOv8 `.pt` model weights (default: `yolov8n.pt`).
- `model.confidence_threshold`: Detection confidence threshold (default: `0.45`).
- `alerts.cooldown_seconds`: Cooldown interval to prevent spamming notifications.
- `alerts.email` & `alerts.twilio`: Configuration for automated alerts.

---

## 🛠 Tech Stack

- **Computer Vision & Object Detection**: Ultralytics YOLOv8, OpenCV (`opencv-python`)
- **Data Processing & Visualization**: NumPy, Pandas, Matplotlib, Pillow
- **Dashboard & Web UI**: Streamlit
- **Alert System**: `smtplib` (Email), `twilio` (SMS / WhatsApp)
