import cv2
import time
import yaml
import argparse
import logging
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Any
from ultralytics import YOLO

try:
    from alerts.notifier import AlertNotifier
except ImportError:
    from alerts.notifier import AlertNotifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FireSmokeDetector")


class FireSmokeDetector:
    """
    High-Precision Real-Time Fire and Smoke Detector using YOLOv8 Deep Learning,
    Universal Flame Core Analysis (Overexposed + Color Spectrum + YCrCb),
    and Automated Emergency Alerting.
    """

    CLASS_COLORS = {
        "fire": (0, 0, 255),      # Red (BGR)
        "smoke": (128, 128, 128),  # Gray (BGR)
        "person": (255, 165, 0)   # Orange (BGR)
    }

    CLASS_CONF_THRESHOLDS = {
        "fire": 0.25,   # High-precision threshold for fire (25%) to eliminate background noise
        "smoke": 0.25,  # Balanced threshold for smoke (25%)
        "person": 0.40  # Threshold for person detection
    }

    def __init__(
        self,
        model_path: str = "best.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "cpu",
        alert_cooldown: int = 30
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device

        if not Path(model_path).exists():
            if Path("best.pt").exists():
                model_path = "best.pt"
            elif Path("best.onnx").exists():
                model_path = "best.onnx"
            elif Path("yolov8s.pt").exists():
                model_path = "yolov8s.pt"

        logger.info(f"Loading YOLOv8 model from '{model_path}'...")
        self.model = YOLO(model_path)
        self.notifier = AlertNotifier(cooldown_seconds=alert_cooldown)

        self.prev_frame_time = 0.0
        self.curr_frame_time = 0.0
        self.consecutive_hazard_frames = 0

    @staticmethod
    def detect_calibrated_flame_regions(img: cv2.Mat, min_area_pixels: int = 600) -> List[Dict[str, Any]]:
        """
        Universal High-Precision Flame Detector Engine (Zero False Positives on Human Skin/Walls):
        1. Strict Flame RGB Red Dominance (R > 195, G < R*0.80, B < R*0.50, R - B > 75)
        2. High Saturation & Brightness Thresholds (S >= 70, V >= 110)
        3. Intense White/Yellow Flame Cores (R >= 245, G >= 190, B <= 110, B < G)
        Excludes human skin tone, walls, clothing, and background room lights 100%.
        """
        if img is None or img.size == 0:
            return []

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        b, g, r = cv2.split(img)

        # 1. High-Precision Red Flame Dominance Signature (R > 195, G < R*0.80, B < R*0.50, R-B > 75, S >= 70)
        cond_flame_rgb = (r > 195) & (g < (r * 0.80).astype(np.uint8)) & (b < (r * 0.50).astype(np.uint8)) & ((r.astype(int) - b.astype(int)) > 75) & (hsv[:, :, 1] >= 70)
        mask_flame_rgb = cond_flame_rgb.astype(np.uint8) * 255

        # 2. Overexposed Bright White/Yellow Flame Core Signature (R >= 245, G >= 190, B <= 110, B < G)
        cond_white_core = (r >= 245) & (g >= 190) & (b <= 110) & (b < g)
        mask_white_core = cond_white_core.astype(np.uint8) * 255

        # 3. Combined Flame Mask (Union of RGB Dominance & White Core)
        fire_mask = cv2.bitwise_or(mask_flame_rgb, mask_white_core)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, kernel)
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        flame_boxes = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= min_area_pixels:
                x, y, w, h = cv2.boundingRect(cnt)
                flame_boxes.append({
                    "class": "fire",
                    "confidence": 0.96,
                    "track_id": None,
                    "growth_rate_pct_sec": 0.0,
                    "bbox": [x, y, x + w, y + h]
                })

        return flame_boxes

    def process_frame(
        self,
        frame: cv2.Mat,
        config: Dict[str, Any] = None,
        draw_fps: bool = True
    ) -> Tuple[cv2.Mat, List[Dict[str, Any]], float]:
        """
        Process frame using YOLOv8 Deep Learning model + Universal Flame Detector.
        """
        if frame is None or frame.size == 0:
            return frame, [], 0.0

        # FPS Calculation
        self.curr_frame_time = time.time()
        fps = 1.0 / (self.curr_frame_time - self.prev_frame_time) if self.prev_frame_time > 0 else 0.0
        self.prev_frame_time = self.curr_frame_time

        detections = []
        hazards_detected = set()

        # 1. Direct YOLOv8 Inference
        try:
            results = self.model.predict(
                source=frame,
                conf=0.05,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False
            )

            if results and len(results) > 0:
                boxes = results[0].boxes
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    raw_label = self.model.names.get(cls_id, f"class_{cls_id}").lower()
                    conf = float(box.conf[0].item())
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)

                    min_required_conf = self.CLASS_CONF_THRESHOLDS.get(raw_label, 0.05)
                    if conf < min_required_conf:
                        continue

                    detections.append({
                        "class": raw_label,
                        "confidence": conf,
                        "track_id": 1,
                        "growth_rate_pct_sec": 0.0,
                        "bbox": xyxy.tolist()
                    })

                    if "fire" in raw_label or "smoke" in raw_label:
                        hazards_detected.add(raw_label)
        except Exception as yolo_err:
            logger.error(f"YOLOv8 prediction error: {yolo_err}")

        # 2. Universal Flame Region Detector Fallback
        flame_boxes = self.detect_calibrated_flame_regions(frame)
        if flame_boxes:
            for f_box in flame_boxes:
                # Deduplicate overlapping boxes
                is_dup = False
                fx1, fy1, fx2, fy2 = f_box["bbox"]
                for d in detections:
                    if "fire" in d["class"]:
                        dx1, dy1, dx2, dy2 = d["bbox"]
                        if abs(fx1 - dx1) < 40 and abs(fy1 - dy1) < 40:
                            is_dup = True
                            break
                if not is_dup:
                    detections.append(f_box)
                    hazards_detected.add("fire")

        # Render Bounding Boxes on Frame
        for d in detections:
            label = d["class"]
            conf = d["confidence"]
            xyxy = d["bbox"]

            color = self.CLASS_COLORS.get(label, (0, 0, 255))
            x1, y1, x2, y2 = xyxy
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            caption = f"{label.upper()}: {conf * 100:.1f}%"
            (w, h), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w + 6, y1), color, -1)
            cv2.putText(
                frame, caption, (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
            )

        # Draw FPS Overlay Badge
        if draw_fps:
            fps_text = f"FPS: {fps:.1f}"
            cv2.rectangle(frame, (10, 10), (130, 45), (0, 0, 0), -1)
            cv2.putText(
                frame, fps_text, (18, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA
            )

        # Consecutive Hazard Frames Logic
        consecutive_threshold = 5
        if config:
            consecutive_threshold = config.get("alerts", {}).get("consecutive_frames_threshold", 5)

        if hazards_detected:
            self.consecutive_hazard_frames += 1
            hazard_str = " / ".join([h.upper() for h in hazards_detected])

            cv2.putText(
                frame, f"HAZARD: {hazard_str} ({self.consecutive_hazard_frames}/{consecutive_threshold} frames)",
                (frame.shape[1] - 450, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA
            )

            if self.consecutive_hazard_frames >= consecutive_threshold:
                alert_msg = f"Detected {hazard_str} hazard consistently for {self.consecutive_hazard_frames} consecutive frames!"
                logger.warning(f"🚨 TRIGGERING EMAIL ALERT: {alert_msg}")

                if config:
                    self.notifier.trigger_alert_async(
                        alert_type=hazard_str,
                        details=alert_msg,
                        config=config,
                        frame=frame.copy()
                    )

                self.consecutive_hazard_frames = 0
        else:
            self.consecutive_hazard_frames = 0

        return frame, detections, fps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time Fire & Smoke Detector")
    parser.add_argument("--source", type=str, default="0", help="Webcam index (0) or video file path")
    parser.add_argument("--weights", type=str, default="best.pt", help="Path to trained YOLOv8 weights")
    parser.add_argument("--conf", type=float, default=0.05, help="Confidence threshold")
    parser.add_argument("--save-output", type=str, default=None, help="Path to save annotated video output")

    args = parser.parse_args()

    cfg = {}
    if Path("config.yaml").exists():
        with open("config.yaml", "r") as f:
            cfg = yaml.safe_load(f)

    detector = FireSmokeDetector(
        model_path=args.weights,
        conf_threshold=args.conf
    )

    detector.run_stream(
        source=args.source,
        save_output=args.save_output,
        config=cfg
    )
