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
    Calibrated Flame Region Analysis, ByteTrack Multi-Object Tracking,
    and Automated Alert Notifications.
    """

    CLASS_COLORS = {
        "fire": (0, 0, 255),      # Red (BGR)
        "smoke": (128, 128, 128),  # Gray (BGR)
        "person": (255, 165, 0)   # Orange (BGR)
    }

    # Class-specific confidence thresholds for high-sensitivity fire detection
    CLASS_CONF_THRESHOLDS = {
        "fire": 0.10,   # Ultra-high sensitivity for flame detection
        "smoke": 0.25,  # Balanced threshold for smoke
        "person": 0.50
    }

    def __init__(
        self,
        model_path: str = "best.pt",
        conf_threshold: float = 0.15,
        iou_threshold: float = 0.45,
        device: str = "cpu",
        alert_cooldown: int = 30
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device

        if not Path(model_path).exists():
            if Path("best.onnx").exists():
                model_path = "best.onnx"
            elif Path("best.pt").exists():
                model_path = "best.pt"
            elif Path("yolov8s.pt").exists():
                model_path = "yolov8s.pt"

        logger.info(f"Loading YOLOv8 model from '{model_path}' for ByteTrack Object Tracking...")
        self.model = YOLO(model_path)
        self.notifier = AlertNotifier(cooldown_seconds=alert_cooldown)

        # FPS calculation variables
        self.prev_frame_time = 0.0
        self.curr_frame_time = 0.0

        # Consecutive hazard frame counter
        self.consecutive_hazard_frames = 0

        # Object tracking history: {track_id: {'first_seen': float, 'initial_area': float, 'last_area': float, 'last_seen': float}}
        self.track_history: Dict[int, Dict[str, float]] = {}

    @staticmethod
    def detect_calibrated_flame_regions(img: cv2.Mat, min_area_pixels: int = 250) -> List[Dict[str, Any]]:
        """
        Calibrated Flame Region Analysis: Detect bright fire/flame regions (Wide Orange/Red hue
        with flexible saturation & brightness for screen / real fire flames).
        """
        if img is None:
            return []

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # High-Sensitivity Fire HSV ranges (Orange / Red / Flame Yellow)
        lower1 = np.array([0, 70, 130])
        upper1 = np.array([28, 255, 255])

        lower2 = np.array([155, 70, 130])
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
            if area >= min_area_pixels:
                x, y, w, h = cv2.boundingRect(cnt)
                flame_boxes.append({
                    "class": "fire",
                    "confidence": 0.88,
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
        Process frame using YOLOv8 Deep Learning model + ByteTrack multi-object tracking.
        Integrates calibrated flame region detection fallback for isolated fire photos.
        """
        if frame is None:
            return frame, [], 0.0

        # FPS Calculation
        self.curr_frame_time = time.time()
        fps = 1.0 / (self.curr_frame_time - self.prev_frame_time) if self.prev_frame_time > 0 else 0.0
        self.prev_frame_time = self.curr_frame_time

        # Perform YOLOv8 ByteTrack Inference
        results = self.model.track(
            source=frame,
            tracker="bytetrack.yaml",
            persist=True,
            conf=min(self.conf_threshold, 0.10),
            iou=self.iou_threshold,
            device=self.device,
            verbose=False
        )

        detections = []
        hazards_detected = set()

        if results and len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                cls_id = int(box.cls[0].item())
                raw_label = self.model.names.get(cls_id, f"class_{cls_id}").lower()
                conf = float(box.conf[0].item())
                xyxy = box.xyxy[0].cpu().numpy().astype(int)

                # Apply Class-Specific Confidence Calibration Filter
                min_required_conf = self.CLASS_CONF_THRESHOLDS.get(raw_label, self.conf_threshold)
                if conf < min_required_conf:
                    continue

                x1, y1, x2, y2 = xyxy
                box_area = float((x2 - x1) * (y2 - y1))

                track_id = int(box.id[0].item()) if box.id is not None else None
                growth_rate = 0.0

                if track_id is not None:
                    now = self.curr_frame_time
                    if track_id not in self.track_history:
                        self.track_history[track_id] = {
                            "first_seen": now,
                            "initial_area": max(box_area, 1.0),
                            "last_area": box_area,
                            "last_seen": now
                        }
                    else:
                        hist = self.track_history[track_id]
                        dt = now - hist["first_seen"]
                        if dt >= 0.5:
                            growth_rate = ((box_area - hist["initial_area"]) / hist["initial_area"] * 100.0) / dt
                        hist["last_area"] = box_area
                        hist["last_seen"] = now

                detections.append({
                    "class": raw_label,
                    "confidence": conf,
                    "track_id": track_id,
                    "growth_rate_pct_sec": round(growth_rate, 1),
                    "bbox": xyxy.tolist()
                })

                if "fire" in raw_label or "smoke" in raw_label:
                    hazards_detected.add(raw_label)

        # Fallback Check: If no fire detected by YOLO, run Calibrated Flame Region Detector
        has_yolo_fire = any("fire" in d["class"] for d in detections)
        if not has_yolo_fire:
            flame_boxes = self.detect_calibrated_flame_regions(frame)
            if flame_boxes:
                for f_box in flame_boxes:
                    detections.append(f_box)
                    hazards_detected.add("fire")

        # Render Bounding Boxes on Frame
        for d in detections:
            label = d["class"]
            conf = d["confidence"]
            xyxy = d["bbox"]
            track_id = d.get("track_id")
            growth_rate = d.get("growth_rate_pct_sec", 0.0)

            color = self.CLASS_COLORS.get(label, (0, 0, 255))
            x1, y1, x2, y2 = xyxy
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            id_str = f"ID #{track_id} " if track_id is not None else ""
            growth_str = f" ({growth_rate:+.1f}%/s)" if (track_id is not None and abs(growth_rate) > 0.1) else ""
            caption = f"{id_str}{label.upper()}: {conf * 100:.1f}%{growth_str}"

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
    parser.add_argument("--conf", type=float, default=0.15, help="Confidence threshold")
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
