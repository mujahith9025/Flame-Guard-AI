import os
import sys
import time
import yaml
import tempfile
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.inference.detector import FireSmokeDetector


def load_config(config_path="config.yaml"):
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def init_session_state():
    if "alert_logs" not in st.session_state:
        st.session_state["alert_logs"] = []
    if "total_fire_count" not in st.session_state:
        st.session_state["total_fire_count"] = 0
    if "total_smoke_count" not in st.session_state:
        st.session_state["total_smoke_count"] = 0
    if "max_growth_rate" not in st.session_state:
        st.session_state["max_growth_rate"] = 0.0


def add_alert_log(hazard_type: str, confidence: float, source: str, frame_count: int, track_id: int = None, growth_rate: float = 0.0):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    track_str = f"ID #{track_id}" if track_id is not None else "N/A"
    growth_str = f"{growth_rate:+.1f}%/s" if abs(growth_rate) > 0.0 else "0.0%/s"

    st.session_state["alert_logs"].insert(0, {
        "Timestamp": timestamp,
        "Track ID": track_str,
        "Hazard Type": hazard_type.upper(),
        "Confidence Score": f"{confidence * 100:.1f}%",
        "Growth Velocity": growth_str,
        "Source": source,
        "Consecutive Frames": frame_count
    })

    if "FIRE" in hazard_type.upper():
        st.session_state["total_fire_count"] += 1
    if "SMOKE" in hazard_type.upper():
        st.session_state["total_smoke_count"] += 1

    if growth_rate > st.session_state["max_growth_rate"]:
        st.session_state["max_growth_rate"] = growth_rate


def main():
    st.set_page_config(
        page_title="Real-Time Fire & Smoke Detection System (ByteTrack)",
        page_icon="🔥",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    init_session_state()

    st.title("🔥 Real-Time Fire & Smoke Detection & Tracking System")
    st.markdown("Powered by **YOLOv8 + ByteTrack**, **OpenCV**, and **Automated Email/SMS Alerts**.")

    config = load_config()

    # Sidebar Options
    st.sidebar.header("⚙️ Model & Detection Settings")
    weights_path = st.sidebar.text_input(
        "Model Weights Path",
        value=config.get("model", {}).get("weights", "best.pt")
    )

    conf_thresh = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=float(config.get("model", {}).get("confidence_threshold", 0.35)),
        step=0.05
    )

    iou_thresh = st.sidebar.slider(
        "IoU Threshold",
        min_value=0.1,
        max_value=1.0,
        value=float(config.get("model", {}).get("iou_threshold", 0.45)),
        step=0.05
    )

    consecutive_frames = st.sidebar.number_input(
        "Consecutive Frames Threshold",
        min_value=1,
        max_value=30,
        value=int(config.get("alerts", {}).get("consecutive_frames_threshold", 5))
    )

    cooldown = st.sidebar.number_input(
        "Alert Cooldown (Seconds)",
        min_value=5,
        max_value=300,
        value=int(config.get("alerts", {}).get("cooldown_seconds", 30))
    )

    st.sidebar.header("🔔 Alert Notifications")
    email_enabled = st.sidebar.checkbox("Enable Email Alerts", value=config.get("alerts", {}).get("email", {}).get("enabled", False))
    twilio_enabled = st.sidebar.checkbox("Enable Twilio SMS Alerts", value=config.get("alerts", {}).get("twilio", {}).get("enabled", False))

    runtime_config = config.copy()
    if "alerts" not in runtime_config:
        runtime_config["alerts"] = {}
    runtime_config["alerts"]["cooldown_seconds"] = cooldown
    runtime_config["alerts"]["consecutive_frames_threshold"] = consecutive_frames
    runtime_config["alerts"]["email"] = {"enabled": email_enabled}
    runtime_config["alerts"]["twilio"] = {"enabled": twilio_enabled}

    # Initialize YOLOv8 Detector
    @st.cache_resource
    def get_detector(weights, conf, iou, cd):
        return FireSmokeDetector(
            model_path=weights,
            conf_threshold=conf,
            iou_threshold=iou,
            alert_cooldown=cd
        )

    try:
        detector = get_detector(weights_path, conf_thresh, iou_thresh, cooldown)
    except Exception as e:
        st.error(f"Failed to load YOLO model from '{weights_path}': {e}")
        return

    # Top Metric Banner
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Active Model", weights_path)
    m2.metric("Tracker Engine", "ByteTrack")
    m3.metric("Total Fire Alerts", st.session_state["total_fire_count"])
    m4.metric("Total Smoke Alerts", st.session_state["total_smoke_count"])
    m5.metric("Peak Growth Rate", f"{st.session_state['max_growth_rate']:+.1f}%/s")

    st.markdown("---")

    # Main Area Tabs
    tab1, tab2, tab3 = st.tabs(["📹 ByteTrack Stream & Video Detection", "🖼️ Single Image Test", "📊 Alert & Growth Logs"])

    with tab1:
        col_video, col_logs = st.columns([2, 1])

        with col_video:
            st.subheader("🎥 Video Feed with ByteTrack Object Tracking")
            source_option = st.radio("Select Input Stream", ["Upload Video File", "Live Webcam Feed"], horizontal=True)

            if source_option == "Upload Video File":
                uploaded_video = st.file_uploader("Upload Video (.mp4, .avi, .mov)", type=["mp4", "avi", "mov"])
                if uploaded_video:
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                    tfile.write(uploaded_video.read())
                    tfile.flush()

                    cap = cv2.VideoCapture(tfile.name)
                    st_frame = st.empty()
                    stop_btn = st.button("Stop Video Feed")

                    while cap.isOpened() and not stop_btn:
                        ret, frame = cap.read()
                        if not ret:
                            break

                        annotated_frame, detections, fps = detector.process_frame(
                            frame, config=runtime_config, draw_fps=True
                        )

                        # Log hazard detections
                        for d in detections:
                            if "fire" in d["class"] or "smoke" in d["class"]:
                                add_alert_log(
                                    hazard_type=d["class"],
                                    confidence=d["confidence"],
                                    source="Uploaded Video",
                                    frame_count=detector.consecutive_hazard_frames,
                                    track_id=d.get("track_id"),
                                    growth_rate=d.get("growth_rate_pct_sec", 0.0)
                                )

                        st_frame.image(
                            cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
                            channels="RGB",
                            use_container_width=True
                        )
                    cap.release()

            elif source_option == "Live Webcam Feed":
                run_webcam = st.checkbox("Launch Webcam Stream")
                if run_webcam:
                    cap = cv2.VideoCapture(0)
                    st_frame = st.empty()

                    while run_webcam and cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            st.error("Unable to capture frame from webcam.")
                            break

                        annotated_frame, detections, fps = detector.process_frame(
                            frame, config=runtime_config, draw_fps=True
                        )

                        for d in detections:
                            if "fire" in d["class"] or "smoke" in d["class"]:
                                add_alert_log(
                                    hazard_type=d["class"],
                                    confidence=d["confidence"],
                                    source="Webcam Feed",
                                    frame_count=detector.consecutive_hazard_frames,
                                    track_id=d.get("track_id"),
                                    growth_rate=d.get("growth_rate_pct_sec", 0.0)
                                )

                        st_frame.image(
                            cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
                            channels="RGB",
                            use_container_width=True
                        )
                        time.sleep(0.01)

                    cap.release()

        with col_logs:
            st.subheader("🚨 ByteTrack Hazard Log")
            if st.session_state["alert_logs"]:
                df_logs = pd.DataFrame(st.session_state["alert_logs"]).head(15)
                st.dataframe(df_logs, use_container_width=True, height=400)
            else:
                st.info("No hazard detections logged yet. Start a video stream or webcam feed.")

            if st.button("Clear Log History"):
                st.session_state["alert_logs"] = []
                st.session_state["total_fire_count"] = 0
                st.session_state["total_smoke_count"] = 0
                st.session_state["max_growth_rate"] = 0.0
                st.rerun()

    with tab2:
        st.subheader("Single Image Detection Test")
        uploaded_image = st.file_uploader("Upload Image File (.jpg, .png)", type=["jpg", "jpeg", "png"])
        if uploaded_image:
            file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)

            c1, c2 = st.columns(2)
            with c1:
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_container_width=True)

            annotated_img, detections, _ = detector.process_frame(img.copy(), config=runtime_config, draw_fps=False)

            with c2:
                st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), caption="YOLOv8 Detections", use_container_width=True)

            if detections:
                st.success(f"Detected {len(detections)} objects!")
                st.dataframe(pd.DataFrame(detections))

    with tab3:
        st.subheader("📊 ByteTrack Hazard Log Export & Growth Analytics")
        if st.session_state["alert_logs"]:
            df_full = pd.DataFrame(st.session_state["alert_logs"])
            st.dataframe(df_full, use_container_width=True)

            csv_data = df_full.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download ByteTrack Hazard Log (.csv)",
                data=csv_data,
                file_name=f"bytetrack_hazard_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No log data available to export.")


if __name__ == "__main__":
    main()
