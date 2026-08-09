import os
import sys
import time
import json
import yaml
import base64
import logging
import asyncio
import tempfile
import requests
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse, Response, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Configure Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("APIServer")

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.inference.detector import FireSmokeDetector

app = FastAPI(
    title="Fire & Smoke Detection System API",
    description="Real-Time Computer Vision & Hazard Analytics Engine",
    version="2.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache-Control Middleware to prevent browser disk caching of HTML/CSS/JS
@app.middleware("http")
async def add_no_cache_header(request: Request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

# Mount Frontend Static Directory
FRONTEND_DIR = Path(PROJECT_ROOT) / "frontend"
FRONTEND_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# Global Application State
CONFIG_PATH = Path(PROJECT_ROOT) / "config.yaml"


def load_config():
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Error loading config.yaml: {e}")
    return {}


state = {
    "config": load_config(),
    "detector": None,
    "alert_logs": [],
    "total_fire": 0,
    "total_smoke": 0,
    "max_growth_rate": 0.0
}


def get_detector():
    if state["detector"] is None:
        cfg = state["config"]
        best_pt_path = Path(PROJECT_ROOT) / "best.pt"
        if best_pt_path.exists():
            weights = str(best_pt_path)
        else:
            weights = cfg.get("model", {}).get("weights", "best.pt")

        conf = float(cfg.get("model", {}).get("confidence_threshold", 0.05))
        iou = float(cfg.get("model", {}).get("iou_threshold", 0.45))
        cd = int(cfg.get("alerts", {}).get("cooldown_seconds", 30))

        logger.info(f"Initializing YOLOv8 detector using absolute path '{weights}'...")
        state["detector"] = FireSmokeDetector(
            model_path=weights,
            conf_threshold=conf,
            iou_threshold=iou,
            alert_cooldown=cd
        )
    return state["detector"]


def log_hazard(hazard_type: str, confidence: float, source: str, frame_count: int, track_id: int = None, growth_rate: float = 0.0):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    track_str = f"ID #{track_id}" if track_id is not None else "N/A"
    growth_str = f"{growth_rate:+.1f}%/s" if abs(growth_rate) > 0.0 else "0.0%/s"

    entry = {
        "timestamp": timestamp,
        "track_id": track_str,
        "hazard_type": hazard_type.upper(),
        "confidence": f"{confidence * 100:.1f}%",
        "growth_velocity": growth_str,
        "source": source,
        "consecutive_frames": frame_count
    }

    state["alert_logs"].insert(0, entry)
    if len(state["alert_logs"]) > 500:
        state["alert_logs"].pop()

    if "FIRE" in hazard_type.upper():
        state["total_fire"] += 1
    if "SMOKE" in hazard_type.upper():
        state["total_smoke"] += 1
    if growth_rate > state["max_growth_rate"]:
        state["max_growth_rate"] = growth_rate


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(
            str(index_path),
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
        )
    return HTMLResponse("<h2>Fire & Smoke Detection Dashboard</h2>")


@app.get("/login", response_class=HTMLResponse)
@app.get("/login.html", response_class=HTMLResponse)
@app.get("/account", response_class=HTMLResponse)
async def serve_login():
    login_path = FRONTEND_DIR / "login.html"
    if login_path.exists():
        return FileResponse(
            str(login_path),
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
        )
    return HTMLResponse("<h2>Telegram Push Notification Portal Login</h2>")


@app.get("/api/status")
async def get_system_status():
    """
    Lightweight instant status API. Never triggers blocking model load.
    """
    if state["detector"] is not None:
        detector = state["detector"]
        model_name = detector.model.model_name if hasattr(detector.model, 'model_name') else "YOLOv8s"
        conf_thresh = detector.conf_threshold
    else:
        model_name = "YOLOv8s Engine"
        conf_thresh = float(state["config"].get("model", {}).get("confidence_threshold", 0.05))

    telegram_cfg = state["config"].get("alerts", {}).get("telegram", {})
    telegram_id = telegram_cfg.get("chat_id", "")
    telegram_name = telegram_cfg.get("user_name", f"User #{telegram_id}" if telegram_id else "Security Officer")
    sms_phone = state["config"].get("alerts", {}).get("sms", {}).get("to_number", "")

    return {
        "status": "ONLINE",
        "active_model": model_name,
        "confidence_threshold": conf_thresh,
        "total_fire_alerts": state["total_fire"],
        "total_smoke_alerts": state["total_smoke"],
        "max_growth_rate": round(state["max_growth_rate"], 1),
        "total_logged_events": len(state["alert_logs"]),
        "registered_telegram_id": telegram_id,
        "registered_telegram_name": telegram_name,
        "registered_sms_phone": sms_phone
    }


@app.post("/api/stream-frame")
async def process_stream_frame(payload: Dict[str, Any]):
    """
    HTTP POST Frame Streaming API (Cloud Fallback for Render.com when WebSockets are throttled).
    """
    frame_b64 = payload.get("frame_b64")
    if not frame_b64:
        raise HTTPException(status_code=400, detail="Missing frame_b64")

    try:
        b64_str = frame_b64.split(",")[1] if "," in frame_b64 else frame_b64
        img_bytes = base64.b64decode(b64_str)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {e}")

    if frame is None or frame.size == 0:
        raise HTTPException(status_code=400, detail="Frame decode failed.")

    detector = get_detector()
    annotated_frame, detections, fps = detector.process_frame(
        frame, config=state["config"], draw_fps=True
    )

    has_fire_or_smoke = False
    hazards_found = set()

    for d in detections:
        cls_name = d["class"].lower()
        if "fire" in cls_name or "smoke" in cls_name:
            has_fire_or_smoke = True
            hazards_found.add(cls_name.upper())
            log_hazard(
                hazard_type=d["class"],
                confidence=d["confidence"],
                source="Live Camera Stream",
                frame_count=detector.consecutive_hazard_frames,
                track_id=d.get("track_id"),
                growth_rate=d.get("growth_rate_pct_sec", 0.0)
            )

    _, buffer = cv2.imencode(".jpg", annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
    out_b64 = base64.b64encode(buffer).decode("utf-8")

    return {
        "success": True,
        "fps": round(fps, 1),
        "has_fire": has_fire_or_smoke,
        "hazard_labels": list(hazards_found),
        "status_message": f"🚨 CRITICAL INCIDENT // FIRE DETECTED: {', '.join(hazards_found)}" if has_fire_or_smoke else "✅ FACILITY SECURE // NO INCIDENTS DETECTED",
        "detections": detections,
        "consecutive_frames": detector.consecutive_hazard_frames,
        "frame_b64": f"data:image/jpeg;base64,{out_b64}"
    }


@app.post("/api/detect-image")
async def detect_uploaded_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file format.")

    detector = get_detector()
    annotated_img, detections, _ = detector.process_frame(img.copy(), config=state["config"], draw_fps=False)

    has_fire_or_smoke = False
    hazards_found = set()

    for d in detections:
        cls_name = d["class"].lower()
        if "fire" in cls_name or "smoke" in cls_name:
            has_fire_or_smoke = True
            hazards_found.add(cls_name.upper())
            log_hazard(
                hazard_type=d["class"],
                confidence=d["confidence"],
                source="Image Upload",
                frame_count=1,
                track_id=d.get("track_id"),
                growth_rate=d.get("growth_rate_pct_sec", 0.0)
            )

    _, buffer = cv2.imencode(".jpg", annotated_img)
    b64_image = base64.b64encode(buffer).decode("utf-8")

    return {
        "success": True,
        "has_fire": has_fire_or_smoke,
        "hazard_labels": list(hazards_found),
        "status_message": f"🚨 CRITICAL INCIDENT // FIRE DETECTED: {', '.join(hazards_found)}" if has_fire_or_smoke else "✅ FACILITY SECURE // NO INCIDENTS DETECTED",
        "detection_count": len(detections),
        "detections": detections,
        "annotated_image_b64": f"data:image/jpeg;base64,{b64_image}"
    }


@app.get("/api/logs")
async def get_alert_logs():
    return {
        "logs": state["alert_logs"][:50],
        "total_fire": state["total_fire"],
        "total_smoke": state["total_smoke"],
        "max_growth_rate": round(state["max_growth_rate"], 1)
    }


@app.get("/api/export-csv")
async def export_csv():
    if not state["alert_logs"]:
        raise HTTPException(status_code=404, detail="No log entries to export.")

    df = pd.DataFrame(state["alert_logs"])
    csv_str = df.to_csv(index=False)

    return Response(
        content=csv_str,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=hazard_log_{int(time.time())}.csv"}
    )


@app.post("/api/test-telegram")
async def test_telegram_push(payload: Dict[str, Any]):
    chat_id = payload.get("chat_id")
    user_name = payload.get("user_name") or state["config"].get("alerts", {}).get("telegram", {}).get("user_name", f"User #{chat_id}")
    bot_token = payload.get("bot_token") or state["config"].get("alerts", {}).get("telegram", {}).get("bot_token", "8850365473:AAHPme9b8jteFySKl7j2hkDa1pu3PXJ_Wp8")

    if not chat_id or not bot_token:
        raise HTTPException(status_code=400, detail="Telegram Chat ID is required.")

    text_msg = (
        f"🚨 *FLAME-GUARD AI TELEGRAM PUSH VERIFIED*\n\n"
        f"👤 *Registered Officer:* `{user_name}`\n"
        f"🆔 *Telegram Chat ID:* `{chat_id}`\n"
        f"📱 *Mobile Alert Push:* ACTIVE 24/7\n"
        f"⏰ *Timestamp:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    url_msg = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    try:
        resp = requests.post(
            url_msg,
            data={
                "chat_id": chat_id,
                "text": text_msg,
                "parse_mode": "Markdown"
            },
            timeout=15
        )

        if resp.status_code == 200:
            return {"success": True, "message": f"Test push alert sent to {user_name} (Chat ID: {chat_id})"}
        else:
            err_body = resp.text
            logger.error(f"Telegram sendMessage API Error ({resp.status_code}): {err_body}")
            if "chat not found" in err_body.lower():
                return {
                    "success": False,
                    "detail": "Chat not started! Open @fireflame_guard_bot on Telegram and click START first, then try again."
                }
            return {"success": False, "detail": f"Telegram Error: {err_body}"}
    except requests.exceptions.Timeout:
        logger.error("Telegram API connection timed out.")
        return {
            "success": False,
            "detail": "Network Timeout: api.telegram.org did not respond within 15s. Please check internet connection or retry."
        }
    except Exception as e:
        logger.error(f"Telegram test push failed: {e}")
        return {"success": False, "detail": str(e)}


@app.post("/api/config")
async def update_config(payload: Dict[str, Any]):
    if "confidence_threshold" in payload and state["detector"] is not None:
        state["detector"].conf_threshold = float(payload["confidence_threshold"])
    if "iou_threshold" in payload and state["detector"] is not None:
        state["detector"].iou_threshold = float(payload["iou_threshold"])

    if "alerts" not in state["config"]:
        state["config"]["alerts"] = {}

    if "cooldown_seconds" in payload:
        if state["detector"] is not None:
            state["detector"].notifier.cooldown_seconds = int(payload["cooldown_seconds"])
        state["config"]["alerts"]["cooldown_seconds"] = int(payload["cooldown_seconds"])
    if "consecutive_frames_threshold" in payload:
        state["config"]["alerts"]["consecutive_frames_threshold"] = int(payload["consecutive_frames_threshold"])

    if "telegram_chat_id" in payload or "telegram_user_name" in payload:
        if "telegram" not in state["config"]["alerts"]:
            state["config"]["alerts"]["telegram"] = {}
        if "telegram_chat_id" in payload:
            chat_id_val = str(payload["telegram_chat_id"]).strip()
            state["config"]["alerts"]["telegram"]["chat_id"] = chat_id_val

            raw_name = str(payload.get("telegram_user_name", "")).strip()
            user_name_val = raw_name if raw_name else f"User #{chat_id_val}"
            state["config"]["alerts"]["telegram"]["user_name"] = user_name_val

        state["config"]["alerts"]["telegram"]["enabled"] = True

    if "sms_to_number" in payload:
        if "sms" not in state["config"]["alerts"]:
            state["config"]["alerts"]["sms"] = {}
        state["config"]["alerts"]["sms"]["to_number"] = str(payload["sms_to_number"]).strip()
        state["config"]["alerts"]["sms"]["enabled"] = True

    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            yaml.safe_dump(state["config"], f)
    except Exception as e:
        logger.warning(f"Could not persist config.yaml to disk: {e}")

    return {"status": "SUCCESS", "updated_config": state["config"]}


@app.websocket("/ws/stream")
async def websocket_stream_endpoint(websocket: WebSocket):
    """
    Real-Time WebSocket Detection Stream.
    Supports receiving live webcam frames directly from the browser client (Cloud & Render compatible).
    """
    await websocket.accept()
    logger.info("WebSocket client connected to live detection stream.")

    detector = get_detector()
    cap = None

    try:
        while True:
            raw_msg = await websocket.receive_text()
            data = json.loads(raw_msg)

            frame = None
            if "frame_b64" in data and data["frame_b64"]:
                try:
                    b64_str = data["frame_b64"].split(",")[1] if "," in data["frame_b64"] else data["frame_b64"]
                    img_bytes = base64.b64decode(b64_str)
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                except Exception as b64_err:
                    logger.error(f"Error decoding client frame: {b64_err}")

            if frame is None:
                if cap is None:
                    cap = cv2.VideoCapture(0)
                ret, frame = cap.read()
                if not ret or frame is None:
                    await asyncio.sleep(0.05)
                    continue

            annotated_frame, detections, fps = detector.process_frame(
                frame, config=state["config"], draw_fps=True
            )

            has_fire_or_smoke = False
            hazards_found = set()

            for d in detections:
                cls_name = d["class"].lower()
                if "fire" in cls_name or "smoke" in cls_name:
                    has_fire_or_smoke = True
                    hazards_found.add(cls_name.upper())
                    log_hazard(
                        hazard_type=d["class"],
                        confidence=d["confidence"],
                        source="Live CCTV Matrix Stream",
                        frame_count=detector.consecutive_hazard_frames,
                        track_id=d.get("track_id"),
                        growth_rate=d.get("growth_rate_pct_sec", 0.0)
                    )

            _, buffer = cv2.imencode(".jpg", annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            b64_frame = base64.b64encode(buffer).decode("utf-8")

            payload = {
                "fps": round(fps, 1),
                "has_fire": has_fire_or_smoke,
                "hazard_labels": list(hazards_found),
                "status_message": f"🚨 CRITICAL INCIDENT // FIRE DETECTED: {', '.join(hazards_found)}" if has_fire_or_smoke else "✅ FACILITY SECURE // NO INCIDENTS DETECTED",
                "detections": detections,
                "consecutive_frames": detector.consecutive_hazard_frames,
                "frame_b64": f"data:image/jpeg;base64,{b64_frame}"
            }

            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(0.01)

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
    except Exception as e:
        logger.error(f"WebSocket streaming error: {e}")
    finally:
        if cap is not None:
            cap.release()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=True)
