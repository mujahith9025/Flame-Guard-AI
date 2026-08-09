import time
import smtplib
import logging
import base64
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import cv2
import threading

logger = logging.getLogger("AlertNotifier")


class AlertNotifier:
    """
    Multi-Channel Emergency Alert Dispatcher: Supports Telegram Instant Mobile Push Notifications with Photo Snapshots,
    SMTP Email with Snapshots, and Mobile SMS with automatic timeout fallback protection.
    """

    def __init__(self, cooldown_seconds: int = 30):
        self.cooldown_seconds = cooldown_seconds
        self.last_alert_time = 0.0
        self.lock = threading.Lock()

    def can_send_alert(self) -> bool:
        with self.lock:
            now = time.time()
            if now - self.last_alert_time >= self.cooldown_seconds:
                return True
            return False

    def trigger_alert_async(self, alert_type: str, details: str, config: dict, frame=None):
        if not self.can_send_alert():
            logger.info("Alert suppressed due to cooldown throttle.")
            return

        with self.lock:
            self.last_alert_time = time.time()

        thread = threading.Thread(
            target=self._send_all_alerts,
            args=(alert_type, details, config, frame),
            daemon=True
        )
        thread.start()

    def _send_all_alerts(self, alert_type: str, details: str, config: dict, frame=None):
        logger.warning(f"🚨 HAZARD ALERT TRIGGERED: {alert_type} - {details}")

        # 1. Dispatch Telegram Instant Mobile Push Notification (Photo with Text Fallback)
        self._send_telegram(alert_type, details, config, frame)

        # 2. Dispatch Mobile SMS Alert
        self._send_sms(alert_type, details, config)

        # 3. Dispatch Email Alert with Snapshot
        self._send_email(alert_type, details, config, frame)

    def _send_telegram(self, alert_type: str, details: str, config: dict, frame=None):
        """
        Dispatches Instant Telegram Push Notification with robust timeout fallback.
        """
        telegram_cfg = config.get("alerts", {}).get("telegram", {})
        if not telegram_cfg.get("enabled", False):
            logger.info("Telegram notification skipped (disabled in config).")
            return

        bot_token = telegram_cfg.get("bot_token")
        chat_id = telegram_cfg.get("chat_id")
        user_name = telegram_cfg.get("user_name", f"User #{chat_id}" if chat_id else "Security Officer")

        if not bot_token or not chat_id:
            logger.warning("Telegram configuration incomplete (bot_token and chat_id required).")
            return

        text_msg = (
            f"🚨 *FLAME-GUARD AI EMERGENCY HAZARD ALERT*\n\n"
            f"👤 *Registered Officer:* `{user_name}`\n"
            f"🔥 *Hazard Detected:* `{alert_type}`\n"
            f"📍 *Location:* Zone 1 - Facility Main Bay\n"
            f"📝 *Details:* {details}\n"
            f"⏰ *Timestamp:* {time.strftime('%Y-%m-%d %H:%M:%S')}"
        )

        # Attempt 1: Try sending Photo Snapshot first if frame is available (with 15s timeout)
        if frame is not None:
            try:
                ret, jpeg_buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                if ret:
                    url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
                    data = {
                        "chat_id": chat_id,
                        "caption": text_msg + "\n\n📸 *Live CCTV frame snapshot attached below!*",
                        "parse_mode": "Markdown"
                    }
                    files = {
                        "photo": ("hazard_snapshot.jpg", jpeg_buf.tobytes(), "image/jpeg")
                    }
                    resp = requests.post(url, data=data, files=files, timeout=15)
                    if resp.status_code == 200:
                        logger.info(f"📸 📲 ✅ Telegram photo snapshot alert dispatched to {user_name} ({chat_id}) successfully!")
                        return
            except Exception as photo_err:
                logger.warning(f"Telegram photo push timed out/failed ({photo_err}). Falling back to instant text alert...")

        # Attempt 2 / Fallback: Send Text Notification (with 15s timeout)
        try:
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            data = {
                "chat_id": chat_id,
                "text": text_msg,
                "parse_mode": "Markdown"
            }
            resp = requests.post(url, data=data, timeout=15)
            if resp.status_code == 200:
                logger.info(f"📲 ✅ Telegram text alert dispatched to {user_name} ({chat_id}) successfully!")
            else:
                logger.error(f"Telegram sendMessage failed: {resp.text}")
        except Exception as e:
            logger.error(f"Failed to dispatch Telegram mobile push alert: {e}")

    def _send_sms(self, alert_type: str, details: str, config: dict):
        sms_cfg = config.get("alerts", {}).get("sms", {})
        if not sms_cfg.get("enabled", False):
            logger.info("SMS notification skipped (disabled in config).")
            return

        account_sid = sms_cfg.get("account_sid")
        auth_token = sms_cfg.get("auth_token")
        from_number = sms_cfg.get("from_number")
        to_number = sms_cfg.get("to_number")

        if not account_sid or not auth_token or not from_number or not to_number:
            logger.warning("SMS configuration incomplete.")
            return

        try:
            sms_body = f"🚨 EMERGENCY ALERT: FLAME-GUARD AI detected {alert_type.upper()} in Zone 1. Details: {details}. Time: {time.strftime('%H:%M:%S')}"
            url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json"

            resp = requests.post(
                url,
                data={
                    "From": from_number,
                    "To": to_number,
                    "Body": sms_body
                },
                auth=(account_sid, auth_token),
                timeout=10
            )

            if resp.status_code in [200, 201]:
                logger.info(f"📱 ✅ SMS hazard alert successfully sent to mobile phone: {to_number}")
            else:
                logger.error(f"SMS send failed: {resp.text}")
        except Exception as e:
            logger.error(f"Failed to dispatch Mobile SMS alert: {e}")

    def _send_email(self, alert_type: str, details: str, config: dict, frame=None):
        email_cfg = config.get("alerts", {}).get("email", {})

        if not email_cfg.get("enabled", False):
            logger.info("Email notification skipped (disabled in config).")
            return

        sender = email_cfg.get("sender_email")
        password = email_cfg.get("sender_password")
        recipients = email_cfg.get("recipients", [])

        if not sender or not password or not recipients:
            logger.warning("Email configuration incomplete. Skipping email dispatch.")
            return

        msg = MIMEMultipart()
        msg['Subject'] = f"🚨 EMERGENCY HAZARD ALERT: {alert_type.upper()} DETECTED"
        msg['From'] = sender
        msg['To'] = ", ".join(recipients)

        body = (
            f"FLAME-GUARD AI HAZARD DETECTION SYSTEM ALERT\n\n"
            f"Hazard Type: {alert_type.upper()}\n"
            f"Details: {details}\n"
            f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f"Immediate safety action required."
        )
        msg.attach(MIMEText(body, 'plain'))

        if frame is not None:
            ret, jpeg = cv2.imencode('.jpg', frame)
            if ret:
                img_data = jpeg.tobytes()
                image = MIMEImage(img_data, name="hazard_snapshot.jpg")
                msg.attach(image)

        try:
            server_host = email_cfg.get("smtp_server", "smtp.gmail.com")
            server_port = email_cfg.get("smtp_port", 587)

            with smtplib.SMTP(server_host, server_port) as server:
                server.starttls()
                server.login(sender, password)
                server.sendmail(sender, recipients, msg.as_string())
            logger.info(f"✅ Email hazard alert successfully sent to {recipients}")
        except Exception as e:
            logger.error(f"Failed to send alert email: {e}")
