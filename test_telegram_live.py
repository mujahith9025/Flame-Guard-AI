import urllib.request
import urllib.parse
import json

def test_telegram_alert():
    bot_token = "8850365473:AAHPme9b8jteFySKl7j2hkDa1pu3PXJ_Wp8"
    chat_id = "5730957885"

    text_msg = (
        "🚨 *FLAME-GUARD AI SYSTEM ONLINE*\n\n"
        "✅ *Status:* Mobile Push Notification System Successfully Linked!\n"
        "🔥 *Monitoring:* Live 4-Camera CCTV Hazard Surveillance\n"
        "⏰ *Timestamp:* 2026-08-08 20:10:55"
    )

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = urllib.parse.urlencode({
        "chat_id": chat_id,
        "text": text_msg,
        "parse_mode": "Markdown"
    }).encode("utf-8")

    print("Sending live test notification to Telegram...")
    try:
        req = urllib.request.Request(url, data=payload)
        resp = urllib.request.urlopen(req, timeout=5)
        res_data = json.loads(resp.read().decode("utf-8"))
        print("Telegram API Response:", res_data.get("ok"))
        if res_data.get("ok"):
            print("✅ TEST ALERT DELIVERED TO USER TELEGRAM PHONE APP!")
    except Exception as e:
        print("Error sending Telegram alert:", e)

if __name__ == "__main__":
    test_telegram_alert()
