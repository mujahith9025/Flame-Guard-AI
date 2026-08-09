import urllib.request
import urllib.parse
import json

bot_token = "8850365473:AAHPme9b8jteFySKl7j2hkDa1pu3PXJ_Wp8"
chat_id = "5730957885"

url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
payload = urllib.parse.urlencode({
    "chat_id": chat_id,
    "text": "🚨 FLAME-GUARD AI TEST PUSH: Mobile Alert System Verified!",
}).encode("utf-8")

try:
    req = urllib.request.Request(url, data=payload)
    resp = urllib.request.urlopen(req, timeout=5)
    print("SUCCESS:", resp.read().decode())
except urllib.error.HTTPError as e:
    err_body = e.read().decode()
    print("HTTP ERROR:", e.code, err_body)
except Exception as e:
    print("ERROR:", e)
