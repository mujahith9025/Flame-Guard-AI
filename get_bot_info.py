import urllib.request
import json

bot_token = "8850365473:AAHPme9b8jteFySKl7j2hkDa1pu3PXJ_Wp8"
url = f"https://api.telegram.org/bot{bot_token}/getMe"

try:
    resp = urllib.request.urlopen(url, timeout=5)
    data = json.loads(resp.read().decode())
    print("BOT INFO:", json.dumps(data, indent=2))
except Exception as e:
    print("ERROR:", e)
