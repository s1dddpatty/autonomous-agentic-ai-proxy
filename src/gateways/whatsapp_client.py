import requests
from typing import Optional
from src.core.config import WHATSAPP_ACCESS_TOKEN, WHATSAPP_PHONE_ID, TARGET_PHONE_NUMBER

def send_whatsapp_message(to_number: Optional[str] = None, text: str = ""):
    recipient = to_number or TARGET_PHONE_NUMBER
    if not WHATSAPP_ACCESS_TOKEN or not WHATSAPP_PHONE_ID:
        print("Error: Missing WhatsApp tokens in configuration.")
        return None

    if not recipient:
        print("Error: No recipient phone number provided or configured.")
        return None

    url = f"https://graph.facebook.com/v19.0/{WHATSAPP_PHONE_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": recipient,
        "type": "text",
        "text": {"preview_url": False, "body": text}
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        print(f"Message sent successfully to {recipient}!")
        return response.json()
    except requests.exceptions.HTTPError as e:
        print(f"Failed to send message: {e.response.text}")
        return None

if __name__ == "__main__":
    if TARGET_PHONE_NUMBER:
        response = send_whatsapp_message(TARGET_PHONE_NUMBER, "Hello from your Personal AI Proxy! 🚀")
        print(f"Response from Meta: {response}")
    else:
        print("Please configure TARGET_PHONE_NUMBER in .env to test.")
