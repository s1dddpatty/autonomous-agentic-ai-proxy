import requests
from typing import Optional
from src.core.config import WHATSAPP_ACCESS_TOKEN, WHATSAPP_PHONE_ID, get_whatsapp_token_for_number, get_contact_display_name, TARGET_PHONE_NUMBER

def send_whatsapp_message(to_number: str, text: str = ""):
    recipient = str(to_number).strip() if to_number else ""
    token = get_whatsapp_token_for_number(recipient) or WHATSAPP_ACCESS_TOKEN
    if not token or not WHATSAPP_PHONE_ID:
        print("Error: Missing WhatsApp tokens in configuration.")
        return None

    if not recipient:
        print("Error: No dynamic recipient phone number provided for outbound WhatsApp message.")
        return None

    url = f"https://graph.facebook.com/v19.0/{WHATSAPP_PHONE_ID}/messages"
    headers = {
        "Authorization": f"Bearer {token}",
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
        err_text = e.response.text if hasattr(e, "response") and e.response is not None else str(e)
        if "190" in err_text or "OAuthException" in err_text:
            print("\n" + "!" * 65)
            print("🚨 [META ERROR 190]: WHATSAPP ACCESS TOKEN HAS EXPIRED!")
            print("The 24-hour Meta temporary token in .env has expired.")
            print("Generate a fresh token in Meta Developer Dashboard, or create a")
            print("Permanent System User Token so it never expires.")
            print("!" * 65 + "\n")
            
            # If a specific contact token failed, retry with primary WHATSAPP_ACCESS_TOKEN
            if token != WHATSAPP_ACCESS_TOKEN and WHATSAPP_ACCESS_TOKEN:
                print(f"[WhatsApp Client] 🔄 Retrying with primary WHATSAPP_ACCESS_TOKEN for {recipient}...")
                headers["Authorization"] = f"Bearer {WHATSAPP_ACCESS_TOKEN}"
                try:
                    retry_resp = requests.post(url, headers=headers, json=payload, timeout=10)
                    retry_resp.raise_for_status()
                    print(f"Message sent successfully to {recipient} on retry!")
                    return retry_resp.json()
                except Exception as retry_err:
                    print(f"[WhatsApp Client] Retry failed: {retry_err}")
        print(f"Failed to send message: {err_text}")
        return None

if __name__ == "__main__":
    if TARGET_PHONE_NUMBER:
        response = send_whatsapp_message(TARGET_PHONE_NUMBER, "Hello from your Personal AI Proxy! 🚀")
        print(f"Response from Meta: {response}")
    else:
        print("Please configure TARGET_PHONE_NUMBER in .env to test.")
