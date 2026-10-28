import os
import sys
from pyngrok import ngrok
import uvicorn

import multiprocessing

def start_telegram_bot():
    from src.gateways.telegram_bot import main as telegram_main
    telegram_main()

def main():
    port = int(os.environ.get("PORT", 8000))

    from src.core.config import NGROK_AUTHTOKEN
    ngrok_token = NGROK_AUTHTOKEN

    # In cloud environments, PORT is set and public URLs are assigned by the cloud provider.
    # Ngrok is used when NGROK_AUTHTOKEN is present and ENABLE_NGROK is not disabled.
    enable_ngrok = os.environ.get("ENABLE_NGROK", "True").lower() in ("true", "1", "yes") and bool(ngrok_token)
    if enable_ngrok:
        try:
            try:
                ngrok.set_auth_token(ngrok_token)
            except Exception:
                pass
            # Start ngrok tunnel pointing to dynamic port
            public_url = ngrok.connect(port).public_url
            print("\n" + "="*60)
            print("🚀 NGROK TUNNEL ESTABLISHED 🚀")
            print("Please copy the URL below and paste it into the Meta Developer Dashboard:")
            print(f"Callback URL: {public_url}/webhook")
            print("="*60 + "\n")
        except Exception as e:
            print(f"⚠️ Warning: Could not establish ngrok tunnel on port {port}: {e}")
    else:
        print(f"ℹ️ Cloud deployment mode: Listening directly on port {port}.")

    # Start Telegram Bot in a separate process
    p = multiprocessing.Process(target=start_telegram_bot)
    p.start()

    # Run the FastAPI server via Uvicorn with dynamic port
    # The import string "src.gateways.whatsapp_webhook:app" relies on being run from the project root.
    uvicorn.run("src.gateways.whatsapp_webhook:app", host="0.0.0.0", port=port, reload=False)

if __name__ == "__main__":
    main()
