import os
import sys
from pyngrok import ngrok
import uvicorn

import multiprocessing

def start_telegram_bot():
    from src.gateways.telegram_bot import main as telegram_main
    telegram_main()

def main():
    from src.core.config import NGROK_AUTHTOKEN
    ngrok_token = NGROK_AUTHTOKEN
    if ngrok_token:
        ngrok.set_auth_token(ngrok_token)
    else:
        print("⚠️ WARNING: NGROK_AUTHTOKEN not found in .env. ngrok may fail to start if it requires authentication.")
        
    # Start ngrok tunnel pointing to port 8000
    public_url = ngrok.connect(8000).public_url
    
    print("\n" + "="*60)
    print("🚀 NGROK TUNNEL ESTABLISHED 🚀")
    print("Please copy the URL below and paste it into the Meta Developer Dashboard:")
    print(f"Callback URL: {public_url}/webhook")
    print("="*60 + "\n")
    
    # Start Telegram Bot in a separate process
    p = multiprocessing.Process(target=start_telegram_bot)
    p.start()
    
    # Run the FastAPI server via Uvicorn
    # The import string "src.gateways.whatsapp_webhook:app" relies on being run from the project root.
    uvicorn.run("src.gateways.whatsapp_webhook:app", host="0.0.0.0", port=8000, reload=False)

if __name__ == "__main__":
    main()
