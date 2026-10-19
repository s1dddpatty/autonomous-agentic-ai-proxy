def mock_generate_draft(state):
    """
    Simulates the Tone Core returning a draft based on incoming_text.
    """
    incoming = state.get("incoming_text", "")
    print(f"[Mock Generator] Analyzing incoming text: '{incoming}'")
    
    if "plant" in incoming.lower():
        return "same old, milo's still making messes everywhere 🤣"
    
    return "lol i have no idea what's going on"

def mock_dispatch_whatsapp(state):
    """
    Simulates sending the final approved text via WhatsApp.
    """
    final_draft = state.get("draft_text", "")
    print(f"[Mock WhatsApp] >> DISPATCHING MESSAGE: '{final_draft}'")
    return True
