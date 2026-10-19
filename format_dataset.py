import os
import re
import json

CHATS_DIR = "chats"
OUTPUT_FILE = "dataset.jsonl"
YOUR_NAME = "You"
FRIEND_NAME = "Aarushi Ghosh SPIT"

# 🧠 THE ANCHOR PROMPT: This defines the entire relationship dynamic
SYSTEM_PROMPT = "You are Siddharth texting your girlfriend Aarushi. You are affectionate, use sweet emojis, type in casual Hinglish, and reply warmly. You never use formal punctuation and you are chill most of the times."

def parse_whatsapp_files():
    all_messages = []
    pattern = re.compile(r'^(?:\[)?\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}.*?(?:\]\s|-\s)(.*?):\s(.*)')
    
    for root, _, files in os.walk(CHATS_DIR):
        for filename in sorted(files):
            if filename != "chat.txt": continue
            filepath = os.path.join(root, filename)
            
            with open(filepath, "r", encoding="utf-8") as f:
                last_sender = None
                last_message = ""
                
                for line in f:
                    line = line.strip()
                    if not line or "Messages and calls are end-to-end encrypted" in line or "<Media omitted>" in line: continue
                        
                    match = pattern.match(line)
                    if match:
                        sender, text = match.groups()
                        sender = sender.strip()
                        text = text.strip()
                        
                        # ✨ THE FIX: Merge consecutive bubbles from the same sender
                        if sender == last_sender:
                            last_message += "\n" + text
                        else:
                            if last_sender and last_message:
                                role = "gpt" if last_sender == YOUR_NAME else "human"
                                all_messages.append({"from": role, "value": last_message})
                            last_sender = sender
                            last_message = text
                    else:
                        if last_sender:
                            last_message += "\n" + line
                            
                if last_sender and last_message:
                    role = "gpt" if last_sender == YOUR_NAME else "human"
                    all_messages.append({"from": role, "value": last_message})

    dataset = []
    chunk_size = 15 
    for i in range(0, len(all_messages), chunk_size):
        chunk = all_messages[i:i + chunk_size]
        
        # Ensure the chunk strictly starts with her to maintain alternating sequence
        if chunk and chunk[0]["from"] == "gpt":
            chunk = chunk[1:]
            
        if len(chunk) >= 2 and any(m["from"] == "human" for m in chunk) and any(m["from"] == "gpt" for m in chunk):
            # ✨ THE FIX: Inject the System Prompt at the start of every chunk
            final_chunk = [{"from": "system", "value": SYSTEM_PROMPT}] + chunk
            dataset.append({"conversations": final_chunk})

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for conv in dataset:
            out.write(json.dumps(conv, ensure_ascii=False) + "\n")
            
    print(f"✅ Dataset rebuilt with Merged Bubbles & System Prompts! Total blocks: {len(dataset)}")

if __name__ == "__main__":
    parse_whatsapp_files()