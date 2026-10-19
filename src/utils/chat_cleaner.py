import re
import json
import os

def clean_chat(input_filepath, output_filepath):
    # Matches both standard WhatsApp export formats:
    # 1: [dd/mm/yy, hh:mm:ss] Name: Message
    # 2: dd/mm/yy, hh:mm - Name: Message
    pattern = re.compile(
        r'^(?:\[?(\d{1,2}[\/\.]\d{1,2}[\/\.]\d{2,4})[, ]\s*(\d{1,2}:\d{2}(?::\d{2})?\s*(?:[aApP][mM])?)\]?)\s*[-:]?\s*([^:]+):\s*(.*)$'
    )
    
    cleaned_messages = []
    
    if not os.path.exists(input_filepath):
        print(f"Error: Could not find {input_filepath}")
        return
        
    with open(input_filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            match = pattern.match(line)
            if match:
                date_str, time_str, name, message = match.groups()
                
                # Drop system and media messages
                if any(x in message for x in ["<Media omitted>", "image omitted", "video omitted"]):
                    continue
                    
                cleaned_messages.append({
                    "name": name.strip(),
                    "message": message.strip()
                })
            else:
                # Handle multi-line messages by appending to the previous message
                if cleaned_messages:
                    cleaned_messages[-1]['message'] += f"\n{line}"

    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(cleaned_messages, f, indent=2, ensure_ascii=False)
    
    print(f"Cleaned {len(cleaned_messages)} messages and saved to {output_filepath}")
    return cleaned_messages

if __name__ == "__main__":
    os.makedirs(os.path.dirname("data/cleaned_chat.json"), exist_ok=True)
    clean_chat("data/test_chat.txt", "data/cleaned_chat.json")
