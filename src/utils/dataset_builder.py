import json
import os

def build_dataset(input_filepath, output_filepath, target_persona_name, user_persona_name):
    if not os.path.exists(input_filepath):
        print(f"Error: Could not find {input_filepath}")
        return

    with open(input_filepath, 'r', encoding='utf-8') as f:
        messages = json.load(f)
        
    system_prompt = "You are texting your sister. Keep replies casual, mostly lowercase, no formal punctuation."
    
    dataset = []
    
    # 1. Group consecutive messages by the same user to form complete paragraphs/thoughts
    grouped_messages = []
    current_name = None
    current_text = []
    
    for msg in messages:
        if msg['name'] != current_name:
            if current_name is not None:
                grouped_messages.append({"name": current_name, "message": " ".join(current_text)})
            current_name = msg['name']
            current_text = [msg['message']]
        else:
            current_text.append(msg['message'])
            
    if current_name is not None:
         grouped_messages.append({"name": current_name, "message": " ".join(current_text)})
         
    # 2. Build ShareGPT pairs (Instruction/Response)
    for i in range(len(grouped_messages) - 1):
        msg_1 = grouped_messages[i]
        msg_2 = grouped_messages[i+1]
        
        # We only care when the Sister sends a message and Me replies
        if msg_1['name'] == user_persona_name and msg_2['name'] == target_persona_name:
            record = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": msg_1['message']},
                    {"role": "assistant", "content": msg_2['message']}
                ]
            }
            dataset.append(record)
            
    with open(output_filepath, 'w', encoding='utf-8') as f:
        for record in dataset:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
    print(f"Generated {len(dataset)} ShareGPT pairs and saved to {output_filepath}")

if __name__ == "__main__":
    os.makedirs(os.path.dirname("data/dataset.jsonl"), exist_ok=True)
    build_dataset(
        input_filepath="data/cleaned_chat.json", 
        output_filepath="data/dataset.jsonl", 
        target_persona_name="Me", 
        user_persona_name="Sister"
    )
