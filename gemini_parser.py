import json

# Load your authentic 451-block dataset
with open("dataset.jsonl", "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

gemini_dataset = []
for row in data:
    messages = []
    for msg in row["conversations"]:
        # Skip the system prompt here (we will paste it into the Google AI Studio UI later)
        if msg["from"] == "system":
            continue
        
        # Convert Llama roles to Gemini roles
        role = "user" if msg["from"] == "human" else "model"
        messages.append({"role": role, "content": msg["value"]})
        
    gemini_dataset.append({"messages": messages})

# Save the Gemini-ready file
with open("gemini_tuning_dataset.jsonl", "w", encoding="utf-8") as f:
    for item in gemini_dataset:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
print("✅ Dataset converted for Google AI Studio!")