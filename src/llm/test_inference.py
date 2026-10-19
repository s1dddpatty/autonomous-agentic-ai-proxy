import time
import ollama

def run_inference(label):
    system_prompt = "You are texting your sister. Keep replies under 12 words, casual, mostly lowercase, no formal punctuation."
    context = "Context: Sister adopted a cat named Milo; lives in New York."
    user_input = "how is your day going? milo just knocked over another plant 😭"

    print(f"\n--- {label} ---")
    start_time = time.perf_counter()
    
    response = ollama.chat(
        model='persona-llama',
        messages=[
            {'role': 'system', 'content': f"{system_prompt}\n\n{context}"},
            {'role': 'user', 'content': user_input}
        ]
    )
    
    end_time = time.perf_counter()
    latency = end_time - start_time
    
    output = response.get('message', {}).get('content', '')
    
    # Calculate tokens per second
    eval_count = response.get('eval_count', 0)
    eval_duration = response.get('eval_duration', 0) / 1e9  # nanoseconds to seconds
    
    tps = (eval_count / eval_duration) if eval_duration > 0 else 0
    
    print(f"Output: {output}")
    print(f"Latency: {latency:.2f} seconds")
    print(f"Speed: {tps:.2f} tokens/second")
    print(f"Tokens Evaluated: {eval_count}")

# Run cold start
run_inference("Cold Start")

# Wait a brief moment
time.sleep(1)

# Run warm start
run_inference("Warm Start")
