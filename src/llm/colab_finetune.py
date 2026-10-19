"""
Google Colab Unsloth Fine-Tuning Pipeline
Instructions:
1. Open Google Colab and select the Free T4 GPU (Runtime > Change runtime type > T4 GPU).
2. Create a new code cell and run the following pip installs:
   !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
   !pip install --no-deps xformers trl peft accelerate bitsandbytes
3. Upload your generated `dataset.jsonl` to the Colab environment's file explorer.
4. Copy and run the code below in a new cell.
"""

from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
from unsloth.chat_templates import get_chat_template

# 1. Configuration
max_seq_length = 2048 # Can be increased if needed
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# 2. Load Base Model
# Using Llama 3.2 3B Instruct model, optimized by Unsloth for fast 4-bit loading
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-3B-Instruct",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# 3. Configure LoRA Adapter
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

# 4. Format Dataset using Chat Template
# This maps the ShareGPT formatted JSONL dataset natively into Llama 3 prompt structures
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3",
    mapping = {"role": "role", "content": "content", "user": "user", "assistant": "assistant"}
)

def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
    return {"text": texts}

dataset = load_dataset("json", data_files={"train": "dataset.jsonl"}, split="train")
dataset = dataset.map(formatting_prompts_func, batched=True)

# 5. Configure SFTTrainer
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60, # Increase for real dataset (e.g. 100-300 depending on size)
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

# 6. Run Training
print("Starting training...")
trainer_stats = trainer.train()
print("Training complete!")

# 7. Export to GGUF for Ollama
# This automatically merges the adapter and exports it in 4-bit quantized GGUF format
print("Exporting model to GGUF format for local Ollama usage...")
model.save_pretrained_gguf("persona_llama_gguf", tokenizer, quantization_method = "q4_k_m")
print("Export complete! Download the 'persona_llama_gguf' folder/file to your local Mac.")
