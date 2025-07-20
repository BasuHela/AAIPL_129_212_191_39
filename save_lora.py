import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

# --- 1. Define Your Paths ---
# Directory of the base model (e.g., the original Qwen3-4B)
MODEL_DIR = "/jupyter-tutorial/hf_models/Qwen3-4B"

# Directory where your trained LoRA adapter is saved
# This contains files like 'adapter_model.bin' and 'adapter_config.json'
LORA_ADAPTER_DIR = "./ckpt/checkpoint-15050" # Example name for your LoRA adapter

# Directory to save the final, merged model
# This will be your new, standalone model checkpoint
MERGED_MODEL_DIR = "Qwen3-4B-merged-lora"

print(f"Base model directory: {MODEL_DIR}")
print(f"LoRA adapter directory: {LORA_ADAPTER_DIR}")
print(f"Output directory for merged model: {MERGED_MODEL_DIR}")

# --- 2. Load the Base Model and Tokenizer ---
print("\nLoading base model and tokenizer...")
# Load the base model in a higher precision for stable merging.
# Using bfloat16 is recommended if your hardware supports it.
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

# --- 3. Load the PEFT Model (Apply LoRA) ---
print(f"\nLoading PeftModel and applying LoRA adapter: '{LORA_ADAPTER_DIR}'...")
# This loads the base model and applies the LoRA adapter on top.
peft_model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_DIR)

# You can optionally print the model architecture here to see the LoRA layers
# print(peft_model)

# --- 4. Merge the LoRA Weights ---
print("\nMerging LoRA weights into the base model...")
# The `merge_and_unload` method combines the adapter weights with the base model
# and returns the merged model.
merged_model = peft_model.merge_and_unload()
print("Merge complete.")

# You can optionally print the model architecture again to confirm LoRA layers are gone
# print(merged_model)

# --- 5. Save the Merged Model and Tokenizer ---
print(f"\nSaving the merged model to: '{MERGED_MODEL_DIR}'")
os.makedirs(MERGED_MODEL_DIR, exist_ok=True)
merged_model.save_pretrained(MERGED_MODEL_DIR, safe_serialization=True)
tokenizer.save_pretrained(MERGED_MODEL_DIR)

print("\nProcess finished successfully!")
print(f"Your merged model is now saved in '{MERGED_MODEL_DIR}'.")
print("You can now load and use this model as a standard Hugging Face model.")