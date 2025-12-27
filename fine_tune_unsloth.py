import os
from datasets import load_from_disk
import unsloth
# Disable xformers backend if present — some GPU/driver combos (very new GPUs) have
# xformers ops incompatibilities; force fallback to PyTorch SDPA/Flash if available.
try:
    import unsloth.models._utils as _unsloth_utils
    _unsloth_utils.HAS_XFORMERS = False
    _unsloth_utils.xformers = None
    _unsloth_utils.xformers_attention = None
    _unsloth_utils.xformers_version = None
    import unsloth.utils.attention_dispatch as _attn
    _attn.HAS_XFORMERS = False
    _attn.XFORMERS_BLOCK_DIAG_CLS = None
except Exception:
    pass
from transformers import AutoTokenizer, AutoConfig
from unsloth import FastLanguageModel
from peft import LoraConfig, get_peft_model
from utils import format_dolly
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

# Paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data", "dolly")

train_full = load_from_disk(os.path.join(DATA_DIR, "train"))
dev_full = load_from_disk(os.path.join(DATA_DIR, "dev"))
train = train_full.select(range(100))
dev = dev_full.select(range(20))

# Cargo el modelo
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
config = AutoConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(config)

# Tokenizar dataset
def tokenize_function(examples):
    return format_dolly(examples, tokenizer)

train = train.map(
    tokenize_function,
    batched=True,
    batch_size=500,
    remove_columns=train.column_names
)
dev = dev.map(
    tokenize_function,
    batched=True,
    batch_size=500,
    remove_columns=dev.column_names
)

train.set_format("torch")
dev.set_format("torch")

# Load Unsloth optimized model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=2048,
    load_in_4bit=False,  # Cambiar a True si quieres QLoRA
)

# Add LoRA adapter using PEFT
lora_config = LoraConfig(
    r=4,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Dataloaders
train_loader = DataLoader(train, batch_size=1, shuffle=True)
dev_loader = DataLoader(dev, batch_size=1)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Optimizer
optimizer = AdamW(model.parameters(), lr=2e-4)

# Training loop similar lógica al script HF
EPOCHS = 5

print("\nStarting Unsloth training...\n")

# Prepare training history logging
training_history = []

for epoch in range(EPOCHS):
    model.train()
    progress = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for batch in progress:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        progress.set_postfix({"loss": loss.item()})
        # Record a training step
        step = len(training_history) + 1
        training_history.append({
            "epoch": epoch + 1,
            "step": step,
            "loss": loss.item(),
            # optionally include learning rate if scheduler used
        })

print("\nTraining completed.")


# Save model
OUTPUT_DIR = os.path.join(PROJECT_DIR, "results", "unsloth_lora")
os.makedirs(OUTPUT_DIR, exist_ok=True)

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Save training history to disk for plotting/collection
import json
with open(os.path.join(OUTPUT_DIR, "training_log.json"), "w") as f:
    json.dump(training_history, f, indent=2)

print(f"Training log saved to: {os.path.join(OUTPUT_DIR, 'training_log.json')}")

print(f"Model saved to: {OUTPUT_DIR}")
