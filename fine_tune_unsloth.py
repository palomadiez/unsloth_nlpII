import os
import json
import torch
from datasets import load_from_disk
from unsloth import FastLanguageModel
from utils import format_dolly
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import AdamW

# Paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data", "dolly")
OUTPUT_DIR = os.path.expanduser("~/unsloth_results")

# Hyperparams
R_LORA = 32
ALPHA_LORA= 4*R_LORA
BATCH_SIZE = 16
LEARNING_RATE = 2e-4

# Datasets
train_full = load_from_disk(os.path.join(DATA_DIR, "train"))
dev_full = load_from_disk(os.path.join(DATA_DIR, "dev"))
train = train_full.select(range(100))
dev = dev_full.select(range(20))

# Cargo el modelo
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=2048,
    dtype=torch.float16,
    load_in_4bit=True,
)

FastLanguageModel.for_training(model)

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

# Add LoRA adapter using PEFT
model = FastLanguageModel.get_peft_model(
    model,
    r=R_LORA,
    lora_alpha=ALPHA_LORA,
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    bias="none",
    use_gradient_checkpointing=True,
)

# Dataloaders
train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
dev_loader = DataLoader(dev, batch_size=BATCH_SIZE)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Optimizer
optimizer = AdamW(model.parameters(), lr=2e-4)

# Training loop
print("\nStarting Unsloth training...\n")
EPOCHS = 5
training_history = []
global_step = 0

for epoch in range(EPOCHS):
    model.train()
    progress = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for batch in progress:
        global_step += 1
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        progress.set_postfix({"loss": loss.item()})
        # Record a training step
        training_history.append({
            "epoch": epoch + 1,
            "step": global_step,
            "loss": loss.item(),
            "learning_rate": optimizer.param_groups[0]['lr']
        })

print("\nTraining completed.")

# Save model
os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Save training history to disk for plotting/collection
with open(os.path.join(OUTPUT_DIR, "training_log.json"), "w") as f:
    json.dump(training_history, f, indent=2)