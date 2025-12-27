import os
import json
import matplotlib.pyplot as plt

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(PROJECT_DIR, "results", "unsloth_lora", "training_log.json")
OUT_FILE = os.path.join(PROJECT_DIR, "results", "unsloth_lora", "training_loss_curve.png")

with open(DATA_FILE) as f:
    history = json.load(f)

steps = [h["step"] + (h["epoch"] - 1) * 100 for h in history]
losses = [h["loss"] for h in history]

plt.figure(figsize=(8, 5))
plt.plot(steps, losses, marker="o")
plt.title("Unsloth LoRA Training Loss")
plt.xlabel("Synthetic Step")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT_FILE)
print(f"Training loss plot saved to: {OUT_FILE}")
