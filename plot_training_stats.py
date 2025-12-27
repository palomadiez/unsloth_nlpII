import os
import json
import matplotlib.pyplot as plt

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE_FILE = os.path.join(PROJECT_DIR, "results", "hf_lora", "checkpoint-125", "trainer_state.json")
OUT_DIR = os.path.join(PROJECT_DIR, "results", "hf")

with open(STATE_FILE) as f:
    state = json.load(f)

log_history = state["log_history"]

steps = []
losses = []
grad_norms = []
learning_rates = []

for entry in log_history:
    if "loss" in entry:
        steps.append(entry["step"])
        losses.append(entry["loss"])
        grad_norms.append(entry.get("grad_norm", None))
        learning_rates.append(entry.get("learning_rate", None))

# -----------------------------
# 1. LOSS
# -----------------------------
plt.figure(figsize=(8, 5))
plt.plot(steps, losses)
plt.title("Training Loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "loss_curve.png"))
plt.close()

# -----------------------------
# 2. GRAD NORM
# -----------------------------
plt.figure(figsize=(8, 5))
plt.plot(steps, grad_norms)
plt.title("Gradient Norm")
plt.xlabel("Step")
plt.ylabel("Grad Norm")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "grad_norm_curve.png"))
plt.close()

# -----------------------------
# 3. LEARNING RATE
# -----------------------------
plt.figure(figsize=(8, 5))
plt.plot(steps, learning_rates)
plt.title("Learning Rate Schedule")
plt.xlabel("Step")
plt.ylabel("Learning Rate")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "learning_rate_curve.png"))
plt.close()

print("Gráficas generadas en /results")
