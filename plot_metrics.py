import os
import json
import matplotlib.pyplot as plt

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS_FILE = os.path.join(PROJECT_DIR, "results", "unsloth_metrics.json")
OUTPUT = os.path.join(PROJECT_DIR, "results", "unsloth_metrics_plot.png")

with open(METRICS_FILE) as f:
    metrics = json.load(f)

# Cargo las métricas
bleu = metrics["BLEU"]["score"]
rouge = metrics["ROUGE-L"]
bert_f1 = metrics["BERTScore"]["f1"]

names = ["BLEU", "ROUGE-L", "BERTScore-F1"]
values = [bleu, rouge, bert_f1]

plt.figure(figsize=(7,5))
plt.bar(names, values)
plt.title("Evaluation Metrics (Unsloth LoRA)")
plt.ylabel("Score")
plt.tight_layout()
plt.savefig(OUTPUT)

print(f"Gráfica guardada en: {OUTPUT}")
