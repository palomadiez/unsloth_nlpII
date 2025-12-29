import matplotlib.pyplot as plt
import os

SAVE_FILE = os.path.expanduser("~/comparisons")
os.makedirs(SAVE_FILE, exist_ok=True)

# BLEU
models = ["Base", "r16_b32", "r32_b128", "r32_b16", "r4_b16", "r4_b64", "r8_b128"]
scores = [5.4, 3.96, 3.07, 4.26, 2.57, 2.68, 2.98]

plt.bar(models, scores, color=plt.cm.Set2.colors)
plt.ylabel("Score")
plt.title("Model comparison - BLEU")
plt.savefig(os.path.join(SAVE_FILE, "bleu_comparison"))
plt.show()

# ROUGE
models = ["Base", "r16_b32", "r32_b128", "r32_b16", "r4_b16", "r4_b64", "r8_b128"]
scores = [0.22, 0.21, 0.22, 0.26, 0.26, 0.21, 0.21]

plt.bar(models, scores, color=plt.cm.Set2.colors)
plt.ylabel("Score")
plt.title("Model comparison - ROUGE")
plt.savefig(os.path.join(SAVE_FILE, "rouge_comparison"))
plt.show()

# BERT - F1
models = ["Base", "r16_b32", "r32_b128", "r32_b16", "r4_b16", "r4_b64", "r8_b128"]
scores = [0.83, 0.85, 0.86, 0.86, 0.87, 0.86,0.86]

plt.bar(models, scores, color=plt.cm.Set2.colors)
plt.ylabel("Score")
plt.title("Model comparison - BERT F1")
plt.savefig(os.path.join(SAVE_FILE, "bert_f1_comparison"))
plt.show()
