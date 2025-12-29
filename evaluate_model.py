import os
import json
import torch
from datasets import load_from_disk
from unsloth import FastLanguageModel
from transformers import AutoTokenizer
import evaluate
from tqdm import tqdm

# Paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data", "dolly")
MODEL_DIR = os.path.join(PROJECT_DIR, "results", "unsloth_lora")
OUTPUT_FILE = os.path.join(PROJECT_DIR, "results", "unsloth_metrics.json")

# Load test set
test = load_from_disk(os.path.join(DATA_DIR, "test"))
test = test.select(range(20))

# Format data function
def format_instruction(example):
    prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n"
    return prompt

# Load model + LoRA adapter
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_seq_length=2048,
    dtype=torch.float16,
    load_in_4bit=True,
)

# Cargar LoRA entrenado
model.load_adapter(MODEL_DIR)

# Activar modo inferencia
FastLanguageModel.for_inference(model)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.eval()

# Evaluators
bleu = evaluate.load("sacrebleu")
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")

# Evaluation
predictions = []
references = []

print("Generating predictions on the test set...")
# print(test[0])
# prompt = format_instruction(test[0])
# inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

# with torch.no_grad():
#     output_ids = model.generate(**inputs, max_new_tokens=128)

# print(tokenizer.decode(output_ids[0], skip_special_tokens=True))

for sample in tqdm(test):
    prompt = format_instruction(sample)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.2,
            do_sample=False,
        )

    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    pred = decoded.split("### Response:")[-1].strip()
    ref = sample["response"]

    predictions.append(pred)
    references.append(ref)

# Metrics
print("Calculating metrics...")

bleu_score = bleu.compute(predictions=predictions, references=[[r] for r in references])
rouge_score = rouge.compute(predictions=predictions, references=references)
bertscore_res = bertscore.compute(predictions=predictions, references=references, lang="en")

# Save results
results = {
    "BLEU": bleu_score,
    "ROUGE-L": rouge_score["rougeL"],
    "BERTScore": {
        "precision": float(sum(bertscore_res["precision"]) / len(bertscore_res["precision"])),
        "recall": float(sum(bertscore_res["recall"]) / len(bertscore_res["recall"])),
        "f1": float(sum(bertscore_res["f1"]) / len(bertscore_res["f1"]))
    }
}

with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=4)

print(f"\nEvaluation complete. Results saved to {OUTPUT_FILE}")
