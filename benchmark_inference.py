import csv
import os
import time
import psutil
import torch
import json
from unsloth import FastLanguageModel
import numpy as np
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM

# Configuración
PATH_ARROW = "data/dolly/test/data-00000-of-00001.arrow"

MAX_EJEMPLOS = 5
N_INTENTOS = 2
MAX_NEW_TOKENS = 100
TEMPERATURE = 0.7
TOP_P = 0.9

# Paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_DIR, "results", "unsloth_lora")
CSV_OUTPUT = os.path.join(PROJECT_DIR, "results", "unsloth_metrics", "unsloth_peft_comparative_metrics.csv")

# Modelo base usado también en HF
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

#  Cargar modelo
print("Loading model for inference benchmark...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = BASE_MODEL,
    max_seq_length = 2048,
    load_in_4bit = False      # Cambia a True si usaste QLoRA en Unsloth
)
model.eval()

print("Loading PEFT model...")
peft_base_model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device_map="cpu"
)
peft_model = PeftModel.from_pretrained(
    peft_base_model,
    MODEL_DIR
)
peft_model.eval()

# Preparación de prompts Dolly
def prepare_dolly_prompt(ejemplo):
    instruccion = ejemplo.get("instruction", "")
    contexto = ejemplo.get("context", "")
    if contexto:
        return f"Instruction: {instruccion}\nContext: {contexto}\nResponse:"
    else:
        return f"Instruction: {instruccion}\nResponse:"
    
# Memoria pico (RSS)
def get_peak_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

# Benchmark por intento
def run_single_inference(prompt, model):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

    start_time = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            top_p=TOP_P
        )
    end_time = time.time()

    latency = end_time - start_time
    prompt_len = inputs["input_ids"].shape[1]
    tokens_generated = output_ids.shape[1] - prompt_len
    tps = tokens_generated / latency if latency > 0 else 0

    response = tokenizer.decode(
        output_ids[0][prompt_len:],
        skip_special_tokens=True
    )

    return latency, tps, tokens_generated, response

# Benchmark completo
def benchmark_prompt(prompt, model, n_iterations):
    latencies = []
    throughputs = []
    last_response = ""

    print(f"  -> Evaluando {n_iterations} intentos...")

    for _ in range(n_iterations):
        latency, tps, _, response = run_single_inference(prompt, model)
        latencies.append(latency)
        throughputs.append(tps)
        last_response = response

    peak_mem = get_peak_memory_mb()

    return {
        "output_text": last_response,
        "latency_per_request_s": np.mean(latencies),
        "throughput_tokens_sec": np.mean(throughputs),
        "peak_memory_mb": peak_mem
    }


# Run benchmark
if __name__ == "__main__":
    print("Loading Dolly dataset...")
    dataset = Dataset.from_file(PATH_ARROW)

    print("Running Unsloth inference benchmark...\n")
    results = []

    for i in range(min(MAX_EJEMPLOS, len(dataset))):
        ejemplo = dataset[i]
        prompt = prepare_dolly_prompt(ejemplo)
        print(f"\n[Ejemplo {i+1}] Categoría: {ejemplo['category']}")

        base_metrics = benchmark_prompt(prompt, base_model, N_INTENTOS)
        peft_metrics = benchmark_prompt(prompt, peft_model, N_INTENTOS)

        results.append({
        "id": i + 1,
        "category": ejemplo["category"],
        "prompt": prompt,

        # Base model
        "base_latency_s": base_metrics["latency_per_request_s"],
        "base_throughput_tps": base_metrics["throughput_tokens_sec"],
        "base_peak_memory_mb": base_metrics["peak_memory_mb"],
        "base_output_text": base_metrics["output_text"],

        # PEFT model
        "peft_latency_s": peft_metrics["latency_per_request_s"],
        "peft_throughput_tps": peft_metrics["throughput_tokens_sec"],
        "peft_peak_memory_mb": peft_metrics["peak_memory_mb"],
        "peft_output_text": peft_metrics["output_text"],
        })


    # Guardo CSV
    if results:
        with open(CSV_OUTPUT, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    print(f"\nResultados detallados guardados en: {CSV_OUTPUT}")

    # Print results
    print("\n===== Unsloth Inference Benchmark Results =====")
    for r in results:
        print("\nPrompt:", r["prompt"])
        print("Total Time (s):", r["total_time_seconds"])
        print("Tokens/s:", r["tokens_per_second"])
        print("Memory Used (MB):", r["memory_used_MB"]) 