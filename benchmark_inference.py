import csv
import os
import time
import psutil
import torch
import json
from unsloth import FastLanguageModel
import numpy as np
from datasets import Dataset

# Configuración
PATH_ARROW = "data/dolly/test/data-00000-of-00001.arrow"

MAX_EJEMPLOS = 5
N_INTENTOS = 2
MAX_NEW_TOKENS = 100
TEMPERATURE = 0.7
TOP_P = 0.9

# Paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.expanduser("~/unsloth_results")
CSV_OUTPUT = os.path.expanduser("~/benchmark_unsloth_base_vs_ft")

# Modelo base usado también en HF
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

#  Cargar modelo
print("Loading model for inference benchmark...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = BASE_MODEL,
    max_seq_length = 2048,
    dtype = torch.float16,
    load_in_4bit = True,      # Cambia a True si usaste QLoRA en Unsloth
)
FastLanguageModel.for_inference(model)
device = "cuda" if torch.cuda.is_available() else "cpu"
# model.to(device)
model.eval()

# FT MODEL
ft_model, _ = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=2048,
    dtype=torch.float16,
    load_in_4bit=True,
)

ft_model.load_adapter(MODEL_DIR)
FastLanguageModel.for_inference(ft_model)
ft_model.eval()


# Preparación de prompts Dolly
def prepare_dolly_prompt(ejemplo):
    instruccion = ejemplo.get("instruction", "")
    contexto = ejemplo.get("context", "")
    if contexto:
        return f"Instruction: {instruccion}\nContext: {contexto}\nResponse:"
    else:
        return f"Instruction: {instruccion}\nResponse:"
    
# Benchmark por intento
def run_single_inference(prompt, model):
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    torch.cuda.reset_peak_memory_stats()
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
    peak_memory_mb = torch.cuda.max_memory_allocated() / 1024**2

    prompt_len = inputs["input_ids"].shape[1]
    tokens_generated = output_ids.shape[1] - prompt_len
    tps = tokens_generated / latency if latency > 0 else 0

    response = tokenizer.decode(
        output_ids[0][prompt_len:],
        skip_special_tokens=True
    )

    return latency, tps, peak_memory_mb, response

# Benchmark completo
def benchmark_prompt(prompt, model, n_iterations):
    latencies = []
    throughputs = []
    memories = []
    last_response = ""

    print(f"  -> Evaluando {n_iterations} intentos...")

    for _ in range(n_iterations):
        latency, tps, memory, response = run_single_inference(prompt, model)
        latencies.append(latency)
        throughputs.append(tps)
        memories.append(memory)
        last_response = response

    return {
        "output_text": last_response,
        "latency_per_request_s": np.mean(latencies),
        "throughput_tokens_sec": np.mean(throughputs),
        "peak_memory_mb": max(memories)
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

        base_metrics = benchmark_prompt(prompt, model, N_INTENTOS)
        ft_metrics = benchmark_prompt(prompt, ft_model, N_INTENTOS)

        results.append({
        "id": i + 1,
        "category": ejemplo["category"],
        "prompt": prompt,

        # Base model
        "base_latency_s": base_metrics["latency_per_request_s"],
        "base_throughput_tps": base_metrics["throughput_tokens_sec"],
        "base_peak_memory_mb": base_metrics["peak_memory_mb"],
        "base_output_text": base_metrics["output_text"],

        # FINE-TUNED MODEL
        "ft_latency_s": ft_metrics["latency_per_request_s"],
        "ft_throughput_tps": ft_metrics["throughput_tokens_sec"],
        "ft_peak_memory_mb": ft_metrics["peak_memory_mb"],
        "ft_output_text": ft_metrics["output_text"],
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
        print("\nCategory:", r["category"])
        print("BASE---------------")
        print("Total Time (s):", r["base_latency_s"])
        print("Tokens/s:", r["base_throughput_tps"])
        print("Memory Used (MB):", r["base_peak_memory_mb"])
        print("Response Unsloth: ", r["base_output_text"])
        print("FT-----------------")
        print("Total Time (s):", r["ft_latency_s"])
        print("Tokens/s:", r["ft_throughput_tps"])
        print("Memory Used (MB):", r["ft_peak_memory_mb"])
        print("Response Unsloth: ", r["ft_output_text"])