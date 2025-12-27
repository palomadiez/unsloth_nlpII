import os
import time
import psutil
import torch

from unsloth import FastLanguageModel

# Paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_DIR, "results", "unsloth_lora")

# Modelo base usado también en HF
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


print("Loading Unsloth model for inference benchmark...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = BASE_MODEL,
    max_seq_length = 2048,
    load_in_4bit = False      # Cambia a True si usaste QLoRA en Unsloth
)

model.eval()

# Benchmark function
def benchmark(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    # Ensure inputs are on the same device as the model parameters
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Measure memory before
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)

    # Latency: time to first token
    start = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.2
        )
    end = time.time()

    mem_after = process.memory_info().rss / (1024 * 1024)

    total_time = end - start
    num_tokens = len(output_ids[0]) - len(inputs["input_ids"][0])
    tps = num_tokens / total_time if total_time > 0 else 0.0

    return {
        "prompt": prompt,
        "total_time_seconds": total_time,
        "tokens_generated": num_tokens,
        "tokens_per_second": tps,
        "memory_before_MB": mem_before,
        "memory_after_MB": mem_after,
        "memory_used_MB": mem_after - mem_before,
        }


# Run benchmark
if __name__ == "__main__":
    prompts = [
        "Explain the concept of reinforcement learning.",
        "Write a short poem about the ocean.",
        "What is the difference between supervised and unsupervised learning?",
    ]

    print("Running Unsloth inference benchmark...\n")

    results = []
    for p in prompts:
        print(f"- Prompt: {p[:30]}...")
        results.append(benchmark(p))

    import json
    out_dir = os.path.join(PROJECT_DIR, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "benchmark_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n===== Unsloth Inference Benchmark Results =====")
    for r in results:
        print("\nPrompt:", r["prompt"])
        print("Total Time (s):", r["total_time_seconds"])
        print("Tokens generated:", r["tokens_generated"])
        print("Tokens/s:", r["tokens_per_second"])
        print("Memory Used (MB):", r["memory_used_MB"]) 

    print(f"Saved benchmark results to: {out_file}")