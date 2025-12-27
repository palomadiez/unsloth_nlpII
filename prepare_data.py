import os
from datasets import load_dataset

# Ruta del proyecto
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data", "dolly")

os.makedirs(DATA_DIR, exist_ok=True)

print("Saving dataset to:", DATA_DIR)

# Cargar dataset
dataset = load_dataset("databricks/databricks-dolly-15k")
dataset = dataset.shuffle(seed=42)

# Crear splits
train = dataset["train"].select(range(12000))
dev = dataset["train"].select(range(12000, 14000))
test = dataset["train"].select(range(14000, 15000))

# Guardar a disco
train.save_to_disk(os.path.join(DATA_DIR, "train"))
dev.save_to_disk(os.path.join(DATA_DIR, "dev"))
test.save_to_disk(os.path.join(DATA_DIR, "test"))

print("Dataset guardado correctamente.")
