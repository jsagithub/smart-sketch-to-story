import json
import os

import pandas as pd

PARQUET_PATH = "../data/datasets/train-00000-of-00001.parquet"
OUT_DIR = "cnn"
OUT_PATH = os.path.join(OUT_DIR, "idx_to_class.json")

# Ensure output directory exists
os.makedirs(OUT_DIR, exist_ok=True)
# Load parquet
df = pd.read_parquet(PARQUET_PATH)

# Detect label column
if "category" in df.columns:
    labels = sorted(df["category"].unique())
    idx_to_class = {i: label for i, label in enumerate(labels)}

elif "label" in df.columns:
    # Numeric labels case (less common)
    labels = sorted(df["label"].unique())
    idx_to_class = {int(label): str(label) for label in labels}

else:
    raise ValueError("No 'category' or 'label' column found in parquet.")

# Save
with open(OUT_PATH, "w") as f:
    json.dump(idx_to_class, f, indent=2)

print(f"Saved class mapping to {OUT_PATH}")
print(f"Total classes: {len(idx_to_class)}")
print("First 10 classes:", list(idx_to_class.items())[:10])
