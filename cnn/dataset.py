import pandas as pd
import io
from PIL import Image
from torch.utils.data import Dataset

class TUBerlinParquet(Dataset):
    def __init__(self, parquet_path, transform=None):
        self.df = pd.read_parquet(parquet_path)
        self.transform = transform

        # Detect label column
        if "label" in self.df.columns:
            self.label_col = "label"
        elif "category" in self.df.columns:
            self.label_col = "category"
        else:
            raise ValueError("No label column found in parquet.")

        self.num_samples = len(self.df)

        # Build label → index mapping
        if self.df[self.label_col].dtype == 'object':
            # string labels
            classes = sorted(self.df[self.label_col].unique())
            self.class_to_idx = {c: i for i, c in enumerate(classes)}
        else:
            # numeric labels
            classes = sorted(self.df[self.label_col].unique())
            self.class_to_idx = {int(c): int(c) for c in classes}

        # Reverse mapping
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # --- Load image ---
        img_bytes = None

        if isinstance(row["image"], (bytes, bytearray)):
            img_bytes = row["image"]
        elif isinstance(row["image"], dict) and "bytes" in row["image"]:
            img_bytes = row["image"]["bytes"]
        else:
            raise ValueError("Unsupported image format in parquet row.")

        img = Image.open(io.BytesIO(img_bytes)).convert("L")

        # --- Transform ---
        if self.transform:
            img = self.transform(img)

        # --- Label ---
        label_value = row[self.label_col]
        label = self.class_to_idx[label_value]

        return img, label
