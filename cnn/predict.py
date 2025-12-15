import json

import torch
from .dataset import TUBerlinParquet
from .model import SketchCNN
from torchvision import transforms
from PIL import Image
import io
from data.datasets.items import items

def load_idx_to_class(path="cnn/cnn/idx_to_class.json"):
    with open(path, "r") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}

# load dataset to get class names
dummy = TUBerlinParquet("data/datasets/train-00000-of-00001.parquet")
idx_to_class = load_idx_to_class()

# load model
model = SketchCNN(num_classes=len(idx_to_class))
model.load_state_dict(torch.load("cnn/cnn_best.pt", map_location="cpu"))
model.eval()

# transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def predict_image_bytes(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        out = model(x)
        pred = out.argmax(1).item()
    return [items[pred]]

