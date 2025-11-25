import torch
from dataset import TUBerlinParquet
from model import SketchCNN
from torchvision import transforms
from PIL import Image
import io

# load dataset to get class names
dummy = TUBerlinParquet("../data/datasets/train-00000-of-00001.parquet")
idx_to_class = dummy.idx_to_class

# load model
model = SketchCNN(num_classes=len(idx_to_class))
model.load_state_dict(torch.load("cnn_best.pt", map_location="cpu"))
model.eval()

# transforms
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def predict_image_bytes(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        out = model(x)
        pred = out.argmax(1).item()
    return idx_to_class[pred]

print(predict_image_bytes(open("../data/sketches_raw/car.jpeg","rb").read()))