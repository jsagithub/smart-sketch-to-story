# app/pipeline.py
import json
import os
import argparse
import io

import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

from cnn.model import SketchCNN
from rnn.train_rnn import StoryGenerator, StoryDataset, DEVICE

# ---------------------------
# Utilities
# ---------------------------

def load_cnn_model(checkpoint_path, num_classes, device=DEVICE):
    model = SketchCNN(num_classes).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

def preprocess_image(img: Image.Image, size=(128,128)):
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    return transform(img).unsqueeze(0)  # add batch dim

def predict_topk_from_pil(model, pil_img, idx_to_class, topk=3, device=DEVICE):
    x = preprocess_image(pil_img).to(device)
    with torch.no_grad():
        logits = model(x)  # (1, C)
        probs = F.softmax(logits, dim=-1)
        top_probs, top_idx = torch.topk(probs, k=topk, dim=-1)
        top_probs = top_probs.cpu().numpy().flatten().tolist()
        top_idx = top_idx.cpu().numpy().flatten().tolist()
    top_classes = [idx_to_class[i] for i in top_idx]
    return list(zip(top_classes, top_probs))

def load_idx_to_class(path="cnn/idx_to_class.json"):
    with open(path, "r") as f:
        raw = json.load(f)
    # JSON keys are strings → convert to int
    return {int(k): v for k, v in raw.items()}

# ---------------------------
# RNN resources + generation (greedy, re-using train_rnn logic)
# ---------------------------

def load_rnn_resources(jsonl_path, model_path):
    dataset = StoryDataset(jsonl_path)
    vocab_size = len(dataset.word2idx)
    obj_count = len(dataset.obj2idx)

    model = StoryGenerator(vocab_size=vocab_size, obj_count=obj_count).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model, dataset

def generate_story_from_objects_rnn(model, dataset, object_names, max_len=30):
    # Convert object names to ids; if an object is unknown, skip it
    obj_ids = []
    for o in object_names:
        if o in dataset.obj2idx:
            obj_ids.append(dataset.obj2idx[o])
        else:
            # try normalized forms (lowercase)
            if o.lower() in dataset.obj2idx:
                obj_ids.append(dataset.obj2idx[o.lower()])
    if len(obj_ids) == 0:
        raise ValueError(f"No provided objects found in RNN object vocab. Given: {object_names}")

    obj_tensor = torch.tensor(obj_ids, device=DEVICE)

    sos_id = dataset.word2idx["<SOS>"]
    eos_id = dataset.word2idx["<EOS>"]

    generated_ids = []
    input_id = torch.tensor([[sos_id]], device=DEVICE)
    hidden = None

    with torch.no_grad():
        for _ in range(max_len):
            word_emb = model.word_embed(input_id)  # (1,1,embed_dim)
            obj_emb = model.obj_embed(obj_tensor)  # (num_objs, embed_dim)
            obj_mean = obj_emb.mean(dim=0).unsqueeze(0).unsqueeze(1)  # (1,1,embed_dim)

            conditioned = word_emb + obj_mean
            out, hidden = model.lstm(conditioned, hidden)
            logits = model.fc(out[:, -1, :])  # (1, vocab)
            next_id = int(torch.argmax(logits, dim=-1).item())

            if next_id == eos_id:
                break
            generated_ids.append(next_id)
            input_id = torch.tensor([[next_id]], device=DEVICE)

    # Convert ids to words
    words = []
    for wid in generated_ids:
        w = dataset.idx2word.get(wid, "")
        if w not in ("<PAD>", "<SOS>", "<EOS>"):
            words.append(w)
    return " ".join(words)


# ---------------------------
# Main pipeline orchestration
# ---------------------------

def run_pipeline_from_image(
    sketch_path,
    cnn_checkpoint,
    cnn_parquet_for_map,
    rnn_jsonl,
    rnn_checkpoint,
    topk=3,
    save_output_dir=None,
    generate_image=False
):
    # 1) Load CNN class name map
    idx_to_class = load_idx_to_class("cnn/cnn/idx_to_class.json")

    # 2) Load CNN model
    num_classes = len(idx_to_class)
    cnn_model = load_cnn_model(cnn_checkpoint, num_classes)

    # 3) Load image
    pil = Image.open(sketch_path).convert("L")

    # 4) Predict top-K
    topk_preds = predict_topk_from_pil(cnn_model, pil, idx_to_class, topk=topk)
    predicted_objects = [c for c, p in topk_preds]
    print("Top-K predictions (class, prob):", topk_preds)

    # 5) Load RNN resources
    rnn_model, rnn_dataset = load_rnn_resources(rnn_jsonl, rnn_checkpoint)

    # 6) Generate story (use predicted objects)
    try:
        story = generate_story_from_objects_rnn(rnn_model, rnn_dataset, predicted_objects)
    except ValueError as e:
        # fallback: try lowercased object names
        story = generate_story_from_objects_rnn(rnn_model, rnn_dataset, [o.lower() for o in predicted_objects])

    print("Generated story:", story)

    return {
        "predicted_objects": topk_preds,
        "story": story,
    }

def run_pipeline_from_parquet_row(
    parquet_file,
    row_index,
    cnn_checkpoint,
    rnn_jsonl,
    rnn_checkpoint,
    topk=3,
    save_output_dir=None,
    generate_image=False
):
    # load image bytes from parquet row
    ds = pd.read_parquet("data/datasets/train-00000-of-00001.parquet")
    row = ds.df.iloc[row_index]
    # Try to extract image bytes (mirrors parquet_dataset.py logic)
    img_field = row["image"]
    if isinstance(img_field, (bytes, bytearray)):
        img_bytes = img_field
    elif isinstance(img_field, dict) and "bytes" in img_field:
        img_bytes = img_field["bytes"]
    else:
        raise ValueError("Unsupported image format in parquet row.")

    pil = Image.open(io.BytesIO(img_bytes)).convert("L")
    # For mapping we need parquet file to obtain idx->class
    return run_pipeline_from_image(
        sketch_path=None,
        cnn_checkpoint=cnn_checkpoint,
        cnn_parquet_for_map=parquet_file,
        rnn_jsonl=rnn_jsonl,
        rnn_checkpoint=rnn_checkpoint,
        topk=topk,
    )

# ---------------------------
# CLI
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="End-to-end: sketch -> objects -> story -> (image)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", type=str, help="Path to sketch image (PNG/JPG)")

    args = ap.parse_args()

    if args.image:
        out = run_pipeline_from_image(
            sketch_path=args.image,
            cnn_checkpoint="cnn/cnn_best.pt",
            cnn_parquet_for_map="data/datasets/train-00000-of-00001.parquet",
            rnn_jsonl="data/stories/story_dataset.jsonl",
            rnn_checkpoint="rnn/rnn_checkpoints/story_rnn.pt",
            topk=3,
        )
    else:
        raise NotImplementedError("Only --image input is currently supported in CLI.")


    print("Pipeline output:", out)

if __name__ == "__main__":
    main()
