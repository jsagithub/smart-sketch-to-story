import json
import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


######################################################################
# 1. Load dataset
######################################################################

class StoryDataset(Dataset):
    def __init__(self, jsonl_path):
        self.samples = []
        with open(jsonl_path, "r") as f:
            for line in f:
                self.samples.append(json.loads(line))

        # build vocab
        self.word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2}
        self.idx2word = {}

        # collect all words
        all_text = []
        for sample in self.samples:
            words = sample["story"].lower().split()
            all_text.extend(words)

        for word in sorted(set(all_text)):
            self.word2idx[word] = len(self.word2idx)

        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

        # build object vocab
        self.obj2idx = {}
        all_objs = set()
        for sample in self.samples:
            for obj in sample["objects"]:
                all_objs.add(obj)

        self.obj2idx = {obj: i for i, obj in enumerate(sorted(all_objs))}

    def encode_text(self, text):
        tokens = text.lower().split()
        ids = [self.word2idx[t] for t in tokens]
        return torch.tensor([self.word2idx["<SOS>"]] + ids + [self.word2idx["<EOS>"]])

    def encode_objects(self, objs):
        ids = [self.obj2idx[o] for o in objs]
        return torch.tensor(ids)

    def __getitem__(self, idx):
        item = self.samples[idx]
        obj_ids = self.encode_objects(item["objects"])
        story_ids = self.encode_text(item["story"])
        return obj_ids, story_ids

    def __len__(self):
        return len(self.samples)


######################################################################
# 2. Collate function for batching
######################################################################

def collate_fn(batch):
    obj_seqs = [x[0] for x in batch]
    txt_seqs = [x[1] for x in batch]

    # pad text
    txt_padded = pad_sequence(txt_seqs, batch_first=True, padding_value=0)
    txt_lengths = torch.tensor([len(x) for x in txt_seqs])

    return obj_seqs, txt_padded, txt_lengths


######################################################################
# 3. LSTM Story Generator Model
######################################################################

class StoryGenerator(nn.Module):
    def __init__(self, vocab_size, obj_count, embed_dim=128, hidden_dim=256):
        super().__init__()

        # object embeddings (for the input conditioning)
        self.obj_embed = nn.Embedding(obj_count, embed_dim)

        # word embeddings (for the decoder)
        self.word_embed = nn.Embedding(vocab_size, embed_dim)

        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, object_lists, text_input):

        # ---- 1. Encode objects ----
        # Each sample has variable number of objects
        batch_obj_vectors = []
        for obj_ids in object_lists:
            emb = self.obj_embed(obj_ids)          # (num_objs, embed_dim)
            mean_emb = emb.mean(dim=0)             # (embed_dim,)
            batch_obj_vectors.append(mean_emb)

        # (batch, embed_dim)
        obj_tensor = torch.stack(batch_obj_vectors, dim=0)

        # ---- 2. Word embeddings ----
        text_emb = self.word_embed(text_input)

        # Add conditioning by adding object vector to every time step
        batch, seq, dim = text_emb.shape
        obj_expand = obj_tensor.unsqueeze(1).expand(-1, seq, -1)
        conditioned = text_emb + obj_expand

        # ---- 3. LSTM ----
        out, _ = self.lstm(conditioned)

        # ---- 4. Output logits ----
        logits = self.fc(out)
        return logits


######################################################################
# 4. Training Loop
######################################################################

def train_model(jsonl_path, epochs=10, batch_size=32, lr=1e-3):

    dataset = StoryDataset(jsonl_path)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, collate_fn=collate_fn)

    model = StoryGenerator(
        vocab_size=len(dataset.word2idx),
        obj_count=len(dataset.obj2idx)
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore PAD
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("Training started...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for obj_seqs, txt_padded, _ in tqdm(dataloader):
            txt_padded = txt_padded.to(DEVICE)
            obj_seqs = [o.to(DEVICE) for o in obj_seqs]
            logits = model(obj_seqs, txt_padded[:, :-1])  # shift input
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                txt_padded[:, 1:].reshape(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss / len(dataloader):.4f}")

    os.makedirs("rnn_checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "rnn_checkpoints/story_rnn.pt")

    print("Model saved → rnn_checkpoints/story_rnn.pt")


######################################################################
# 5. Main
######################################################################

if __name__ == "__main__":
    train_model("../data/stories/story_dataset.jsonl", epochs=10)
