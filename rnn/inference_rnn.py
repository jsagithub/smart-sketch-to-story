import torch
from .train_rnn import StoryGenerator, StoryDataset, DEVICE


######################################################################
# Load trained model + vocab
######################################################################

def load_resources(jsonl_path="data/stories/story_dataset.jsonl",
                   model_path="rnn/rnn_checkpoints/story_rnn.pt"):

    # Load dataset (for vocab + object map)
    dataset = StoryDataset(jsonl_path)

    vocab_size = len(dataset.word2idx)
    obj_count = len(dataset.obj2idx)

    model = StoryGenerator(vocab_size=vocab_size,
                           obj_count=obj_count).to(DEVICE)

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    return model, dataset


######################################################################
# Convert tokens → text
######################################################################

def ids_to_sentence(ids, idx2word):
    words = []
    for idx in ids:
        word = idx2word.get(idx, "")
        if word == "<EOS>":
            break
        if word not in ("<PAD>", "<SOS>"):
            words.append(word)
    return " ".join(words)


######################################################################
# Text generation (greedy decoding)
######################################################################

def generate_story(model, dataset, objects, max_len=30):

    # 1. Convert objects → tensor of ids
    obj_ids = [dataset.obj2idx[o] for o in objects]
    obj_tensor = torch.tensor(obj_ids).to(DEVICE)

    # 2. Prepare initial input: <SOS>
    sos_id = dataset.word2idx["<SOS>"]
    eos_id = dataset.word2idx["<EOS>"]

    input_id = torch.tensor([[sos_id]], device=DEVICE)

    # Hidden state for LSTM
    hidden = None

    # Collect generated ids
    generated = []

    for _ in range(max_len):

        # Embed + condition with object embeddings
        word_emb = model.word_embed(input_id)

        # Condition on objects
        with torch.no_grad():
            obj_emb = model.obj_embed(obj_tensor)   # (num_objs, embed_dim)
            obj_mean = obj_emb.mean(dim=0)          # (embed_dim,)
            obj_mean = obj_mean.unsqueeze(0).unsqueeze(1)  # (1,1,embed_dim)

        conditioned = word_emb + obj_mean

        # LSTM step
        with torch.no_grad():
            output, hidden = model.lstm(conditioned, hidden)
            logits = model.fc(output[:, -1, :])  # last time step only
            next_id = torch.argmax(logits, dim=-1).item()

        if next_id == eos_id:
            break

        generated.append(next_id)

        input_id = torch.tensor([[next_id]], device=DEVICE)

    # Convert ids → words
    return ids_to_sentence(generated, dataset.idx2word)

