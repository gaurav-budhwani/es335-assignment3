import torch
import numpy as np
import re

class NextWordMLP(torch.nn.Module):
    def __init__(self, vocab_size, block_size, embedding_dim=32, hidden_dim=1024, num_hidden_layers=2, activation='relu'):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        input_dim = embedding_dim * block_size

        layers = []
        for i in range(num_hidden_layers):
            layers.append(torch.nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
            if activation == 'relu':
                layers.append(torch.nn.ReLU())
            elif activation == 'tanh':
                layers.append(torch.nn.Tanh())
            else:
                raise ValueError("Unknown activation type")
        self.mlp = torch.nn.Sequential(*layers)
        self.classifier = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embeds = self.embedding(x)
        flat = embeds.view(embeds.size(0), -1)
        features = self.mlp(flat)
        logits = self.classifier(features)
        return logits

def load_checkpoint(path):
    return torch.load(path, map_location=torch.device('cpu'))

def scan_option_vals(file_list, regex):
    vals = set()
    for fname in file_list:
        m = re.search(regex, fname)
        if m:
            vals.add(m.group(1))
    return sorted(list(vals))

def get_closest_vocab_word(word, stoi, model, itos):
    # Find the vocab token whose embedding is closest to <UNK> (just demonstrative)
    unk_idx = stoi.get("<UNK>")
    if unk_idx is None:
        return "<UNK>"
    emb_weights = model.embedding.weight.cpu().detach().numpy()
    unk_emb = emb_weights[unk_idx]
    min_dist = float('inf')
    closest_idx = unk_idx
    for idx, tok in enumerate(itos):
        if idx in [stoi.get("<PAD>", -1), stoi.get("<UNK>", -1), stoi.get("<start>", -1)]:
            continue
        dist = np.linalg.norm(unk_emb - emb_weights[idx])
        if dist < min_dist:
            min_dist = dist
            closest_idx = idx
    return itos[closest_idx]

def prepare_context(context_words, handle_oov, stoi, model, itos):
    clean_context = []
    affected = []
    for w in context_words:
        if w in stoi:
            clean_context.append(w)
        else:
            affected.append(w)
            if handle_oov == "Skip word":
                continue
            elif handle_oov == "Mask as <UNK>":
                clean_context.append("<UNK>")
            elif handle_oov == "Find closest (embedding similarity)":
                close_word = get_closest_vocab_word(w, stoi, model, itos)
                clean_context.append(close_word)
    return clean_context, affected

def generate_next_k_words(model, context, stoi, itos, block_size, k, temperature=1.0, device='cpu', skip_tokens=None, seed=42):
    # Set default skip tokens if None
    if skip_tokens is None:
        skip_tokens = {"<start>", "<PAD>", "<UNK>", "<eos>", "<EOS>", "<BOS>", "<bos>"}
    
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    generated = []
    local_context = context[:]  # Copy to avoid modifying input
    
    for _ in range(k):
        context_idx = [stoi.get(w, stoi['<UNK>']) for w in local_context[-block_size:]]
        if len(context_idx) < block_size:
            context_idx = [stoi['<PAD>']] * (block_size - len(context_idx)) + context_idx
            
        x = torch.tensor(context_idx, dtype=torch.long).unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            logits = model(x)
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            
            # Filter out special tokens
            filtered_probs = probs.copy()
            for idx, tok in enumerate(itos):
                if tok in skip_tokens:
                    filtered_probs[idx] = 0
            
            # Renormalize probabilities
            filtered_probs = filtered_probs / filtered_probs.sum()
            
            next_idx = np.random.choice(len(itos), p=filtered_probs)
            next_word = itos[next_idx]
            
        generated.append(next_word)
        local_context.append(next_word)
        
    return generated
