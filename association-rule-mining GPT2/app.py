import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math

# Sample data (same as before)
transactions = [
    ['bread', 'butter', 'milk'],
    ['bread', 'butter', 'eggs'],
    ['milk', 'cereal'],
    ['bread', 'jam', 'eggs'],
    ['butter', 'milk', 'yogurt'],
    ['bread', 'butter', 'cheese'],
    ['milk', 'yogurt', 'cereal'],
    ['bread', 'eggs', 'bacon'],
    ['butter', 'cheese', 'yogurt'],
    ['bread', 'milk', 'cereal']
]

# Create a vocabulary of unique items
vocab = sorted(list(set([item for transaction in transactions for item in transaction])))
item2idx = {item: idx for idx, item in enumerate(vocab)}
idx2item = {idx: item for item, idx in item2idx.items()}

# Add special tokens
special_tokens = ['<pad>', '<sos>', '<eos>']
for token in special_tokens:
    item2idx[token] = len(item2idx)  # {'<pad>' : 9, '<sos>': 10, ...}
    idx2item[len(idx2item)] = token
    vocab.append(token)

# print(item2idx)


# Dataset class
class TransactionDataset(Dataset):
    def __init__(self, transactions, item2idx, max_len):
        self.transactions = transactions
        self.item2idx = item2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.transactions)

    def __getitem__(self, idx):
        transaction = ['<sos>'] + self.transactions[idx] + ['<eos>']
        input_seq = [self.item2idx[item] for item in transaction]

        # Pad sequence if necessary
        if len(input_seq) < self.max_len:
            input_seq = input_seq + [self.item2idx['<pad>']] * (self.max_len - len(input_seq))
        else:
            input_seq = input_seq[:self.max_len]

        return torch.tensor(input_seq[:-1]), torch.tensor(input_seq[1:])


# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]


# Transformer Decoder Model
class TransformerDecoderModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        decoder_layers = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers)
        self.output = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(self, tgt, tgt_mask=None):
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)
        memory = torch.zeros_like(tgt)  # Self-attention only, no encoder memory
        output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)
        return self.output(output)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


# Training function
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        tgt_mask = model.generate_square_subsequent_mask(inputs.size(1)).to(device)
        outputs = model(inputs, tgt_mask=tgt_mask)
        loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


# Hyperparameters
D_MODEL = 64
NHEAD = 4
NUM_LAYERS = 2
DIM_FEEDFORWARD = 256
BATCH_SIZE = 4
EPOCHS = 100
LEARNING_RATE = 0.001
MAX_LEN = 12  # Maximum sequence length (including special tokens)

# Create dataset and dataloader
dataset = TransactionDataset(transactions, item2idx, MAX_LEN)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerDecoderModel(len(vocab), D_MODEL, NHEAD, NUM_LAYERS, DIM_FEEDFORWARD).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=item2idx['<pad>'])
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(EPOCHS):
    loss = train(model, dataloader, criterion, optimizer, device)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {loss:.4f}")

print("Training completed.")


def predict_next_items(model, input_items, item2idx, idx2item, max_len, device, top_k=5):
    model.eval()

    # Convert input items to indices and add special tokens
    input_seq = ['<sos>'] + input_items
    input_indices = [item2idx.get(item, item2idx['<pad>']) for item in input_seq]  # Use '<pad>' for unknown items

    # Pad sequence if necessary
    if len(input_indices) < max_len:
        input_indices = input_indices + [item2idx['<pad>']] * (max_len - len(input_indices))
    else:
        input_indices = input_indices[:max_len]

    input_tensor = torch.tensor(input_indices).unsqueeze(0).to(device)

    with torch.no_grad():
        tgt_mask = model.generate_square_subsequent_mask(input_tensor.size(1)).to(device)
        output = model(input_tensor, tgt_mask=tgt_mask)

    # Get probabilities for the next item
    probabilities = torch.softmax(output[0, -1, :], dim=-1)
    top_k_probs, top_k_indices = torch.topk(probabilities, k=top_k)

    # Decode predictions
    predicted_items = [idx2item[idx.item()] for idx in top_k_indices]

    return predicted_items, top_k_probs.tolist()


# Test prediction
test_input = ['bread', 'butter']
predicted_items, probabilities = predict_next_items(model, test_input, item2idx, idx2item, MAX_LEN, device)

print(f"Input: {test_input}")
print("Predicted next items:")
for item, prob in zip(predicted_items, probabilities):
    if item not in ['<pad>', '<sos>', '<eos>']:
        print(f"{item}: {prob:.4f}")