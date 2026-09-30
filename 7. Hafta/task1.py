import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib

with open('C:\\Users\\toprak\\Documents\\yz50\\7. Hafta\\shakespeare.txt') as input_file:
    text_data = input_file.read()

chars = sorted(list(set(text_data)))

torch.set_default_device('cuda')

def encode(str):
    out = []
    for i in str: out.append(chars.index(i))
    return out
def decode(list):
    out = ''
    for i in list: out = f'{out}{chars[i]}'
    return out

block_size = 8
batch_size = 4

torch.manual_seed(1234)

split_rate = 0.9
split_idx = int(split_rate * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

def get_batch(batchsize, blocksize, data):
    idx = torch.randint(len(data) - blocksize, (batchsize, ))
    x = torch.stack([torch.tensor(encode(data[i:i+blocksize])) for i in idx])
    y = torch.stack([torch.tensor(encode(data[i+1:i+blocksize+1])) for i in idx])
    return x,y

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding_table = nn.Embedding(vocab_size, vocab_size)
    def forward(self, idx, targets=None):
        logits = self.embedding_table(idx)
        if targets is not None: 
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
            return logits, loss
        else:
            return logits, None
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx

model = BigramLanguageModel(len(chars))
batch = get_batch(4,8,train_data)

optim = torch.optim.AdamW(model.parameters(), lr=1e-3)

total_steps = 10000
for step in range(total_steps):
    x, y = get_batch(4,8,train_data)
    logits, loss = model(x,y)
    optim.zero_grad(set_to_none=True)
    loss.backward()
    if not step%1000: print(f"{step}/{total_steps}: {loss.item()}")
    optim.step()

print(loss.item())
