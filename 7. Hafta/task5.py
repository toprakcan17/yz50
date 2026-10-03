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
embedding_dims = 32
head_size = 16

torch.manual_seed(1234)

split_rate = 0.9
split_idx = int(split_rate * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

val_step = 0

def get_batch(batchsize, blocksize, data):
    idx = torch.randint(len(data) - blocksize, (batchsize, ))
    x = torch.stack([torch.tensor(encode(data[i:i+blocksize])) for i in idx])
    y = torch.stack([torch.tensor(encode(data[i+1:i+blocksize+1])) for i in idx])
    return x,y

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(embedding_dims, head_size, bias=False)
        self.value = nn.Linear(embedding_dims, head_size, bias=False)
        self.query = nn.Linear(embedding_dims, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    def forward(self, ins):
        k = self.key(ins)
        q = self.query(ins)
        wei = q @ k.transpose(-2, -1)
        wei = wei * head_size **-0.5
        wei = F.softmax(wei, dim=-1)
        v = self.value(ins)
        out = wei @ v
        if val_step == 4999: print(out[0])
        return out


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_dims)
        self.positional_encoding = nn.Embedding(block_size, embedding_dims)
        self.sa_head = Head(head_size)
        self.lm_head = nn.Linear(head_size, len(chars))
    def forward(self, idx, targets=None):
        token_embeddings = self.token_embedding_table(idx)
        pos_embeddings = self.positional_encoding(torch.arange(idx.shape[1]))
        x = token_embeddings + pos_embeddings
        x = self.sa_head(x)
        logits = self.lm_head(x)
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
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
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


xval = torch.tensor([encode(val_data[:-1])])
yval = torch.tensor([encode(val_data[1:])])
loss_val = 0
for _ in range(5000):
    batch = get_batch(batch_size, block_size, val_data)
    _, loss = model(batch[0], batch[1])
    loss_val += loss.item()
    val_step += 1

print(f'val loss: {loss_val/5000}')

ctx = torch.zeros(1,1, dtype=torch.long)
print(decode(model.generate(ctx, 1000)[0].tolist()))
