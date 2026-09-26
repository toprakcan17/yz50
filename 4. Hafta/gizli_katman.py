import torch

with open('/Users/toprak/Documents/yz50/4. Hafta/names.txt') as names_file:
    names = names_file.read().splitlines()

chars = sorted(list(set(''.join(names))))
chars.append('.')
def encode(char):
    return chars.index(char)

def decode(index):
    return chars[index]
def encode_str(string):
    return [encode(i) for i in string]
def softmax(tensor):
    return tensor.exp()/tensor.exp().sum(1, keepdims=True)
x,y = list(), list()
block_size = 4
dims = 2
for i in names[:5]:
    i = f"{'.'*block_size}{i}."
    i = encode_str(i)
    for j,k in enumerate(i[:-block_size]):
        ctx = i[j:j+block_size]
        x.append(ctx)
        y.append(i[j+block_size])

x = torch.tensor(x)
y = torch.tensor(y)
embedding_matrix = torch.randn(len(chars), dims)
emb = embedding_matrix[x]
emb_flattened = emb.view(len(x), block_size*dims)


l1_size = 1000
W1 = torch.randn(emb_flattened.shape[1], l1_size)
B1 = torch.randn(l1_size)
W2 = torch.randn(l1_size, len(chars))
B2 = torch.randn(len(chars))
act1 = ((emb_flattened @ W1) + B1).tanh()
logits = act1 @ W2 + B2
probs = softmax(logits)

loss_manual = -probs[torch.arange(x.shape[0]), y].log().mean()
loss = torch.nn.functional.cross_entropy(logits, y)
print(f'Elle hesaplanan loss: {loss_manual} - Fonksiyonla hesaplanan loss: {loss}')