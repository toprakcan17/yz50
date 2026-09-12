import torch
import matplotlib.pyplot as plt


with open('/Users/toprak/Documents/yz50/4. Hafta/names.txt') as names_file:
    names = names_file.read().splitlines()

chars = sorted(list(set(''.join(names))))
chars.append('.')
def split_dataset(n):
    train = []
    valid = []
    test = []
    for i,j in enumerate(n):
        if i%10 == 0: valid.append(j)
        elif i%10 == 1: test.append(j)
        else: train.append(j)
    return (train, valid, test)
def encode(char):
    return chars.index(char)
def decode(index):
    return chars[index]
def encode_str(string):
    return [encode(i) for i in string]

xtr,ytr,xval,yval,xtest,ytest = list(), list(), list(), list(), list(), list()
block_size = 3
dims = 2
dataset = split_dataset(names)
for a, dset in enumerate(dataset):
    for i in dataset[a]:
        i = f"{'.'*block_size}{i}."
        i = encode_str(i)
        for j,k in enumerate(i[:-block_size]):
            ctx = i[j:j+block_size]
            if a == 0: x = xtr; y = ytr
            elif a == 1: x = xval; y = yval
            else: x = xtest; y = ytest
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
params = [embedding_matrix, W1, B1, W2, B2]
for i in params: i.requires_grad = True
def forward_pass():
    act1 = ((emb_flattened @ W1) + B1).tanh()
    logits = act1 @ W2 + B2
    return logits


lre = torch.linspace(-3,0,10000)
lr = 10**lre
lr_stats = []
loss_stats = []
print(f"Parametre sayisi: {sum(p.nelement() for p in params)}")
"""for step in range(1000):
    idx = torch.randint(0, 256, (256,))

    emb = embedding_matrix[x[idx]]
    emb_flattened = emb.view(256, block_size*dims)

    logits = forward_pass()
    loss = torch.nn.functional.cross_entropy(logits, y[idx])
    for i in params: i.grad = None
    loss.backward()
    for i in params:
        i.data -= i.grad * 0.1 # lr[step]
    lr_stats.append(lr[step])
    loss_stats.append(loss.item())"for step in range(1000):
    idx = torch.randint(0, 256, (256,))

    emb = embedding_matrix[x[idx]]
    emb_flattened = emb.view(256, block_size*dims)

    logits = forward_pass()
    loss = torch.nn.functional.cross_entropy(logits, y[idx])
    for i in params: i.grad = None
    loss.backward()
    for i in params:
        i.data -= i.grad * 0.1 # lr[step]
    lr_stats.append(lr[step])
    loss_stats.append(loss.item())"""
    
for step in range(1000):
    idx = torch.randint(0, x.shape[0], (256,))

    emb = embedding_matrix[x[idx]]
    emb_flattened = emb.view(256, block_size*dims)

    logits = forward_pass()
    loss = torch.nn.functional.cross_entropy(logits, y[idx])
    for i in params: i.grad = None
    loss.backward()
    for i in params:
        i.data -= i.grad * 0.01 # lr[step] # 
    lr_stats.append(lr[step])
    loss_stats.append(loss.item())
    
#plt.plot(lr_stats, loss_stats)
plt.plot(loss_stats)
plt.pause(10)