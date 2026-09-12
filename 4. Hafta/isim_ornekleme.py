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
dims = 10
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


embedding_matrix = torch.randn(len(chars), dims)
xtr = torch.tensor(xtr)
ytr = torch.tensor(ytr)
xval = torch.tensor(xval)
yval = torch.tensor(yval)
xtest = torch.tensor(xtest)
ytest = torch.tensor(ytest)

g = torch.Generator().manual_seed(2147483647)

l1_size = 100
W1 = torch.randn(dims*block_size, l1_size, generator=g)
B1 = torch.randn(l1_size, generator=g)
W2 = torch.randn(l1_size, len(chars), generator=g)
B2 = torch.randn(len(chars), generator=g)
params = [embedding_matrix, W1, B1, W2, B2]
for i in params: i.requires_grad = True

def forward_pass(embd):
    act1 = ((embd @ W1) + B1).tanh()
    logits = act1 @ W2 + B2
    return logits

step_count = 200000
lre = torch.linspace(-3,0,step_count)
lr = 10**lre
lr_stats = []
loss_stats = []
print(f"Parametre sayisi: {sum(p.nelement() for p in params)}")
    
batch_size = 256
for step in range(step_count):
    idx = torch.randint(0, xtr.shape[0], (batch_size,), generator=g)
    idx_dev = torch.randint(0, xval.shape[0], (batch_size,), generator=g)

    emb = embedding_matrix[xtr[idx]]
    emb_flattened = emb.view(batch_size, block_size*dims)

    emb_dev = embedding_matrix[xval[idx_dev]]
    emb_flattened_dev = emb_dev.view(batch_size, block_size*dims)
    logits = forward_pass(emb_flattened)
    loss = torch.nn.functional.cross_entropy(logits, ytr[idx])
    with torch.no_grad():
        logits_dev = forward_pass(emb_flattened_dev)
        devloss = torch.nn.functional.cross_entropy(logits, yval[idx_dev])
    for i in params: i.grad = None
    loss.backward()
    for i in params:
        i.data -= i.grad * 0.1
    lr_stats.append(lr[step])
    loss_stats.append(devloss.log10().item())

model_loss = torch.nn.functional.cross_entropy(forward_pass(embedding_matrix[xtest].view(-1, block_size*dims)), ytest)
print("Loss:", model_loss.item())

plt.plot(loss_stats)
plt.show()

plt.figure(figsize=(8,8))
plt.scatter(embedding_matrix[:,0].data, embedding_matrix[:,1].data, s=200)
for i in range(embedding_matrix.shape[0]):
    plt.text(embedding_matrix[i,0].item(), embedding_matrix[i,1].item(), decode(i), ha="center", va="center", color="white")
plt.grid('minor')
plt.show()

for _ in range(10):
    ctx = '.'*block_size
    word = ""
    while True:
        probs = torch.nn.functional.softmax(forward_pass((embedding_matrix[torch.tensor([encode_str(ctx)])]).view(1,-1)), dim=-1)
        new_char = torch.multinomial(probs, 1, generator=g).item()
        if decode(new_char) == '.': break
        else: word = word+decode(new_char)
        ctx = ctx[1:]+decode(new_char)
    print(word)