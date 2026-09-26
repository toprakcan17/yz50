import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F


with open('/Users/toprak/Documents/yz50/3. Hafta/names.txt') as names_file:
    names = names_file.read().splitlines()

chars = sorted(list(set(''.join(names))))
print(len(chars))
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

class Linear:
    def __init__(self, fan_in, fan_out, bias=False):
        self.w = torch.randn((fan_in, fan_out)) * torch.sqrt(torch.tensor((2/fan_in)))
        self.b = (torch.zeros(fan_out) if bias else None)
    def __call__(self, ins):
        if self.b: self.out = ins @ self.w + self.b
        else: self.out = ins @ self.w
        return self.out
    def params(self):
        return [self.w] + ([] if self.b is None else [self.b])
class BatchNorm1d:
    def __init__(self, dim, m=0.1, training=True):
        self.eps = 1e-5
        self.m = m
        self.training = training
        self.bngain = torch.ones(dim)
        self.bnbias = torch.zeros(dim)
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)
    def __call__(self, x):
        if self.training:
            if x.ndim == 2:
                dim = 0
            elif x.ndim == 3:
                dim = (0,1)
            mean = x.mean(dim, keepdim=True)
            var = x.var(dim, keepdim=True)
        else: mean = self.running_mean; var = self.running_var
        diff = x - mean
        self.out = self.bngain * (diff / torch.sqrt(var + self.eps)) + self.bnbias
        if self.training: 
            with torch.no_grad(): self.running_mean = (1-self.m)*self.running_mean + self.m*mean; self.running_var = (1-self.m)*self.running_var + self.m*var
        return self.out
    def params(self):
        return [self.bngain, self.bnbias]

class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out
    def params(self):
        return []

class Embedding:
    def __init__(self, size, dim):
        self.w = torch.randn((size, dim))
    def __call__(self, idx):
        self.out = self.w[idx]
        return self.out
    def params(self):
        return [self.w]

class Flatten:
    def __init__(self, n):
        self.n = n
    def __call__(self, emb):
        emb = emb.view(emb.shape[0], -1, self.n*emb.shape[2])
        if emb.shape[1] == 1: emb = torch.squeeze(emb)
        self.out = emb
        return self.out
    def params(self): return []

class Sequential:
    def __init__(self, layers):
        self.layers = layers
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out
    def params(self):
        return [p for layer in self.layers for p in layer.params()]

xtr,ytr,xval,yval,xtest,ytest = list(), list(), list(), list(), list(), list()
block_size = 8
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

torch.manual_seed(2147483647)

xtr, ytr, xval, yval, xtest, ytest = map(
    torch.tensor,
    [xtr, ytr, xval, yval, xtest, ytest]
)


l1_size = 100

model = Sequential([
    Embedding(len(chars), dims),
    Flatten(2),
    Linear(block_size * dims, l1_size),
    BatchNorm1d(l1_size),
    Tanh(),
    Linear(l1_size, len(chars))
])

steps = 20000
batch_size = 256
lr = 0.1
parameters =  model.params()

loss_stats = []

print('Parametre sayisi:', sum(p.nelement() for p in parameters))
for param in parameters: param.requires_grad = True

# gradient descent
for step in range(steps):
    idx = torch.randint(0,xtr.shape[0], (batch_size,))
    batch = (xtr[idx], ytr[idx])

    logits = model(batch[0])
    loss = F.cross_entropy(logits, batch[1])
    for param in parameters: param.grad = None
    loss.backward()

    for param in parameters:
        param.data -= lr*param.grad

    if not step%1000: print(f'Loss: {loss.item():.4f}')

    loss_stats.append(torch.log10(loss).item())

loss = F.cross_entropy(model(xtest), ytest)
print(f'Test loss: {loss.item():.4f}')

plt.plot(torch.tensor(loss_stats).view(-1, 1000).mean(1))
plt.show()

