import torch
import matplotlib.pyplot as plt


with open('/Users/toprak/Documents/yz50/3. Hafta/isimler.txt') as names_file:
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

g = torch.Generator().manual_seed(2147483647)

xtr = torch.tensor(xtr)
ytr = torch.tensor(ytr)
xval = torch.tensor(xval)
yval = torch.tensor(yval)
xtest = torch.tensor(xtest)
ytest = torch.tensor(ytest)



l1_size = 100
embedding_matrix = torch.randn(len(chars), dims, generator=g)
W1 = torch.randn(dims*block_size, l1_size, generator=g) * ((5/3)/(dims*block_size)) **0.5
B1 = torch.randn(l1_size, generator=g) * 0.01
W2 = torch.randn(l1_size, len(chars), generator=g) * 0.01
B2 = torch.randn(len(chars), generator=g) * 0.01
bn_gain = torch.ones((1,l1_size)) 
bn_bias = torch.zeros((1, l1_size))
bn_running_var = torch.ones((1,l1_size))
bn_running_mean = torch.zeros((1, l1_size))


params = [embedding_matrix, W1, B1, W2, B2, bn_bias, bn_gain]
for i in params: i.requires_grad = True

def forward_pass(embd):
    act1 = ((embd @ W1) + B1).tanh()
    logits = act1 @ W2 + B2
    return logits

step_count = 20000
lre = torch.linspace(-3,0,step_count)
lr = 10**lre
lr_stats = []
loss_stats = []
print(f"Parametre sayisi: {sum(p.nelement() for p in params)}")
    
batch_size = 256
m = 0.1

for step in range(step_count):
    idx = torch.randint(0, xtr.shape[0], (batch_size,), generator=g)
    idx_dev = torch.randint(0, xval.shape[0], (batch_size,), generator=g)

    emb = embedding_matrix[xtr[idx]]
    emb_flattened = emb.view(batch_size, block_size*dims)

    emb_dev = embedding_matrix[xval[idx_dev]]
    emb_flattened_dev = emb_dev.view(batch_size, block_size*dims)

    h = ((emb_flattened @ W1) + B1)
    bn_h = bn_gain * (h-h.mean(dim=0, keepdim=True))/torch.sqrt(h.var(dim=0, keepdim=True)) + bn_bias
    logits = bn_h.tanh() @ W2 + B2
    # loss = torch.nn.functional.cross_entropy(logits, ytr[idx])    
    logits_exp = logits.exp()
    logits_exp_sum = logits_exp.sum(dim=1, keepdim=True)
    probs = logits_exp * logits_exp_sum**-1
    logprobs = probs.log()
    loss = -logprobs[range(batch_size), ytr[idx]].mean() 
    for i in [emb_flattened, emb, h, bn_h, logits, logits_exp, logits_exp_sum, probs, logprobs]: i.retain_grad()
    loss.backward()

    grad_logprobs = torch.zeros_like(logprobs)
    grad_logprobs[range(batch_size), ytr[idx]] = -batch_size**-1
    grad_probs = grad_logprobs/probs
    grad_logits_exp = logits_exp_sum**-1 * grad_probs
    grad_counts_sum_inv = logits_exp * grad_probs
    grad_counts_sum = grad_counts_sum_inv*(-logits_exp_sum**-2)
    grad_logits_exp += torch.ones_like(logits)*grad_counts_sum # broadcasting icin onemli
    grad_logits = logits_exp*grad_logits_exp


    with torch.no_grad():
        bn_running_mean = (1-m) * bn_running_mean + m * h.mean(dim=0, keepdim=True)
        bn_running_var = (1-m) * bn_running_var + m * h.var(dim=0, keepdim=True)

        h = ((emb_flattened_dev @ W1) + B1)
        bn_h = bn_gain * (h-bn_running_mean)/torch.sqrt(bn_running_var) + bn_bias
        logits_dev = bn_h.tanh() @ W2 + B2
        devloss = torch.nn.functional.cross_entropy(logits_dev, yval[idx_dev])
    for i in params: i.grad = None
    loss.backward()
    for i in params:
        i.data -= i.grad * 0.1
    #lr_stats.append(lr[step])
    if step % 50 == 0: loss_stats.append(devloss.log10().item())

emb_test = embedding_matrix[xtest]
embcat_test = emb_test.view(-1, block_size*dims)
h = ((embcat_test @ W1) + B1)
bn_h = bn_gain * (h-bn_running_mean)/torch.sqrt(bn_running_var) + bn_bias
logits_test= bn_h.tanh() @ W2 + B2

model_loss = torch.nn.functional.cross_entropy(logits_test, ytest)
print("Loss:", model_loss.item())


plt.plot(loss_stats)
plt.show()

"""
plt.figure(figsize=(8,8))
plt.scatter(embedding_matrix[:,0].data, embedding_matrix[:,1].data, s=200)
for i in range(embedding_matrix.shape[0]):
    plt.text(embedding_matrix[i,0].item(), embedding_matrix[i,1].item(), decode(i), ha="center", va="center", color="white")
plt.grid('minor')
plt.show()
"""
for _ in range(10):
    ctx = '.'*block_size
    word = ""
    while True:
        emb = embedding_matrix[encode_str(ctx)]
        embcat = emb.view(-1, block_size*dims)
        h = ((embcat @ W1) + B1)
        bn_h = bn_gain * (h-bn_running_mean)/torch.sqrt(bn_running_var) + bn_bias
        _logits = bn_h.tanh() @ W2 + B2
        probs = torch.nn.functional.softmax(_logits)
        new_char = torch.multinomial(probs, 1, generator=g).item()
        if decode(new_char) == '.': break
        else: word = word+decode(new_char)
        ctx = ctx[1:]+decode(new_char)
    print(word)