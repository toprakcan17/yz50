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

def cmp(s, dt, t):
  ex = torch.all(dt == t.grad).item()
  app = torch.allclose(dt, t.grad)
  maxdiff = (dt - t.grad).abs().max().item()
  print(f'{s:15s} | exact: {str(ex):5s} | approximate: {str(app):5s} | maxdiff: {maxdiff}')

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
    var = h.var(dim=0, keepdim=True)
    mean = h.mean(dim=0, keepdim=True)
    sqrt_var = torch.sqrt(var)
    inv_sqrt_var = sqrt_var**-1
    diff = h-mean
    bn_h = bn_gain * (diff * inv_sqrt_var) + bn_bias
    logits = bn_h.tanh() @ W2 + B2
    # loss = torch.nn.functional.cross_entropy(logits, ytr[idx])    
    logits_exp = logits.exp()
    logits_exp_sum = logits_exp.sum(dim=1, keepdim=True)
    probs = logits_exp * logits_exp_sum**-1
    logprobs = probs.log()
    loss = -logprobs[range(batch_size), ytr[idx]].mean() 
    for i in [emb_flattened, emb, h, bn_h, logits, logits_exp, logits_exp_sum, probs, logprobs, diff, inv_sqrt_var, sqrt_var, var, h, mean]: i.retain_grad()
    for i in params: i.grad = None
    loss.backward()
    if step==0:
        grad_logprobs = torch.zeros_like(logprobs)
        grad_logprobs[range(batch_size), ytr[idx]] = -batch_size**-1
        grad_probs = grad_logprobs/probs
        grad_logits_exp = logits_exp_sum**-1 * grad_probs
        grad_counts_sum_inv = (logits_exp * grad_probs).sum(dim=1, keepdim=True)
        grad_counts_sum = grad_counts_sum_inv*(-logits_exp_sum**-2)
        grad_logits_exp += torch.ones_like(logits)*grad_counts_sum # broadcasting icin onemli
        grad_logits = logits_exp*grad_logits_exp
        grad_w2 = bn_h.tanh().T @ grad_logits
        grad_b2 = grad_logits.sum(dim=0, keepdim=False)
        grad_bn_h = grad_logits @ W2.T
        grad_bn_h*= 1 - bn_h.tanh()**2
        grad_diff = grad_bn_h * bn_gain * inv_sqrt_var
        grad_inv_sqrt_var = (grad_bn_h*bn_gain*diff).sum(dim=0, keepdim=True)
        grad_sqrt_var = grad_inv_sqrt_var*-sqrt_var**-2
        grad_var = grad_sqrt_var * sqrt_var * 1/2 * var**-1
        grad_mean = -grad_diff.sum(dim=0, keepdim=True)
        print(mean.shape, diff.shape, h.shape)
        grad_h = grad_diff + ((batch_size**-1)*grad_mean)*torch.ones_like(h)+grad_var * 2 * diff / (batch_size-1)
        grad_w1 = emb_flattened.T @ grad_h
        grad_b1 = grad_h.sum(dim=0, keepdim=False)
        grad_emb_flattened = grad_h @ W1.T
        grad_emb = grad_emb_flattened.view(batch_size, block_size, dims)
        grad_embedding_matrix = torch.zeros_like(embedding_matrix)
        for k in range(xtr[idx].shape[0]): 
            for j in range(xtr[idx].shape[1]):
                ix = xtr[idx][k,j]
                grad_embedding_matrix[ix] += grad_emb[k, j]

        cmp('logprobs', grad_logprobs, logprobs)
        cmp('probs', grad_probs, probs)
        cmp('logits_exp', grad_logits_exp, logits_exp)
        cmp('logits', grad_logits, logits)
        cmp('w2', grad_w2, W2)
        cmp('b2', grad_b2, B2)
        cmp('bn_h', grad_bn_h, bn_h)
        cmp('diff', grad_diff, diff)
        cmp('inv_sqrt_var', grad_inv_sqrt_var, inv_sqrt_var)
        cmp('sqrt_var', grad_sqrt_var, sqrt_var)
        cmp('var', grad_var, var)
        cmp('mean', grad_mean, mean)
        cmp('h', grad_h, h)
        cmp('w1', grad_w1, W1)
        cmp('b1', grad_b1, B1)
        cmp('emb_flattened', grad_emb_flattened, emb_flattened)
        cmp('emb', grad_emb, emb)
        cmp("embedding_matrix", grad_embedding_matrix, embedding_matrix)




    with torch.no_grad():
        bn_running_mean = (1-m) * bn_running_mean + m * h.mean(dim=0, keepdim=True)
        bn_running_var = (1-m) * bn_running_var + m * h.var(dim=0, keepdim=True)

        h = ((emb_flattened_dev @ W1) + B1)
        bn_h = bn_gain * (h-bn_running_mean)/torch.sqrt(bn_running_var) + bn_bias
        logits_dev = bn_h.tanh() @ W2 + B2
        devloss = torch.nn.functional.cross_entropy(logits_dev, yval[idx_dev])

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