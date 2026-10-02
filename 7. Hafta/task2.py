import torch
from torch.nn import functional as F
torch.manual_seed(1234)

x = torch.randn(1,5,2)
xbow_forloop = torch.zeros_like(x)
for batch in range(x.shape[0]):
    for time in range(x.shape[1]):
        prev = x[batch, :time+1]
        xbow_forloop[batch,time] = torch.mean(prev, 0)


a = torch.tril(torch.ones(5,5))
a = a / a.sum(1, keepdim=True)
b = a @ x

tril = torch.tril(torch.ones(5,5))
wei = torch.zeros_like(tril)
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, 1)
xbow_softmax = wei @ x

print(torch.allclose(b, xbow_forloop))
print(torch.allclose(b, xbow_softmax))
        
        
