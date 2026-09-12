import torch
from torch.nn.functional import one_hot
import matplotlib.pyplot as plt

counts = torch.zeros((27,27), dtype=torch.int16)
with open('/Users/toprak/Documents/yz50/3. Hafta/names.txt') as names_file:
    names = names_file.read().splitlines()

chars = sorted(list(set(''.join(names))))
chars.append('.')

def encode_char(char):
    return chars.index(char)

def decode_char(index):
    return chars[index]

for i in names:
    i = f'.{i}.'
    for j,k in enumerate(i[:-1]):
        next_char = i[j+1]
        counts[encode_char(k)][encode_char(next_char)] += 1

counts+=1
normalised = counts.float() / counts.sum(1, keepdim=True)
char_count = 0

xs = []
ys = []

for i in names:
    i = f'.{i}.'
    for j,k in enumerate(i[:-1]):
        next_char = i[j+1]
        xs.append(encode_char(k))
        ys.append(encode_char(next_char))
        char_count+=1

xs = torch.tensor(xs)
ys = torch.tensor(ys)

W = torch.randn(27,27, requires_grad=True)

learning_step = 10
step_count = 100
losses = []
for i in range(step_count):
    x_one_hot = one_hot(xs, num_classes=27).float()
    act = x_one_hot @ W
    fake_counts = act.exp()
    normalised_counts = fake_counts/fake_counts.sum(1, keepdim=True)
    loss = -normalised_counts[torch.arange(len(ys)),ys].log().mean()
    W.grad = None
    loss.backward()
    W.data -= W.grad * learning_step
    losses.append(loss.data)
    print(f'Step: {i+1} - Loss: {loss.data:.2f}')
plt.plot(losses)
plt.waitforbuttonpress()
