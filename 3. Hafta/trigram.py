import torch
from torch.nn.functional import one_hot
import matplotlib.pyplot as plt

LEARNING_STEP_COUNT = 500
LEARNING_STEP_SIZE = 100
DEV_STEP_SIZE = .125

with open('/Users/toprak/Documents/yz50/3. Hafta/names.txt') as names_file:
    names = names_file.read().splitlines()
    names = [name for i in names for name in i.split()]


def split_dataset(n):
    split = ([],[],[])
    for i,j in enumerate(n):
        if i%10==0: split[1].append(j)
        elif i%10==1: split[2].append(j)
        else: split[0].append(j)
    return split

dataset = split_dataset(names)
chars = sorted(list(set(''.join(names))))
chars.append('.')


counts = torch.zeros((len(chars)**2, len(chars)), dtype=torch.int16)

def encode_char(char):
    return chars.index(char)
def encode_two_chars(str):
    return chars.index(str[0])*len(chars)+chars.index(str[1])
def decode_char(index):
    return chars[index]
def load_data(words):
    xs=[]
    ys=[]
    char_count = 0
    for i in words:
        i = f'..{i}.' 
        for j,k in enumerate(i[:-2]):
            next_char = i[j+2]
            prev_char = i[j+1]
            xs.append(encode_two_chars(k+prev_char))
            ys.append(encode_char(next_char))
            char_count+=1
    return (torch.tensor(xs), torch.tensor(ys), char_count)

xtrain, ytrain, n_chars_train = load_data(dataset[0])
xdev, ydev, _ = load_data(dataset[1])
xtest, ytest, _ = load_data(dataset[2])
W = torch.randn(len(chars)**2,len(chars), requires_grad=True)

def forward_pass(x,w,smoothing=0):
    x_one_hot = one_hot(x, num_classes=len(chars)**2).float()
    act = x_one_hot @ w
    fake_counts = act.exp()
    fake_counts=fake_counts + smoothing
    normalised_counts = fake_counts/fake_counts.sum(1, keepdim=True)
    return normalised_counts

def compute_loss(counts, y):
    return -counts[torch.arange(len(y)), y].log().mean()

def train_model(x,y,weights, step_size, step_count):
    losses = []
    for i in range(step_count):
        counts = forward_pass(x,weights)
        loss = compute_loss(counts,y)
        weights.grad = None
        loss.backward()
        weights.data -= weights.grad * step_size
        losses.append(loss.data)
        print(f'Step: {i+1} - Loss: {loss.data:.2f}')
    return weights 

def tune_smoothing(x,y,weights,step_size):
    smoothing = 0
    prev_loss = -1
    while True:
        loss = 0
        loss = compute_loss(forward_pass(x, weights,smoothing=smoothing),y).data
        if prev_loss < loss and prev_loss!=-1: return smoothing-step_size
        prev_loss = loss
        smoothing+=step_size

def generate_word(weights,smoothing):
    prev_chars = '..'
    word = ''
    while True:
        probs = forward_pass(torch.tensor([encode_two_chars(prev_chars)]), weights, smoothing=smoothing)
        new_char = decode_char(torch.multinomial(probs,1,replacement=False))
        if new_char == '.': return word
        word+=new_char
        prev_chars = prev_chars[1] + new_char
trained_model = train_model(xtrain,ytrain,W,LEARNING_STEP_SIZE,LEARNING_STEP_COUNT)
smoothing = tune_smoothing(xdev,ydev,trained_model,DEV_STEP_SIZE)
loss = compute_loss(forward_pass(xtest,trained_model,smoothing=smoothing),ytest).data
print(f'Loss: {loss}')
for i in range(55):
    print(generate_word(trained_model,smoothing))
