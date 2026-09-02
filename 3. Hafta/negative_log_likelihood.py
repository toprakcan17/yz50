import torch
from torch.nn.functional import one_hot
import matplotlib.pyplot as plt

counts = torch.zeros((27,27), dtype=torch.int16)
with open('/Users/macbookair/Documents/yz50/3. Hafta/names.txt') as names_file:
    names = names_file.read().splitlines()

chars = sorted(list(set(''.join(names))))
chars.insert(26, '.')

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
print(counts.shape)
normalised = counts.float() / counts.sum(1, keepdim=True)
char_count = 0
negative_log_likelihood = float()

for i in names:
    i = f'.{i}.'

    for j,k in enumerate(i[:-1]):
        next_char = i[j+1]
        prob = normalised[encode_char(k),encode_char(next_char)]
        negative_log_likelihood-=prob.log()
        char_count+=1

average_nll = negative_log_likelihood/char_count
print(negative_log_likelihood)
