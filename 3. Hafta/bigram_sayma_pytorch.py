import torch
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

plt.imshow(counts)
plt.pause(20)
