import torch
import matplotlib

counts = {}
with open('/Users/macbookair/Documents/yz50/3. Hafta/names.txt') as names_file:
    names = names_file.read().splitlines()


for i in names:
    i = f'.{i}.'
    for j,k in enumerate(i[:-1]):
        next_char = i[j+1]
        counts[k] = counts.get(k, {}) | {i[j+1]: counts.get(k, {}).get(next_char, 0)+1}
