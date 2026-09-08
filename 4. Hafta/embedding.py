import torch

with open('/Users/macbookair/Documents/yz50/4. Hafta/names.txt') as names_file:
    names = names_file.read().splitlines()

chars = sorted(list(set(''.join(names))))
chars.append('.')
def encode(char):
    return chars.index(char)

def decode(index):
    return chars[index]
def encode_str(string):
    return [encode(i) for i in string]

x,y = list(), list()
block_size = 4

for i in names[:5]:
    i = f"{'.'*block_size}{i}."
    i = encode_str(i)
    for j,k in enumerate(i[:-block_size]):
        ctx = i[j:j+block_size]
        x.append(ctx)
        y.append(i[j+block_size])

x = torch.tensor(x)
y = torch.tensor(y)
embedding_matrix = torch.randn(len(chars), 2)
emb = embedding_matrix[x].shape
print(emb)