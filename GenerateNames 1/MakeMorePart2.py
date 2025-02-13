import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import random

words = open("C:/Users/Lachlan/Documents/python/zero to hero/MakeMore/names.txt", 'r').read().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = {s:(i+1) for (i, s) in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for (s, i) in stoi.items()}

def build_dataset(words):
    block_size = 3
    X, Y = [], []
    for w in words:
        context = [0] * block_size 
        for ch in w+'.':
            ix = stoi[ch]
            X.append(context) 
            Y.append(ix)
            context = context[1:] + [ix]
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y

random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))                #Shuffles words, then picks 80%, 10%, 10% for training, validation/development, test respectively
Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])



C = torch.randn((27, 10))
W1 = torch.randn((30, 300))
b1 = torch.rand(300)

W2 = torch.randn(300, 27)
b2 = torch.randn(27)

parameters = [C, W1, b1, W2, b2]

for p in parameters:
    p.requires_grad=True

for _ in range(50000):
    ix = torch.randint(0, Xtr.shape[0], (32, ))

    emb = C[Xtr[ix]]
    h = torch.tanh(emb.view(-1, 30) @ W1+b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Ytr[ix])
    for p in parameters:
        p.grad = None

    loss.backward()
    lr = 0.1 if _ < 40000 else 0.01:
    for p in parameters:
        p.data += -lr*p.grad

#In late stages of training, repeat with lower training rate. This is called learning rate decay

#Dont want to train on all the data / with a v large model to avoid overfitting. We split data into training, validation and the test split
#80%, 10%, 10% roughly usually

emb = C[Xdev]
h = torch.tanh(emb.view(-1, 30) @ W1+b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Ydev)
print('Dev loss', loss.item()) #Remember we use test data set to eval model EXTREMELY sparingly. Otherwise model fits to it.
#If loss about = dev loss, we are not overfitting. As they are equal, we are underfitting, so should scale up model (neurons:100->300)

emb = C[Xte]
h = torch.tanh(emb.view(-1, 30) @ W1+b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Yte)
print('Test loss', loss.item())

newword = torch.tensor([0, 0, 0])
for letters in range(10):
    embed = C[newword]
    h = torch.tanh(embed.view(-1, 30) @ W1+b1 )
    logits = h @ W2 + b2
    counts = logits.exp()
    probs = counts/counts.sum(1, keepdim=True)
    print(sum(probs[0]))
print(probs[torch.arange(5), Y])