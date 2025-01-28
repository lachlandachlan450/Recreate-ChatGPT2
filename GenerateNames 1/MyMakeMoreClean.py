#My makeMore
import os
import sys
import time
import math
import argparse
from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

words = open("C:/Users/Lachlan/Documents/python/zero to hero/MakeMore/names.txt", 'r').read().splitlines()

N = torch.zeros((27, 27), dtype=torch.int32)
chars = sorted(list(set(''.join(words))))
stoi = {s:(i+1) for (i, s) in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for (s, i) in stoi.items()}

xs = [] 
ys = [] 

for w in words:
    chs = ['.'] + list(w) + ['.']
    for  (ch1, ch2) in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1) 
        ys.append(ix2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement() #num elements in xs
print('number of examples:', num)

#Initialise the network
W = torch.randn((27, 27), requires_grad=True) 

#gradient descent
for k in range(50):
    #Forward pass
    xenc = F.one_hot(xs, num_classes=27).float()
    logits = xenc @ W
    counts = logits.exp()
    probs = counts/counts.sum(1, keepdims=True)
    loss = -probs[torch.arange(num), ys].log().mean() + 0.01*(W**2).mean()
    print(loss.item())
    W.grad = None
    loss.backward()
    W.data += -10 * W.grad

#sampling from the model

for i in range(15):
    out=[]
    ix = 0
    while True:

        xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
        logits = xenc @ W
        counts = logits.exp()
        p = counts / counts.sum(1, keepdims=True)

        ix = torch.multinomial(p, num_samples=1, replacement=True).item()
        out.append(itos[ix])
        if ix==0:
            break
    print(''.join(out))
