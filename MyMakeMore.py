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

#so its doing zip('emma', 'mma' which gives em, mm, ma) (then the final a hasnt got a pair so doesnt exist)
# b = {}
# for w in words:
#     chs = ['<S>'] + list(w) + ['<E>'] #start and end list
#     for (ch1, ch2) in zip(chs, chs[1:]):
#         bigram = (ch1, ch2)
#         b[bigram] = b.get(bigram, 0) + 1    #Counting how often ch2 follows ch1, by adding it to dictionary 'b'. If bigram don't exist, make it 0

# maximum = max([b[big] for big in b])
# print([big for big in b if b[big] == maximum])

#alternatively
#sort by the count of these elements
# print(sorted(b.items(), key=lambda kv: -kv[1])) # lambda takes in kv (key value) and returns kv[1] (-kv[1] for reversed order)
# plt.figure(figsize=(16, 16)) #makes a grid, indexed by j, i
# plt.imshow(N, cmap='Blues')
# for i in range(27):
#     for j in range(27):
#         chstr = itos[i] + itos[j]
#         plt.text(j, i, chstr, ha='center', va='bottom', color='gray')   #ha/va = horizontal/vertical alignment
#         plt.text(j, i, N[i, j].item(), ha='center', va='top', color='gray') #orientation of the i, j doesn't matter, just transposes the output 'matrix'
# #you can read this as ab - 2332 -> b follows a 2332 times
# plt.axis('off')

#normalise, for probabilities
#for example, for one row p=N[0] = N[0, :]
# p = p/p.sum(), then 
# each p is now a probability (p.sum() = 1)

#use torch multinomal: probabilities -> integers. it picks samples and returns index of it
# (example in test.py)

#Make efficient
# P = N.float()
# P /= P.sum(1, keepdim=True) #This gives P divided by the sum of each row which is really clever
#carefully search broadcasting semantics. We are dividing [27, 27] by [27, 1]. It expands the 1 out into 27, so each element can be divided by the sum of its row
#Video discusses it at ~47 minutes

#generating names
N = torch.zeros((27, 27), dtype=torch.int32)




chars = sorted(list(set(''.join(words))))
stoi = {s:(i+1) for (i, s) in enumerate(chars)}
stoi['.'] = 0

# for w in words:
#     chs = ['.'] + list(w) + ['.'] #start and end list
#     for (ch1, ch2) in zip(chs, chs[1:]):
#         ix1 = stoi[ch1]
#         ix2 = stoi[ch2]
#         N[ix1, ix2] += 1

# P = (N+1).float()
# P /= P.sum(1, keepdim=True)
        
# itos = {i:s for (s, i) in stoi.items()}

# g = torch.Generator().manual_seed(2147483647)

# for i in range(5):

#     ix = 0
#     out = []
#     while True:
#         p = P[ix]
#         ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
#         out.append(itos[ix])
#         if ix == 0:
#             break
#     print(''.join(out))

# # for evaluation
# log_likelihood = 0
# n = 0



#get probability of words, so we can gauge how good a response is



# for w in words:
#     chs = ['.'] + list(w) + ['.'] #start and end list
#     for (ch1, ch2) in zip(chs, chs[1:]):
#         ix1 = stoi[ch1]
#         ix2 = stoi[ch2]
#         N[ix1, ix2] += 1
#         prob = P[ix1, ix2]
#         logprob = torch.log(prob)
#         n+= 1
#         log_likelihood += logprob #more negative for bad, closer to 0 for better

# nll = -log_likelihood # for convention
# normalisednll = nll/n  #loss function




#Now we are going to create training set of bigrams (x, y), where x is the input, and ys is the target
xs = [] #inputs converted to integers
ys = [] #targets as integers


#This iterates through everything so can make our data
for w in words[:1]:
    chs = ['.'] + list(w) + ['.']
    for  (ch1, ch2) in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1) #adding integer input
        ys.append(ix2) #adding integer target

xs = torch.tensor(xs) #We dont want lists of integers, we are using tensors instead, easier to use
ys = torch.tensor(ys) #Be aware that torch.tensor and torch.Tensor both exist, and have small differences so stick with .tensor (datatype is inferred automatically)

#For each 5 letter word, for example, we will get FIVE training data points - The first letter starts a word, the 2nd follows the first, ..., the last follows the 4th.
#So here we have that for the first element in xs (.) we want the network to go to the first element of ys(e) from that
#And so on, so the second element of xs(e) should go to ys[1], which is (m), giving us .->e, e->m, m->m, m->a, a->.
#Remember they are stored as integers, so its more like 0->5, 5->13, 13->13, 13-> 1, 1->0

#We can't just plug our current integers into a neural net, it would make more sense to encode the integers as vectors (explained in a few lines)
#We will encode our integers as vectors, ie n -> [0, ..., 1, 0, ... ] - we have n<= k. We encode n as a vector of k zeros, but the nth element is 1
#So for our 27 digits, we would encode d as d=4 -> [0, 0, 0, 0, 4, 0, 0, 0, ...] (26 0s, 1 1). We can do this with pytorch,one_hot
#This allows the network to give more MEANING to each input - think about how inputs in a NN are vectors, each element corresponds to some characteristic of it!
#Without encoding, the NN might infer that later letters (z, y, x) are associated with larger integers and therefore are more important.
#Also consider how we use weights and biases, these would affect the input and meaning would be LOST! if we just had them as integers.

xenc = F.one_hot(xs, num_classes=27)

#try showing this with 
# plt.imshow(xenc)
# plt.show()

#We are not finished encoding! 
#It similarly makes sense for the elements of xenc NOT to be integers, but FLOATS instead, so we can apply operations to them (think abt it)#

xenc = xenc.float()

#Now its time to determine the initial weights for the neurons in the network!

#This gets numbers from normal distribution, around 0, sd=1 etc
W = torch.randn((27, 27), requires_grad=True) #So we get a random tensor of 27 rows by 27 columns, for our weights. 
#Required grad=true enables a gradient so we can back propogate later
#So we have the weights of 27 neurons, each neuron WITH 27 WEIGHTS. For the weights for 1 neuron, it would be torch.randn(27, 1)
#This also means we get a larger output from applying the weights - (5, 27) @ (27, 27) -> (5, 27), so we can also reapply

#Usually for weights and operations on inputs, we do a dot product. There's a life hack here.
#If we do matrix multiplication, we end up doing MULTIPLE DOT PRODUCTS IN PARALLEL (think about how matrix multiplication works)
#so we can now apply the weights to the inputs(xenc) with matrix multiplication. So they will be multiplied n added like dot product
xenc @ W
#The output is (5, 27). This is telling us for each input (each row, 5 rows, 5 letters), what the firing rate of each neuron is (27 columns for each row)
# (27 neurons for each input). The firing rate is the probability that each neuron is fired given the input (so once trained, 
#The firing rate for m with input m should be higher than average)

#Now that we have our neurons with their values, we ideally want them to be some sort of probability field
#So we cant have negative numbers, and integers (as a count) don't work as they are too big. What works is a log count
#Then to get the count (for the probability of each letter), we can exponentiate the log count
#
#We can pretend that xenc @ w are all log counts. Then we can exponentiate. Note all previous negative values are now <1, and vice versa
#We call log counts logits
logits = xenc @ W
#Then by the above logic
counts = logits.exp()
#Then the probabilities are normalised counts (div total in row etc, we have already done this)
probs = counts/counts.sum(1, keepdims=True)
#Now we have a probability field. It sums to 1 etc.

#Now as said before, if we take some row of prob (eg prob[0]), we get 27 probabilities. These will rep. probabilities of each letter following
#The input letter

#Now we can make a loss function so that we can start to backpropogate and optimise the model

#BTW: the combination of counts=logits.exp() and prob=counts/... is called a 'softmax'. Softmax activation function is v commonly used
# It takes in any real inputs and outputs a probability field

#Everything we have just done is differentiable, so backpropogation should be a BREEZE!

#Now we can train the model with words (eg for emma, we have input .e and we train to get a higher probability for m coming next)
#The loss function is the log likelihood, for more loss we have a higher negative log likelihood. So lets minimise it
#log likelihood is literally just the log of the probability. So less p = lower log likelihood, higher negative log likelihood
#we are looking at the p that we want. we want to minimise negative log likelihood for that p
#The total loss per word is the average negative log likelihood, for each letter

#Its gonna be just like neural network. We want a single number to represent loss and back propogate, but instead of mse use avg neg log lik. for loss

#Example for training with .-e-m-m-m-a-.

probs[0, 5], probs[1, 13], probs[2, 13], probs[3, 1], probs[4, 0]
#So here we have probabilities for each step. We have the target value of some high probabilities.

#Our loss looks at the probability of each wanted letter. We are trying to make it as high as possible. We arent explicitly comparing it
#To a target value, we just want it as high as possible. So we look at each letter (torch.arange(n)), and the NEXT (correct) letter (ys)
#And make it as big as possible probability

#We can list these instead of a tuple, as a vector in pytorch
probs[torch.arange(5), ys] # ys is integer version of strings

loss = -probs[torch.arange(5), ys].log().mean() #Now we have loss which is the average negative log likelihood

print(loss)

#Now its time for the backwards pass

W.grad = None #same as 0, just a bit more efficient
loss.backward() #This backpropogates, and gives gradient to each element in W and all the intermediates. so clever! it knows how to differentiate

#Now we update according to gradient
W.data += -0.1 * W.grad

#recalculate loss - notice its slightly lower
# logits = xenc @ W
# counts = logits.exp()
# probs = counts/counts.sum(1, keepdims=True)
# loss = -probs[torch.arange(5), ys].log().mean()
# print(loss)



#Now we can repeat the training process, say 10 times, and on the whole data set. 
#Im doing this in MyMakeMoreClean.py, as its too messy with comments here.
#We can afford larger learning rate (10) on this example as the loss isnt decreasing as quickly as i would like

#We will also do regularisation loss
#If all elements in W are 0, the probabilities will become uniform.
#We want to bias the weights towards being 0 and away from being large so we add on W**2.mean() to the loss, so bigger weights mean more loss
#This is to get a 'smoother' probability curve. P much this means the model will be more simplified.
#Therefore the theory is that it will be better at 'general' predictions instead of specific 'spiky probability' predictions, so should
#perform better

#Hypothetically if we did not train the model with regularisation loss, we would end up with a probability for "zm", for example, of 0,
#as "zm" doesnt appear in the training data set. Therefore if we used the model for training with new names and it came across a name with "zm",
#it would give an error of infinity, as -log(0) = \infty. This would ruin the training via backpropogation, as the weights would change by infinity.
#so the model would become demolished. best to avoid in this case.
#This would be using the model for inference - making the AI make descisions based on new data.

#Think like a human mind, we dont analyse to see what letter should come next, we use the vibe and gist of things to continue the word