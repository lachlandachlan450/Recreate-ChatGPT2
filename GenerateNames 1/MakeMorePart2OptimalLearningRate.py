import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math

words = open("C:/Users/Lachlan/Documents/python/zero to hero/MakeMore/names.txt", 'r').read().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = {s:(i+1) for (i, s) in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for (s, i) in stoi.items()}

#build the dataset

block_size = 3 #how many context characters do we take to predict the next one?
X, Y = [], []
for w in words:
    context = [0] * block_size #for block_size=3, context = [0, 0, 0]
    for ch in w+'.':
        ix = stoi[ch]
        X.append(context) #first time round gives [[0, 0, 0]]
        Y.append(ix)
        #print(''.join(itos[i] for i in context), '---->', itos[ix])
        context = context[1:] + [ix] #crop and append
X = torch.tensor(X) #the tensor ..., ..e, .em, emm, mma (INTEGER FORM). we can get the nth elements with X[[2, 4, 1]]
Y = torch.tensor(Y) #the tensor emma (INTEGER FORM)
C = torch.randn((27, 2))
 #This embeds X. The 2nd & 3rd bits of the shape will be (per "block"): block_size(3) * the "embedding" of C (2). So for each combo, we have 3 letters each
# mapped to 2 numbers, so we get 6 inputs per character, not including the ... at the start.
# (we are just deciding to make the embedding of C=2). More embedding means more complex if u think abt it
# print(C[X]) #maps a random pair of numbers to each thing in X. each [1, 2, 3] in X makes 3 pairs of numbers (6)

#weights
W1 = torch.randn((6, 300)) # We have 6 inputs per character as discussed above. We are using 100 neurons.
b1 = torch.rand(300) #biases

#With W1 + b1, we have a 6x100 matrix + a 100 vector. So what happens is the vector gets added to every row
#we want to perform emb @ W1+b1 to get things rolling in the MLP. But we need emb to have the shape (x, 6) instead of (x, 3, 2)

#torch.cat  conCATenates
# emb = torch.cat((emb[:, 0, :], emb[:, 1, :], emb[:, 2, :]), dim=1)
#torch.unbind generalises this
# emb = torch.cat(torch.unbind(emb, dim=1), dim=1)
#alternatively can do emb.view(32, 6) - this is so much better, it lists the elements as 1dim vector then re distributes them in order to a different shape, as long as the total is the same
#eg it could do (3, 3, 2) <-> (9, 2) <-> (18)
#emb = emb.view(5, 6) could generalise with emb = emb.view(int(math.prod([n for n in emb.shape])/6), 6) 
#doh - or emb=emb.view(emb.shape[0], 6), or emb.shape(-1, 6) as python can infer. .view is more efficient as well as cat creates a whole new tensor
#we want to concatenate the middle bit (0, 1, 2), so the dimension is 1

 # lit just tanh it all for the tanh layer

#tanh layer is done

#softmax layer
W2 = torch.randn(300, 27)
b2 = torch.randn(27)

parameters = [C, W1, b1, W2, b2]

#Example for indexing into probs. 
# print(probs.shape) #This is 5 rows by 27 columns
# #then
# print(sum(probs[0]))
# print(probs[torch.arange(5), Y]) #Gives us the probabilities for each letter in EMMA

# loss = -probs[torch.arange(5), Y].log().mean()
# print(loss)

#This can be done with a shortcut!
#print(F.cross_entropy(logits, Y))

for p in parameters:
    p.requires_grad=True

learnings = torch.linspace(-3, 0, 1000)
lrs = 10**learnings

pastlr = []
lastloss = []

#forward pass
for _ in range(5000):

    #Making minibatch
    ix = torch.randint(0, X.shape[0], (32, ))

    emb = C[X[ix]] #(32, 3, 2   )
    h = torch.tanh(emb.view(-1, 6) @ W1+b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Y[ix])
    print(loss.item())
    #backward pass
    for p in parameters:
        p.grad = None #=0 in pytorch

    loss.backward()
    #learningrate = lrs[_]
    for p in parameters:
        p.data += -0.1*p.grad
    
    pastlr.append(_)
    lastloss.append(loss.item())
#This is not very quick as we update the weights with thousands of data points each time. What is faster is picking batches at random and training on them
plt.plot(pastlr, lastloss)
plt.show()
#Generating batch
#torch.randint(0, X.shape[0], (32, )) # makes 32 integers in the range of X.shape

#Now we can run MUCH faster and decrease the loss MUCH faster. As well as this, it makes sense that we are maybe more general and less overfitting
#So it is better to approx. the gradient and take many steps than to calculate the exact gradient and take less steps

#Final loss

# emb = C[X] #(32, 3, 2   )
# h = torch.tanh(emb.view(-1, 6) @ W1+b1)
# logits = h @ W2 + b2
# loss = F.cross_entropy(logits, Y)
# print('final loss')
# print(loss.item())