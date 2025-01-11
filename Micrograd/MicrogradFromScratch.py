import torch
import random
import math

class Value():

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other) # Lets us run Valueobject + integer
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        #No brackets - we are setting a function, not calling it
        out._backward = _backward
        return out
    
    def __radd__(self, other):
        return self + other
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward(): # we do += instead of = to account for the fact that each variable could be used more than once (see video 1)
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def __rmul__(self, other): #For other * self instead of self * other
        return self * other
    
    def __truediv__(self, other): # self / other
        return self * (other**(-1))
    
    def __rtruediv__(self, other):
        return other* (self**(-1))
    
    def __neg__(self):
        return self *-1
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return -self + other
    
    def tanh(self):
        x = self.data; t=(math.exp(2*x)-1)/(math.exp(2*x)+1)
        out = Value(t, (self, ), 'tanh')
        def _backward():
            self.grad += (1- t**2) * out.grad
        out._backward = _backward
        return out
    
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), 'Only supporting int/float powers for now'
        out = Value(self.data**other, (self, ), f'**{other}')
        def _backward():
            self.grad += other*self.data**(other-1) * out.grad
        out._backward = _backward
        return out

    
    def backward(self):
            #Automate it

        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        #Now its ordered in first to last p much, so we can call backward through the list backwards ( depth first )

        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

#Like in backpropogation.py, pytorch does the same thing, but with tensors (multiple values, apart from that the same). Tensors are n-dimensional arrays of scalars
#so instead of x1 = Value(2), we have
#x1 = torch.Tensor([2.0])
#can also have arrays, try x1.shape, or .dtype
#we use double to get everything in the datatype float64, which is nicer (64 bits, so more decimal points)

#By default pytorch assumes they do not need gradients, so we add in that they do

# x1 = torch.Tensor([2.0]).double()                ; x1.requires_grad = True
# x2 = torch.Tensor([0.0]).double()                ; x2.requires_grad = True
# w1 = torch.Tensor([-3.0]).double()               ; w1.requires_grad = True
# w2 = torch.Tensor([1.0]).double()                ; w2.requires_grad = True
# b = torch.Tensor([6.8813735870195432]).double()  ; b.requires_grad = True
# n = x1*w1 + x2*w2 + b
# o = torch.tanh(n)

# print(o.data.item())
# o.backward()

# print('---')
# print('x2', x2.grad.item())
# print('w2', w2.grad.item())
# print('x1', x1.grad.item())
# print('w1', w1.grad.item())

#Note o.item() = o.data.item() - they both return o.data. If we take o.grad, we get a tensor, so we use o.grad.item() to get the value of the gradient

class Neuron:

    def __init__(self, nin): #nin = number of inputs
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x): #say we have n = Neuron(2), then we can do n(x) = n.__call__()
        #activation function
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)# + self.b (its in the sum)# zip() pairs each w with each x, sum gets the total. This gives the dot product of self.w and x
        #Now we make it non linear by using tanh
        out = act.tanh()
        return out
    
    def parameters(self):
        return self.w + [self.b]

class Layer:
    #A layer is simply multiple neurons - look at diagram in video

    def __init__(self, nin, nout):
        #nin (number of inputs) is dimensionality of each neuron, nout is number of neurons 
        #(same as number of outputs - each neuron converts n dimensional input (n inputs) to 1 output)
        self.neurons = [Neuron(nin) for _ in range(nout)]
    
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()] #wow efficient

class MLP:
    #A Multi-Layer Perceptron is an input, some number of layers, and an output. The input feeds into 1st layer -> 2nd layer -> ... -> output
    #nin is number of inputs, nouts is a list of the size of each layer (in neurons)
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
        #This makes a layer with input sz[i] (outputs of prev), and outputs sz[i+1] (inputs of next)

    #After each layer, for the next layer the number of inputs = num outputs of prev layer

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

# x = [2.0, 3.0, -1.0]
# n = MLP(3, [4, 4, 1])
# print(n(x).data)

#NOW LETS DO SOME AI!

#example:

n = MLP(3, [4, 4, 1])

xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]

ys = [1.0, -1.0, -1.0, 1.0] #desired targets

def waffle():
    pass
    # #make a loss function - mse

    # loss = sum((yout-ygt)**2 for ygt, yout in zip(ys, ypred)) # zip format is the same as enumerate format
    # print(loss)
    # #Now for magic:
    # loss.backward() #Now the weights etc have gradients!!! you know what this means don't you boy
    # #So I think we could do self.data = self.data-self.grad for each weight

    # for p in n.parameters():
    #     p.data += -0.01*p.grad  # Yep I was right! just by a smaller amount (1 would be massive lol)

    # #test loss again?
    # ypred = [n(x) for x in xs]
    # loss = sum((yout-ygt)**2 for ygt, yout in zip(ys, ypred)) # zip format is the same as enumerate format
    # print(loss) # In my test, its gone from 2.698 to 1.791!! So cool!!

#training loop:
for k in range(50):
    ypred = [n(x) for x in xs]
    loss = sum((yout-ygt)**2 for ygt, yout in zip(ys, ypred))
    for p in n.parameters():
        p.grad = 0 #Remeber .grad is +=, so every iteration it needs to be set to 0. Do it here, just before reset (other wise ypred is affected)
    loss.backward()
    #update
    for p in n.parameters():
        p.data += -0.05*p.grad
    print(k, loss.data)
print(ypred) #Perfect
