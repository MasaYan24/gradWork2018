#!/usr/local/bin/python3

#p.139

import numpy as np
import chainer.functions as F
import chainer.links as L
from chainer import Variable, optimizers
from chainer import Chain #from chainer import FunctionSet
import matplotlib.pyplot as plt
from sklearn import datasets

MNIST = datasets.load_digits()
X = MNIST.data.astype(np.float32)
t = MNIST.target

print(X.shape)
print(t)

#plt.imshow(X[0,:].reshape(8,8))
#plt.show()

[N,M] = X.shape #N: number of source, M: number of 2D space
C = np.max(t)+1 #number of character kind
target = np.zeros([N,C]).astype(np.float32)
for i in range(N):
    target[i,t[i]] = 1.0

Ntrain = 1000
Ntest = N - Ntrain
index = np.random.permutation(range(N))
xtrain = X[index[0:Ntrain],:]
print(xtrain.shape)
ytrain = target[index[0:Ntrain],:]
xtest = X[index[Ntrain:N],:]
yans = target[index[Ntrain:N]]

NNset = Chain(l1=L.Linear(M,20),l2=L.Linear(20,C))

def model(x):
    h = F.sigmoid(NNset.l1(x))
    y = F.sigmoid(NNset.l2(h))
    return y

x = Variable(xtrain)
t = Variable(ytrain)

optimizer = optimizers.Adam()
optimizer.setup(NNset)

Tall = 1000

train_loss = []
for i in range(Tall):
    NNset.cleargrads()
    y = model(x)
    loss = F.mean_squared_error(y,t)
    loss.backward()
    optimizer.update()
    train_loss.append(loss.data)

plt.figure(figsize=(8,6))
plt.plot(range(Tall),train_loss)
plt.title('optimizatin vol4')
plt.xlabel('step')
plt.ylabel('loss function')
plt.xlim([0,Tall])
plt.ylim([0,0.5])
plt.show()

