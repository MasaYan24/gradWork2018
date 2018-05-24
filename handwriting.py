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

[N,M] = X.shape
C = np.max(t)+1
target = np.zeros([N,C]).astype(np.float32)
for i in range(N):
    target[i,t[i]] = 1.0

Ntrain = 1000
Ntest = N - Ntrain
index = np.random.permutation(range(N))
xtrain = X[index[0:Ntrain],:]
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

Nh = int(0.5*M)
NNset = Chain(conv1 = L.Convolution2D(1,32,5),
        conv2 = L.Convolution2D(32,32,5),
        l1 = L.Linear(13*13*32,Nh),
        l2 = L.Linear(Nh,C))

def model(x):
    h = F.max_pooling_2d(L.relu(NNet.conv1(x)),2)
    h = F.max_pooling_2d(F.relu(NNset.conv2(h)),2)
    h = F.relu(NNset.l1(h))
    y = NNset.l2(h)
    return y

Tall = 100
mb = 100

train_loss = []
for i in range(Tall):
    index = np.random.permutation(range(Ntrain))
    for j in range(0,Ntrain,mb):
        x = Variable(xtrain[index[j:j+mb]].reshape(mb,1,64,64))
        t = Variable(ytrain[index[j:j+mb]])
        NNset.cleargrads()
        y = model(x)
        loss = F.softmax_cross_entropy(y,t)
        loss.backward()
        optimizer.update()
    train_loss.append(loss.data)

