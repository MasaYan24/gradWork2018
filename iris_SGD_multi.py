#!/usr/local/bin/python3

import numpy as np
import chainer.functions as F
import chainer.links as L
from chainer import Variable, optimizers
from chainer import FunctionSet
import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data.astype(np.float32)
t = iris.target

print(X.shape)

target = np.zeros([150,3]).astype(np.float32)
for i in range(150):
    target[i,t[i]] = 1.0

index = np.random.permutation(range(150))
xtrain = X[index[0:75],:]
ytrain = target[index[0:75],:]
xtest = X[index[75:150],:]
yans = target[index[75:150]]

NNset = FunctionSet(l1 = F.Linear(4,6), l2 = F.Linear(6,3))

def model(x):
    h = F.sigmoid(NNset.l1(x))
    y = F.sigmoid(NNset.l2(x))
    return y

x = Variable(xtrain)
t = Variable(ytrain)

optimizer = optimizers.SGD()
optimizer.setup(NNset)

Tall = 1000

train_loss = []
for i in range(Tall):
    NNset.zerograds()
    y = model(x)
    loss = F.mean_squared_error(y,t)
    loss.backward()
    optimizer.update()
    train_loss.append(loss.data)

plt.figure(figsize=(8,6))
plt.plot(range(Tall),train_loss)
plt.title('optimization vol2')
plt.xlabel('step')
plt.ylabel('loss function')
plt.xlim([0,Tall])
plt.ylim([0,0.5])
plt.show()

xtest = Variable(xtest)
ytest = model(xtest)
ok = 0
for i in range(75):
    cls1 = np.argmax(ytest.data[i,:])
    cls2 = np.argmax(yans[i,:])
    if cls1 == cls2:
        ok += 1
print(ok,"/",75,"=",(ok*1.0)/75)

