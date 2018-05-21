#!/usr/local/bin/python3

import numpy as np
import chainer.functions as F
import chainer.links as L
from chainer import Variable, optimizers

model = L.Linear(1,1)
print(model.W.data)
print(model.b.data)
x = Variable(np.array([[i] for i in range(10)],dtype=np.float32))
#x = Variable(np.array([[i] for i in range(10)],dtype=np.dtype(float).type))
y = model(x)
print(y.data)
t = Variable(np.array([[i*2] for i in range(10)],dtype=np.float32))
#t = Variable(np.array([[i*2] for i in range(10)],dtype=np.dtype(float).type))
optimizer = optimizers.SGD()
optimizer.setup(model)
time = 1000
for i in range(0,time):
#    optimizer.zero_grads()
#    optimizer.zerograd()
#    optimizer.reallocate_cleared_grads()
#    optimizer.cleargrads()
    model.cleargrads() # model.zerograds()
    y = model(x)
    loss = F.mean_squared_error(y,t)
    loss.backward()
    optimizer.update()

xtest = Variable(np.array([[i+0.5] for i in range(10)],dtype=np.float32))
#xtest = Variable(np.array([[i+0.5] for i in range(10)],dtype=np.dtype(float).type))
ytest = model(xtest)
print(ytest.data)
print(model.W.data)
