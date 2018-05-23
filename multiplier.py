#!/usr/local/bin/python3

import numpy as np
import chainer.functions as F
import chainer.links as L
from chainer import Variable, optimizers

model = L.Linear(1,1)
print(model.W.data)
print(model.b.data)
x = Variable(np.array([[i] for i in range(10)],dtype=np.float32))
y = model(x)
print(y.data)
t = Variable(np.array([[i*2] for i in range(10)],dtype=np.float32))
optimizer = optimizers.SGD()
optimizer.setup(model)
time = 1
for i in range(0,time):
    optimizer.zero_grads()
    y = model(x)
    loss = F.mean_squared_error(y,t)
    loss.backward()
    optimizer.update()
    print("type:",type(loss))

xtest = Variable(np.array([[i+0.5] for i in range(10)],dtype=np.float32))
ytest = model(xtest)
print(ytest.data)
print(model.W.data)

