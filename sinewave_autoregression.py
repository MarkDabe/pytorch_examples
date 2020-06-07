# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

N = 1000
series = 2 * np.sin(0.1*np.arange(N)) + np.random.randn(N)*0.1

plt.plot(series)
plt.show()

T = 10
X = []
Y = []

for t in range(len(series)-T):
  X.append(series[t:t+T])
  Y.append(series[t+T])

X = np.array(X).reshape(-1, T)
Y = np.array(Y).reshape(-1, 1)
N = len(X)


model = nn.Linear(T, 1)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

X_train = torch.from_numpy(X[:N//2].astype(np.float32))
Y_train = torch.from_numpy(Y[:N//2].astype(np.float32))
X_test = torch.from_numpy(X[N//2:].astype(np.float32))
Y_test = torch.from_numpy(Y[N//2:].astype(np.float32))

n_epochs = 10000

training_loss = np.zeros(n_epochs)
testing_loss = np.zeros(n_epochs)

for it in range(n_epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, Y_train)
    loss.backward()
    optimizer.step()
    training_loss[it] = loss.item()

    test_outputs = model(X_test)
    loss = criterion(test_outputs, Y_test)
    testing_loss[it] = loss.item()
    print(f'Epoch: {it+1}/{n_epochs}. Training loss: {training_loss[it]:.4f}. Testing loss: {testing_loss[it]:.4f}')

plt.plot(training_loss,label='training loss')
plt.plot(testing_loss, label='testing loss')
plt.legend()
plt.show()

validation_target = Y[-N//2:]
validation_prediction = []
last_sequence = torch.from_numpy(X[-N//2].astype(np.float32))

while len(validation_prediction) < len(validation_target):
    input_ = last_sequence.view(1,-1)
    output = model(input_)
    validation_prediction.append(output[0,0].item())
    last_sequence = torch.cat((last_sequence[1:], output[0]))

plt.plot(validation_target, label='validation target')
plt.plot(validation_prediction, label='validation prediction')
plt.legend()
plt.show()