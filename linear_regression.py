import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

N = 20

X = np.random.random(N)*10 - 5
Y = 0.5 * X - 1 + np.random.randn(N)

model = nn.Linear(1, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

X = X.reshape(N, 1)
Y = Y.reshape(N, 1)

inputs = torch.from_numpy(X.astype(np.float32))
targets = torch.from_numpy(Y.astype(np.float32))

n_epochs = 60
loses = []
for it in range(n_epochs):
    optimizer.zero_grad()

    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loses.append(loss.item())

    loss.backward()
    optimizer.step()
    print(f'Epoch {it+1}/{n_epochs}, Loss {loss.item(): .4f}')

# plt.plot(loses)
# plt.show()

# w = model.weight.data.numpy()
# b = model.bias.data.numpy()

predicted = model(inputs).detach().numpy()
plt.scatter(X, Y, label='Original Data')
plt.plot(X, predicted, label='Fitted Line')
plt.legend()
plt.show()