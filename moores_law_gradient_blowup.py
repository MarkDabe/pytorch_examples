import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('moores.csv', header=None, delimiter=' ').values
X = data[:, 0].reshape(-1, 1)
Y = data[:, 1].reshape(-1, 1)
Y = np.log(Y)

# plt.scatter(X,Y)
# plt.show()

# m_X = X.mean()
# sd_X = X.std()
# m_Y = Y.mean()
# sd_Y = Y.std()
#
# X = (X-m_X)/sd_X
# Y = (Y-m_Y)/sd_Y

inputs = torch.from_numpy(X.astype(np.float32))
targets = torch.from_numpy((Y.astype(np.float32)))

model = nn.Linear(1, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0000001, momentum=0.7)

n_epochs = 100
losses = []

for it in range(n_epochs):
    model.zero_grad()

    outputs = model(inputs)

    loss = criterion(outputs, targets)
    losses.append(loss.item())

    loss.backward()
    optimizer.step()

    print(f'Epoch {it+1}/{n_epochs}, Loss {loss.item(): .4f}')

plt.plot(losses)
plt.show()
