import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plot

train_dataset = torchvision.datasets.MNIST(root='.', train=True, transform=transforms.ToTensor(), download=False)
test_dataset = torchvision.datasets.MNIST(root='.', train=False, transform=transforms.ToTensor(), download=False)

model = nn.Sequential(
    nn.Linear(784,128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

batch_size = 128
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size, shuffle=False)

n_epochs = 10
training_losses = np.zeros(n_epochs)
testing_losses = np.zeros(n_epochs)
for it in range(n_epochs):

  training_loss = []
  for inputs, targets in train_loader:
    inputs, targets = inputs.to(device), targets.to(device)
    inputs = inputs.view(-1,784)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    training_loss.append(loss.item())

  training_losses[it] = np.mean(training_loss)

  testing_loss =[]
  for inputs, targets in test_loader:
    inputs, targets = inputs.to(device), targets.to(device)
    inputs = inputs.view(-1,784)
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    testing_loss.append(loss.item())
  testing_loss[it] = np.mean(testing_loss)
  print(f'Epoch: {it+1}/{n_epochs}. Training loss: {training_losses[it]:.4f}. Testing loss {testing_loss[it]:.4f}')

# plt.plot(training_losses, label='training loss')
# plt.plot(testing_losses, label='testing loss')
# plt.legend()
# plt.show()