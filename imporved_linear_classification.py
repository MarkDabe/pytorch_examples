from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt

data = load_breast_cancer()

X, Y = data.data, data.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

N, D = X_train.shape

model = nn.Sequential(
    nn.Linear(D,1),
)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters())

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
Y_train = torch.from_numpy((Y_train.astype(np.float32).reshape(-1,1)))
Y_test = torch.from_numpy((Y_test.astype(np.float32).reshape(-1,1)))

n_epochs = 1000

train_losses = np.zeros(n_epochs)
test_losses = np.zeros(n_epochs)

for it in range(n_epochs):
    model.zero_grad()

    outputs = model(X_train)

    loss = criterion(outputs, Y_train)

    loss.backward()
    optimizer.step()

    test_outputs = model(X_test)

    test_loss = criterion(test_outputs, Y_test)

    train_losses[it] = loss.item()

    test_losses[it] = test_loss.item()

    if (it+1) % 50 == 0:
        print(f'Epoch {it+1}/{n_epochs}, Training Loss {loss.item(): .4f}, Test Loss{test_loss.item(): .4f}')

# plt.plot(train_losses, 'b', label='train loss')
# plt.plot(test_losses, 'r', label='test loss')
# plt.legend()
# plt.show()

with torch.no_grad():
    training_outputs = model(X_train)
    training_results = (training_outputs.numpy() > 0)
    training_accuracy = np.mean([training_results == Y_train.numpy()])

    testing_outputs = model(X_test)
    testing_results = np.round(testing_outputs.numpy() > 0)
    testing_accuracy = np.mean([testing_results == Y_test.numpy()])

    print(f'Training Accuracy {training_accuracy: .4f}, Test Accuracy {testing_accuracy: .4f}')

torch.save(model.state_dict(), 'breastcancer_improved.pt')