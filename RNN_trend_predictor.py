import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


class RNN_(nn.Module):
    def __init__(self, n_type, n_inputs, n_outputs, n_hidden, n_layers, nonlinearity='relu'):
        if n_type != 'RNN' and n_type != 'GRU' and n_type != 'LSTM':
            raise ValueError("Invalid value for n_type")
        super(RNN_, self).__init__()
        self.TYPE = n_type
        self.D = n_inputs
        self.M = n_hidden
        self.K = n_outputs
        self.L = n_layers
        self.A = nonlinearity

        if self.TYPE == 'RNN':
            self.rnn = nn.RNN(
                input_size=self.D,
                hidden_size=self.M,
                num_layers=self.L,
                nonlinearity=self.A,
                batch_first=True
            )

        elif self.TYPE == 'GRU':
            self.rnn = nn.GRU(
                input_size=self.D,
                hidden_size=self.M,
                num_layers=self.L,
                batch_first=True
            )
        elif self.TYPE == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=self.D,
                hidden_size=self.M,
                num_layers=self.L,
                batch_first=True
            )
        self.fc = nn.Linear(self.M, self.K)

    def forward(self, X):
        h0 = torch.zeros(self.L, X.size(0), self.M).to(device)

        if self.TYPE == 'RNN' or self.TYPE == 'GRU':
            output, _ = self.rnn(X, h0)

        else:
            c0 = torch.zeros(self.L, X.size(0), self.M).to(device)
            output, _ = self.rnn(X, (h0, c0))

        output = self.fc(output[:, -1, :])

        return output

def full_gd(model,
            criterion,
            optimizer,
            X_train,
            y_train,
            X_test,
            y_test,
            epochs=200):
    train_losses = np.zeros(epochs)
    test_losses = np.zeros(epochs)

    for it in range(epochs):
        optimizer.zero_grad()

        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        loss.backward()
        optimizer.step()

        train_losses[it] = loss.item()

        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)
        test_losses[it] = test_loss.item()

        if (it + 1) % 5 == 0:
            print(f'Epoch {it + 1}/{epochs}, Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}')

    return train_losses, test_losses


df = pd.read_csv('USD_CAD_D.csv')

df['PrevClose'] = df['Price'].shift(1)
df['Return'] = (df['Price'] - df['PrevClose']) / df['PrevClose']
input_data = targets = df[['Return']].values[1:]

T = 10
D = input_data.shape[1]
N = len(input_data) - T

Ntrain = len(input_data) * 2 // 3
scaler = StandardScaler()
scaler.fit(input_data[:Ntrain + T - 1])
input_data = scaler.transform(input_data)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = RNN_('RNN', 1, 1, 50, 2)

model.to(device)

X_train = np.zeros((Ntrain, T, D))
Y_train = np.zeros((Ntrain, 1))

for t in range(Ntrain):
    X_train[t, :, :] = input_data[t:t + T]
    Y_train[t] = (targets[t + T] > 0)

# Setup X_test and Y_test
X_test = np.zeros((N - Ntrain, T, D))
Y_test = np.zeros((N - Ntrain, 1))

for u in range(N - Ntrain):
    # u counts from 0...(N - Ntrain)
    # t counts from Ntrain...N
    t = u + Ntrain
    X_test[u, :, :] = input_data[t:t + T]
    Y_test[u] = (targets[t + T] > 0)

X_train = torch.from_numpy(X_train.astype(np.float32))
y_train = torch.from_numpy(Y_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_test = torch.from_numpy(Y_test.astype(np.float32))
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.5)

train_losses, test_losses = full_gd(model,
                                    criterion,
                                    optimizer,
                                    X_train,
                                    y_train,
                                    X_test,
                                    y_test,
                                    epochs=300)

plt.plot(train_losses, label='train loss')
plt.plot(test_losses, label='test loss')
plt.legend()
plt.show()

# model.load_state_dict(torch.load('RNN_trend_predictor.pt'))

with torch.no_grad():
    p_train = model(X_train)
    p_train = (p_train.cpu().numpy() > 0)
    train_acc = np.mean(y_train.cpu().numpy() == p_train)

    p_test = model(X_test)
    p_test = (p_test.cpu().numpy() > 0)
    test_acc = np.mean(y_test.cpu().numpy() == p_test)

print(f"Train acc: {train_acc:.4f}, Test acc: {test_acc:.4f}")

# torch.save(model.state_dict(), 'RNN_trend_predictor.pt')