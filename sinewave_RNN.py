import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

N = 1000
series = 2 * np.sin(0.1 * np.arange(N)) + np.random.randn(N) * 0.1

plt.plot(series)
plt.show()

T = 10
X = []
Y = []

for t in range(len(series) - T):
    X.append(series[t:t + T])
    Y.append(series[t + T])

X = np.array(X).reshape(-1, T, 1)
Y = np.array(Y).reshape(-1, 1)
N = len(X)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class SimpleRNN(nn.Module):
    def __init__(self, n_inputs, n_outputs, n_hidden, n_rnnlayers, ):
        super(SimpleRNN, self).__init__()
        self.D = n_inputs
        self.M = n_hidden
        self.K = n_outputs
        self.L = n_rnnlayers
        self.rnn = nn.RNN(
            input_size=self.D,
            hidden_size=self.M,
            num_layers=self.L,
            nonlinearity='relu',
            batch_first=True,
        )
        self.fc = nn.Linear(self.M, self.K)

    def forward(self, X):
        h0 = torch.zeros(self.L, X.size(0), self.M).to(device)

        output, _ = self.rnn(X, h0)

        output = self.fc(output[:, -1, :])

        return output


model = SimpleRNN(1, 1, 5, 1)
model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

X_train = torch.from_numpy(X[:N // 2].astype(np.float32))
Y_train = torch.from_numpy(Y[:N // 2].astype(np.float32))
X_test = torch.from_numpy(X[N // 2:].astype(np.float32))
Y_test = torch.from_numpy(Y[N // 2:].astype(np.float32))

X_train, Y_train = X_train.to(device), Y_train.to(device)
X_test, Y_test = X_test.to(device), Y_test.to(device)

n_epochs = 1000

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

plt.plot(training_loss, label='training loss')
plt.plot(testing_loss, label='testing loss')
plt.legend()
plt.show()

validation_target = Y[-N // 2:]
validation_prediction = []
last_sequence = torch.from_numpy(X[-N // 2].astype(np.float32)).view(T)

while len(validation_prediction) < len(validation_target):
    input_ = last_sequence.view(1, T, -1)
    output = model(input_)
    validation_prediction.append(output[0, 0].item())
    last_sequence = torch.cat((last_sequence[1:], output[0]))

plt.plot(validation_target, label='validation target')
plt.plot(validation_prediction, label='validation prediction')
plt.legend()
plt.show()
