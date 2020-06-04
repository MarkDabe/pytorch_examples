from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import numpy as np

data = load_breast_cancer()

X, Y = data.data, data.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


N, D = X_train.shape

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
Y_train = torch.from_numpy((Y_train.astype(np.float32).reshape(-1,1)))
Y_test = torch.from_numpy((Y_test.astype(np.float32).reshape(-1,1)))


model = nn.Sequential(
    nn.Linear(D,1),
    nn.Sigmoid()
)

model.load_state_dict(torch.load('breastcancer.pt'))

with torch.no_grad():
    training_probs = model(X_train)
    training_results = np.round(training_probs.numpy())
    training_accuracy = np.mean([training_results == Y_train.numpy()])

    testing_probs = model(X_test)
    testing_results = np.round(testing_probs.numpy())
    testing_accuracy = np.mean([testing_results == Y_test.numpy()])

    print(f'Training Accuracy {training_accuracy: .4f}, Test Accuracy {testing_accuracy: .4f}')
