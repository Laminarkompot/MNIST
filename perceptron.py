import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA_PATH = '/home/den/data/MNIST/'
BATCH_SIZE = 2000
DEVICE = 'cuda'

# Read Data
data = pd.read_csv(DATA_PATH + 'train.csv')

X_train = data.iloc[:21000, 1:].values
X_val = data.iloc[21000:, 1:].values

y_train = data.label[:21000].values
y_val = data.label[21000:].values

print('X_train.shape', X_train.shape)
print('y_train.shape', y_train.shape)
print('X_val.shape', X_val.shape)
print('y_val.shape', y_val.shape)
print('y_val.dtype', y_val.dtype)

# Plot
# for i in range(10):
#     img = X_train[i].reshape(28, 28)
#     plt.title(y_train[i])
#     plt.imshow(img, cmap='gray')
#     plt.show()

# Normalize data
mean = X_train.mean()
std = X_train.std()
X_train = (X_train - mean) / std
X_val = (X_val - mean) / std

# Create model
class Perceptron(nn.Module):
    def __init__(
        self,
        num_classes,
    ):
        super().__init__()
        self.layer_1 = nn.Linear(784, 500)
        self.layer_2 = nn.Linear(500, 100)
        self.layer_3 = nn.Linear(100, num_classes)
        self.activation = nn.ReLU()

    def forward(self, inp):
        '''
        inp: tensor of shape [batch, 784]
        '''
        x = self.layer_1(inp) # [batch, 128]
        x = self.activation(x) # [batch, 128]
        x = self.layer_2(x) # [batch, 100]
        x = self.activation(x) # [batch, 100]
        x = self.layer_3(x) # [batch, num_classes]
        return x

model = Perceptron(num_classes=10)
model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(100):
    print('Epoch', epoch)
    for i in range(len(X_train) // BATCH_SIZE):
        # Get batch of data
        batch_X = X_train [i * BATCH_SIZE: i * BATCH_SIZE+BATCH_SIZE] # [B, 784]
        batch_y = y_train [i * BATCH_SIZE: i * BATCH_SIZE+BATCH_SIZE] # [B]
        # Convert to torch tensors
        batch_X = torch.tensor(batch_X, dtype=torch.float32, device=DEVICE)
        batch_y = torch.tensor(batch_y, dtype=torch.long, device=DEVICE)
        # Pass to the neural network
        out = model(batch_X) # [B, 10]
        # Compute loss
        loss = loss_fn(out, batch_y)
        # Gradient descent step
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    # Predict on validation set
    model_input = torch.tensor(X_val, dtype=torch.float32, device=DEVICE)
    out = model(model_input) # [21000, 10]
    out = out.detach().cpu().numpy()
    out = np.argmax(out, axis=1)
    # Compute accuracy
    acc = (out == y_val).mean()
    print('ACCURACY    ', acc)

# Plot validation
# for i in range(100):
#     model_input = torch.tensor(X_val[i: i + 1], dtype=torch.float32, device=DEVICE)
#     out = model(model_input) # [1, 10]
#     out = out.detach().cpu().numpy()
#     out = np.argmax(out)

#     img = X_val[i].reshape(28, 28)
#     plt.title(f'Model: {out}, GT: {y_val[i]}')
#     plt.imshow(img, cmap='gray')
#     plt.show()


