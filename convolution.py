import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA_PATH = '/home/den/data/MNIST/'
BATCH_SIZE = 1000
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
class ConvNet(nn.Module):
    def __init__(
        self,
        num_classes,
    ):
        super().__init__()
        self.conv_1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv_2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv_3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv_4 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv_5 = nn.Conv2d(64, 128, 3, padding=1)
        self.pooling = nn.AvgPool2d(2)

        self.dense_1 = nn.Linear(128, 128)
        self.dense_2 = nn.Linear(128, num_classes)

        self.dropout = nn.Dropout(0.5)

        self.activation = nn.ReLU()
        

    def forward(self, inp):
        '''
        inp: tensor of shape [batch, 1, 28, 28]
        '''
        # Convolutional layers
        x = self.conv_1(inp) # [batch, 32, 28, 28]
        x = self.activation(x)
        x = self.conv_2(x) # [batch, 32, 28, 28]
        x = self.activation(x)
        x = self.pooling(x) # [batch, 32, 14, 14]
        x = self.conv_3(x) # [batch, 64, 14, 14]
        x = self.activation(x)
        x = self.conv_4(x) # [batch, 64, 14, 14]
        x = self.activation(x)
        x = self.pooling(x) # [batch, 64, 7, 7] 
        x = self.conv_5(x) # [batch, 128, 7, 7]

        # Global average pooling
        x = x.mean(dim=[2, 3]) # [batch, 128]

        # Dense layers
        x = self.dropout(x)
        x = self.dense_1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.dense_2(x)

        return x

model = ConvNet(num_classes=10)
model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(100):
    print('Epoch', epoch)
    model.train()
    for i in range(len(X_train) // BATCH_SIZE):
        # Get batch of data
        batch_X = X_train [i * BATCH_SIZE: i * BATCH_SIZE+BATCH_SIZE] # [B, 784]
        batch_y = y_train [i * BATCH_SIZE: i * BATCH_SIZE+BATCH_SIZE] # [B]
        batch_X = batch_X.reshape (BATCH_SIZE, 1,28, 28)
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
    outputs = []
    model.eval()
    for i in range(len(X_val) // BATCH_SIZE):
        batch_X = X_val [i * BATCH_SIZE: i * BATCH_SIZE+BATCH_SIZE] # [B, 784]
        model_input = torch.tensor(batch_X, dtype=torch.float32, device=DEVICE)
        model_input = model_input.reshape(BATCH_SIZE, 1, 28, 28)
        out = model(model_input) # [21000, 10]
        out = out.detach().cpu().numpy()
        out = np.argmax(out, axis=1)
        outputs.append(out)
    outputs = np.concatenate(outputs)
    # Compute accuracy
    acc = (outputs == y_val).mean()
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


