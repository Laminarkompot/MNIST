import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Define constants
DATA_PATH = '/home/den/data/MNIST/'
SAVE_PATH = '/home/den/code/edu/python/'
NUM_EPOCHS = 100
BATCH_SIZE_TRAIN = 7000
BATCH_SIZE_TEST = 7000
DEVICE = 'cuda'

# Read data
data = pd.read_csv(DATA_PATH + 'train.csv')
X_train = data.iloc[:21000,1:].values
X_test = data.iloc[21000:,1:].values
Y_train = data.label[:21000].values
Y_test = data.label[21000:].values

# Normalize data
mean = X_train.mean()
std = X_train.std()
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# Create model
class Perceptron(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.layer_1 = nn.Linear(784, 500)
        self.layer_2 = nn.Linear(500, 100)
        self.layer_3 = nn.Linear(100, num_classes)
        self.activation = nn.ReLU()
    
    def forward(self, inp):
        x = self.layer_1(inp)
        x = self.activation(x)
        x = self.layer_2(x)
        x = self.activation(x)
        x = self.layer_3(x)
        return x

# Create an instance of model
model = Perceptron(num_classes=10)

# Pass model to CUDA
model.to(DEVICE)

# Define optimizer
optimizer = torch.optim.Adam(model.parameters())

# Define loss function
loss_fn = nn.CrossEntropyLoss()

# Define arrays for plotting
epochs = range(1, NUM_EPOCHS + 1)
train_acc_array = []
test_acc_array = []
train_loss_array = []
test_loss_array = []

# Learning and testing
for epoch in range(NUM_EPOCHS):
    print('Epoch', epoch, end=" ")

    # Define arrays for intermediate values of loss and accuracy
    epoch_train_loss_array = []
    epoch_train_acc_array = []
    epoch_test_loss_array = []
    epoch_test_acc_array = []
    
    # Training epoch
    for batch_train in range(len(X_train) // BATCH_SIZE_TRAIN):

        # Get the batches X and Y
        batch_X = X_train[batch_train * BATCH_SIZE_TRAIN: (batch_train + 1) * BATCH_SIZE_TRAIN]
        batch_Y = Y_train[batch_train * BATCH_SIZE_TRAIN: (batch_train + 1) * BATCH_SIZE_TRAIN]
        batch_X = torch.tensor(batch_X, dtype=torch.float32, device=DEVICE)
        batch_Y = torch.tensor(batch_Y, dtype=torch.long, device=DEVICE)

        # Make predictions from batch
        out = model(batch_X)

        # Compute mean train loss from batch
        batch_train_loss = loss_fn(out, batch_Y)
        batch_train_loss.backward()

        # Update weights
        optimizer.step()

        # Reset oprimizer
        optimizer.zero_grad()

        # Fill the array of loss values from epoch with the mean loss value from batch
        epoch_train_loss_array.append(float(batch_train_loss))

        # Compute mean train accuracy from batch
        out = out.detach().cpu().numpy()
        out = np.argmax(out, axis=1)
        batch_Y = batch_Y.detach().cpu().numpy()
        train_acc_from_batch = (out == batch_Y).mean()

        # Fill the array of accuracy values from epoch with the mean accuracy value from batch
        epoch_train_acc_array.append(train_acc_from_batch)
    
    # Test epoch
    for batch_test in range(len(X_test) // BATCH_SIZE_TEST):
        
        # Get the batches X and Y
        batch_X = X_test[batch_test * BATCH_SIZE_TEST: (batch_test + 1) * BATCH_SIZE_TEST]
        batch_Y = Y_test[batch_test * BATCH_SIZE_TEST: (batch_test + 1) * BATCH_SIZE_TEST]
        batch_X = torch.tensor(batch_X, dtype=torch.float32, device=DEVICE)
        batch_Y = torch.tensor(batch_Y, dtype=torch.long, device=DEVICE)

        # Make predictions from batch
        out = model(batch_X)

        # Compute mean test loss from batch
        batch_test_loss = loss_fn(out, batch_Y)

        # Fill the array of loss values from epoch with the mean loss value from batch
        epoch_test_loss_array.append(float(batch_test_loss))

        # Compute mean test accuracy from batch
        out = out.detach().cpu().numpy()
        out = np.argmax(out, axis=1)
        batch_Y = batch_Y.detach().cpu().numpy()
        test_acc_from_batch = (out == batch_Y).mean()
        
        # Fill the array of accuracy values from epoch with the mean accuracy value from batch
        epoch_test_acc_array.append(test_acc_from_batch)
        
    # Compute mean train accuracy from epoch
    train_acc = np.array(epoch_train_acc_array).mean()*100

    # Compute mean test accuracy from epoch
    test_acc = np.array(epoch_test_acc_array).mean()*100
    
    # Compute mean train loss from epoch
    train_loss = np.array(epoch_train_loss_array).mean()

    # Compute mean test loss from epoch
    test_loss = np.array(epoch_test_loss_array).mean()

    # Print results
    print('train_acc', round(train_acc, 3), '%', 'test_acc', round(test_acc, 3),'%','train_loss',round(train_loss,3),'test_loss', round(test_loss,3), '\n')

    # Fill arrays for plotting
    train_acc_array.append(train_acc)
    test_acc_array.append(test_acc)
    train_loss_array.append(float(train_loss))
    test_loss_array.append(float(test_loss))

# Save model
torch.save(model.state_dict, SAVE_PATH + 'Perceptron')

# Define figure for plotting
fig = plt.figure(figsize=(20, 10))

# Loss vs number of epochs
plt.subplot(1,2,1,)
plt.plot(epochs, train_loss_array, label='train_loss')
plt.plot(epochs, test_loss_array, label='test_loss')
plt.axis([0, epochs[-1], 0, train_loss_array[0]*1.1])
plt.legend()
plt.xlabel('Количество эпох')
plt.ylabel('loss')

# Acc vs number of epochs
x_hor = [0, epochs[-1]]
y_100 = [100, 100]
plt.subplot(1,2,2)
plt.plot(epochs, train_acc_array, label='train_acc')
plt.plot(epochs, test_acc_array, label='test_acc')
plt.plot(x_hor, y_100, linestyle='--', color='grey', label='100%')
plt.axis([0, epochs[-1], 0, 110])
plt.legend()
plt.xlabel('Количество эпох')
plt.ylabel('Точность, %')

# Save figure
fig.savefig('plot.png')

# Show figure
plt.show()
