import torch.nn as nn

# Dense net
class Perceptron(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.layer_1 = nn.Linear(784, 500)
        self.layer_2 = nn.Linear(500, 100)
        self.layer_3 = nn.Linear(100, num_classes)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, inp):
        x = self.dropout(inp)
        x = self.layer_1(x)
        x = self.activation(x)
        x = self.layer_2(x)
        x = self.activation(x)
        x = self.layer_3(x)
        return x


# Convolutional net
class ConvNet(nn.Module):
    def __init__(self, num_classes):
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