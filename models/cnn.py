import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomConvNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomConvNet, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.leakyrelu2 = nn.LeakyReLU(negative_slope=0.01)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Third convolutional layer
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu3 = nn.ReLU()
        
        # Fourth convolutional layer
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.leakyrelu4 = nn.LeakyReLU(negative_slope=0.01)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Fifth convolutional layer
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu5 = nn.ReLU()
        
        # Sixth convolutional layer
        self.conv6 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.leakyrelu6 = nn.LeakyReLU(negative_slope=0.01)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Fully connected layers
        self.fc1 = nn.Linear(512 * 2 * 2, 4096) # Assuming input size is 32x32x3, after 4 maxpooling layers
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(4096, 4096)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self, x):
        # First layer
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        
        # Second layer
        x = self.conv2(x)
        x = self.leakyrelu2(x)
        x = self.maxpool2(x)
        
        # Third layer
        x = self.conv3(x)
        x = self.relu3(x)
        
        # Fourth layer
        x = self.conv4(x)
        x = self.leakyrelu4(x)
        x = self.maxpool3(x)
        
        # Fifth layer
        x = self.conv5(x)
        x = self.relu5(x)
        
        # Sixth layer
        x = self.conv6(x)
        x = self.leakyrelu6(x)
        x = self.maxpool4(x)
        
        # Flatten the tensor output from the convolutional layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.dropout1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = F.relu(x)
        x = self.fc3(x)
        
        # Apply Softmax to the output layer for multi-class classification
        x = F.softmax(x, dim=1)
        
        return x
