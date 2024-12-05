import torch
import torch.nn as nn
import torch.nn.functional as F

class FastMNIST(nn.Module):
    """
    A lightweight CNN for MNIST digit classification.
    Achieves >99.4% accuracy with less than 20k parameters.
    
    Architecture:
    - 3 Convolutional blocks (1->8->16->32 channels)
    - BatchNorm after each conv
    - MaxPool after each block
    - 2 FC layers (32 neurons in hidden layer)
    - Dropout (0.4) for regularization
    
    Total Parameters: 15,578
    """
    def __init__(self):
        super(FastMNIST, self).__init__()
        # Simple and effective channel progression
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        
        # Efficient FC layers
        self.fc1 = nn.Linear(32 * 3 * 3, 32)
        self.fc2 = nn.Linear(32, 10)
        
        self.dropout = nn.Dropout(0.4)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        
        x = x.view(-1, 32 * 3 * 3)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)