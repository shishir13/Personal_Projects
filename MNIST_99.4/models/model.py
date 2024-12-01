import torch.nn as nn
import torch.nn.functional as F

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        # First conv block - input channels: 1, output channels: 8
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        
        # Second conv block - input channels: 8, output channels: 16
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        
        # Third conv block - input channels: 16, output channels: 20
        self.conv3 = nn.Conv2d(16, 20, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(20)
        
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        
        # Reduced FC layers
        self.fc1 = nn.Linear(20 * 3 * 3, 64)  # Smaller FC layer
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        # First conv block
        x = F.relu(self.bn1(self.conv1(x)))  # 28x28 -> 28x28
        x = F.max_pool2d(x, 2)  # 28x28 -> 14x14
        
        # Second conv block
        x = F.relu(self.bn2(self.conv2(x)))  # 14x14 -> 14x14
        x = F.max_pool2d(x, 2)  # 14x14 -> 7x7
        x = self.dropout1(x)
        
        # Third conv block
        x = F.relu(self.bn3(self.conv3(x)))  # 7x7 -> 7x7
        x = F.max_pool2d(x, 2)  # 7x7 -> 3x3
        x = self.dropout2(x)
        
        # Flatten and FC layers
        x = x.view(x.size(0), -1)  # Flatten: 20 * 3 * 3 = 180
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
