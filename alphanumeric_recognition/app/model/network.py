# import torch
# from torchvision import transforms
# import torch.nn as nn
# import torch.nn.functional as F

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(28, 64, (5,5), padding = 2)
#         self.conv1_bn = nn.BatchNorm2d(64)
#         self.conv2 = nn.Conv2d(64, 128, 2, padding = 2)
#         self.fc1 = nn.Linear(2048, 1024)
#         self.dropout = nn.Dropout(0.3)
#         self.fc2 = nn.Linear(1024, 512)
#         self.bn = nn.BatchNorm1d(1)
#         self.fc3 = nn.Linear(512, 128)
#         self.fc4 = nn.Linear(128,47)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = self.conv1_bn(x)
#         x = F.relu(self.conv2(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = x.view(-1, 2048)
#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.fc2(x)
#         x = x.view(-1, 1, 512)
#         x = self.bn(x)
#         x = x.view(-1, 512)
#         x = self.fc3(x)
#         x = self.fc4(x)

#         return x


# /////////////
import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, (5, 5), padding=2)  # Changed input channels from 28 to 1
        self.conv1_bn = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 2, padding=2)
        self.fc1 = nn.Linear(2048, 1024)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(1024, 512)
        self.bn = nn.BatchNorm1d(1)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 47)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.conv1_bn(x)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 2048)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = x.view(-1, 1, 512)
        x = self.bn(x)
        x = x.view(-1, 512)
        x = self.fc3(x)
        x = self.fc4(x)

        return x

# //////////

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         # Adjust input channels to 1 for grayscale images
#         self.conv1 = nn.Conv2d(1, 64, (5, 5), padding=2)
#         self.conv1_bn = nn.BatchNorm2d(64)
#         self.conv2 = nn.Conv2d(64, 128, (5, 5), padding=2)
#         self.fc1 = nn.Linear(128 * 7 * 7, 1024)  # Adjust input dimensions based on output size
#         self.dropout = nn.Dropout(0.3)
#         self.fc2 = nn.Linear(1024, 512)
#         self.bn = nn.BatchNorm1d(512)  # BatchNorm for 512 features
#         self.fc3 = nn.Linear(512, 128)
#         self.fc4 = nn.Linear(128, 47)  # Assuming 47 classes for output

#     def forward(self, x):
#         # Convolution and pooling layers
#         x = F.relu(self.conv1(x))
#         x = self.conv1_bn(x)
#         x = F.max_pool2d(x, 2, 2)
        
#         x = F.relu(self.conv2(x))
#         x = F.max_pool2d(x, 2, 2)
        
#         # Flatten dynamically
#         x = x.view(x.size(0), -1)
        
#         # Fully connected layers
#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)
        
#         x = F.relu(self.fc2(x))
#         x = self.bn(x)
        
#         x = F.relu(self.fc3(x))
#         x = self.fc4(x)  # Final layer for classification
        
#         return x
