import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms, datasets
import torch.nn.functional as F
import torchvision.models as models


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)  # Sau 2 lớp pooling, kích thước ảnh còn 4x4
        self.fc2 = nn.Linear(128, 10)  # 10 lớp đầu ra cho MNIST

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # Không dùng softmax vì CrossEntropyLoss đã có sẵn softmax
        return x
    
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        # Load pretrained ResNet18
        self.model = models.resnet18(pretrained=True)
        
        # Modify first conv layer to accept 1 channel input (MNIST is grayscale)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Modify final fully connected layer for 10 classes (MNIST)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 10)

    def forward(self, x):
        return self.model(x)