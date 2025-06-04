import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms, datasets
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from typing import Dict, List, Tuple

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

class ResNet50(nn.Module):
    def __init__(self, num_channel=3, num_classes=10):
        super(ResNet50, self).__init__()
        # Khởi tạo model không sử dụng trọng số pre-trained
        self.resnet = resnet50(weights=None)  # tương đương pretrained=False

        # Thay đổi layer đầu vào để phù hợp số channels
        self.resnet.conv1 = nn.Conv2d(num_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Thay đổi fully connected layer để phù hợp số classes
        self.resnet.fc = nn.Linear(2048, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)
