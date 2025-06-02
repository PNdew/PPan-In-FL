import torch 
import torch.nn as nn
from config import *
from torchvision.models import resnet50, ResNet50_Weights

#class MNISTModel(nn.Module):
#     def __init__(self):
#         super(MNISTModel, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=0)
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=0)
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.flatten = nn.Flatten()
#         self.fc1 = nn.Linear(64 * 4 * 4, 128)
#         self.fc2 = nn.Linear(128, 10)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         x = self.relu(self.conv1(x))
#         x = self.pool1(x)
#         x = self.relu(self.conv2(x))
#         x = self.pool2(x)
#         x = self.flatten(x)
#         x = self.relu(self.fc1(x))
#         x = self.fc2(x)  # No softmax here as it's included in CrossEntropyLoss
#         return x


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

def get_model():
    return ResNet50(num_channel=3, num_classes=10)
