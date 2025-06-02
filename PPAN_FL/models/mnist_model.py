import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import  resnet18, ResNet18_Weights  

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
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


class MNISTResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(MNISTResNet, self).__init__()
        # Load pretrained ResNet18
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        # Modify first conv layer to handle single channel input
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Modify final fc layer for MNIST classes
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

def get_model():
    """Create and initialize the ResNet18 model for MNIST"""
    model = MNISTModel()
    return model
