import torch.nn as nn
from config import *
import torch.models as models
import torch.optim as optim
import torch.nn.functional as F
class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)  # No softmax here as it's included in CrossEntropyLoss
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
def get_model():
    return ResNet18()