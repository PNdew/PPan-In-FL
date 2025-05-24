from typing import Dict, List, Tuple
import tensorflow as tf
import flwr as fl
import numpy as np
from flwr.common import Metrics
from flwr.simulation.ray_transport.utils import enable_tf_gpu_growth
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from flwr.server.strategy import DPFedAvgFixed
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import logging
from typing import Optional, Union

from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, Parameters, Scalar
from flwr.common.dp import add_gaussian_noise
from flwr.common.logger import warn_deprecated_feature
from flwr.common.parameter import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import DPFedAvgFixed
from sklearn.feature_selection import mutual_info_regression
from typing import Dict, List, Optional
import random
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion
import threading
from utils.save_utils import write_to_file
from config import *    
from models.mnist_model import get_model


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import flwr as fl
from typing import Dict, List, Tuple
from flwr.common import NDArrays
from models.mnist_model import MNISTModel  # Đảm bảo bạn đã định nghĩa MNISTModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid: str, trainloader: DataLoader, valloader: DataLoader):
        self.cid = cid
        self.model = MNISTModel().to(DEVICE)
        self.trainloader = trainloader
        self.valloader = valloader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=LEARNING_RATE)

    def get_parameters(self, config: Dict[str, str] = {}) -> NDArrays:
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters: NDArrays):
        state_dict = self.model.state_dict()
        new_state_dict = {k: torch.tensor(v, device=DEVICE) for k, v in zip(state_dict.keys(), parameters)}
        self.model.load_state_dict(new_state_dict, strict=True)

    def fit(self, parameters: NDArrays, config: Dict[str, str] = {}) -> Tuple[NDArrays, int, Dict[str, float]]:
        self.set_parameters(parameters)
        self.model.train()

        total_samples = 0
        total_correct = 0

        for images, labels in self.trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()

        accuracy = float(total_correct / total_samples)
        print(f"Client {self.cid}: Sent accuracy = {accuracy}")

        return self.get_parameters(), total_samples, {"accuracy": accuracy}

    def evaluate(self, parameters: NDArrays, config: Dict[str, str] = {}) -> Tuple[float, int, Dict[str, float]]:
        self.set_parameters(parameters)
        self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for images, labels in self.valloader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)

        accuracy = float(total_correct / total_samples)
        avg_loss = float(total_loss / total_samples)

        return avg_loss, total_samples, {"accuracy": accuracy}
