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
import flwr as fl
from flwr.common import Parameters, NDArrays
from flwr.server.client_proxy import ClientProxy
from flwr.common.typing import Config
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms, datasets
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, Parameters, Scalar
from flwr.common.dp import add_gaussian_noise
from flwr.common.logger import warn_deprecated_feature
from flwr.common.parameter import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import DPFedAvgFixed
from sklearn.feature_selection import mutual_info_regression
from typing import Dict, List, Optional, Tuple, Union
from config import *

class AdaptiveLDP:
    def __init__(self, epsilon=INITIAL_EPSILON):
        self.epsilon = epsilon
        self.accuracy_history = deque(maxlen=WINDOW_SIZE)
        self.noise_history = deque(maxlen=WINDOW_SIZE)
        self.leakage_history = deque(maxlen=WINDOW_SIZE)
        self.current_noise = 0.0
        self.total_noise_added = 0.0
        self.noise_samples = 0

    def add_noise(self, data: torch.Tensor) -> torch.Tensor:
        sensitivity = torch.quantile(torch.abs(data), 0.95).item()
        scale = min(sensitivity / max(self.epsilon, MIN_EPSILON), NOISE_CLIP)
        noise = torch.tensor(np.random.laplace(0, scale, data.shape), dtype=data.dtype, device=data.device)
        self.current_noise = torch.mean(torch.abs(noise)).item()
        self.noise_history.append(self.current_noise)
        self.total_noise_added += self.current_noise
        self.noise_samples += 1
        return torch.clamp(data + noise, -1.0, 1.0)

    def compute_privacy_leakage(self, original: torch.Tensor, noisy: torch.Tensor) -> float:
        original_np = original.cpu().numpy().flatten()
        noisy_np = noisy.cpu().numpy().flatten()
        if len(original_np) > 10:
            leakage = mutual_info_regression(original_np.reshape(-1, 1), noisy_np)
            return float(leakage[0])
        return 0.0

    def adjust_epsilon(self, accuracy: float, leakage: float):
        self.accuracy_history.append(accuracy)
        self.leakage_history.append(leakage)
        if len(self.accuracy_history) >= WINDOW_SIZE:
            avg_accuracy = np.mean(self.accuracy_history)
            avg_leakage = np.mean(self.leakage_history)
            accuracy_diff = TARGET_ACCURACY - avg_accuracy
            leakage_penalty = 0.1 * avg_leakage
            delta = ADJUST_RATE * (accuracy_diff - leakage_penalty)
            new_epsilon = self.epsilon * (1.0 + delta)
            self.epsilon = np.clip(new_epsilon, MIN_EPSILON, MAX_EPSILON)

class PrivacyClient(fl.client.NumPyClient):
    def __init__(self, cid: str, model: nn.Module, train_loader: DataLoader):
        self.cid = cid
        self.model = model.to(DEVICE)
        self.train_loader = train_loader
        self.optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        self.criterion = nn.CrossEntropyLoss()
        self.ldp = AdaptiveLDP()
        self.metrics_history = []

    def fit(self, parameters: NDArrays, config: Config):
        self.model.train()
        total_loss, correct, total = 0, 0, 0
        privacy_leakage_values = []

        for data, target in self.train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)

            
            self.optimizer.zero_grad()

            
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            original_grads = [p.grad.clone().detach() for p in self.model.parameters() if p.grad is not None]
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad += self.ldp.add_noise(p.grad)
            noisy_grads = [p.grad.clone().detach() for p in self.model.parameters() if p.grad is not None]
            self.optimizer.step()
            original_grads_flat = torch.cat([g.flatten() for g in original_grads]) if original_grads else torch.tensor([])
            noisy_grads_flat = torch.cat([g.flatten() for g in noisy_grads]) if noisy_grads else torch.tensor([])

            if len(original_grads_flat) > 0 and len(noisy_grads_flat) > 0:
                leakage = self.ldp.compute_privacy_leakage(original_grads_flat, noisy_grads_flat)
                privacy_leakage_values.append(leakage)

            
            total_loss += loss.item()
            correct += (output.argmax(dim=1) == target).sum().item()
            total += target.size(0)

        
        accuracy = correct / total if total > 0 else 0
        avg_loss = total_loss / len(self.train_loader) if len(self.train_loader) > 0 else 0
        avg_leakage = np.mean(privacy_leakage_values) if privacy_leakage_values else 0.0
        self.ldp.adjust_epsilon(accuracy, avg_leakage)
        metrics = {"loss": avg_loss, "accuracy": accuracy, "epsilon": self.ldp.epsilon, "leakage": avg_leakage}
        self.metrics_history.append(metrics)
        return self.get_parameters({}), total, metrics
