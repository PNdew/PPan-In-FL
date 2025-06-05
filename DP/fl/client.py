import flwr as fl
from config import *
from metric.metrics import compute_privacy_leakage, compute_distortion
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple
from flwr.common import NDArrays


def add_gaussian_noise_to_parameters(parameters, epsilon, delta, sensitivity):
    sigma = (sensitivity * np.sqrt(2 * np.log(1.25 / delta))) / epsilon
    noisy_parameters = [
        param + np.random.normal(loc=0.0, scale=sigma, size=param.shape)
        for param in parameters
    ]
    return noisy_parameters


class PrivacyClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, epsilon=1.0, delta=1e-5, sensitivity=1.0):
        self.model = model.to(DEVICE)
        self.train_loader = train_loader
        self.param_shapes = [p.shape for p in self.model.parameters()]
        self.total_params = sum(p.numel() for p in self.model.parameters())
        self.optimizer = optim.SGD(self.model.parameters(), lr=LEARNING_RATE)
        self.criterion = nn.CrossEntropyLoss()
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity

    def get_parameters(self, config=None):
        return [p.detach().cpu().numpy() for p in self.model.parameters()]

    def set_parameters(self, parameters):
        ndarray_params = [torch.tensor(p, dtype=torch.float32, device=DEVICE) for p in parameters]
        params_dict = zip(self.model.state_dict().keys(), ndarray_params)
        state_dict = {k: v for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: NDArrays, config: Dict[str, str] = {}) -> Tuple[NDArrays, int, Dict[str, float]]:
        self.set_parameters(parameters)
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for images, labels in self.train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        accuracy = float(total_correct / total_samples)
        avg_loss = float(total_loss / total_samples)

        # Lấy tham số mô hình và thêm nhiễu DP
        updated_parameters = self.get_parameters()
        noisy_parameters = add_gaussian_noise_to_parameters(
            updated_parameters, self.epsilon, self.delta, self.sensitivity
        )

        # Tính privacy leakage và distortion
        flat_params = np.concatenate([p.flatten() for p in updated_parameters])
        encrypted_np = np.concatenate([p.flatten() for p in noisy_parameters])

        privacy_leakage = float(compute_privacy_leakage(encrypted_np, flat_params))
        distortion = float(compute_distortion(flat_params, encrypted_np))

        return noisy_parameters, total_samples, {
            "loss": avg_loss,
            "accuracy": accuracy,
            "privacy_leakage": privacy_leakage,
            "distortion": distortion,
        }
