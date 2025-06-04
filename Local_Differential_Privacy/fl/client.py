from typing import Dict, List, Tuple
import tensorflow as tf
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from collections import deque
from config import *
from metric.metrics import compute_privacy_leakage

class AdaptiveLDP:
    def __init__(self, epsilon_init=1.0, alpha=0.5, beta=0.2, target_accuracy=0.85, window_size=5, noise_clip=1.0):
        self.epsilon = epsilon_init
        self.epsilon_init = epsilon_init
        self.alpha = alpha
        self.beta = beta
        self.target_accuracy = target_accuracy
        self.window_size = window_size
        self.noise_clip = noise_clip
        self.accuracy_history = deque(maxlen=window_size)

    def _compute_sensitivity(self, image: torch.Tensor) -> float:
        return torch.quantile(torch.abs(image), 0.95).item()

    def _laplace_noise(self, shape, scale, device, dtype):
        noise = np.random.laplace(0, scale, size=shape)
        return torch.tensor(noise, dtype=dtype, device=device)

    def add_noise_to_data(self, image: torch.Tensor) -> torch.Tensor:
        sensitivity = self._compute_sensitivity(image)
        scale = min(sensitivity / max(self.epsilon, 1e-6), self.noise_clip)
        noise = self._laplace_noise(image.shape, scale, image.device, image.dtype)
        return torch.clamp(image + noise, -1.0, 1.0)

    def add_noise_to_gradient(self, gradient: torch.Tensor) -> torch.Tensor:
        sensitivity = self._compute_sensitivity(gradient)
        scale = min(sensitivity / max(self.epsilon, 1e-6), self.noise_clip)
        noise = self._laplace_noise(gradient.shape, scale, gradient.device, gradient.dtype)
        return gradient + noise

    def update_epsilon_dynamic(self, current_accuracy: float):
        """Theo công thức: ε_t = ε_0 * (1 + α * acc / acc_target)"""
        self.accuracy_history.append(current_accuracy)
        if len(self.accuracy_history) >= self.window_size:
            avg_acc = np.mean(self.accuracy_history)
            self.epsilon = self.epsilon_init * (1 + self.alpha * (avg_acc / self.target_accuracy))

    def update_epsilon_on_evaluation(self, current_accuracy: float):
        """Khi acc thấp hơn kỳ vọng: ε_{t+1} = ε_t + β * (label - acc) / label"""
        if current_accuracy < self.target_accuracy:
            delta = self.beta * ((self.target_accuracy - current_accuracy) / self.target_accuracy)
            self.epsilon += delta


class PrivacyClient(fl.client.NumPyClient):
    def __init__(self, cid: str, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader):
        self.cid = cid
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
        self.criterion = nn.CrossEntropyLoss()
        self.ldp = AdaptiveLDP()
        self.metrics_history = []
        self.val_loader = val_loader

    def get_parameters(self, config=None):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        for k, v in zip(state_dict.keys(), parameters):
            state_dict[k] = torch.tensor(v)
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        total_loss, correct, total = 0, 0, 0
        privacy_leakage_values = []

        for image, label in self.train_loader:
            image, label = image.to(DEVICE), label.to(DEVICE)

            # --- THÊM NHIỄU VÀO DỮ LIỆU ---
            image = self.ldp.add_noise_to_data(image)

            self.optimizer.zero_grad()
            output = self.model(image)
            loss = self.criterion(output, label)
            loss.backward()

            # Backup original gradients
            original_grads = [p.grad.clone().detach() for p in self.model.parameters() if p.grad is not None]

            # Add LDP noise to gradients
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad = self.ldp.add_noise_to_gradient(p.grad)

            # Backup noisy gradients for leakage estimation
            noisy_grads = [p.grad.clone().detach() for p in self.model.parameters() if p.grad is not None]
            self.optimizer.step()

            # Flatten gradients to compute leakage
            if original_grads and noisy_grads:
                original_grads_flat = torch.cat([g.flatten() for g in original_grads])
                noisy_grads_flat = torch.cat([g.flatten() for g in noisy_grads])
                leakage = self.compute_privacy_leakage(original_grads_flat, noisy_grads_flat)
                privacy_leakage_values.append(leakage)

            total_loss += loss.item()
            correct += (output.argmax(dim=1) == label).sum().item()
            total += label.size(0)

        accuracy = correct / total if total > 0 else 0
        print(f"Client {self.cid} - Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}")
        avg_loss = total_loss / len(self.train_loader) if len(self.train_loader) > 0 else 0
        avg_leakage = np.mean(privacy_leakage_values) if privacy_leakage_values else 0.0

        self.ldp.update_epsilon_dynamic(accuracy)
        self.ldp.update_epsilon_on_evaluation(accuracy)

        metrics = {
            "loss": avg_loss,
            "accuracy": accuracy,
            "epsilon": self.ldp.epsilon,
            "leakage": avg_leakage
        }
        self.metrics_history.append(metrics)
        return self.get_parameters({}), total, metrics
