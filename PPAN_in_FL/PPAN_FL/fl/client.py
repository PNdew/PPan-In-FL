

import flwr as fl
from config import DEVICE, LEARNING_RATE, NOISE_SCALE
from models.privacy_mechanism import PrivacyMechanism
from utils.metrics import compute_privacy_leakage, compute_distortion
import torch.optim as optim
import torch.nn.functional as F
import torch
import numpy as np
from flwr.common import NDArrays, Scalar, Parameters
from typing import Dict, Tuple


# Define PrivacyClient for Federated Learning
class PrivacyClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader):
        self.model = model.to(DEVICE)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.param_shapes = [p.shape for p in self.model.parameters()]
        self.total_params = sum(p.numel() for p in self.model.parameters())
        self.privacy_mech = PrivacyMechanism(self.total_params, noise_scale=NOISE_SCALE).to(DEVICE)
        self.optimizer = optim.Adam(
            list(self.model.parameters()) + list(self.privacy_mech.parameters()),
            lr=LEARNING_RATE
        )

    def get_parameters(self, config=None):
        """Trả về tham số của mô hình dưới dạng danh sách NumPy arrays"""
        return [p.detach().cpu().numpy() for p in self.model.parameters()]

    def set_parameters(self, parameters):
        """Cập nhật mô hình với danh sách NumPy arrays"""
        ndarray_params = [torch.tensor(p, dtype=torch.float32, device=DEVICE) for p in parameters]
        params_dict = zip(self.model.state_dict().keys(), ndarray_params)
        state_dict = {k: v for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        self.set_parameters(parameters)
        self.model.train()
        self.privacy_mech.train()

        total_loss, correct = 0, 0
        for images, labels in self.train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()

        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / len(self.train_loader.dataset)

        # Get model parameters and apply privacy mechanism
        with torch.no_grad():
            params = [p.detach().cpu().numpy() for p in self.model.parameters()]
            flat_params = np.concatenate([p.flatten() for p in params])
            flat_params_tensor = torch.tensor(flat_params, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            encrypted_params = self.privacy_mech.encrypt(flat_params_tensor)
            encrypted_np = encrypted_params.detach().cpu().numpy()

            privacy_leakage = float(compute_privacy_leakage(encrypted_np, flat_params))
            distortion = float(compute_distortion(flat_params, encrypted_np.flatten()))

        # Return tuple with correct type signature
        return params, len(self.train_loader.dataset), {
            "loss": float(avg_loss),
            "accuracy": float(accuracy),
            "privacy_leakage": privacy_leakage,
            "distortion": distortion
        }