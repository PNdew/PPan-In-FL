

import flwr as fl
from config import *
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
        self.optimizer = optim.SGD(
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

            # Forward qua mô hình chính
            task_outputs = self.model(images)
            task_loss = F.cross_entropy(task_outputs, labels)

            # Lấy trọng số mô hình → flatten → mã hóa + giải mã
            flat_weights = torch.cat([p.view(-1) for p in self.model.parameters()], dim=0).unsqueeze(0).to(DEVICE)
            encrypted, decoded = self.privacy_mech(flat_weights)

            # Privacy loss: MSE giữa bản giải mã và bản gốc (khó phục hồi là tốt)
            privacy_loss = -F.mse_loss(decoded, flat_weights.detach())  # Dấu "-" vì ta muốn tăng lỗi tái tạo

            # Distortion loss: giữ cho bản mã hóa gần bản gốc
            distortion_loss = F.mse_loss(encrypted, flat_weights.detach())

            
            total_batch_loss = task_loss + PRIVACY_WEIGHT * privacy_loss #+ lambda_distortion * distortion_loss
            total_batch_loss.backward()
            self.optimizer.step()

            total_loss += total_batch_loss.item()
            correct += (task_outputs.argmax(dim=1) == labels).sum().item()

        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / len(self.train_loader.dataset)

        # Mã hóa tham số mô hình để gửi về server
        with torch.no_grad():
            params = [p.detach().cpu().numpy() for p in self.model.parameters()]
            flat_params = np.concatenate([p.flatten() for p in params])
            flat_tensor = torch.tensor(flat_params, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            encrypted_params = self.privacy_mech.encrypt(flat_tensor)
            encrypted_np = encrypted_params.detach().cpu().numpy()

            privacy_leakage = float(compute_privacy_leakage(encrypted_np, flat_params))
            distortion = float(compute_distortion(flat_params, encrypted_np.flatten()))

        return params, len(self.train_loader.dataset), {
            "loss": float(avg_loss),
            "accuracy": float(accuracy),
            "privacy_leakage": privacy_leakage,
            "distortion": distortion
        }
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        self.set_parameters(parameters)
        self.model.eval()

        total_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = self.model(images)
                loss = F.cross_entropy(outputs, labels)

                total_loss += loss.item()
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / len(self.test_loader) if len(self.test_loader) > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        # --- Mã hóa tham số mô hình để tính privacy leakage và distortion ---
        with torch.no_grad():
            params = [p.detach().cpu().numpy() for p in self.model.parameters()]
            flat_params = np.concatenate([p.flatten() for p in params])
            flat_tensor = torch.tensor(flat_params, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            encrypted_params = self.privacy_mech.encrypt(flat_tensor)
            encrypted_np = encrypted_params.detach().cpu().numpy()

            privacy_leakage = compute_privacy_leakage(encrypted_np, flat_params)
            distortion = compute_distortion(flat_params, encrypted_np)

        metrics = {
            "test_loss": float(avg_loss),
            "test_accuracy": float(accuracy),
            "test_privacy_leakage": float(privacy_leakage),
            "test_distortion": float(distortion)
        }

        return float(avg_loss), total, metrics

