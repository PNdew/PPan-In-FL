import torch.nn as nn
import torch.optim as optim
from config import *
import numpy as np
import flwr as fl
from models import Net
from privacy.sgp import SignificantGradientProtection
from data import DataLoader
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any
from flwr.common import (
    FitRes, Parameters, Config, Context, Scalar,
    ndarrays_to_parameters, parameters_to_ndarrays
)
from privacy.PrivacyMetrics import PrivacyMetrics
from config import *
class DPFedASGPClient(fl.client.NumPyClient):
    """Client implementing DP-FedASGP"""
    def __init__(self, cid: str, model: Net, trainloader: DataLoader):
        self.cid = cid
        self.model = model.to(DEVICE)
        self.trainloader = trainloader
        privacy_metrics = PrivacyMetrics()
        # Initialize SGP
        total_params = sum(p.numel() for p in model.parameters())
        self.sgp = SignificantGradientProtection(total_params, NUM_ROUNDS)

        # Optimizer with gradient clipping
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=1e-5
        )
        self.criterion = nn.CrossEntropyLoss()
        self.current_round = 0

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Get model parameters"""
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set model parameters"""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
      """Train model with privacy protection"""
      try:
          self.set_parameters(parameters)
          self.current_round += 1

          epoch_metrics = defaultdict(float)
          for _ in range(LOCAL_EPOCHS):  # Multiple local epochs
              self.model.train()
              total_loss = 0
              correct = 0
              total = 0
              protected_grads = 0
              total_grads = 0

              for data, target in self.trainloader:
                  data, target = data.to(DEVICE), target.to(DEVICE)
                  self.optimizer.zero_grad()

                  output = self.model(data)
                  loss = self.criterion(output, target)
                  loss.backward()
                  original_grads = torch.cat([p.grad.view(-1) for p in self.model.parameters() if p.grad is not None])
                  torch.nn.utils.clip_grad_norm_(self.model.parameters(), CLIP_BOUND)
                  for param in self.model.parameters():
                      if param.grad is not None:
                          param.grad.data = self.sgp.protect_gradients(param.grad.data, self.current_round, alpha=ALPHA, beta=BETA)
                          protected_grads += self.sgp.protected_count
                          total_grads += self.sgp.total_count
                  self.privacy_metrics.update(original_grads, protected_grads, self.current_round)

                  # Clip gradients
                  self.optimizer.step()

                  total_loss += loss.item() * len(data)
                  _, predicted = output.max(1)
                  total += target.size(0)
                  correct += predicted.eq(target).sum().item()

              # Accumulate metrics
              epoch_metrics["loss"] += total_loss / total
              epoch_metrics["accuracy"] += correct / total
              epoch_metrics["protected_grads"] += protected_grads
              epoch_metrics["total_grads"] += total_grads

          # Average metrics over epochs
          return self.get_parameters({}), total, {
              k: v / LOCAL_EPOCHS for k, v in epoch_metrics.items()
          }

      except Exception as e:
          logger.error(f"Error in fit for client {self.cid}: {e}")
          return parameters, 0, {}

    def evaluate_test(self) -> Dict[str, float]:
        """Evaluate model on test set"""
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0

        try:
            with torch.no_grad():
                for data, target in self.testloader:
                    data, target = data.to(DEVICE), target.to(DEVICE)
                    output = self.model(data)
                    test_loss += self.criterion(output, target).item()
                    pred = output.argmax(dim=1, keepdim=True)
                    total += target.size(0)
                    correct += pred.eq(target.view_as(pred)).sum().item()

            accuracy = correct / total if total > 0 else 0
            avg_loss = test_loss / len(self.testloader)

            # Adjust privacy budget based on test accuracy
            self.sgp.adjust_privacy_budget(accuracy)

            return {
                "test_loss": avg_loss,
                "test_accuracy": accuracy
            }

        except Exception as e:
            logger.error(f"Error in test evaluation: {e}")
            return {"test_loss": float('inf'), "test_accuracy": 0.0}