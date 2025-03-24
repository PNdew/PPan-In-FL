import torch
import numpy as np
from config import *
from torch.distributions.laplace import Laplace

class SignificantGradientProtection:
    """Enhanced SGP with dynamic privacy budget"""
    def __init__(self, model_params: int, total_rounds: int, epsilon1: float, epsilon2: float, epsilon3: float):
        self.total_params = model_params
        self.total_rounds = total_rounds
        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2
        self.epsilon3 = epsilon3
        self.sensitivity = SENSITIVITY
        self.protected_count = 0
        self.total_count = 0
        self.accuracy_history = []
        logger.info(f"Initialized SGP with {model_params} parameters")


    def compute_threshold(self, round: int, gradients: torch.Tensor) -> float:
        sorted_grads = torch.sort(gradients.abs().flatten())[0]
        total_params = gradients.numel()
        m1 = int((round * total_params) / self.total_rounds)
        m90 = int(0.9 * total_params)
        threshold1 = sorted_grads[m1] if m1 < total_params else sorted_grads[-1]
        threshold2 = sorted_grads[m90] if m90 < total_params else sorted_grads[-1]
        return min(threshold1, threshold2).item()

    def protect_gradients(self, gradients: torch.Tensor, round: int) -> torch.Tensor:
        """Apply gradient protection with dynamic privacy"""
        try:
            # Compute threshold
            threshold = self.compute_threshold(round, gradients)

            # Identify significant gradients
            significant_mask = gradients.abs() > threshold

            # Track protection statistics
            self.total_count += gradients.numel()
            self.protected_count += significant_mask.sum().item()

            # Add noise only to significant gradients
            protected_grads = gradients.clone()
            if significant_mask.any():
                noise = noise = Laplace(0, self.sensitivity / self.epsilon).sample(gradients[significant_mask].shape)
                protected_grads[significant_mask] += noise

            # Clip final values
            protected_grads = torch.clamp(
                protected_grads,
                -CLIP_BOUND,
                CLIP_BOUND
            )

            return protected_grads

        except Exception as e:
            logger.error(f"Error protecting gradients: {e}")
            return gradients

    def adjust_privacy_budget(self, accuracy: float):
        """Dynamically adjust privacy budget based on accuracy trends"""
        self.accuracy_history.append(accuracy)
        if len(self.accuracy_history) >= 2:
            improvement = self.accuracy_history[-1] - self.accuracy_history[-2]

            if improvement > 0.05:  # Good improvement
                self.epsilon = max(self.epsilon * 0.9, MIN_EPSILON)
                logger.info(f"Reducing ε to {self.epsilon:.4f}")
            elif improvement < 0.01:  # Stagnating
                self.epsilon = min(self.epsilon * 1.1, MAX_EPSILON)
                logger.info(f"Increasing ε to {self.epsilon:.4f}")