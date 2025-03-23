import torch
import numpy as np
from config import *

class SignificantGradientProtection:
    """Enhanced SGP with improved threshold computation"""
    def __init__(self, model_params: int, total_rounds: int):
        self.total_params = model_params
        self.total_rounds = total_rounds
        self.epsilon = EPSILON
        self.sensitivity = SENSITIVITY
        self.protected_count = 0
        self.total_count = 0

    def compute_threshold(self, round: int, gradients: torch.Tensor) -> float:
        """Compute threshold with bounds checking"""
        sorted_grads = torch.sort(gradients.abs().flatten())[0]
        total_elements = len(sorted_grads)

        # Compute indices with bounds checking
        idx1 = min(
            int((round * total_elements) / self.total_rounds),
            total_elements - 1
        )
        idx2 = min(
            int(0.9 * total_elements),
            total_elements - 1
        )

        return min(sorted_grads[idx1].item(), sorted_grads[idx2].item())

    def protect_gradients(self, gradients: torch.Tensor, round: int) -> torch.Tensor:
        """Apply gradient protection"""
        try:
            # Reset counters
            self.protected_count = 0
            self.total_count = gradients.numel()

            # Ensure gradients are on correct device
            gradients = gradients.to(DEVICE)

            # Clip gradients
            clipped_grads = torch.clamp(gradients, -CLIP_BOUND, CLIP_BOUND)

            # Compute threshold
            threshold = self.compute_threshold(round, clipped_grads)

            # Add noise for importance evaluation
            alpha = np.random.laplace(0, self.sensitivity/self.epsilon[0])
            beta = np.random.laplace(0, self.sensitivity/self.epsilon[1])

            # Identify significant gradients
            importance_mask = (clipped_grads.abs() + alpha >= threshold + beta)
            self.protected_count = importance_mask.sum().item()

            # Add Laplace noise to significant gradients
            noise = torch.tensor(
                np.random.laplace(0, self.sensitivity/self.epsilon[2], gradients.shape),
                dtype=gradients.dtype,
                device=DEVICE
            )

            return torch.where(importance_mask, clipped_grads + noise, clipped_grads)

        except Exception as e:
            logger.error(f"Error protecting gradients: {e}")
            return clipped_grads