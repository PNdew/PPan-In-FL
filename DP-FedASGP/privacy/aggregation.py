class DynamicAggregation:
    """Implements dynamic gradient aggregation"""
    def __init__(self):
        self.client_losses = {}
        self.total_samples = 0

    def compute_weight(self, client_id: str, num_samples: int, loss: float) -> float:
        """Compute dynamic weight for client"""
        self.client_losses[client_id] = loss
        total_loss = sum(self.client_losses.values())

        if total_loss == 0:
            return num_samples / self.total_samples

        weight = (num_samples * total_loss + self.total_samples * loss) / (2 * self.total_samples * total_loss)
        return weight

    def update_total_samples(self, num_samples: int):
        """Update total number of samples"""
        self.total_samples = num_samples
