from privacy.aggregation import DynamicAggregation
from privacy.metrics import Metrics
from config import NUM_ROUNDS, EPSILON
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import flwr as fl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any
from flwr.common import (
    FitRes, Parameters, Config, Context, Scalar,
    ndarrays_to_parameters, parameters_to_ndarrays
)
from flwr.server.client_proxy import ClientProxy


class DPFedASGPStrategy(fl.server.strategy.FedAvg):
    """Server strategy implementing DP-FedASGP"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aggregator = DynamicAggregation()
        self.metrics = Metrics()
        self.current_round = 0
        self.privacy_cost = 0

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict]:
        """Aggregate results with privacy protection"""
        if not results:
            return None, {}

        self.current_round = server_round

        # Compute weights and aggregate
        weights = {}
        total_samples = sum(fit_res.num_examples for _, fit_res in results)
        self.aggregator.update_total_samples(total_samples)

        accuracy = 0
        loss = 0
        protected_grads = 0
        total_grads = 0

        for client, fit_res in results:
            weights[client.cid] = self.aggregator.compute_weight(
                client.cid,
                fit_res.num_examples,
                fit_res.metrics["loss"]
            )

            accuracy += fit_res.metrics["accuracy"] * weights[client.cid]
            loss += fit_res.metrics["loss"] * weights[client.cid]
            protected_grads += fit_res.metrics["protected_grads"]
            total_grads += fit_res.metrics["total_grads"]

        # Update privacy cost
        self.privacy_cost += sum(EPSILON)

        # Update metrics
        self.metrics.update({
            "accuracy": accuracy,
            "loss": loss,
            "privacy_cost": self.privacy_cost,
            "protected_ratio": (protected_grads / total_grads * 100) if total_grads > 0 else 0
        })
        # Aggregate parameters
        parameters_aggregated = super().aggregate_fit(server_round, results, failures)
        return parameters_aggregated