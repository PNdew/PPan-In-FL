from typing import Callable, Dict, List, Optional, Tuple, Union
from flwr.common import (
    EvaluateRes,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import flwr as fl
import numpy as np
import matplotlib.pyplot as plt
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from flwr_datasets.visualization import plot_label_distributions
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.feature_selection import mutual_info_regression
from typing import List, Dict, Tuple
from flwr.common import NDArrays, EvaluateRes, Parameters, FitRes, Metrics, Context
from torchvision import transforms
import json
import os
from typing import List, Tuple, Dict, Union, Optional, Callable
from flwr.common import Parameters, Scalar, NDArrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,

    FitRes,
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
import threading
from typing import Dict, List, Optional
import random
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion
from config import NOISE_SCALE
from utils.save_utils import save_metric_to_txt


def fit_config(server_round: int) -> Dict[str, str]:
    config = {
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "round": server_round,
    }
    return config

def evaluate_config(server_round: int) -> Dict[str, str]:
    config = {
        "batch_size": BATCH_SIZE,
        "round": server_round,
    }
    return config

def aggregate_fit_metrics(metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
    aggregated_metrics = {}
    for _, client_metrics in metrics:
        for k, v in client_metrics.items():
            aggregated_metrics[k] = aggregated_metrics.get(k, 0) + v
    for k in aggregated_metrics:
        aggregated_metrics[k] /= len(metrics)
    return aggregated_metrics


def aggregate_weighted_parameters(results: List[Tuple[NDArrays, int]]) -> NDArrays:
    """Aggregate model parameters using weighted average."""
    # Calculate total number of examples used during training
    total_examples = sum(num_examples for _, num_examples in results)

    if total_examples == 0:
        return None

    # Get parameter shape from first result
    params_shape = [param.shape for param in results[0][0]]
    weighted_params = [np.zeros_like(param) for param in results[0][0]]

    # Calculate weighted parameters
    for parameters, num_examples in results:
        weight = num_examples / total_examples
        for i, param in enumerate(parameters):
            weighted_params[i] += param * weight

    return weighted_params


class FedAvg_Privacy(fl.server.strategy.FedAvg):
    """Federated Averaging with Privacy strategy."""

    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[[int, NDArrays, Dict[str, Scalar]],
                    Optional[Tuple[float, Dict[str, Scalar]]]]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        noise_scale: float = NOISE_SCALE,
    ) -> None:
        """Initialize FedAvg with Privacy strategy."""
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        self.noise_scale = noise_scale

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[BaseException],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate fit results using weighted average with privacy."""
        if not results:
            return None, {}

        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Call parent's aggregate_fit to get aggregated parameters
        aggregated_result = super().aggregate_fit(server_round, results, failures)
        if aggregated_result is None:
            return None, {}

        parameters_aggregated, metrics = aggregated_result

        # Convert parameters to ndarrays
        ndarrays = parameters_to_ndarrays(parameters_aggregated)

        # Aggregate metrics
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        # Tính loss, accuracy, privacy leakage, distortion trên server
        avg_loss = np.mean([r.metrics["loss"] for _, r in results])
        avg_accuracy = np.mean([r.metrics["accuracy"] for _, r in results])
        avg_privacy_leakage = np.mean([r.metrics["privacy_leakage"] for _, r in results])
        avg_distortion = np.mean([r.metrics["distortion"] for _, r in results])

        # Lưu lại kết quả trên server
        save_metric_to_txt(server_round, "loss", avg_loss, phase="train")
        save_metric_to_txt(server_round, "accuracy", avg_accuracy, phase="train")
        save_metric_to_txt(server_round, "privacy_leakage", avg_privacy_leakage, phase="train")
        save_metric_to_txt(server_round, "distortion", avg_distortion, phase="train")

        # Convert back to Parameters and return
        return ndarrays_to_parameters(ndarrays), metrics_aggregated
