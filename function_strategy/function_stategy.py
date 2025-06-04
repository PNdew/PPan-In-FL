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
import numpy as np
from PPAN_FL.config import LEARNING_RATE, BATCH_SIZE
from typing import List, Tuple
from flwr.common import Metrics

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

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

def aggregate_evaluate_metrics(metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
    total_samples = sum(n for n, _ in metrics)
    aggregated = {}
    for n, client_metrics in metrics:
        for k, v in client_metrics.items():
            aggregated[k] = aggregated.get(k, 0) + v * n
    for k in aggregated:
        aggregated[k] /= total_samples
    return aggregated

def aggregate_weighted_parameters(results: List[Tuple[NDArrays, int]]) -> NDArrays:
    total_examples = sum(num_examples for _, num_examples in results)
    if total_examples == 0:
        return None
    weighted_params = [np.zeros_like(param) for param in results[0][0]]
    for parameters, num_examples in results:
        weight = num_examples / total_examples
        for i, param in enumerate(parameters):
            weighted_params[i] += param * weight
    return weighted_params
