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
import numpy as np
from utils.save_utils import save_metric_to_txt
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
from config import *
from utils.save_utils import save_metric_to_txt
from config import LEARNING_RATE, BATCH_SIZE, NOISE_SCALE
import flwr as fl


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
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        aggregated_result = super().aggregate_fit(server_round, results, failures)
        if aggregated_result is None:
            return None, {}

        parameters_aggregated, metrics = aggregated_result
        ndarrays = parameters_to_ndarrays(parameters_aggregated)

        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        avg_loss = np.mean([r.metrics["loss"] for _, r in results])
        avg_accuracy = np.mean([r.metrics["accuracy"] for _, r in results])
        avg_privacy_leakage = np.mean([r.metrics["privacy_leakage"] for _, r in results])
        avg_distortion = np.mean([r.metrics["distortion"] for _, r in results])

        return ndarrays_to_parameters(ndarrays), metrics_aggregated
        
    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure evaluation for the clients."""
        config = {"round": server_round}
        evaluate_ins = EvaluateIns(parameters=parameters, config=config)
        client_instructions = [(client, evaluate_ins) for client in client_manager.all().values()]
        return client_instructions

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, float]]:
        
        if not results:
            return None, {}

        if not self.accept_failures and failures:
            return None, {}

        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)

        avg_loss = metrics_aggregated.get("loss", None)
        if avg_loss is None:
            avg_loss = np.mean([r.metrics.get("loss", 0) for _, r in results])

        for metric_name in ["loss", "accuracy", "privacy_leakage", "distortion"]:
            metric_val = metrics_aggregated.get(metric_name, None)
            if metric_val is not None:
                save_metric_to_txt(metric_name, metric_val, phase="evaluate")

        return avg_loss, metrics_aggregated
    def _update_best_acc(self, server_round: int, current_acc: float, parameters: Parameters) -> None:
        if not hasattr(self, "best_acc"):
            self.best_acc = 0.0  # Khởi tạo best_acc nếu chưa tồn tại

        if current_acc > self.best_acc:
            self.best_acc = current_acc
    def store_results_and_log(self, server_round: int, results_dict: dict) -> None:
        """
        Lưu và log kết quả sau mỗi vòng training.
        """
        result_line = f"Round {server_round}: " + ", ".join(
            [f"{k}={v:.4f}" for k, v in results_dict.items()]
        )
        with open("results_log.txt", "a") as f:
            f.write(result_line + "\n")

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """
        Evaluate global model parameters using an evaluation function.
        Lưu kết quả evaluate vào file và in log thông tin.

        Args:
            server_round (int): Số vòng hiện tại trên server.
            parameters (Parameters): Tham số của mô hình toàn cục (global model).

        Returns:
            Tuple[float, Dict[str, Scalar]]: Giá trị loss trung bình và các metrics khác.
        """
        # Nếu không có hàm evaluate_fn được truyền vào strategy, bỏ qua evaluation
        if self.evaluate_fn is None:
            return None

        # Chuyển đổi tham số global model về numpy arrays
        ndarrays = parameters_to_ndarrays(parameters)

        # Thực hiện evaluate bằng hàm evaluate_fn đã được định nghĩa trước (bên ngoài)
        loss, metrics = self.evaluate_fn(server_round, ndarrays, {})

        # Cập nhật best accuracy nếu đạt kết quả tốt hơn trước
        if "accuracy" in metrics:
            self._update_best_acc(server_round, metrics["accuracy"], parameters)

        # Lưu kết quả evaluate vào file
        self.store_results_and_log(
            server_round=server_round,
            results_dict={"loss": loss, **metrics},
        )

        # Lưu các metrics cụ thể vào file txt (tuỳ chỉnh)
        for metric_name in ["loss", "accuracy", "privacy_leakage", "distortion"]:
            if metric_name in metrics:
                save_metric_to_txt(metric_name, metrics[metric_name], phase="evaluate")

        # In log kết quả evaluate
        print(f"Round {server_round} - Evaluate Loss: {loss:.4f} - Accuracy: {metrics.get('accuracy', 0):.4f}")

        return loss, metrics

