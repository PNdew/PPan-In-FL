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
import tensorflow as tf
import flwr as fl
import numpy as np
from flwr.common import Metrics
from flwr.simulation.ray_transport.utils import enable_tf_gpu_growth
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from flwr.server.strategy import DPFedAvgFixed
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import logging
from typing import Optional, Union
import flwr as fl
from flwr.common import Parameters, NDArrays
from flwr.server.client_proxy import ClientProxy
from flwr.common.typing import Config
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms, datasets
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, Parameters, Scalar
from flwr.common.dp import add_gaussian_noise
from flwr.common.logger import warn_deprecated_feature
from flwr.common.parameter import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from sklearn.feature_selection import mutual_info_regression
from typing import Dict, List, Optional, Tuple, Union

from utils.save_utils import write_to_file

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

        avg_loss = np.mean([r.metrics.get("loss", 0.0) for _, r in results])
        avg_accuracy = np.mean([r.metrics.get("accuracy", 0.0) for _, r in results])

        privacy_values = [r.metrics["privacy_leakage"] for _, r in results if "privacy_leakage" in r.metrics]
        avg_privacy_leakage = np.mean(privacy_values) if privacy_values else 0.0

        epsilon_values = [r.metrics["epsilon"] for _, r in results if "epsilon" in r.metrics]
        avg_epsilon = np.mean(epsilon_values) if epsilon_values else 0.0

        write_to_file("privacy_leakage.txt", avg_privacy_leakage)
        write_to_file("epsilon.txt", avg_epsilon)


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

        return loss, metrics

