from typing import Callable, Dict, List, Optional, Tuple, Union
from typing import List, Dict, Tuple
from flwr.common import NDArrays, EvaluateRes, Parameters, FitRes, Metrics, Context
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
from typing import Dict, List, Optional
from flwr.server.client_proxy import ClientProxy
from config import *

import flwr as fl
from function_strategy.function_stategy import *
import numpy as np
from utils.save_utils import save_metric_to_txt

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
        epsilon: float = 1.0,
        delta: float = 1e-5,
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
        self.epsilon = epsilon
        self.delta = delta


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
        avg_privacy_leakage = np.mean([r.metrics["privacy_leakage"] for _, r in results])
        avg_distortion = np.mean([r.metrics["distortion"] for _, r in results])
        save_metric_to_txt("privacy_leakage", avg_privacy_leakage, server_round, self.epsilon)
        save_metric_to_txt("distortion", avg_distortion, server_round, self.epsilon)

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
        # In log kết quả evaluate
        print(f"Round {server_round} - Evaluate Loss: {loss:.4f} - Accuracy: {metrics.get('accuracy', 0):.4f}")

        return loss, metrics

