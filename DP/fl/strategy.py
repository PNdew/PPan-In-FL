from typing import Dict, List, Tuple
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

from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, Parameters, Scalar
from flwr.common.dp import add_gaussian_noise
from flwr.common.logger import warn_deprecated_feature
from flwr.common.parameter import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import DPFedAvgFixed
from sklearn.feature_selection import mutual_info_regression
from typing import Dict, List, Optional
import random
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion
import threading
from utils.save_utils import write_to_file
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from flwr.server.strategy.aggregate import aggregate

def compute_privacy_leakage(encrypted_weights, original_weights):
    encrypted_weights = np.array(encrypted_weights)
    original_weights = np.array(original_weights)
    encrypted_2d = encrypted_weights.reshape(-1, 1)
    original_2d = original_weights.reshape(-1, 1)
    min_length = min(len(encrypted_2d), len(original_2d))
    encrypted_2d = encrypted_2d[:min_length]
    original_2d = original_2d[:min_length]
    try:
        mi_score = mutual_info_regression(encrypted_2d, original_2d.ravel())[0]
    except ValueError:
        mi_score = 0.0
    return mi_score

class DPFedAvgPrivacy(DPFedAvgFixed):
    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        aggregated_params, metrics = super().aggregate_fit(server_round, results, failures)

        # Tính độ rò rỉ thông tin
        param_arrays = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
        param_arrays_flat = [np.concatenate([p.flatten() for p in arr]) for arr in param_arrays]

        if len(param_arrays_flat) > 1:
            privacy_leakage = np.mean([
                compute_privacy_leakage(param_arrays_flat[i].reshape(-1, 1), param_arrays_flat[i + 1])
                for i in range(len(param_arrays_flat) - 1)
            ])
        else:
            privacy_leakage = 0.0

        metrics["privacy_leakage"] = privacy_leakage
        
        # Ghi kết quả vào các file riêng biệt
        with open("privacy_leakage.txt", "a") as f:
            f.write(f" {privacy_leakage:.6f}\n")
        
        accuracies = [fit_res.metrics["accuracy"] for _, fit_res in results if fit_res.metrics and "accuracy" in fit_res.metrics]
        avg_accuracy = np.mean(accuracies) if accuracies else 0.0
        metrics["accuracy"] = avg_accuracy

        with open("accuracy.txt", "a") as f:
            f.write(f" {avg_accuracy:.6f}\n")
        
        if failures:
            return None, {}
        
        # Forcing unweighted aggregation, as in https://arxiv.org/abs/1905.03871.
        for _, fit_res in results:
            fit_res.num_examples = 1
            fit_res.parameters = ndarrays_to_parameters(
                add_gaussian_noise(
                    parameters_to_ndarrays(fit_res.parameters),
                    self._calc_client_noise_stddev(),
                )
            )
        
        return aggregated_params, metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[Union[tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> tuple[Optional[float], dict[str, Scalar]]:
        aggregated_loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        accuracies = []

        # for _, res in results:
        #     if res.metrics and isinstance(res.metrics, dict):
        #         acc = res.metrics.get("accuracy", 0.0)
        #         accuracies.append(acc)
        #     else:
        #         logger.warning(f"Missing or invalid metrics in evaluate result: {res.metrics}")

        # avg_accuracy = np.mean(accuracies) if accuracies else 0.0
        # metrics["accuracy"] = avg_accuracy

        # # Ghi file
        # with open("accuracy.txt", "a") as f:
        #     f.write(f"Round {server_round} (Evaluation): {avg_accuracy:.4f}\n")

        # logger.info(f"Round {server_round} (Evaluation): Average accuracy = {avg_accuracy:.4f}")
        # print(f"Round {server_round} (Evaluation): Average accuracy = {avg_accuracy:.4f}")

        return aggregated_loss, metrics