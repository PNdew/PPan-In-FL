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
from flwr.server.strategy import DPFedAvgFixed
from sklearn.feature_selection import mutual_info_regression
from typing import Dict, List, Optional, Tuple, Union

from utils.save_utils import write_to_file

class FedAvg_Privacy(fl.server.strategy.FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metrics_history = []

    def aggregate_fit(self, server_round, results, failures):
        aggregated = super().aggregate_fit(server_round, results, failures)
        if aggregated is None:
            return None, {}
        parameters, _ = aggregated
        total_samples = sum(fit_res.num_examples for _, fit_res in results)
        metrics = {
            "round": server_round,
            "accuracy": sum(fit_res.metrics["accuracy"] * fit_res.num_examples for _, fit_res in results) / total_samples,
            "epsilon": np.mean([fit_res.metrics["epsilon"] for _, fit_res in results]),
            "leakage": np.mean([fit_res.metrics["leakage"] for _, fit_res in results]),
            "loss": sum(fit_res.metrics["loss"] * fit_res.num_examples for _, fit_res in results) / total_samples
        }

        print(f"Server received metrics: {metrics}")  # Debug

        write_to_file("loss.txt", metrics["loss"])
        write_to_file("accuracy.txt", metrics["accuracy"])
        write_to_file("epsilon.txt", metrics["epsilon"])
        write_to_file("leakage.txt", metrics["leakage"])
        self.metrics_history.append(metrics)
        return parameters, metrics
