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

    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[BaseException]):
        if not results:
            return None, {}
        self.aggregator.reset()
        parameters_list = []
        weights = []
        for client, fit_res in results:
            params = parameters_to_ndarrays(fit_res.parameters)
            metrics = fit_res.metrics
            num_samples = fit_res.num_examples
            weight = self.aggregator.compute_weight(client.cid, num_samples, metrics["loss"])
            parameters_list.append(params)
            weights.append(weight)
        aggregated_params = [
            np.sum([w * p[i] for w, p in zip(weights, parameters_list)], axis=0)
            for i in range(len(parameters_list[0]))
        ]
        return ndarrays_to_parameters(aggregated_params), {}    