from config import *
from fl.client import DPFedASGPClient
from fl.strategy import DPFedASGPStrategy
from data.loader import *
import flwr as fl
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from flwr.common import (
    FitRes, Parameters, Config, Context, Scalar,
    ndarrays_to_parameters, parameters_to_ndarrays
)
from models.net import Net

def fit_config(server_round: int) -> Dict[str, str]:
    """Return training configuration for round"""
    return {
        "learning_rate": str(LEARNING_RATE),
        "batch_size": str(BATCH_SIZE),
        "round": str(server_round),
    }

def evaluate_config(server_round: int) -> Dict[str, str]:
    """Return evaluation configuration for round"""
    return {
        "batch_size": str(BATCH_SIZE),
        "round": str(server_round),
    }

def aggregate_fit_metrics(metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
    """Aggregate training metrics from clients"""
    if not metrics:
        return {}

    aggregated = {}
    for _, client_metrics in metrics:
        for k, v in client_metrics.items():
            if k not in aggregated:
                aggregated[k] = []
            aggregated[k].append(v)

    return {
        k: np.mean(v) for k, v in aggregated.items()
    }


def weighted_average(metrics: List[Tuple[int, Dict]]) -> Dict:
    """Compute weighted average of metrics"""
    if not metrics:
        return {}

    total_examples = sum(num_examples for num_examples, _ in metrics)
    weighted_metrics = {}

    for metric_name in metrics[0][1].keys():
        weighted_sum = sum(
            value * num_examples
            for num_examples, metric_dict in metrics
            for name, value in metric_dict.items()
            if name == metric_name and not np.isnan(value)
        )
        weighted_metrics[metric_name] = weighted_sum / total_examples if total_examples > 0 else 0

    return weighted_metrics

def client_fn(cid: str) -> fl.client.Client:
    """Create a Flower client"""
    try:
        # Get client data
        if f"client_{cid}" not in federated_data:
            raise ValueError(f"Invalid client ID: {cid}")

        client_data = federated_data[f"client_{cid}"]
        trainloader = get_dataloader(client_data)

        # Create and return client
        client = DPFedASGPClient(
            cid=cid,
            model=Net(),
            trainloader=trainloader
        )

        return client.to_client()

    except Exception as e:
        logger.error(f"Error creating client: {e}")
        raise

def main():
    """Main training function"""
    try:
        logger.info("Starting DP-FedASGP training...")

        # Create federated data partitions
        global federated_data, fds
        try:
            fds, federated_data = split_mnist_dirichlet_flwr()
            logger.info("Successfully created federated data partitions")
        except Exception as e:
            logger.error(f"Error creating data partitions: {e}")
            raise

        # Create strategy with improved parameters
        strategy = DPFedASGPStrategy(
            fraction_fit=0.5,  # Increased participation
            fraction_evaluate=1.0,
            min_fit_clients=5,  # More clients per round
            min_evaluate_clients=NUM_CLIENTS,
            min_available_clients=NUM_CLIENTS,
            evaluate_metrics_aggregation_fn=weighted_average,
            on_fit_config_fn=fit_config,
            on_evaluate_config_fn=evaluate_config
        )

        # Start simulation
        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=NUM_CLIENTS,
            config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
            strategy=strategy,
            client_resources={"num_cpus": 1}
        )

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        logger.error(traceback.format_exc())
if __name__ == "__main__":
    main()