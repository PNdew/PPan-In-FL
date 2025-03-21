from config import *
from models.mnist_model import MNISTModel
from fl.client import PrivacyClient
from fl.strategy import *
from fl.manager import SimpleClientManager
from utils.data_handling import split_mnist_dirichlet_flwr, get_dataloader
import flwr as fl

# Config functions for fit and evaluate


fds, federated_data = split_mnist_dirichlet_flwr()  
# Federated Learning Simulation
def client_fn(context: Context) -> fl.client.Client:
    """Tạo một Flower client đại diện cho một tổ chức."""
    partition_id = context.node_config["partition-id"]

    # Kiểm tra nếu partition_id hợp lệ
    if f"client_{partition_id}" not in federated_data:
        raise ValueError(f"Client ID {partition_id} không tồn tại trong dữ liệu!")
    client_data = federated_data[f"client_{partition_id}"]
    train_loader = get_dataloader(client_data)
    test_loader = get_dataloader(client_data)
    model = MNISTModel()

    # Trả về client
    return PrivacyClient(model, train_loader, test_loader).to_client()

def main():
    client_manager = SimpleClientManager()
    strategy = FedAvg_Privacy(
        fraction_fit= 0.1,
        fraction_evaluate=0.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=0.0,
        min_available_clients=NUM_CLIENTS,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        fit_metrics_aggregation_fn=aggregate_fit_metrics,
        noise_scale=NOISE_SCALE,
    )
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_manager = client_manager,
        client_resources = {'num_cpus': 1, 'num_gpus': 0},
    )

if __name__ == "__main__":
    main()