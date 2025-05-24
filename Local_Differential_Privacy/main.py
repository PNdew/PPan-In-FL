from config import *
from utils.data_hanlding import load_data
from fl.client import PrivacyClient
from fl.strategy import FedAvg_Privacy
import flwr as fl
from fl.manager import SimpleClientManager
from models.mnist_model import Net, ResNet18
from flwr.common import Context
from  utils.data_hanlding import get_dataloader

def main():
    federated_data = load_data(NUM_CLIENTS)
    client_manager = SimpleClientManager()

    def client_fn(context: Context) -> fl.client.Client:
        """Create a Flower client representing an organization."""
        partition_id = int(context.node_config["partition-id"])
        
        # Check if partition_id is valid
        if partition_id >= NUM_CLIENTS:
            raise ValueError(f"Client ID {partition_id} does not exist in the data!")
            
        # Get client data
        client_data = federated_data[partition_id]
        
        # Create train and test dataloaders
        train_loader = get_dataloader(client_data, train=True)
        test_loader = get_dataloader(client_data, train=False)
        
        model = ResNet18()
        
        # Return client
        return PrivacyClient(str(partition_id), model, train_loader, test_loader)


    strategy = FedAvg_Privacy(min_available_clients=NUM_CLIENTS, min_fit_clients=K_CLIENTS)
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        client_manager=client_manager,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        client_resources={'num_cpus': 1, 'num_gpus': 0},
        strategy=strategy
    )

if __name__ == "__main__":
    main()