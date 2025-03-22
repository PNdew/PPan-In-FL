from config import *
from utils.data_hanlding  import load_data
from fl.client import PrivacyClient
from fl.strategy import FedAvg_Privacy
import flwr as fl
from fl.manager import SimpleClientManager

def main():
    client_data = load_data(NUM_CLIENTS)
    client_manager = SimpleClientManager()
    def client_fn(cid: str): return PrivacyClient(cid, Net(), client_data[int(cid)])
    strategy = FedAvg_Privacy(min_available_clients=NUM_CLIENTS, min_fit_clients=K_CLIENTS)
    history = fl.simulation.start_simulation(client_fn=client_fn, num_clients=NUM_CLIENTS,client_manager=client_manager,
                                                     config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
                                                     client_resources = {'num_cpus': 1, 'num_gpus': 0},
                                                     strategy=strategy)

if __name__ == "__main__":
    main()
