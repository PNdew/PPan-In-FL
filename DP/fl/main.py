from client import PrivacyClient
from Strategy import FedAvg_Privacy
import flwr as fl
from clientmanager.manager import SimpleClientManager
from models.mnist_model import Net
from preprocessing.data_handling import split_mnist_dirichlet_flwr, get_dataloader
from function_strategy.function_stategy import *
from flwr.common import Context
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from config import *

fds, federated_data = split_mnist_dirichlet_flwr()
# Federated Learning Simulation
def get_client_fn(epsilon,delta, sensitivity):
    def client_fn(context: Context) -> fl.client.Client:
        partition_id = context.node_config["partition-id"]
        if f"client_{partition_id}" not in federated_data:
            raise ValueError(f"Client ID {partition_id} không tồn tại trong dữ liệu!")
        client_data = federated_data[f"client_{partition_id}"]
        train_loader = get_dataloader(client_data)
        test_loader = get_dataloader(client_data)
        model = Net()
        return PrivacyClient(model, train_loader, epsilon, delta, sensitivity).to_client()
    return client_fn


def get_evaluate_fn(testset, epsilon):
    """Hàm evaluate trung tâm cho CIFAR-10 với ResNet50"""
    testloader = DataLoader(testset, batch_size=64, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: dict):
        
        model = Net()
        model.to(device)
        model.eval()

        # Lấy state_dict từ tham số FL
        model_keys = list(model.state_dict().keys())
        state_dict = {}

        for key, weight in zip(model_keys, parameters):
            model_weight_shape = model.state_dict()[key].shape
            weight_tensor = torch.tensor(weight, device=device)  # Đưa tensor về đúng device
            if weight_tensor.shape == model_weight_shape:
                state_dict[key] = weight_tensor
            else:
                print(f"[Warning] Shape mismatch at {key}: expected {model_weight_shape}, got {weight_tensor.shape}")

        # Load tham số vào model
        model.load_state_dict(state_dict, strict=False)

        # Đánh giá model trên testset
        correct, total, total_loss = 0, 0, 0.0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total if total else 0.0
        avg_loss = total_loss / total if total else 0.0

        print(f"[Evaluation] Round {server_round}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}")

        with open(os.path.join(RESULTS_DIR, f"accuracy_get_evaluate_fn{epsilon}.txt"), "a") as f:
            f.write(f"{accuracy:.6f}\n")

        with open(os.path.join(RESULTS_DIR, f"loss_get_evaluate_fn{epsilon}.txt"), "a") as f:
            f.write(f"{avg_loss:.6f}\n")

        return avg_loss, {"accuracy": accuracy}

    return evaluate

def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    centralized_testset = datasets.FashionMNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )
   
    client_manager = SimpleClientManager()
    strategy = FedAvg_Privacy(
        fraction_fit=0.1,
        fraction_evaluate=0.1,
        min_fit_clients=int(0.1 * NUM_CLIENTS),
        min_evaluate_clients=int(0.1 * NUM_CLIENTS),
        min_available_clients=NUM_CLIENTS,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        fit_metrics_aggregation_fn=aggregate_fit_metrics,
        evaluate_metrics_aggregation_fn=aggregate_evaluate_metrics,
        evaluate_fn=get_evaluate_fn(centralized_testset),
        epsilon= eplison,
        delta=delta,
    )

    # Tạo thư mục riêng cho mỗi privacy_weight
    subdir = os.path.join(RESULTS_DIR, f"epsilon_{eplison}")
    os.makedirs(subdir, exist_ok=True)
    with open(os.path.join(subdir, "log.txt"), "w") as f:
        f.write(f"Running simulation with epsilon={eplison}\n")

    fl.simulation.start_simulation(
        client_fn=get_client_fn(eplison, delta, sensitivity),
         # Truyền hàm client với privacy_weight
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_manager=client_manager,
        client_resources={'num_cpus': 1, 'num_gpus': 0.1},
    )
if __name__ == "__main__":
    main()
