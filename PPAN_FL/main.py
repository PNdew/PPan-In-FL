from config import *
from models.mnist_model import MNISTModel, MNISTResNet
from fl.client import PrivacyClient
from fl.strategy import *
from fl.manager import SimpleClientManager
from utils.data_handling import split_mnist_dirichlet_flwr, get_dataloader
import flwr as fl
from torchvision import datasets, transforms
from utils.metrics import *

# Config functions for fit and evaluate


def fit_config(server_round: int) -> Dict[str, str]:
    config = {
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "round": server_round,
    }
    return config

def evaluate_config(server_round: int) -> Dict[str, str]:
    config = {
        "batch_size": BATCH_SIZE,
        "round": server_round,
    }
    return config

def aggregate_fit_metrics(metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
    aggregated_metrics = {}
    for _, client_metrics in metrics:
        for k, v in client_metrics.items():
            aggregated_metrics[k] = aggregated_metrics.get(k, 0) + v
    for k in aggregated_metrics:
        aggregated_metrics[k] /= len(metrics)
    return aggregated_metrics

def aggregate_evaluate_metrics(metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
    total_samples = sum(n for n, _ in metrics)
    aggregated = {}
    for n, client_metrics in metrics:
        for k, v in client_metrics.items():
            aggregated[k] = aggregated.get(k, 0) + v * n
    for k in aggregated:
        aggregated[k] /= total_samples
    return aggregated

def aggregate_weighted_parameters(results: List[Tuple[NDArrays, int]]) -> NDArrays:
    total_examples = sum(num_examples for _, num_examples in results)
    if total_examples == 0:
        return None
    weighted_params = [np.zeros_like(param) for param in results[0][0]]
    for parameters, num_examples in results:
        weight = num_examples / total_examples
        for i, param in enumerate(parameters):
            weighted_params[i] += param * weight
    return weighted_params

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
    model =  MNISTModel()

    # Trả về client
    return PrivacyClient(model, train_loader, test_loader).to_client()
def get_evaluate_fn(testset: torch.utils.data.Dataset, batch_size: int = 32):
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]):
        model = MNISTModel()  # Khởi tạo model PyTorch của bạn
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        loss_fn = torch.nn.CrossEntropyLoss()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for data, target in testloader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                loss = loss_fn(outputs, target)
                total_loss += loss.item() * data.size(0)
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == target).sum().item()
                total_samples += data.size(0)

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        with torch.no_grad():
            # Lấy tham số model thành list tensor
            
            params = [p.detach().cpu().numpy() for p in model.parameters()]
            flat_params = np.concatenate([p.flatten() for p in params])
            input_dim = flat_params.shape[0] 
            privacy_mech = PrivacyMechanism(input_dim)

            flat_tensor = torch.tensor(flat_params, dtype=torch.float32).unsqueeze(0).to(device)
            encrypted_tensor = privacy_mech.encrypt(flat_tensor)
            encrypted_np = encrypted_tensor.detach().cpu().numpy().flatten()

            privacy_leakage = float(compute_privacy_leakage(encrypted_np, flat_params))
            distortion = float(compute_distortion(flat_params, encrypted_np))

        metrics = {
            "accuracy": accuracy,
            "privacy_leakage": privacy_leakage,
            "distortion": distortion,
        }

        return avg_loss, metrics

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
        fraction_fit=0.1,  # 10% clients
        fraction_evaluate=0.1,  # Không evaluate
        min_fit_clients=int(0.1 * NUM_CLIENTS),  # Tối thiểu 10 clients
        min_evaluate_clients=int(0.1 * NUM_CLIENTS),
        min_available_clients=NUM_CLIENTS,  # Tổng số clients có sẵn
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        fit_metrics_aggregation_fn=aggregate_fit_metrics,
        evaluate_metrics_aggregation_fn=aggregate_evaluate_metrics,
        evaluate_fn=get_evaluate_fn(centralized_testset),
    )

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_manager = client_manager,
        client_resources = {'num_cpus': 1, 'num_gpus': 0.1},
    )

if __name__ == "__main__":
    main()
