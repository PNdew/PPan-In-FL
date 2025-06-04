from config import *
from models.mnist_model import MNISTModel, MNISTResNet
from fl.client import PrivacyClient
from fl.strategy import *
from clientmanager.manager import SimpleClientManager
from preprocessing.data_handling import split_mnist_dirichlet_flwr, get_dataloader
import flwr as fl
from torchvision import datasets, transforms
from metric.metrics import *
from function_strategy.function_stategy import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader




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
    PRIVACY_WEIGHT = PRIVACY_WEIGHT  # Sử dụng biến toàn cục PRIVACY_WEIGHT
    # Trả về client
    return PrivacyClient(model, train_loader, test_loader,PRIVACY_WEIGHT).to_client()


def get_evaluate_fn(testset):
    """Hàm evaluate trung tâm cho CIFAR-10 với ResNet50"""
    testloader = DataLoader(testset, batch_size=64, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: dict):
        
        model = MNISTModel().to(device)
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

        # Ghi kết quả vào file
        with open(f"accuracy_get_evaluate_fn{PRIVACY_WEIGHT}.txt", "a") as f:
            f.write(f"{accuracy:.6f}\n")
        with open(f"loss_get_evaluate_fn{PRIVACY_WEIGHT}.txt", "a") as f:
            f.write(f"{avg_loss:.6f}\n")

        return avg_loss, {"accuracy": accuracy}

    return evaluate
def run_simulation_for_privacy_weight(p_weight):
    global PRIVACY_WEIGHT
    PRIVACY_WEIGHT = p_weight  # Gán lại giá trị để dùng trong client

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
        PRIVACY_WEIGHT=p_weight,  # Truyền privacy_weight vào strategy
    )

    # Tạo thư mục riêng cho mỗi privacy_weight
    subdir = os.path.join(RESULTS_DIR, f"privacy_{p_weight}")
    os.makedirs(subdir, exist_ok=True)
    with open(os.path.join(subdir, "log.txt"), "w") as f:
        f.write(f"Running simulation with privacy_weight={p_weight}\n")

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_manager=client_manager,
        client_resources={'num_cpus': 1, 'num_gpus': 0.1},
    )
if __name__ == "__main__":
    PRIVACY_WEIGHT_LIST = [500, 200, 100, 10, 1, 0.1, 0.01, 0.001]
    for pw in PRIVACY_WEIGHT_LIST:
        print(f" Đang chạy với privacy_weight = {pw}")
        run_simulation_for_privacy_weight(pw)

