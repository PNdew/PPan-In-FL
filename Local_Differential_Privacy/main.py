from config import *
from preprocessing.data_handling  import *
from fl.client import PrivacyClient
from fl.strategy import FedAvg_Privacy
import flwr as fl
from clientmanager.manager import SimpleClientManager
from models.mnist_model import *
from function_strategy.function_stategy import *
from flwr.common import Context


def client_fn(context: Context) -> fl.client.Client:
    partition_id = context.node_config["partition-id"]

    if f"client_{partition_id}" not in federated_data:
        raise ValueError(f"Client ID {partition_id} không tồn tại trong dữ liệu!")
    
    client_data = federated_data[f"client_{partition_id}"]
    train_loader = get_dataloader_cifar10(client_data)
    test_loader = get_dataloader_cifar10(client_data)
    model = ResNet50(3,10).to(DEVICE)

    # Truyền đúng tham số cho PrivacyClient: cid, model, train_loader
    return PrivacyClient(cid=str(partition_id), model=model, train_loader=train_loader, val_loader = test_loader).to_client()


def get_evaluate_fn(testset):
    """Hàm evaluate trung tâm cho CIFAR-10 với ResNet50"""
    testloader = DataLoader(testset, batch_size=64, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: dict):
        # Khởi tạo model ResNet50 với num_classes=10
        model = ResNet50(num_channel=3, num_classes=10).to(device)
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
        with open("accuracy_get_evaluate_fn.txt", "a") as f:
            f.write(f"Round {server_round}: {accuracy:.6f}\n")

        return avg_loss, {"accuracy": accuracy}

    return evaluate


federated_data = None
def main():
    global federated_data
    client_data = load_data(NUM_CLIENTS)
    cifar10_fds, federated_data = split_mnist_dirichlet_flwr()
   # Define CIFAR-10 transform
    transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    # Load CIFAR-10 testset
    centralized_testset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform
)
    client_manager = SimpleClientManager()
    strategy = FedAvg_Privacy(
            fraction_fit=0.1,
            fraction_evaluate=0.1,
            min_fit_clients=int(0.1 * NUM_CLIENTS),
            min_evaluate_clients=int(0.1 * NUM_CLIENTS),
            min_available_clients=NUM_CLIENTS*0.75,
            fit_metrics_aggregation_fn=weighted_average,
            evaluate_fn=get_evaluate_fn(centralized_testset)
        )
    client_manager = SimpleClientManager()
    history = fl.simulation.start_simulation(client_fn=client_fn, num_clients=NUM_CLIENTS,client_manager=client_manager,
                                                     config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
                                                     client_resources = {'num_cpus': 1, 'num_gpus': 0.2},
                                                     strategy=strategy)

if __name__ == "__main__":
    main()


