from config import *
from models.mnist_model import *
from fl.strategy import *
from fl.manager import SimpleClientManager
from utils.data_handling import split_mnist_dirichlet_flwr, get_client_fn
import flwr as fl
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import transforms, datasets
import torch
from torch.utils.data import DataLoader

def weighted_average(metrics):
    if not metrics:
        print("⚠️ No metrics received!")  # Debug lỗi
        return {}

    total_samples = sum(num_examples for num_examples, _ in metrics)

    avg_metrics = {
        key: sum(num_examples * metric[key] for num_examples, metric in metrics) / total_samples
        for key in metrics[0][1] if all(key in metric for _, metric in metrics)
    }

    print(f"✅ Aggregated Metrics: {avg_metrics}")  # Debug
    return avg_metrics

def get_evaluate_fn(testset):
    testloader = DataLoader(testset, batch_size=64, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]):
        model = ResNet50(num_channel=3, num_classes=10).to(device)

        model_keys = list(model.state_dict().keys())
        state_dict = {}

        for key, weight in zip(model_keys, parameters):
            model_weight_shape = model.state_dict()[key].shape
            weight_tensor = torch.tensor(weight)
            if weight_tensor.shape == model_weight_shape:
                state_dict[key] = weight_tensor
            else:
                print(f"[Warning] Shape mismatch at {key}: expected {model_weight_shape}, got {weight_tensor.shape}")

        model.load_state_dict(state_dict, strict=False)  # dùng strict=False để bỏ qua layer lỗi

        model.eval()
        correct, total, total_loss = 0, 0, 0.0
        criterion = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total if total else 0.0
        avg_loss = total_loss / total if total else 0.0
        print(f"Central Evaluation - Round {server_round}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}")
        with open("accuracy_get_evaluate_fn.txt", "a") as f:
            f.write(f" {accuracy:.6f}\n")
        return avg_loss, {"accuracy": accuracy}

    return evaluate



def main():
    cifar10_fds ,federated_data = split_mnist_dirichlet_flwr()
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    centralized_testset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )
    client_manager = SimpleClientManager()
    strategy = DPFedAvgPrivacy(
        fl.server.strategy.FedAvg(
            fraction_fit=0.1,
            fraction_evaluate=0.1,
            min_fit_clients=int(0.1 * NUM_CLIENTS),
            min_evaluate_clients=int(0.1 * NUM_CLIENTS),
            min_available_clients=NUM_CLIENTS*0.75,
            fit_metrics_aggregation_fn=weighted_average,
            evaluate_fn=get_evaluate_fn(centralized_testset)
        ),
        num_sampled_clients=10,
        server_side_noising=True,
        clip_norm=CLIP_NORM,
        noise_multiplier=NOISE_MULTIPLIER
    )

    history = fl.simulation.start_simulation(
        client_fn=get_client_fn(cifar10_fds, federated_data),
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_manager=client_manager,
        client_resources={"num_cpus": 1, "num_gpus": 0.1}
    )

    logger.info("Training completed successfully")
    print(history)
    return history

if __name__ == "__main__":
    main()
