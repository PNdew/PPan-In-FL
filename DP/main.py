from config import *
from models.mnist_model import get_model
from fl.strategy import *
from fl.manager import SimpleClientManager
from utils.data_handling import split_mnist_dirichlet_flwr, get_client_fn
import flwr as fl
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

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
    """Return an evaluation function for server-side (i.e. centralized) evaluation."""

    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]):
        model = get_model()
        state_dict = dict(zip(model.state_dict().keys(), [torch.tensor(w) for w in parameters]))
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        correct, total, total_loss = 0, 0, 0.0
        criterion = torch.nn.CrossEntropyLoss()
        with torch.no_grad():
            for batch in testset:
                if isinstance(batch, dict):
                    images, labels = batch["image"], batch["label"]
                else:
                    images, labels = batch
                images = images.to(device).float() / 255.0   
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, {"accuracy": accuracy}

    return evaluate


def main():
    mnist_fds ,federated_data = split_mnist_dirichlet_flwr()
    client_manager = SimpleClientManager()
    strategy = DPFedAvgPrivacy(
        fl.server.strategy.FedAvg(
            fraction_fit=0.1,
            fraction_evaluate=0.0,
            min_fit_clients=NUM_CLIENTS,
            min_evaluate_clients=0,
            min_available_clients=NUM_CLIENTS,
            fit_metrics_aggregation_fn=weighted_average
        ),
        num_sampled_clients=10,
        server_side_noising=True,
        clip_norm=CLIP_NORM,
        noise_multiplier=NOISE_MULTIPLIER
    )

    history = fl.simulation.start_simulation(
        client_fn=get_client_fn(mnist_fds, federated_data),
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_manager=client_manager,
        client_resources={"num_cpus": 1, "num_gpus": 0.0}
    )

    logger.info("Training completed successfully.")
    logger.info(f"Final training history: {history}")
    return history

if __name__ == "__main__":
    main()
