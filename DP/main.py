from config import *
from models.mnist_model import *
from fl.client import FlowerClient
from fl.strategy import *
from fl.manager import SimpleClientManager
from utils.data_handling import split_mnist_dirichlet_flwr, get_client_fn
import flwr as fl

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



def main():
    mnist_fds, federated_data = split_mnist_dirichlet_flwr()
    def get_evaluate_fn(testset):
      """Return an evaluation function for server-side (i.e. centralised) evaluation."""

      # The `evaluate` function will be called after every round by the strategy
      def evaluate(
          server_round: int,
          parameters: fl.common.NDArrays,
          config: Dict[str, fl.common.Scalar],
      ):
          model = get_model()  # Construct the model
          model.set_weights(parameters)  # Update model with the latest parameters
          loss, accuracy = model.evaluate(testset, verbose=VERBOSE)
          return loss, {"accuracy": accuracy}

      return evaluate
    mnist_fds = FederatedDataset(dataset="mnist", partitioners={"train": NUM_CLIENTS})
    centralized_testset = mnist_fds.load_split("test").to_tf_dataset(
        columns="image", label_cols="label", batch_size=64
    )

    client_manager = SimpleClientManager()
    strategy = DPFedAvgPrivacy(
        fl.server.strategy.FedAvg(evaluate_fn=get_evaluate_fn(centralized_testset),
            fraction_fit= 0.1,
        fraction_evaluate=0.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=0.0,
        min_available_clients=NUM_CLIENTS,
        fit_metrics_aggregation_fn=weighted_average
        ),
        num_sampled_clients=10,
        server_side_noising=True,
        clip_norm=CLIP_NORM,
        noise_multiplier=NOISE_MULTIPLIER
    )

    history = fl.simulation.start_simulation(
        client_fn=get_client_fn(mnist_fds),
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_manager=client_manager,
        client_resources={"num_cpus": 1, "num_gpus": 0.0}
    )

    logger.info("Training completed successfully")
    print(history)
    return history

if __name__ == "__main__":
    main()
