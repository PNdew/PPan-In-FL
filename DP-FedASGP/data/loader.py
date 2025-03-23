from config import *
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import datasets, transforms
import traceback

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner

def split_mnist_dirichlet_flwr(num_clients=NUM_CLIENTS, alpha=DIRICHLET_ALPHA, seed=42):
    """Split MNIST using Dirichlet partitioning"""
    partitioner = DirichletPartitioner(
        num_partitions=num_clients,
        partition_by="label",
        alpha=alpha,
        seed=seed
    )

    fds = FederatedDataset(
        dataset="mnist",
        partitioners={"train": partitioner}
    )

    # Load partitions for each client
    federated_data = {
        f"client_{i}": fds.load_partition(i)
        for i in range(num_clients)
    }

    return fds, federated_data

def get_dataloader(client_data, batch_size=BATCH_SIZE):
    """Create DataLoader from client data"""
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Convert images to tensors
    x_tensor = torch.stack([transform(img) for img in client_data["image"]])
    y_tensor = torch.tensor(client_data["label"], dtype=torch.long)

    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)