import tensorflow as tf
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from config import NUM_CLIENTS, BATCH_SIZE
from models.mnist_model import ResNet50
from flwr.common import NDArrays
import flwr as fl
from typing import Dict, Tuple
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.feature_selection import mutual_info_regression
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from flwr_datasets.visualization import plot_label_distributions
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.feature_selection import mutual_info_regression
from typing import List, Dict, Tuple
from flwr.common import NDArrays, EvaluateRes, Parameters, FitRes, Metrics, Context 
from typing import Callable, Dict, List, Optional, Tuple, Union
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
import numpy as np
from fl.client import FlowerClient

def split_mnist_dirichlet_flwr(num_clients=NUM_CLIENTS, alpha=0.5, seed=42):
    partitioner = DirichletPartitioner(
        num_partitions=num_clients, partition_by="label", alpha=alpha, seed=seed
    )
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": partitioner})
    federated_data = {f"client_{i}": fds.load_partition(i) for i in range(num_clients)}
    return fds, federated_data


def get_dataloader(client_data, batch_size=BATCH_SIZE):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    x_tensor = torch.stack([transform(img) for img in client_data["img"]])
    y_tensor = torch.tensor(client_data["label"], dtype=torch.long)
    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def get_dataloader_Cifar10(client_data, batch_size=BATCH_SIZE):
    transform = transforms.Compose([
        transforms.Lambda(lambda x: Image.fromarray((np.array(x) * 255).astype(np.uint8)).convert("RGB")),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    x_tensor = torch.stack([transform(img) for img in client_data["img"]])
    y_tensor = torch.tensor(client_data["label"], dtype=torch.long)

    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)



def split_mnist_dirichlet_flwr_cifar_10(num_clients=NUM_CLIENTS, alpha=0.5, seed=42):
    partitioner = DirichletPartitioner(
        num_partitions=num_clients, partition_by="label", alpha=alpha, seed=seed
    )
    fds = FederatedDataset(dataset="cifar_10", partitioners={"train": partitioner})
    federated_data = {f"client_{i}": fds.load_partition(i) for i in range(num_clients)}
    return fds, federated_data

def get_client_fn(dataset: FederatedDataset,federated_data: Dict[str, List[Dict[str, torch.Tensor]]]) -> Callable[[Context], fl.client.Client]:
    def client_fn(context: Context) -> fl.client.Client:
        # Lấy dữ liệu của client từ dataset
        partition_id = context.node_config["partition-id"]

        # Kiểm tra nếu partition_id hợp lệ
        if f"client_{partition_id}" not in federated_data:
            raise ValueError(f"Client ID {partition_id} không tồn tại trong dữ liệu!")
        client_data = federated_data[f"client_{partition_id}"]
        train_loader = get_dataloader_Cifar10(client_data)
        test_loader = get_dataloader_Cifar10(client_data)
        model = ResNet50()

        # Trả về client
        return FlowerClient (model, train_loader, test_loader).to_client()

    return client_fn
