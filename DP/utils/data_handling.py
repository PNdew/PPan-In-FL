import tensorflow as tf
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from config import NUM_CLIENTS, BATCH_SIZE
from models.mnist_model import MNISTModel
from flwr.common import NDArrays
from fl.client import FlowerClient
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
from models.mnist_model import *

def split_mnist_dirichlet_flwr(num_clients=NUM_CLIENTS, alpha=0.5, seed=42):
    partitioner = DirichletPartitioner(
        num_partitions=num_clients,
        partition_by="label",
        alpha=alpha,
        seed=seed
    )
    # Changed dataset from fashion_mnist to cifar10
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": partitioner})
    federated_data = {f"client_{i}": fds.load_partition(i) for i in range(num_clients)}
    return fds, federated_data


def get_dataloader(client_data, batch_size=BATCH_SIZE):
    # CIFAR-10 specific transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],  # CIFAR-10 mean
            std=[0.2023, 0.1994, 0.2010]    # CIFAR-10 std
        ),
        transforms.Resize(224),  # For ResNet50
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
    ])
    
    # Transform and stack images (CIFAR-10 is RGB)
    x_tensor = torch.stack([transform(img) for img in client_data["image"]])
    y_tensor = torch.tensor(client_data["label"], dtype=torch.long)
    
    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )


def get_client_fn(dataset: FederatedDataset,federated_data: Dict[str, List[Dict[str, torch.Tensor]]]) -> Callable[[Context], fl.client.Client]:
    def client_fn(context: Context) -> fl.client.Client:
        # Lấy dữ liệu của client từ dataset
        partition_id = context.node_config["partition-id"]

        # Kiểm tra nếu partition_id hợp lệ
        if f"client_{partition_id}" not in federated_data:
            raise ValueError(f"Client ID {partition_id} không tồn tại trong dữ liệu!")
        client_data = federated_data[f"client_{partition_id}"]
        train_loader = get_dataloader(client_data)
        test_loader = get_dataloader(client_data)
        model = ResNet18()

        # Trả về client
        return FlowerClient (model, train_loader, test_loader).to_client()

    return client_fn