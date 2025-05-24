from typing import Dict, List, Tuple
import tensorflow as tf
import flwr as fl
import numpy as np
from flwr.common import Metrics
from flwr.simulation.ray_transport.utils import enable_tf_gpu_growth
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from typing import Optional, Union
import flwr as fl
from flwr.common import Parameters, NDArrays
from flwr.server.client_proxy import ClientProxy
from flwr.common.typing import Config
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms, datasets
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, Parameters, Scalar
from flwr.common.dp import add_gaussian_noise
from flwr.common.logger import warn_deprecated_feature
from flwr.common.parameter import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import DPFedAvgFixed
from sklearn.feature_selection import mutual_info_regression
from typing import Dict, List, Optional
from config import *


def split_mnist_dirichlet_flwr(num_clients=100, alpha=0.5, seed=42):
    partitioner = DirichletPartitioner(
        num_partitions=num_clients, 
        partition_by="label", 
        alpha=alpha, 
        seed=seed
    )
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": partitioner})
    federated_data = {f"client_{i}": fds.load_partition(i) for i in range(num_clients)}
    return fds, federated_data

def get_dataloader(client_data, batch_size=16):
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

def load_data(num_clients: int):
    """Load and preprocess CIFAR-10 data for federated learning"""
    try:
        _, federated_data = split_mnist_dirichlet_flwr(num_clients)
        client_loaders = {}
        
        for i in range(num_clients):
            try:
                partition = federated_data[f"client_{i}"]
                dataloader = get_dataloader(partition)
                client_loaders[i] = dataloader
                
                # Log progress
                if i % 10 == 0:
                    print(f"Processed {i}/{num_clients} clients")
                    
            except Exception as e:
                print(f"Error processing client {i}: {str(e)}")
                raise
                
        return client_loaders
        
    except Exception as e:
        print(f"Error in load_data: {str(e)}")
        raise
