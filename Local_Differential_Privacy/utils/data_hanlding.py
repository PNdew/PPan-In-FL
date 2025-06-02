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
from PIL import Image
from flwr.common.dp import add_gaussian_noise
from flwr.common.logger import warn_deprecated_feature
from flwr.common.parameter import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import DPFedAvgFixed
from sklearn.feature_selection import mutual_info_regression
from typing import Dict, List, Optional
from config import *


def split_mnist_dirichlet_flwr(num_clients=NUM_CLIENTS, alpha=0.5, seed=42):
    partitioner = DirichletPartitioner(
        num_partitions=num_clients, partition_by="label", alpha=alpha, seed=seed
    )
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": partitioner})
    federated_data = {f"client_{i}": fds.load_partition(i) for i in range(num_clients)}
    return fds, federated_data  # Trả về cả fds và dữ liệu phân vùng

# def get_dataloader(client_data, batch_size=BATCH_SIZE):
#     """Create DataLoader from client data with proper image handling"""
#     try:
#         # Convert PIL images to tensors
#         transform = transforms.Compose([
#             transforms.ToTensor(),  # Converts PIL Image to tensor and scales to [0,1]
#             transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
#         ])
        
#         # Handle images
#         images = []
#         for img in client_data["image"]:
#             # Convert to tensor and normalize
#             img_tensor = transform(img)
#             images.append(img_tensor)
            
#         # Stack all images into a single tensor
#         images = torch.stack(images)
        
#         # Convert labels to tensor
#         labels = torch.tensor(client_data["label"], dtype=torch.long)
        
#         # Create dataset and dataloader
#         dataset = TensorDataset(images, labels)
#         return DataLoader(
#             dataset,
#             batch_size=batch_size,
#             shuffle=True,
#             num_workers=0
#         )
        
#     except Exception as e:
#         print(f"Error in get_dataloader: {str(e)}")
#         raise

def get_dataloader_cifar10(client_data, batch_size=BATCH_SIZE):
    try:
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        images = []
        for img in client_data["img"]:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            img_tensor = transform(img)
            images.append(img_tensor)
            
        images = torch.stack(images)
        labels = torch.tensor(client_data["label"], dtype=torch.long)
        
        dataset = TensorDataset(images, labels)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    except Exception as e:
        print(f"Error in get_dataloader: {str(e)}")
        raise

def load_data(num_clients: int):
    """Load and preprocess MNIST data for federated learning"""
    try:
        _, federated_data = split_mnist_dirichlet_flwr(num_clients)
        client_loaders = {}
        
        for i in range(num_clients):
            try:
                partition = federated_data[f"client_{i}"]
                dataloader = get_dataloader_cifar10(partition)
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

