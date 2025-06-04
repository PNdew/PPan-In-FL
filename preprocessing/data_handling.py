import torch
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from PIL import Image
import numpy as np
from PPAN_FL.config import NUM_CLIENTS, BATCH_SIZE

def split_mnist_dirichlet_flwr(num_clients=100, alpha=0.5, seed=42):
    partitioner = DirichletPartitioner(
        num_partitions=num_clients, partition_by="label", alpha=alpha, seed=seed
    )
    fds = FederatedDataset(dataset="fashion_mnist", partitioners={"train": partitioner})
    federated_data = {f"client_{i}": fds.load_partition(i) for i in range(num_clients)}
    return fds, federated_data

def get_dataloader(client_data, batch_size=16):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    x_tensor = torch.stack([transform(img) for img in client_data["image"]])
    y_tensor = torch.tensor(client_data["label"], dtype=torch.long)
    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


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