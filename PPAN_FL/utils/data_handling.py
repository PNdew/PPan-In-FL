# import torch
# from torchvision import transforms
# from torch.utils.data import DataLoader, TensorDataset
# from flwr_datasets import FederatedDataset
# from flwr_datasets.partitioner import DirichletPartitioner

# def split_mnist_dirichlet_flwr(num_clients=100, alpha=0.5, seed=42):
#     partitioner = DirichletPartitioner(
#         num_partitions=num_clients, partition_by="label", alpha=alpha, seed=seed
#     )
#     fds = FederatedDataset(dataset="fashion_mnist", partitioners={"train": partitioner})
#     federated_data = {f"client_{i}": fds.load_partition(i) for i in range(num_clients)}
#     return fds, federated_data

# def get_dataloader(client_data, batch_size=16):
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.1307,), (0.3081,))
#     ])
#     x_tensor = torch.stack([transform(img) for img in client_data["image"]])
#     y_tensor = torch.tensor(client_data["label"], dtype=torch.long)
#     dataset = TensorDataset(x_tensor, y_tensor)
#     return DataLoader(dataset, batch_size=batch_size, shuffle=True)

import torch
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner

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
    x_tensor = torch.stack([transform(img) for img in client_data["img"]])
    y_tensor = torch.tensor(client_data["label"], dtype=torch.long)
    
    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )