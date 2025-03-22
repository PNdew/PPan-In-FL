import tensorflow as tf
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner

def split_mnist_dirichlet_flwr(num_clients=NUM_CLIENTS, alpha=0.5, seed=42):
    partitioner = DirichletPartitioner(
        num_partitions=num_clients, partition_by="label", alpha=alpha, seed=seed
    )
    fds = FederatedDataset(dataset="mnist", partitioners={"train": partitioner})
    federated_data = {f"client_{i}": fds.load_partition(i) for i in range(num_clients)}
    return fds, federated_data  # Return both fds and federated_data

def get_client_fn(dataset: FederatedDataset):
    def client_fn(cid: str) -> fl.client.Client:
        client_dataset = dataset.load_partition(int(cid), "train")
        splits = client_dataset.train_test_split(test_size=0.1)

        # Add reshape to ensure correct dimensions (batch_size, height, width, channels)
        trainset = splits["train"].to_tf_dataset(
            columns="image",
            label_cols="label",
            batch_size=BATCH_SIZE
        ).map(lambda x, y: (tf.expand_dims(x, axis=-1), y)).prefetch(tf.data.AUTOTUNE)

        valset = splits["test"].to_tf_dataset(
            columns="image",
            label_cols="label",
            batch_size=BATCH_SIZE*2
        ).map(lambda x, y: (tf.expand_dims(x, axis=-1), y)).prefetch(tf.data.AUTOTUNE)

        return FlowerClient(cid, trainset, valset).to_client()
    return client_fn