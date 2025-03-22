import flwr as fl
import threading
import random
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
from flwr.server.criterion import Criterion

class SimpleClientManager(ClientManager):
    def __init__(self):
        self.clients = {}
        self._cv = threading.Condition()
        self.seed = 0

    def __len__(self):
        return len(self.clients)

    def num_available(self):
        return len(self)

    def wait_for(self, num_clients, timeout=86400):
        with self._cv:
            return self._cv.wait_for(lambda: len(self.clients) >= num_clients, timeout=timeout)

    def register(self, client):
        if client.cid in self.clients:
            return False
        self.clients[client.cid] = client
        with self._cv:
            self._cv.notify_all()
        return True

    def unregister(self, client):
        if client.cid in self.clients:
            del self.clients[client.cid]
            with self._cv:
                self._cv.notify_all()

    def all(self):
        return self.clients

    def sample(self, num_clients, min_num_clients=None, criterion=None):
        if min_num_clients is None:
            min_num_clients = num_clients
        self.wait_for(min_num_clients)
        available_cids = list(self.clients)
        sampled_cids = random.sample(available_cids, num_clients)
        self.seed += 1
        return [self.clients[cid] for cid in sampled_cids]