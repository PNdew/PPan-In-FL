import torch
import os

# Training parameters
NUM_CLIENTS = 100
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 0.01
PRIVACY_WEIGHT = [500,200,100,10,1,0.1,0.01,0.001]
NUM_ROUNDS = 300
eplison = 0.1
delta = 1e-5
sensitivity = 1.0

# Results directory
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
