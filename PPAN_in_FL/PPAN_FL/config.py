import torch
import os

# Training parameters
NUM_CLIENTS = 100
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 0.1
PRIVACY_WEIGHT = 0.01
NUM_ROUNDS = 200
NOISE_SCALE = 0.01

# Results directory
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)