
import torch


# Constants
NUM_CLIENTS = 100
K_CLIENTS = 10
BATCH_SIZE = 16
NUM_ROUNDS = 200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 0.1

# Privacy parameters
INITIAL_EPSILON = 2.0
MIN_EPSILON = 0.1
MAX_EPSILON = 5.0
TARGET_ACCT = 0.9
ADJUST_RATE = 0.05
NOISE_CLIP = 0.01
WINDOW_SIZE = 3
