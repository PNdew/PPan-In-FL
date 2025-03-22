
import torch


# Constants
NUM_CLIENTS = 100
K_CLIENTS = 10
BATCH_SIZE = 16
NUM_ROUNDS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 0.001

# Privacy parameters
INITIAL_EPSILON = 2.0
MIN_EPSILON = 0.1
MAX_EPSILON = 5.0
TARGET_ACCURACY = 0.85
ADJUST_RATE = 0.05
NOISE_CLIP = 0.1
WINDOW_SIZE = 3
