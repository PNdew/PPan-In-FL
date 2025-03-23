import logging
import torch

# Constants
NUM_CLIENTS = 10
LOCAL_EPOCHS = 5
BATCH_SIZE = 32
NUM_ROUNDS = 5
LEARNING_RATE = 0.001
EPSILON = [0.1, 0.1, 0.8]  # Privacy budget distribution
CLIP_BOUND = 1.0
SENSITIVITY = 2.0
DIRICHLET_ALPHA = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('dp_fedasgp.log')
    ]
)
logger = logging.getLogger(__name__)