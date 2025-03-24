import logging
import torch

# Constants
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('dp_fedasgp.log')
    ]
)
logger = logging.getLogger(__name__)
# Constants for MNIST and FL training
# Constants for MNIST and FL training
NUM_CLIENTS = 10
LOCAL_EPOCHS = 5
BATCH_SIZE = 32
NUM_ROUNDS = 10
LEARNING_RATE = 0.001

# Privacy parameters
INITIAL_EPSILON = 1.0  # Initial privacy budget
MIN_EPSILON = 0.1     # Minimum privacy budget
MAX_EPSILON = 2.0     # Maximum privacy budget
EPSILON = [0.1, 0.1, 0.8]  # Privacy budget distribution
CLIP_BOUND = 1.0      # Gradient clipping bound
SENSITIVITY = 2.0     # Privacy sensitivity
ALPHA = 0.05         # α parameter for CLDP
DIRICHLET_ALPHA = 0.5  # Dirichlet distribution parameter

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
