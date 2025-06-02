import os
import logging  

VERBOSE = 0
NUM_CLIENTS = 100
BATCH_SIZE = 32
NUM_ROUNDS = 1000
CLIP_NORM = 14.142135623730953
NOISE_MULTIPLIER = 0.01  # Có thể thay đổi
LEARNING_RATE = 0.01

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
