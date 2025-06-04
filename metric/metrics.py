import numpy as np
from sklearn.feature_selection import mutual_info_regression

def compute_privacy_leakage(encrypted_weights, original_weights):
    encrypted_weights = np.array(encrypted_weights)
    original_weights = np.array(original_weights)
    encrypted_2d = encrypted_weights.reshape(-1, 1)
    original_2d = original_weights.reshape(-1, 1)
    min_length = min(len(encrypted_2d), len(original_2d))
    encrypted_2d = encrypted_2d[:min_length]
    original_2d = original_2d[:min_length]
    try:
        mi_score = mutual_info_regression(encrypted_2d, original_2d.ravel())[0]
    except ValueError:
        mi_score = 0.0
    return mi_score

def compute_distortion(original_weights, encrypted_weights):
    original_weights = np.array(original_weights)
    encrypted_weights = np.array(encrypted_weights)
    return np.mean((original_weights - encrypted_weights) ** 2)