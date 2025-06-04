import torch
import torch.nn as nn
from .ppan import PPAN_Encoder, PPAN_Adversary
from config import *

class PrivacyMechanism(nn.Module):
    def __init__(self, input_dim, noise_scale=NOISE_SCALE):
        super().__init__()
        self.encoder = PPAN_Encoder(input_dim, 64).to(DEVICE)
        self.adversary = PPAN_Adversary(input_dim, 64).to(DEVICE)
        self.noise_scale = noise_scale
        self.device = DEVICE

    def encrypt(self, x):
        x = x.to(self.device)               # Chuyển input lên đúng device
        generated = self.encoder(x)         # encoder đã ở đúng device rồi
        if self.training:
            noise = torch.randn_like(generated) * self.noise_scale
            generated = generated + noise
        return generated

    def decrypt(self, encrypted):
        encrypted = encrypted.to(self.device)  # Chuyển input lên device nếu chưa
        return self.adversary(encrypted)

    def forward(self, x):
        encrypted = self.encrypt(x)
        decoded = self.decrypt(encrypted)
        return encrypted, decoded

