import torch
import torch.nn as nn
from .ppan import PPAN_Encoder, PPAN_Adversary
from config import NOISE_SCALE

class PrivacyMechanism(nn.Module):
    def __init__(self, input_dim, noise_scale=NOISE_SCALE):
        super().__init__()
        self.encoder = PPAN_Encoder(input_dim, 64)
        self.adversary = PPAN_Adversary(input_dim, 64)
        self.noise_scale = noise_scale

    def encrypt(self, x):
        generated = self.encoder(x)
        if self.training:
            noise = torch.randn_like(generated) * self.noise_scale
            generated = generated + noise
        return generated

    def decrypt(self, encrypted):
        return self.adversary(encrypted)

    def forward(self, x):
        encrypted = self.encrypt(x)
        decoded = self.decrypt(encrypted)
        return encrypted, decoded