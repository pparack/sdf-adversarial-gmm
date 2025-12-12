import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm

class SDFNet(nn.Module):
    """Positive SDF with mean normalization E[m]=1 (batch-wise)."""
    def __init__(self, input_dim, hidden=20, drop=0.2, eps=1e-6):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden, 1)
        self.eps = eps

    def forward(self, x):
        h = self.drop(self.relu(self.fc1(x)))
        m_raw = torch.exp(self.fc2(h)) + self.eps
        m = m_raw / (m_raw.mean(dim=0, keepdim=True) + self.eps)
        return m  # (B,1)

class CriticNet(nn.Module):
    """Critic g(x,y): spectral norm + tanh + standardization."""
    def __init__(self, input_dim, hidden=32, dropout=0.2, use_spectral_norm=True):
        super().__init__()
        Linear = (lambda *args, **kw: spectral_norm(nn.Linear(*args, **kw))) if use_spectral_norm else nn.Linear
        self.fc1 = Linear(input_dim + 1, hidden)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = Linear(hidden, 1)

    def forward(self, x, y):
        y = y.view(-1, 1)
        h = torch.cat([x, y], dim=1)
        h = self.drop(self.act(self.fc1(h)))
        g = self.fc2(h)
        g = torch.tanh(g)
        g = g - g.mean(dim=0, keepdim=True)
        g = g / (g.std(unbiased=False) + 1e-6)
        return g
