import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from config import cfg

# Sinusoidal positional embedding for time steps
# Converts timestep t into a sinusoidal representation similar to transformer encodings
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        if half_dim == 0:
            raise ValueError("dim must be at least 2")
        emb_factor = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_factor)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        return emb


# Time embedding block using sinusoidal input and linear layers
class TimeEmbedding(nn.Module):
    def __init__(self, time_dim=cfg.time_dim, embed_dim=cfg.embed_dim):
        super().__init__()
        self.sinusoidal = SinusoidalPosEmb(time_dim)
        self.linear1 = nn.Linear(time_dim, embed_dim)
        self.linear2 = nn.Linear(embed_dim, embed_dim)

    def forward(self, t):
        emb = self.sinusoidal(t)           # Get sinusoidal embedding
        emb = F.relu(self.linear1(emb))    # Pass through first linear + ReLU
        emb = self.linear2(emb)            # Final linear projection
        return emb


# Dynamic timestep sampling strategy based on training epoch progress
def t_strategy_v3(epoch, max_epochs, batch_size, T=cfg.T, device=cfg.device):
    ratio = epoch / max_epochs
    if ratio < 0.3:
        t_min, t_max = 10, 300
    elif ratio < 0.5:
        t_min, t_max = 100, 500
    elif ratio < 0.7:
        t_min, t_max = 200, 700
    elif ratio < 0.9:
        t_min, t_max = 10, 900
    else:
        t_min, t_max = 10, T

    return torch.randint(t_min, t_max, (batch_size,), device=device).long()