import torch
import copy

# Exponential Moving Average (EMA) model wrapper.
# Maintains a shadow copy of the model whose weights are updated as a moving average
#   of the original model's weights. This helps stabilize training and often leads to better generalization at inference time.
# EMA model is kept in eval mode  and is not updated by backpropagation.

class EMA:
    def __init__(self, model, decay=0.999):
        self.ema_model = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.ema_model.parameters():
            p.requires_grad = False  # Prevent gradient computation for EMA model

    def update(self, model):
        with torch.no_grad():
            for ema_p, p in zip(self.ema_model.parameters(), model.parameters()):
                ema_p.data.mul_(self.decay).add_(p.data, alpha=1 - self.decay)  # Update EMA weights
            for ema_buf, buf in zip(self.ema_model.buffers(), model.buffers()):
                ema_buf.copy_(buf)  # Copy non-parameter buffers (e.g., running stats)

    def to(self, device):
        self.ema_model.to(device)  # Move EMA model to specified device