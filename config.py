from types import SimpleNamespace
import torch

cfg = SimpleNamespace(
    # Model & Architecture Parameters
    embed_dim = 1280,
    time_dim = 1280,
    num_groups = 8,
    T = 1000,
    num_timesteps = 1000,

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),

    # Training Configuration
    epoch = 20,
    epoch_diff = 10,
    batch_size = 8,
    subset_size = 20000,
    num_workers = 4,

    # Model parameters
    latent_channels=4,
    in_channels=3,
    out_channels=3,

    # Optimization
    optimizer_lr=1e-4,
    vae_lr=1e-4,
    kl_weight=0.3,
    min_kl_loss=0.08,

    # Scheduler & Early Stopping
    early_stopping_patience = 15,
    scheduler_patience = 7,
    scheduler_factor = 0.4,
)

cfg.warmup_epochs = cfg.epoch_diff // 4

