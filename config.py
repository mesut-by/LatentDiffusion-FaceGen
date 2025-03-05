from types import SimpleNamespace

cfg = SimpleNamespace(
    # Training parameters
    epochs=30,
    batch_size=8,
    subset_size=20000,
    num_workers=4,

    # Model parameters
    latent_channels=256,
    in_channels=3,
    out_channels=3,
    num_groups=32,

    # Optimization
    optimizer_lr=1e-4,
    vae_lr=1e-4,
    kl_weight=0.3,
    min_kl_loss=0.08,

    # Scheduler and early stopping
    early_stopping_patience=12,
    scheduler_patience=5,
    scale_factor=0.34,
    scheduler_factor=0.4
)
