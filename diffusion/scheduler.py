# DiffusionScheduler class
# This module implements the core logic for training denoising diffusion probabilistic
# models (DDPM) by defining a linear beta schedule for noise addition and supporting
# deterministic DDIM sampling steps.
# It provides utilities for:
# - forward noise scheduling (q_sample),
# - uniform and weighted noise prediction loss computation,
# - posterior mean and variance calculations for reverse diffusion steps,
# - DDIM sampling loop and denoising loop.
# The scheduler uses a linear beta schedule and controls how noise is added to or removed
# from latent representations. It supports time-dependent loss weighting to enhance
# stability and performance in later timesteps.


import torch
import torch.nn.functional as F

class DiffusionScheduler:
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02, device='cuda'):
        # Define the noise schedule (linear beta schedule)
        self.device = device
        self.num_timesteps = num_timesteps

        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0).to(device)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars).to(device)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars).to(device)

    # Add noise to a clean image at timestep t
    def q_sample(self, x_start, t):
        t = t.long()
        alpha_bar_t = self.sqrt_alpha_bars[t].view(-1, 1, 1, 1)
        one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1, 1)
        noise = torch.randn_like(x_start)
        x_noisy = alpha_bar_t * x_start + one_minus_alpha_bar_t * noise
        return x_noisy, noise

    # Return cumulative product of alpha at timestep t
    def get_alpha_bar(self, t):
        t = t.long()
        return self.alpha_bars[t].view(-1, 1, 1, 1)

    # Compute MSE loss between predicted noise and actual noise (uniform)
    def loss_uniform(self, model, x_start, t):
        x_noisy, noise = self.q_sample(x_start, t)
        pred = model(x_noisy, t.float())  # Predict noise
        return F.mse_loss(pred, noise)

    # Compute weighted loss based on alpha_bar importance
    def loss_weighted(self, model, x_start, t):
        x_noisy, noise = self.q_sample(x_start, t)
        pred = model(x_noisy, t.float())
        mse = F.mse_loss(pred, noise, reduction='none')  # Pixel-wise MSE
        mse_per_image = mse.view(mse.size(0), -1).mean(dim=1)
        alpha_bar = self.get_alpha_bar(t).view(-1)
        w = torch.sqrt(1.0 - alpha_bar + 1e-8)  # Weight by timestep uncertainty
        w = w / (w.mean() + 1e-8)
        return (w * mse_per_image).mean()

    # Switch between uniform and weighted loss during training
    def loss_fn(self, model, x_start, t, epoch, warmup_epochs=5):
        if epoch < warmup_epochs:
            return self.loss_uniform(model, x_start, t)
        else:
            return self.loss_weighted(model, x_start, t)

    # Compute posterior mean and variance (used in DDPM sampling)
    def posterior_mean_variance(self, x_start, x_t, t):
        beta_t = self.betas[t].view(-1, 1, 1, 1)
        alpha_t = self.alphas[t].view(-1, 1, 1, 1)
        alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1, 1)
        alpha_bar_prev = self.alpha_bars[torch.clamp(t - 1, min=0)].view(-1, 1, 1, 1)

        # Compute variance and mean for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t + 1e-8)
        posterior_mean = (
            torch.sqrt(alpha_bar_prev) * beta_t / (1.0 - alpha_bar_t + 1e-8) * x_start +
            torch.sqrt(alpha_t) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t + 1e-8) * x_t
        )
        return posterior_mean, posterior_variance

    # Perform one step of DDIM sampling (reverse step)
    def ddim_sample_step(self, model, x_t, t, t_prev, eta=0.0):
        alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1, 1)
        alpha_bar_prev = self.alpha_bars[t_prev].view(-1, 1, 1, 1)
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)

        predicted_noise = model(x_t, t.float())  # Predict noise
        x0_pred = (x_t - sqrt_one_minus_alpha_bar_t * predicted_noise) / sqrt_alpha_bar_t
        x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

        # Add noise if eta > 0 (stochastic DDIM)
        sigma = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev + 1e-8))
        noise = torch.randn_like(x_t) if eta > 0 else 0

        # Compute x_{t-1}
        x_prev = torch.sqrt(alpha_bar_prev) * x0_pred + \
                 torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * predicted_noise + \
                 sigma * noise
        return x_prev, x0_pred

    # Perform full DDIM sampling loop from pure noise
    def ddim_sampling_loop(self, model, shape, eta=0.0):
        x = torch.randn(shape, device=self.device)
        time_schedule = torch.linspace(self.num_timesteps - 1, 0, steps=self.num_timesteps, dtype=torch.long, device=self.device)
        for i in range(self.num_timesteps - 1):
            t = time_schedule[i].expand(shape[0])
            t_prev = time_schedule[i + 1].expand(shape[0])
            x, _ = self.ddim_sample_step(model, x, t, t_prev, eta=eta)
        return x

    # Denoise a given noisy latent over specified steps
    def denoising_loop(self, model, x_noisy, eta=0.0, num_steps=50):
        device = x_noisy.device
        b = x_noisy.size(0)
        img = x_noisy
        t_array = torch.linspace(self.num_timesteps - 1, 0, steps=num_steps, device=device).long()
        for i in range(num_steps):
            t = t_array[i].repeat(b)
            t_prev = t_array[i + 1].repeat(b) if i < num_steps - 1 else torch.zeros_like(t)
            img, _ = self.ddim_sample_step(model, img, t, t_prev, eta=eta)
        return img