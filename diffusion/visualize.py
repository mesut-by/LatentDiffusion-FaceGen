import torch
import random
import matplotlib.pyplot as plt

# Visualize original, VAE reconstruction, noisy output, and denoised UNet₂D output
def visualize_unet2d_output(
    vae, diffusion_model,
    dataloader, scheduler, device, current_epoch=0, max_epochs=1,
    sampling_steps=300
):
    vae.eval();  diffusion_model.eval();

    dataset = dataloader.dataset
    idx = random.randint(0, len(dataset) - 1)
    orig = dataset[idx].to(device)  # Select a random input image

    with torch.no_grad():
        _, _, latent, skips = vae.encoder(orig.unsqueeze(0))  # Encode input image
        vae_recon = vae.decoder(latent, skips).squeeze(0)  # Reconstruct using VAE

        # Apply noise at a scheduled timestep
        t = t_strategy_v3(current_epoch, max_epochs, 1, T=scheduler.num_timesteps, device=device)
        x_noisy, noise = scheduler.q_sample(latent, t)

        noisy_dec = vae.decoder(x_noisy, skips).squeeze(0)  # Decode noisy latent

        # Apply denoising using UNet₂D
        denoised_latent = scheduler.denoising_loop(
            diffusion_model, x_noisy, eta=0.0, num_steps=sampling_steps
        )

        # Match latent stats to improve decoding
        denoised_latent_fixed = denoised_latent / (denoised_latent.std() + 1e-8) * (latent.std() + 1e-8)
        denoised_latent_fixed = torch.clamp(denoised_latent_fixed, latent.min().item(), latent.max().item())

        unet_dec = vae.decoder(denoised_latent_fixed, skips).squeeze(0)  # Final decoded output

    def norm(x): return torch.clamp((x + 1) / 2, 0, 1)  # Normalize to [0, 1]
    imgs = [orig, vae_recon, noisy_dec, unet_dec]
    titles = [
        "Original",
        "VAE",
        "Decoder(x_noisy)",
        f"UNet₂D+Decoder ({sampling_steps} steps)"
    ]

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    for ax, img, t in zip(axes, imgs, titles):
        ax.imshow(norm(img).permute(1, 2, 0).cpu().numpy())
        ax.set_title(t)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


# Generate and visualize a sample using DDIM scheduler and UNet₂D
def generate_and_visualize_ddim_sample(decoder, diffusion_model, scheduler, device, latent_shape=(1, 256, 64, 64), eta=0.0, title_suffix=""):
    decoder.eval()
    diffusion_model.eval()

    with torch.no_grad():
        sampled_latent = scheduler.ddim_sampling_loop(
            model=diffusion_model,
            shape=latent_shape,
            eta=eta
        )

        recon_image = decoder(sampled_latent, skips=None)  # Decode sampled latent
        recon_image = recon_image.squeeze(0) if recon_image.shape[0] == 1 else recon_image

    def norm(x):
        return torch.clamp((x + 1) / 2.0, 0, 1)  # Normalize to [0, 1]

    recon_image = norm(recon_image)

    plt.figure(figsize=(6, 6))
    plt.imshow(recon_image.permute(1, 2, 0).cpu().numpy())
    plt.title(f"DDIM Sampled Output (eta={eta}) {title_suffix}")
    plt.axis("off")
    plt.show()