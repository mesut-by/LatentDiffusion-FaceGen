import torch
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

# Tracks min, max, and mean values through encoder and decoder blocks
# Helps monitor numerical behavior and diagnose training instability
def track_tensor_values(model, dataloader, device):
    model.eval()
    dataset = dataloader.dataset
    idx = random.randint(0, len(dataset) - 1)
    original = dataset[idx].to(device).unsqueeze(0)

    print("### Tracking Tensor Values ###")
    print(f"Input        : min={original.min():.4f}, max={original.max():.4f}, mean={original.mean():.4f}")

    with torch.no_grad():
        encoder_layers = list(model.encoder.encoder.children())

        e1 = encoder_layers[0](original)
        print(f"Block1       : min={e1.min():.4f}, max={e1.max():.4f}, mean={e1.mean():.4f}")

        e2 = encoder_layers[1](e1)
        print(f"Block2       : min={e2.min():.4f}, max={e2.max():.4f}, mean={e2.mean():.4f}")

        e3 = encoder_layers[2](e2)
        e4 = encoder_layers[3](e3)
        print(f"Block3       : min={e4.min():.4f}, max={e4.max():.4f}, mean={e4.mean():.4f}")

        e5 = encoder_layers[4](e4)
        e6 = encoder_layers[5](e5)
        e7 = encoder_layers[6](e6)
        print(f"Block4       : min={e7.min():.4f}, max={e7.max():.4f}, mean={e7.mean():.4f}")

        e8 = encoder_layers[7](e7)
        e9 = encoder_layers[8](e8)
        e10 = encoder_layers[9](e9)
        print(f"Block5       : min={e10.min():.4f}, max={e10.max():.4f}, mean={e10.mean():.4f}")

        e11 = encoder_layers[10](e10)
        e12 = encoder_layers[11](e11)
        e13 = encoder_layers[12](e12)
        e14 = encoder_layers[13](e13)
        e15 = encoder_layers[14](e14)
        e16 = encoder_layers[15](e15)
        e17 = encoder_layers[16](e16)
        print(f"Compressed   : min={e17.min():.4f}, max={e17.max():.4f}, mean={e17.mean():.4f}")

        mean, logvar = torch.chunk(e17, 2, dim=1)
        print(f"Mean         : min={mean.min():.4f}, max={mean.max():.4f}, mean={mean.mean():.4f}")
        print(f"LogVar       : min={logvar.min():.4f}, max={logvar.max():.4f}, mean={logvar.mean():.4f}")

        sampled = torch.tanh(mean + logvar.exp().sqrt() * torch.randn_like(mean)) * model.encoder.scale
        print(f"Sampled      : min={sampled.min():.4f}, max={sampled.max():.4f}, mean={sampled.mean():.4f}")

        decoder_layers = list(model.decoder.decoder.children())

        d1 = decoder_layers[0](sampled)
        d2 = decoder_layers[1](d1)
        print(f"Dec-Block1   : min={d2.min():.4f}, max={d2.max():.4f}, mean={d2.mean():.4f}")

        d3 = decoder_layers[2](d2)
        d4 = decoder_layers[3](d3)
        d5 = decoder_layers[4](d4)
        d6 = decoder_layers[5](d5)
        print(f"Dec-Block2   : min={d6.min():.4f}, max={d6.max():.4f}, mean={d6.mean():.4f}")

        d7 = decoder_layers[6](d6)
        d8 = decoder_layers[7](d7)
        d9 = decoder_layers[8](d8)
        print(f"Dec-Block3   : min={d9.min():.4f}, max={d9.max():.4f}, mean={d9.mean():.4f}")

        d10 = decoder_layers[9](d9)
        d11 = decoder_layers[10](d10)
        d12 = decoder_layers[11](d11)
        d13 = decoder_layers[12](d12)
        d14 = decoder_layers[13](d13)
        d15 = decoder_layers[14](d14)
        d16 = decoder_layers[15](d15)
        d17 = decoder_layers[16](d16)
        print(f"Output       : min={d17.min():.4f}, max={d17.max():.4f}, mean={d17.mean():.4f}")

# Calculates MSE, SSIM, and PSNR for model evaluation
# Used to quantitatively measure reconstruction quality
from skimage.metrics import structural_similarity as ssim
import torch.nn.functional as F
import numpy as np

def calculate_metrics(model, dataloader, device):
    """
    Calculates MSE, SSIM, and PSNR metrics for the given model and dataloader.
    Automatically prints the evaluation and interpretation.
    """
    model.eval()
    mse_list = []
    ssim_list = []
    psnr_list = []

    with torch.no_grad():
        for data in dataloader:
            original_image = data.to(device)
            _, _, latent, skips = model.encoder(original_image)
            reconstructed_image = model.decoder(latent, skips)

            original_image_norm = torch.clamp((original_image + 1) / 2.0, 0, 1)
            reconstructed_image_norm = torch.clamp((reconstructed_image + 1) / 2.0, 0, 1)

            mse = F.mse_loss(reconstructed_image_norm, original_image_norm).item()
            mse_list.append(mse)

            original_np = original_image_norm.permute(0, 2, 3, 1).cpu().numpy()
            reconstructed_np = reconstructed_image_norm.permute(0, 2, 3, 1).cpu().numpy()

            for i in range(original_np.shape[0]):
                ssim_value = ssim(
                    original_np[i],
                    reconstructed_np[i],
                    multichannel=True,
                    data_range=1.0,
                    win_size=3
                )
                psnr_value = 10 * np.log10(1 / mse)
                ssim_list.append(ssim_value)
                psnr_list.append(psnr_value)

    avg_mse = np.mean(mse_list)
    avg_ssim = np.mean(ssim_list)
    avg_psnr = np.mean(psnr_list)

    # Print results and interpretation
    print(f"Mean Squared Error (MSE): {avg_mse:.4f}")
    print(f"Structural Similarity Index (SSIM): {avg_ssim:.4f}")
    print(f"Peak Signal-to-Noise Ratio (PSNR): {avg_psnr:.4f}\n")

    print("""
Evaluation Ranges:
1. MSE:   0.0 - 0.1 (Good),   0.1 - 0.5 (Moderate),   0.5 - 1.0 (Low),   1.0 - 2.0 (Poor),   >2.0 (Very Poor)
2. SSIM:  0.9 - 1.0 (Good),   0.7 - 0.9 (Moderate),   0.5 - 0.7 (Low),   0.3 - 0.5 (Poor),   <0.3 (Very Poor)
3. PSNR:  >30 (Good),   25 - 30 (Moderate),   20 - 25 (Low),   15 - 20 (Poor),   <15 (Very Poor)
""")

    return avg_mse, avg_ssim, avg_psnr

# Visualizes a random sample and its reconstruction
# Used to qualitatively assess VAE output quality
def visualize_random_input_output(model, dataloader, device):
    model.eval()
    dataset = dataloader.dataset
    idx = random.randint(0, len(dataset) - 1)
    orig = dataset[idx].to(device)

    with torch.no_grad():
        _, _, latent, skips = model.encoder(orig.unsqueeze(0))
        recon = model.decoder(latent, skips).squeeze(0)

    orig_disp = torch.clamp((orig + 1) / 2, 0, 1)
    recon_disp = torch.clamp((recon + 1) / 2, 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(10,5))
    axes[0].imshow(orig_disp.permute(1,2,0).cpu().numpy())
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(recon_disp.permute(1,2,0).cpu().numpy())
    axes[1].set_title("Reconstructed")
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()


# Plots training and validation loss curves
# Includes both epoch-based and batch-based visualizations
def plot_losses(loss_dict):
    plt.figure(figsize=(16, 9))
    plt.plot(loss_dict['train'], label='Train Loss (epoch)', color='orange', linewidth=3)
    plt.plot(loss_dict['val'], label='Validation Loss (epoch)', color='purple', linewidth=3)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Losses (Per Epoch)")
    plt.legend()
    plt.ylim(0, 0.4)
    plt.grid()
    plt.show()

    plt.figure(figsize=(16, 9))
    plt.plot(loss_dict['total'], label='Total Loss (batch)', color='blue')
    plt.plot(loss_dict['kl'], label='KL Loss (batch)', color='red')
    plt.plot(loss_dict['recon'], label='Recon Loss (batch)', color='green')
    plt.xlabel("Steps (batch)")
    plt.ylabel("Loss")
    plt.title("Batch Losses (Total, KL, Reconstruction)")
    plt.legend()
    plt.ylim(0, 0.4)
    plt.grid()
    plt.show()