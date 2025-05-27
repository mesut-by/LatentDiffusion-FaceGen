
---

# Latent Diffusion Model Project for High-Quality Human Face Generation (Under Development)

This project integrates a **VAE + Diffusion** architecture to generate high-quality human faces in the latent space using a Latent Diffusion Model (LDM). The two-stage architecture:

1. **VAE:** Learns a latent representation by compressing and reconstructing images.
2. **Diffusion (UNet₂D-based):** Iteratively removes noise in the latent space to produce realistic samples.

In future stages, it will be extended into a full text-to-image generation system by integrating CLIP.

---

## Table of Contents

* [Project Overview](#project-overview)
* [Installation & Requirements](#installation--requirements)
* [Project Structure](#project-structure)
* [VAE Model Details](#vae-model-details)
* [Diffusion Model Details](#diffusion-model-details)

  * [Time Embedding & t-Strategy](#time-embedding--t-strategy)
  * [Diffusion Scheduler](#diffusion-scheduler)
  * [UNet₂D Architecture](#unet₂d-architecture)
  * [Training Configuration](#training-configuration)
* [Training & Usage](#training--usage)
* [Visualization & Evaluation](#visualization--evaluation)
* [Future Plans](#future-plans)
* [Additional Notes](#additional-notes)
* [Dataset License](#dataset-license)
* [Acknowledgements](#acknowledgements)

---

## Project Overview

The goal of this LDM project is to generate high-quality human faces through **UNet₂D-based Diffusion** steps applied on latent representations learned by a VAE.

* The VAE compresses and reconstructs images, producing a mean/logvar distribution in the latent space.
* The Diffusion module adds and removes noise step-by-step in the latent space.

---

## Installation & Requirements

1. **Environment:**

   * GPU-supported environment (e.g., Colab or local CUDA).
   * Python ≥3.8.

2. **Required Packages:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Dataset Download & Preparation:**

   ```bash
   kaggle datasets download -d arnaud58/flickrfaceshq-dataset-ffhq
   unzip flickrfaceshq* -d ./data/ffhq
   ```

4. **Configuration (`config.py`):**

   * Adjust values for `batch_size`, `epochs`, `lr`, `kl_weight`, etc.
   * Image preprocessing: resizing, normalization.

---

## Project Structure

```
LatentDiffusion-FaceGen/
├── checkpoints/                  
│   ├── best_diff_model.pth
│   └── best_vae_model.pth
│
├── diffusion/                    
│   ├── model.py                 # UNet2D + all layers (Residual, Attention, Down/Up, Bottleneck)
│   ├── scheduler.py             # q_sample, denoising_loop, loss_fn, DDIM sampling
│   ├── time.py                  # TimeEmbedding, SinusoidalPosEmb, t_strategy_v3
│   ├── ema.py                   # EMA class
│   ├── visualize.py             # visualize_unet2d_output, generate_and_visualize_ddim_sample
│   └── training_diff.ipynb      # Diffusion training script  
│
├── vae/                          
│   ├── model_vae.py             # VAE, Encoder, Decoder
│   ├── evaluate_vae.py
│   └── training_vae.ipynb
│
├── test/                         
│   └── example.png
│
├── config.py                     # Hyperparameters (SimpleNamespace/dict-based)
├── requirements.txt
├── requirements.in               
├── README.md
├── CONTRIBUTING.md
└── .gitignore
```

---

## VAE Model Details

* **Encoder:**

  * Downsample, ResidualBlock, AttentionBlock, `mean` + `logvar`
  * Reparameterization trick for sampling the latent vector

* **Decoder:**

  * Upsample, Skip Connections, Tanh activation

* **Loss Functions:**

  * MSE Loss (reconstruction)
  * KL Divergence (latent regularization)
  * KL Annealing (gradual increase of the KL weight)

* **Additional Techniques:**

  * Gradient Clipping
  * Early Stopping
  * Learning Rate Scheduler

---

## Diffusion Model Details

In the second stage of the LDM, a UNet₂D-based diffusion module operates on the latent samples produced by the VAE.

### Time Embedding & t-Strategy

* **SinusoidalPosEmb:** Generates positional embeddings for time steps.
* **TimeEmbedding:** Enhances embeddings via an MLP.
* **t\_strategy\_v3:** Dynamically sets `t` values from low to high based on the epoch.

### Diffusion Scheduler

| Function                  | Description                           |
| ------------------------- | ------------------------------------- |
| `q_sample`                | Adds noise to latent samples          |
| `loss_uniform`            | Uniform MSE loss                      |
| `loss_weighted`           | √-weighted MSE                        |
| `loss_fn`                 | Chooses the appropriate loss by epoch |
| `posterior_mean_variance` | Predicts mean and variance for DDIM   |
| `ddim_sample_step`        | Single-step DDIM sampling             |
| `ddim_sampling_loop`      | End-to-end DDIM sample generation     |
| `denoising_loop`          | x\_noisy → x₀ prediction              |

### UNet₂D Architecture

```
[Latent] → Downsample_diff → ResidualBlock_diff → BottleneckBlock → SelfAttention2D → Upsample_diff → [Denoised Latent]
```

* **ResidualBlock\_diff:** Residual layer with time embedding support
* **SelfAttention2D:** Standard convolutional attention
* **Downsample\_diff / Upsample\_diff:** Resolution transition blocks
* **BottleneckBlock:** Central block with dilated conv and attention

### Training Configuration

| Setting         | Values                                                         |
| --------------- | -------------------------------------------------------------- |
| Optimizer       | Adam(lr=2e-4, betas=(0.9, 0.999))                              |
| EMA             | decay=0.999                                                    |
| GradScaler      | Mixed precision (autocast + GradScaler)                        |
| Early Stopping  | patience=4, min\_delta=1e-4                                    |
| LR Scheduler    | 10% warm-up → cosine decay                                     |
| Loss Transition | Uniform MSE for the first 5 epochs → √-weighted MSE afterwards |

* **Monitoring Hooks:**

  * `stats`: Layer output std values
  * `grads`: Gradient norms

---

## Training & Usage

1. **VAE Training:**

   * Run `training_vae.ipynb` to train the VAE on FFHQ.
   * The best weights are saved to `checkpoints/best_vae_model.pth`.

2. **Diffusion Training:**

   * Use `diffusion/model.py`, `diffusion/scheduler.py`, `diffusion/time.py`, and `diffusion/ema.py` to start the training loop.
   * EMA, GradScaler, LR scheduler, and EarlyStopping are integrated.
   * The best diffusion weights are saved to `checkpoints/best_diff_model.pth`.

3. **Latent Sample Generation:**

   ```python
   from diffusion.visualize import generate_and_visualize_ddim_sample

   generate_and_visualize_ddim_sample(
       decoder=vae.decoder,
       diffusion_model=diff_model,
       scheduler=scheduler,
       device=device,
       latent_shape=(1, 256, 64, 64),
       eta=0.0, title_suffix="Deterministic"
   )
   ```

---

## Visualization & Evaluation

* **`visualize_unet2d_output(...)`:**

  * Shows original image, VAE reconstruction, x\_noisy → decoder output, and final result.

* **Metrics:**

  * **MSE, SSIM, PSNR** calculation → `calculate_metrics.py`
  * Loss curves → `plot_losses.py`
  * Intermediate std/gradient tracking → `track_tensor_values`

---

## Future Plans

* More advanced attention mechanisms
* Progressive denoising strategies
* Additional quality metrics like FID and LPIPS
* CLIP-based text-guided latent search

---

## Additional Notes

* Always mark code changes with comments.
* Configuration changes in `config.py` affect model behavior.
* Inspect logs and visual outputs first when debugging.

---

## Dataset License

This project uses the **FFHQ** dataset under the **CC BY-SA 3.0** license. Please comply with the license terms.

---

## Acknowledgements

Thanks to everyone who contributed to this project and to the open-source community. Stay tuned on GitHub for updates and collaboration.
