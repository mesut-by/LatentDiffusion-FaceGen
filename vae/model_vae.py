import torch
import torch.nn as nn
import torch.nn.functional as F

from config import cfg

# Multi-mode attention block designed for adaptive efficiency.
# Chooses CBAM, Linear, or Full attention based on input size and channels,
# ensuring optimal use of memory and computation at different feature map scales.

class AttentionBlock(nn.Module):
    def __init__(self, channels, heads=1, attention_type=None, max_full_res=64):
        super().__init__()
        self.channels = channels
        self.heads = heads
        self.max_full_res = max_full_res  # Limit for full attention resolution

        # Auto-select attention type if not provided
        if attention_type is not None:
            self.attention_type = attention_type
        elif channels == 64:
            self.attention_type = 'full'
        elif channels == 128:
            self.attention_type = 'linear'
        else:
            self.attention_type = 'cbam'

        if self.attention_type == 'cbam':
            # CBAM: Channel and spatial attention
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.max_pool = nn.AdaptiveMaxPool2d(1)
            self.shared_mlp = nn.Sequential(
                nn.Conv2d(channels, channels // 8, 1, bias=False),
                nn.ReLU(),
                nn.Conv2d(channels // 8, channels, 1, bias=False)
            )
            self.sigmoid_channel = nn.Sigmoid()
            self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
            self.sigmoid_spatial = nn.Sigmoid()

        elif self.attention_type == 'linear':
            # Linear attention using 1D projections
            self.norm = nn.GroupNorm(8, channels)
            self.q_proj = nn.Conv1d(channels, channels, kernel_size=1)
            self.k_proj = nn.Conv1d(channels, channels, kernel_size=1)
            self.v_proj = nn.Conv1d(channels, channels, kernel_size=1)
            self.out_proj = nn.Conv1d(channels, channels, kernel_size=1)

        elif self.attention_type == 'full':
            # Full attention (dot-product based)
            self.norm = nn.GroupNorm(8, channels)
            self.qkv_proj = nn.Conv1d(channels, channels * 3, kernel_size=1)
            self.out_proj = nn.Conv1d(channels, channels, kernel_size=1)
            self.scale = (channels // heads) ** 0.5

    def forward(self, x):
        n, c, h, w = x.shape

        if self.attention_type == 'cbam':
            # --- Channel Attention ---
            avg_out = self.shared_mlp(self.avg_pool(x))
            max_out = self.shared_mlp(self.max_pool(x))
            channel_attn = self.sigmoid_channel(avg_out + max_out)
            x = x * channel_attn

            # --- Spatial Attention ---
            avg_out = torch.mean(x, dim=1, keepdim=True)
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            spatial_attn = self.sigmoid_spatial(self.spatial_conv(torch.cat([avg_out, max_out], dim=1)))
            return x * spatial_attn

        x_norm = self.norm(x)
        x_reshaped = x_norm.view(n, c, h * w)

        if self.attention_type == 'linear':
            # Linear attention computation
            q = self.q_proj(x_reshaped).transpose(1, 2)
            k = self.k_proj(x_reshaped)
            v = self.v_proj(x_reshaped).transpose(1, 2)

            attn = torch.softmax(torch.bmm(q, k) / (c ** 0.5), dim=-1)
            out = torch.bmm(attn, v)
            out = out.transpose(1, 2).contiguous()
            out = self.out_proj(out).view(n, c, h, w)
            return x + out

        elif self.attention_type == 'full':
            # Full attention with optional SDPA fallback
            if h > self.max_full_res or w > self.max_full_res:
                raise RuntimeError(f"Spatial size {h}x{w} too large for full attention")

            qkv = self.qkv_proj(x_reshaped)
            q, k, v = qkv.chunk(3, dim=1)

            q = q.view(n, self.heads, c // self.heads, h * w)
            k = k.view(n, self.heads, c // self.heads, h * w)
            v = v.view(n, self.heads, c // self.heads, h * w)

            try:
                from torch.nn.functional import scaled_dot_product_attention as sdpa
                out = sdpa(q, k, v, is_causal=False)
            except ImportError:
                attn = torch.einsum('nhcw,nhdw->nhcd', q, k) / self.scale
                attn = torch.softmax(attn, dim=-1)
                out = torch.einsum('nhcd,nhdw->nhcw', attn, v)

            out = out.contiguous().view(n, c, h * w)
            out = self.out_proj(out).view(n, c, h, w)
            return x + out

        return x

# Standard residual block with optional channel projection
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=cfg.num_groups):
        super().__init__()
        out_channels = out_channels or in_channels

        self.norm1 = nn.GroupNorm(num_groups=min(num_groups, in_channels), num_channels=in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.norm2 = nn.GroupNorm(num_groups=min(num_groups, out_channels), num_channels=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # Use 1x1 conv if input/output channels differ
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.activation = nn.SiLU()

    def forward(self, x):
        residual = self.residual_layer(x)

        x = self.norm1(x)
        x = self.activation(x)
        x = self.conv1(x)

        x = self.norm2(x)
        x = self.activation(x)
        x = self.conv2(x)

        return x + residual

# Downsampling block with stride-2 conv and lightweight residual shortcut
class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=cfg.num_groups):
        super(Downsample, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(num_groups=min(num_groups, out_channels), num_channels=out_channels),
            nn.SiLU()
        )
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)

    def forward(self, x):
        return self.main(x) + 0.1 * self.skip(x)  # Add weak skip path to reduce info loss and stabilize feature blending

# Upsampling block with nearest interpolation and residual shortcut
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=cfg.num_groups):
        super(Upsample, self).__init__()

        self.main = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(num_groups, out_channels), num_channels=out_channels),
            nn.SiLU()
        )

        self.skip = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        return self.main(x) + 0.1 * self.skip(x)

# Encoder module that maps input image to latent distribution
# Combines residual blocks, downsampling, and multi-mode attention
class Encoder(nn.Module):
    def __init__(self, in_channels=cfg.in_channels, latent_channels=cfg.latent_channels):
        super(Encoder, self).__init__()
        self.scale = 0.18215  # Latent scaling factor

        self.encoder = nn.Sequential(
            ResidualBlock(3, 8),                         # 512×512
            Downsample(8, 64),                           # 256×256
            ResidualBlock(64, 64),
            AttentionBlock(64, attention_type='cbam'),   # Lightweight CBAM attention

            Downsample(64, 128),                         # 128×128
            ResidualBlock(128, 128),
            AttentionBlock(128, heads=4, attention_type='linear'),  # Linear attention

            Downsample(128, 256),                        # 64×64
            ResidualBlock(256, 256),
            AttentionBlock(256, attention_type='full'),  # Full attention for low-res

            ResidualBlock(256, 128),
            ResidualBlock(128, 128),

            ResidualBlock(128, 64),
            ResidualBlock(64, 64),

            ResidualBlock(64, 16),
            ResidualBlock(16, 16),

            ResidualBlock(16, 8),
        )

    def forward(self, x):
        x = self.encoder(x)  # Feature extraction through downsampling

        # Latent distribution: mean and log-variance
        mean, log_variance = torch.chunk(x, 2, dim=1)
        log_variance = torch.clamp(log_variance, -5, 5)
        stdev = log_variance.exp().sqrt()

        # Reparameterization trick with tanh for bounded output
        noise = torch.randn_like(mean)
        sampled = torch.tanh(mean + stdev * noise) * self.scale

        return mean, log_variance, sampled, None


# Decoder module that reconstructs image from latent representation
# Uses residual blocks, upsampling, and attention to recover spatial detail
class Decoder(nn.Module):
    def __init__(self, use_skip=False, latent_channels=cfg.latent_channels, out_channels=cfg.out_channels):
        super(Decoder, self).__init__()

        self.decoder = nn.Sequential(
            ResidualBlock(4, 8),                         # 64×64
            AttentionBlock(8, heads=1, attention_type='full'),  # Full attention

            ResidualBlock(8, 8),
            Upsample(8, 64),                             # 128×128
            ResidualBlock(64, 64),
            AttentionBlock(64, heads=4, attention_type='linear'),  # Linear attention

            Upsample(64, 128),                           # 256×256
            ResidualBlock(128, 128),
            AttentionBlock(128, attention_type='cbam'),  # CBAM attention

            Upsample(128, 256),                          # 512×512
            ResidualBlock(256, 256),

            ResidualBlock(256, 128),
            ResidualBlock(128, 128),

            ResidualBlock(128, 64),
            ResidualBlock(64, 64),

            nn.Conv2d(64, 3, kernel_size=1),             # Final projection
            nn.Tanh(),                                   # Output normalized to [-1, 1]
        )

    def forward(self, x, skips=None):
        return self.decoder(x)


# Variational Autoencoder combining encoder and decoder modules
class VAE(nn.Module):
    def __init__(self, in_channels=cfg.in_channels, latent_channels=cfg.latent_channels, out_channels=cfg.out_channels, use_skip=False):
        super(VAE, self).__init__()
        self.encoder = Encoder(in_channels, latent_channels)

        self.decoder = Decoder(
            use_skip=use_skip,
            latent_channels=latent_channels,
            out_channels=out_channels
        )

    def forward(self, x):
        mean, log_variance, sampled, skips = self.encoder(x)       # Encode to latent
        reconstructed = self.decoder(sampled, skips)               # Decode to image
        return mean, log_variance, reconstructed