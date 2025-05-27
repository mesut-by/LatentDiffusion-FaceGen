import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from config import cfg
from diffusion.time import TimeEmbedding


# Time-conditioned residual block with normalization and optional channel change
class ResidualBlock_diff(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.gn1   = nn.GroupNorm(num_groups=cfg.num_groups, num_channels=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.gn2   = nn.GroupNorm(num_groups=cfg.num_groups, num_channels=out_channels)

        # Project time embedding to match spatial feature map
        self.time_mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(cfg.time_dim, out_channels)
        )

        # Match input/output channels or use identity
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) \
                            if in_channels != out_channels else nn.Identity()

        self.gamma = nn.Parameter(torch.tensor(1e-5))  # Scaling factor for residual
        self.layernorm = nn.GroupNorm(1, out_channels)  # Final layer norm

        self._initialize()

    # He initialization for conv/linear layers
    def _initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, t_emb):
        # Project time embedding to spatial dimensions
        t_proj = self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)

        h = self.conv1(x)
        h = self.gn1(h)
        h = F.gelu(h)

        h = self.conv2(h)
        h = h + t_proj  # Inject time info before norm
        h = self.gn2(h)
        h = F.gelu(h)

        out = self.shortcut(x) + self.gamma * h  # Residual connection with scaling
        out = self.layernorm(out)
        out = F.gelu(out)
        return out


# Multi-head self-attention over 2D feature maps
class SelfAttention2D(nn.Module):
    def __init__(self, in_channels, num_heads=4):
        super().__init__()
        assert in_channels % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Conv2d(in_channels, in_channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.norm = nn.GroupNorm(cfg.num_groups, in_channels)

    def forward(self, x):
        b, c, h, w = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x).reshape(b, 3, self.num_heads, self.head_dim, h * w)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        # Shapes: (B, heads, head_dim, HW)
        q = q.permute(0, 1, 3, 2)
        k = k.permute(0, 1, 2, 3)
        v = v.permute(0, 1, 3, 2)

        # Scaled dot-product attention
        attn = torch.softmax(torch.matmul(q, k) * self.scale, dim=-1)
        out = torch.matmul(attn, v)

        # Reshape back to spatial map
        out = out.permute(0, 1, 3, 2).reshape(b, c, h, w)
        out = self.proj(out)

        return self.norm(out + x)  # Residual + normalization


# Bottleneck block with residual, dilated conv, time conditioning, and attention
class BottleneckBlock(nn.Module):
    def __init__(self, channels, time_emb_dim):
        super().__init__()
        self.block1 = ResidualBlock_diff(channels, channels, time_emb_dim)

        self.dilated_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2)
        self.dilated_norm = nn.GroupNorm(cfg.num_groups, channels)

        self.dilated_time_proj = nn.Sequential(
            nn.GELU(),
            nn.Linear(cfg.time_dim, channels)
        )

        self.attn = SelfAttention2D(channels)

    def forward(self, x, t_emb):
        x = self.block1(x, t_emb)  # Residual block with time conditioning

        x = self.dilated_conv(x)
        x = self.dilated_norm(x)

        # Inject time info into dilated conv output
        t_proj = self.dilated_time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        x = x + t_proj

        x = self.attn(x)  # Spatial self-attention
        return x


# Upsampling block with nearest interpolation and lightweight residual path
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.main = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(cfg.num_groups, out_channels), num_channels=out_channels),
            nn.SiLU()
        )

        self.skip = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        return self.main(x) + 0.04 * self.skip(x)  # Add weak skip connection to preserve signal and stabilize training


# Diffusion-style upsampling block with time embedding and residual blocks
class Upsample_diff(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.up = Upsample(in_channels, out_channels)
        self.skip_norm = nn.GroupNorm(cfg.num_groups, in_channels)
        self.up_norm = nn.GroupNorm(cfg.num_groups, out_channels)
        self.resblock1 = ResidualBlock_diff(out_channels + in_channels, out_channels, time_emb_dim)
        self.norm = nn.GroupNorm(cfg.num_groups, out_channels)
        self.resblock2 = ResidualBlock_diff(out_channels, out_channels, time_emb_dim)

    def forward(self, x, skip, t_emb):
        x = self.up(x)                   # Upsample input feature
        skip = self.skip_norm(skip)      # Normalize skip feature
        x = self.up_norm(x)              # Normalize upsampled feature
        x = torch.cat([skip, x], dim=1)  # Concatenate features
        x = self.resblock1(x, t_emb)
        x = self.norm(x)
        x = self.resblock2(x, t_emb)
        return x


# Downsampling block with convolution, normalization, and residual shortcut
class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(num_groups=min(cfg.num_groups, out_channels), num_channels=out_channels),
            nn.SiLU()
        )
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)

    def forward(self, x):
        return self.main(x) + 0.2 * self.skip(x)  # Add moderate skip connection to downsample while preserving features and aiding gradient flow


# Diffusion-style downsampling block with time embedding and residual blocks
class Downsample_diff(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.down = Downsample(in_channels, out_channels)
        self.resblock1 = ResidualBlock_diff(out_channels, out_channels, time_emb_dim)
        self.resblock2 = ResidualBlock_diff(out_channels, out_channels, time_emb_dim)

    def forward(self, x, t_emb):
        x = self.down(x)               # Downsample input
        x = self.resblock1(x, t_emb)   # Apply residual block 1
        x = self.resblock2(x, t_emb)   # Apply residual block 2
        return x


# Advanced UNet₂D architecture for diffusion models
class AdvancedUNet2d_diff(nn.Module):
    def __init__(self, latent_channels=cfg.latent_channels, time_emb_dim=cfg.time_dim):
        super().__init__()
        self.time_emb = TimeEmbedding(time_dim=time_emb_dim, embed_dim=time_emb_dim)

        # ENCODER
        self.init_conv = nn.Sequential(
            nn.Conv2d(latent_channels, 320, kernel_size=3, padding=1),
            nn.GroupNorm(cfg.num_groups, 320),
            nn.GELU(),
            nn.Conv2d(320, 320, kernel_size=3, padding=1),
            nn.GELU()
        )  # 64x64x320

        self.dres1 = ResidualBlock_diff(320, 320, time_emb_dim)   # 64x64x320
        self.down1 = Downsample_diff(320, 640, time_emb_dim)      # 32x32x640

        self.dres2 = ResidualBlock_diff(640, 640, time_emb_dim)   # 32x32x640
        self.attn2 = SelfAttention2D(640, num_heads=4)            # 32x32x640
        self.down2 = Downsample_diff(640, 1280, time_emb_dim)     # 16x16x1280

        self.dres3 = ResidualBlock_diff(1280, 1280, time_emb_dim) # 16x16x1280
        self.attn3 = SelfAttention2D(1280, num_heads=4)           # 16x16x1280
        self.down3 = Downsample_diff(1280, 1280, time_emb_dim)    # 8x8x1280

        # BOTTLENECK
        self.bottleneck = BottleneckBlock(1280, time_emb_dim)     # 8x8x1280

        # DECODER
        self.ures1 = ResidualBlock_diff(1280, 1280, time_emb_dim) # 8x8x1280
        self.up1 = Upsample_diff(1280, 640, time_emb_dim)         # 16x16x640

        self.ures2 = ResidualBlock_diff(640, 640, time_emb_dim)   # 16x16x640
        self.attn2d = SelfAttention2D(640, num_heads=4)           # 16x16x640
        self.up2 = Upsample_diff(640, 320, time_emb_dim)          # 32x32x320

        self.ures3 = ResidualBlock_diff(320, 320, time_emb_dim)   # 32x32x320
        self.attn3d = SelfAttention2D(320, num_heads=4)           # 32x32x320
        self.up3 = Upsample_diff(320, 320, time_emb_dim)          # 64x64x320

        self.final_norm = nn.GroupNorm(cfg.num_groups, 320)
        self.final_conv = nn.Sequential(
            nn.Conv2d(320, latent_channels, kernel_size=3, padding=1),
            # nn.Tanh()  # Optional output activation
        )

    def forward(self, x, t):
        t_emb = self.time_emb(t)  # Embed time step

        x0 = self.init_conv(x)  # Initial convolution

        dr1 = self.dres1(x0, t_emb)
        d1 = self.down1(dr1, t_emb)     # 64 -> 32
        d1 = F.gelu(d1)

        dr2 = self.dres2(d1, t_emb)
        dr2 = self.attn2(dr2)
        d2 = self.down2(dr2, t_emb)     # 32 -> 16
        d2 = F.gelu(d2)

        dr3 = self.dres3(d2, t_emb)
        dr3 = self.attn3(dr3)
        d3 = self.down3(dr3, t_emb)     # 16 -> 8
        d3 = F.gelu(d3)

        b = self.bottleneck(d3, t_emb)  # Bottleneck block

        ur1 = self.ures1(b, t_emb)
        u1 = self.up1(ur1, dr3, t_emb)  # 8 -> 16
        u1 = F.gelu(u1)

        ur2 = self.ures2(u1, t_emb)
        ur2 = self.attn2d(ur2)
        u2 = self.up2(ur2, dr2, t_emb)  # 16 -> 32
        u2 = F.gelu(u2)

        ur3 = self.ures3(u2, t_emb)
        ur3 = self.attn3d(ur3)
        u3 = self.up3(ur3, dr1, t_emb)  # 32 -> 64
        u3 = F.gelu(u3)

        u3 = self.final_norm(u3)
        out = self.final_conv(u3)

        return out  # Output of the UNet