import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SynechismCore(nn.Module):
    """
    SynechismCore: A Continuous Latent Space Architecture with φ-Scaling.
    """
    def __init__(self, input_dim=768, base_width=768, depth=12, phi=1.618):
        super().__init__()
        self.phi = phi
        widths = [self._phi_align(base_width * (self.phi ** i)) for i in range(depth)]
        self.layers = nn.ModuleList()
        prev = input_dim
        for w in widths:
            block = nn.Sequential(
                nn.Linear(prev, w), nn.LayerNorm(w), nn.GELU(),
                nn.Linear(w, prev) 
            )
            self.layers.append(block)
        self.norm = nn.LayerNorm(input_dim)
        self.head = nn.Linear(input_dim, input_dim) 

    def _phi_align(self, val):
        """Aligns dimensions to nearest multiple of 8 for GPU efficiency."""
        return int(round(val / 8) * 8)

    def forward(self, x):
        # x shape: [Batch, Seq, Dim]
        for layer in self.layers:
            residual = x
            x = layer(x)
            x = x + residual
        return self.head(self.norm(x))

class LatentDiffusionSmoothing(nn.Module):
    """
    Implements 1D Gaussian smoothing over the temporal dimension (LDS).
    """
    def __init__(self, channels, kernel_size=31, sigma=4.0):
        super().__init__()
        self.padding = kernel_size // 2
        kernel = torch.tensor([
            math.exp(-((x - self.padding)**2) / (2 * sigma**2))
            for x in range(kernel_size)
        ])
        kernel = kernel / kernel.sum()
        self.register_buffer('smooth_kernel', kernel.view(1, 1, -1))
        self.channels = channels

    def forward(self, x):
        # x: [Batch, Seq, Channels] -> [Batch, Channels, Seq] for conv1d
        x_t = x.permute(0, 2, 1)
        x_smooth = F.conv1d(
            x_t, 
            self.smooth_kernel.repeat(self.channels, 1, 1), 
            padding=self.padding, 
            groups=self.channels
        )
        return x_smooth.permute(0, 2, 1)
