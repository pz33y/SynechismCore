"""
HyperAgent — EventDetector + JumpFunction + SmoothCorrection
============================================================
Implements the hybrid architecture of Equation 4:
    h_{t+1} = ODE_integrate(h_t, dt) + E(h_t) * [J(h_t) + C(h_t)]

E(h): EventDetector — Detects discontinuities, gates correction.
J(h): JumpFunction — Discrete jump, unconstrained magnitude.
C(h): SmoothCorrection — Near-discontinuity, Lipschitz-bounded.

Author: Paul E. Harris IV — SynechismCore v20.0
"""

import torch
import torch.nn as nn

class EventDetector(nn.Module):
    """Detects discontinuities, gates correction."""
    def __init__(self, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
            nn.Sigmoid()
        )
        # Initialize with bias=-3.0 so it starts near-inactive
        nn.init.constant_(self.net[-2].bias, -3.0)

    def forward(self, h):
        return self.net(h)

class JumpFunction(nn.Module):
    """Discrete jump, unconstrained magnitude."""
    def __init__(self, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Linear(hidden * 2, hidden)
        )
        # Near-zero initialization ensures small corrections at training start
        nn.init.normal_(self.net[0].weight, std=0.01)
        nn.init.normal_(self.net[-1].weight, std=0.01)

    def forward(self, h):
        return self.net(h)

class SmoothCorrection(nn.Module):
    """Near-discontinuity, Lipschitz-bounded (spectral norm)."""
    def __init__(self, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(hidden, hidden * 2)),
            nn.GELU(),
            nn.utils.spectral_norm(nn.Linear(hidden * 2, hidden))
        )

    def forward(self, h):
        return self.net(h)

class HyperAgent(nn.Module):
    """Combines E, J, and C for the full hybrid correction."""
    def __init__(self, hidden: int):
        super().__init__()
        self.E = EventDetector(hidden)
        self.J = JumpFunction(hidden)
        self.C = SmoothCorrection(hidden)

    def forward(self, h):
        event_prob = self.E(h)
        jump = self.J(h)
        smooth = self.C(h)
        return event_prob * (jump + smooth)
