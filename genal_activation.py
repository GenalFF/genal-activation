"""
╔══════════════════════════════════════════════════════════════════╗
║              GENAL ACTIVATION FAMILY                             ║
║                                                                  ║
║  Author : Genal Ediso Lombano Pineda                             ║
║  Email  : genallombano@gmail.com                                 ║
║  GitHub : github.com/GenalFF/genal-activation                    ║
║  Paper  : zenodo.org/records/20304195                            ║
║  ORCID  : 0009-0009-6495-4085                                    ║
║  Date   : May 2026                                               ║
║  License: GPLv3                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  Formula: Genal(x) = x · sigmoid(x / k)                         ║
║           k = softplus(θ) + ε,  θ trainable                     ║
║                                                                  ║
║  Family:                                                         ║
║  ① GenalActivation  — scalar k                                  ║
║  ② GenalAdvanced    — k per channel                             ║
║  ③ GenalShift       — k + learnable shift β                     ║
║  ④ GenalLeaky       — minimum slope for x ≤ 0                   ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GenalActivation(nn.Module):
    """
    ① Genal Activation — scalar k (one global theta).

    Genal(x) = x · sigmoid(x / k)
    where  k = softplus(clamp(θ)) + 1e-6

    Args:
        theta_init  (float): initial value of theta. Default: 1.0
        clamp_range (tuple): clamp range for theta. Default: (-2, 2)

    Example:
        >>> act = GenalActivation()
        >>> x = torch.randn(4, 128)
        >>> y = act(x)
    """
    def __init__(self, theta_init=1.0, clamp_range=(-2, 2)):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(theta_init, dtype=torch.float32))
        self.clamp_range = clamp_range

    def forward(self, x):
        tc = torch.clamp(self.theta, min=self.clamp_range[0], max=self.clamp_range[1])
        k  = F.softplus(tc) + 1e-6
        return x * torch.sigmoid(x / k)

    def get_k(self):
        with torch.no_grad():
            tc = torch.clamp(self.theta, min=self.clamp_range[0], max=self.clamp_range[1])
            return (F.softplus(tc) + 1e-6).item()


class GenalAdvanced(nn.Module):
    """
    ② Genal Advanced — one independent k per channel.

    Genal(x)_c = x_c · sigmoid(x_c / k_c)
    where  k_c = softplus(θ_c) + 1e-6

    Compatible with:
      - Conv layers: x shape (B, C, H, W)
      - Dense layers: x shape (B, C)

    Args:
        num_channels (int): number of channels / features.
        clamp_range (tuple): clamp range. Default: (-2, 2)

    Example:
        >>> act = GenalAdvanced(num_channels=64)
        >>> x = torch.randn(4, 64, 8, 8)
        >>> y = act(x)
    """
    def __init__(self, num_channels, clamp_range=(-2, 2)):
        super().__init__()
        self.theta = nn.Parameter(torch.randn(num_channels))
        self.clamp_range = clamp_range
        self.num_channels = num_channels

    def forward(self, x):
        tc = torch.clamp(self.theta, min=self.clamp_range[0], max=self.clamp_range[1])
        k  = F.softplus(tc) + 1e-6
        k  = k.view(1, -1, 1, 1) if x.dim() == 4 else k.view(1, -1)
        return x * torch.sigmoid(x / k)

    def get_k(self):
        with torch.no_grad():
            tc = torch.clamp(self.theta, min=self.clamp_range[0], max=self.clamp_range[1])
            return F.softplus(tc) + 1e-6


class GenalShift(nn.Module):
    """
    ③ Genal Shift — learnable curvature k and horizontal shift β.

    Genal(x) = x · sigmoid((x − β) / k)
    where  k = clamp(softplus(θ), 1e-6, 2.0)
           β = learnable shift parameter

    Best result: 85.11% on CIFAR-10 — best in the Genal Family.

    Args:
        theta_init (float): initial theta. Default: 1.0
        beta_init  (float): initial beta.  Default: 0.0

    Example:
        >>> act = GenalShift()
        >>> y = act(torch.randn(4, 128))
    """
    def __init__(self, theta_init=1.0, beta_init=0.0):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(theta_init, dtype=torch.float32))
        self.beta  = nn.Parameter(torch.tensor(beta_init,  dtype=torch.float32))

    def forward(self, x):
        k = torch.clamp(F.softplus(self.theta), min=1e-6, max=2.0)
        return x * torch.sigmoid((x - self.beta) / k)

    def get_k(self):
        with torch.no_grad():
            return torch.clamp(F.softplus(self.theta), 1e-6, 2.0).item()

    def get_beta(self):
        return self.beta.item()


class GenalLeaky(nn.Module):
    """
    ④ Genal Leaky — minimum slope alpha for x <= 0.

    Genal(x) = x · sigmoid(x/k)   if x > 0
             = alpha · x           if x <= 0

    Eliminates dead neurons completely.

    Args:
        theta_init (float): initial theta. Default: 1.0
        alpha      (float): slope for x <= 0. Default: 0.01

    Example:
        >>> act = GenalLeaky(alpha=0.01)
        >>> y = act(torch.randn(4, 128))
    """
    def __init__(self, theta_init=1.0, alpha=0.01):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(theta_init, dtype=torch.float32))
        self.alpha = alpha

    def forward(self, x):
        k = F.softplus(self.theta) + 1e-6
        return torch.where(x > 0, x * torch.sigmoid(x / k), self.alpha * x)

    def get_k(self):
        with torch.no_grad():
            return (F.softplus(self.theta) + 1e-6).item()


# Convenience alias
Genal = GenalActivation

__all__ = [
    "GenalActivation",
    "GenalAdvanced",
    "GenalShift",
    "GenalLeaky",
    "Genal",
]


if __name__ == "__main__":
    print("=" * 55)
    print("  GENAL ACTIVATION FAMILY — Test")
    print("=" * 55)

    for name, act in [
        ("GenalActivation", GenalActivation()),
        ("GenalAdvanced",   GenalAdvanced(16)),
        ("GenalShift",      GenalShift()),
        ("GenalLeaky",      GenalLeaky()),
    ]:
        x = torch.randn(4, 16)
        y = act(x)
        print(f"  {name:<18} | {x.shape} → {y.shape} ✅")

    print("=" * 55)
    print("  Genal Ediso Lombano Pineda — Venezuela 🇻🇪")
    print("  zenodo.org/records/20304195")
    print("=" * 55)