# coding=utf-8
# NKAT Triality Energy Monitor for CALM
# Provides geometric regularization using Spin(8) triality structure

import torch
import torch.nn as nn
from typing import Literal, Dict


class TrialityEnergyMonitor(nn.Module):
    """
    Triality Energy Monitor based on Spin(8) geometry.
    
    This module projects latent vectors into triality space (v, s, c) representing:
    - v: vector representation (logical)
    - s: spinor representation (physical)
    - c: conjugate spinor representation (contextual)
    
    The energy function enforces:
    1. Norm conservation: ||v||² + ||s||² + ||c||² ≈ 3
    2. Triality interaction: Omega invariant from v⊗s⊗c
    
    Args:
        dim: Input dimension of latent vectors
        spin_dim: Dimension of each Spin(8) component (default: 8)
        alpha_init: Initial value for alpha parameter (default: 0.382, golden ratio)
        learnable_alpha: Whether alpha should be learned (default: False)
        invariant_mode: How to compute Omega invariant ("hadamard" or "tensor")
        lambda_reg: Regularization weight for alpha term (default: 0.01)
    """
    
    def __init__(
        self,
        dim: int = 768,
        spin_dim: int = 8,
        alpha_init: float = 0.382,
        learnable_alpha: bool = False,
        invariant_mode: Literal["hadamard", "tensor"] = "hadamard",
        lambda_reg: float = 0.01,
    ):
        super().__init__()
        
        # Projection layers for triality components
        self.proj_v = nn.Linear(dim, spin_dim)
        self.proj_s = nn.Linear(dim, spin_dim)
        self.proj_c = nn.Linear(dim, spin_dim)

        self.invariant_mode = invariant_mode
        self.lambda_reg = lambda_reg

        # Alpha parameter balances norm conservation vs triality interaction
        if learnable_alpha:
            self.alpha_param = nn.Parameter(torch.tensor(alpha_init))
        else:
            self.register_buffer("alpha_param", torch.tensor(alpha_init))

        # Tensor for triality invariant computation
        if invariant_mode == "tensor":
            spin = spin_dim
            self.omega_tensor = nn.Parameter(
                torch.randn(spin, spin, spin) * (1.0 / spin_dim**1.5)
            )
        else:
            self.omega_tensor = None

    @property
    def alpha(self):
        """Alpha clamped to [0, 1] range."""
        return torch.clamp(self.alpha_param, 0.0, 1.0)

    def _compute_omega(self, v, s, c):
        """
        Compute the triality Omega invariant from v⊗s⊗c.
        
        Args:
            v, s, c: Triality components, shape (batch, spin_dim)
            
        Returns:
            Omega invariant, shape (batch,)
        """
        if self.invariant_mode == "hadamard":
            # Simplified Hadamard product version
            return (v * s * c).sum(dim=-1)
        else:
            # Full tensor contraction version
            return torch.einsum("bi,bj,bk,ijk->b", v, s, c, self.omega_tensor)

    def forward(self, latent_emb) -> Dict[str, torch.Tensor]:
        """
        Compute NKAT energy for latent embeddings.
        
        Args:
            latent_emb: Latent vectors, shape (batch, seq_len, dim) or (batch, dim)
            
        Returns:
            Dictionary containing:
                - energy: Total NKAT energy
                - norm_term: Norm conservation term
                - triality_term: Triality interaction term
                - alpha: Current alpha value
        """
        # Handle both 2D and 3D inputs
        original_shape = latent_emb.shape
        if len(original_shape) == 3:
            batch_size, seq_len, dim = original_shape
            latent_emb = latent_emb.reshape(batch_size * seq_len, dim)
        
        # Project to triality space
        v = self.proj_v(latent_emb)
        s = self.proj_s(latent_emb)
        c = self.proj_c(latent_emb)

        # Norm conservation term: (||v||² + ||s||² + ||c||² - 3)²
        v_norm_sq = (v**2).sum(dim=-1)
        s_norm_sq = (s**2).sum(dim=-1)
        c_norm_sq = (c**2).sum(dim=-1)
        total_norm = v_norm_sq + s_norm_sq + c_norm_sq
        norm_term = (total_norm - 3.0) ** 2

        # Triality interaction term: (|Omega|² - 1)²
        omega = self._compute_omega(v, s, c)
        triality_term = (omega.abs() ** 2 - 1.0) ** 2

        # Alpha regularization: keeps alpha near valid range
        alpha = self.alpha
        alpha_term = self.lambda_reg * (alpha**2 + (1.0 - alpha)**2 - 1.0)**2

        # Weighted energy: E = α * norm_term + (1-α) * triality_term + alpha_reg
        energy = alpha * norm_term + (1.0 - alpha) * triality_term + alpha_term

        # Restore original shape if needed
        if len(original_shape) == 3:
            energy = energy.reshape(batch_size, seq_len)
            norm_term = norm_term.reshape(batch_size, seq_len)
            triality_term = triality_term.reshape(batch_size, seq_len)

        return {
            "energy": energy,
            "norm_term": norm_term,
            "triality_term": triality_term,
            "alpha": torch.full_like(energy, alpha),
        }
