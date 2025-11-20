# coding=utf-8
# NKAT utility functions

import torch


def split_latent_triality(latent, method='chunk'):
    """
    Split latent vector into triality components (v, s, c).
    
    Args:
        latent: Latent tensor, shape (..., dim)
        method: Split method ('chunk' or 'project')
        
    Returns:
        Tuple of (v, s, c), each with shape (..., dim//3)
    """
    if method == 'chunk':
        # Simple chunking - assumes dim is divisible by 3
        return torch.chunk(latent, 3, dim=-1)
    else:
        raise NotImplementedError(f"Split method '{method}' not implemented")


def compute_mass_gap(energy_out):
    """
    Compute mass gap from NKAT energy output.
    
    The mass gap indicates how far the latent space is from
    the "vacuum state" (minimum energy configuration).
    
    Args:
        energy_out: Dictionary from TrialityEnergyMonitor.forward()
        
    Returns:
        Mass gap value (scalar)
    """
    return energy_out['energy'].mean()
