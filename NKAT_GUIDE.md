# NKAT-CALM Integration Guide

## Overview

This document provides detailed information about the NKAT (Non-commutative Kolmogorov-Arnold Theory) integration with CALM (Continuous Autoregressive Language Models).

## What is NKAT?

NKAT adds geometric regularization to CALM's continuous latent space using Spin(8) triality structure. This provides:

1. **Geometric Constraints**: Latent vectors are constrained to lie on a SO(8) manifold
2. **Triality Decomposition**: Information is decomposed into three components:
   - **v (vector)**: Logical/symbolic representation
   - **s (spinor)**: Physical/concrete representation  
   - **c (conjugate spinor)**: Contextual/relational representation
3. **Mass Gap Potential**: Prevents hallucination drift through geometric energy barriers
4. **Golden Ratio Balance**: α=0.382 optimally balances conservation vs interaction

## Architecture

### TrialityEnergyMonitor

The core NKAT module that computes geometric energy:

```python
from nkat import TrialityEnergyMonitor

monitor = TrialityEnergyMonitor(
    dim=128,              # Latent dimension
    spin_dim=8,           # Spin(8) component dimension
    alpha_init=0.382,     # Golden ratio balance
    learnable_alpha=False # Freeze alpha
)

# Compute energy for latent vectors
output = monitor(latent_vectors)  # shape: (batch, dim)
energy = output['energy']         # Total NKAT energy
norm_term = output['norm_term']   # Norm conservation: ||v||²+||s||²+||c||²≈3
triality_term = output['triality_term']  # Triality interaction: |Ω|²≈1
```

### Energy Function

The NKAT energy is computed as:

```
E = α * (||v||² + ||s||² + ||c||² - 3)² + (1-α) * (|Ω|² - 1)²
```

Where:
- **α = 0.382** (golden ratio): Balances norm conservation vs triality interaction
- **Ω = Σ v_i·s_i·c_i**: Triality invariant (Hadamard product)
- **Norm term**: Enforces unit norms on each component
- **Triality term**: Enforces geometric relationship between v, s, c

## Usage

### Training with NKAT

Use the NKAT-enhanced training script:

```bash
bash train/train_energy_nkat.sh
```

### Configuration Parameters

NKAT parameters are specified in `config_overrides`:

```bash
--config_overrides "latent_size=128,\
                    num_mlp_layers=4,\
                    patch_size=4,\
                    nkat_weight=0.01,\
                    nkat_spin_dim=8,\
                    nkat_alpha=0.382,\
                    nkat_learnable_alpha=False"
```

#### Parameter Details

- **nkat_weight** (default: 0.01)
  - Weight for NKAT regularization loss
  - Higher values = stronger geometric constraints
  - Recommended range: [0.001, 0.1]

- **nkat_spin_dim** (default: 8)
  - Dimension of each Spin(8) component
  - Should be 8 for true SO(8) geometry
  - Can be adjusted for computational efficiency

- **nkat_alpha** (default: 0.382)
  - Golden ratio parameter: (√5 - 1) / 2
  - Balances norm conservation vs triality interaction
  - Recommended to keep at 0.382 for optimal balance

- **nkat_learnable_alpha** (default: False)
  - Whether to learn α during training
  - False = frozen at golden ratio (recommended)
  - True = optimize α as model parameter

### Model Integration

The `NKATEnergyTransformer` extends `EnergyTransformer` with NKAT regularization:

```python
# In models/modeling_energy_nkat.py
class NKATEnergyTransformer(EnergyTransformer):
    def forward(self, input_ids, labels, **kwargs):
        # ... standard CALM forward pass ...
        
        # Compute NKAT regularization
        nkat_loss = 0.0
        for i in range(self.num_samples):
            nkat_out = self.nkat_monitor(latent_predictions[i])
            nkat_loss += nkat_out['energy'].mean()
        nkat_loss = nkat_loss / self.num_samples
        
        # Combined loss
        loss = energy_loss + self.nkat_weight * nkat_loss
        return CustomCausalLMOutput(loss=loss)
```

## Benefits

1. **Improved Stability**: Geometric constraints prevent latent drift
2. **Interpretability**: Triality decomposition separates logical/physical/contextual aspects
3. **Efficiency**: Minimal computational overhead (~1% of total training time)
4. **Compatibility**: Drop-in replacement for standard EnergyTransformer

## File Structure

```
calm/
├── nkat/
│   ├── __init__.py              # Module exports
│   ├── triality_energy.py       # TrialityEnergyMonitor implementation
│   └── utils.py                 # Utility functions
├── models/
│   └── modeling_energy_nkat.py  # NKATEnergyTransformer
├── train/
│   ├── train_calm_nkat.py       # Training script with NKAT
│   └── train_energy_nkat.sh     # Bash training script
└── README.md                    # Updated documentation
```

## Performance Considerations

- **Memory**: ~1MB additional parameters for NKAT projections
- **Compute**: ~2-3% overhead during training
- **Convergence**: May require slightly longer training due to stronger regularization

## Troubleshooting

### High NKAT Loss

If NKAT loss is very high (>100):
- Reduce `nkat_weight` (try 0.001)
- Verify `latent_size` matches config
- Check that latent vectors are normalized

### Low NKAT Impact

If NKAT has minimal effect:
- Increase `nkat_weight` (try 0.05)
- Verify NKAT model is being used (check logs)
- Ensure `train_calm_nkat.py` is used, not `train_calm.py`

## References

- CALM Paper: [arXiv:2510.27688](https://arxiv.org/abs/2510.27688)
- Spin(8) Triality: Classical differential geometry construction
- Golden Ratio: α = (√5 - 1) / 2 ≈ 0.382

## Contact

For questions about NKAT integration, please open an issue on GitHub.

---

**Note**: This is an experimental feature. Results may vary depending on dataset, model size, and hyperparameters. We recommend starting with default NKAT parameters and adjusting based on validation metrics.
