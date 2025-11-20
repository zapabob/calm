# coding=utf-8
# NKAT-Enhanced Energy Transformer for CALM
# Adds geometric regularization using Spin(8) triality structure

import math
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from .configuration_calm import CALMConfig
from .modeling_energy import EnergyTransformer, CustomCausalLMOutput
from nkat.triality_energy import TrialityEnergyMonitor


class NKATEnergyTransformer(EnergyTransformer):
    """
    NKAT-Enhanced Energy Transformer for CALM.
    
    Extends the base EnergyTransformer with NKAT geometric regularization.
    The NKAT regularizer adds a mass gap potential that:
    1. Constrains latent vectors to lie on SO(8) manifold
    2. Enforces triality structure (v, s, c) decomposition
    3. Prevents hallucination drift through geometric constraints
    
    Args:
        config: CALMConfig with NKAT-specific parameters:
            - nkat_weight: Weight for NKAT loss term (default: 0.01)
            - nkat_spin_dim: Dimension of Spin(8) components (default: 8)
            - nkat_alpha: Alpha parameter for energy balance (default: 0.382)
            - nkat_learnable_alpha: Whether to learn alpha (default: False)
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # NKAT configuration
        self.nkat_weight = getattr(config, 'nkat_weight', 0.01)
        nkat_spin_dim = getattr(config, 'nkat_spin_dim', 8)
        nkat_alpha = getattr(config, 'nkat_alpha', 0.382)
        nkat_learnable_alpha = getattr(config, 'nkat_learnable_alpha', False)
        
        # Initialize NKAT Triality Energy Monitor
        self.nkat_monitor = TrialityEnergyMonitor(
            dim=config.latent_size,
            spin_dim=nkat_spin_dim,
            alpha_init=nkat_alpha,
            learnable_alpha=nkat_learnable_alpha,
            invariant_mode="hadamard",
            lambda_reg=0.01
        )
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Union[Tuple, CustomCausalLMOutput]:
        
        batch_size, seq_length = input_ids.size()
        patch_size = self.patch_size
        latent_length = seq_length // patch_size

        labels = labels[:, patch_size:]
        mask = labels.ne(-100)
        labels = labels[mask].unsqueeze(0)

        # Get ground-truth latent vector from the frozen Autoencoder 
        latent_states = self.ae_model.encoder(input_ids=labels)
        latent_states = latent_states.squeeze(0)
        mean, log_std = torch.chunk(latent_states, 2, dim=-1)

        # Prepare Transformer input
        inputs_embeds = self.transformer.embed_tokens(input_ids).reshape(batch_size, latent_length, -1)[:, :-1, :]
        inputs_embeds = self.embed_proj(inputs_embeds)

        # Get hidden states from the Transformer backbone
        outputs = self.transformer(inputs_embeds = inputs_embeds)
        hidden_states = outputs[0]
        patch_mask = mask.reshape(batch_size, latent_length-1, patch_size)[:, :, 0]
        hidden_states = hidden_states[patch_mask]

        # Generate predictions with the MLP Generator
        hidden_states_repeated = hidden_states.unsqueeze(0).repeat(self.num_samples, 1, 1)
        latent_predictions = self.generative_head.sample(hidden_states_repeated)

        # Compute the original energy loss
        energy_loss = - self.energy_score(latent_predictions, mean, log_std)
        energy_loss = energy_loss.mean()
        
        # Compute NKAT regularization loss on predicted latents
        # Average across samples
        nkat_loss = 0.0
        for i in range(self.num_samples):
            nkat_out = self.nkat_monitor(latent_predictions[i])
            nkat_loss += nkat_out['energy'].mean()
        nkat_loss = nkat_loss / self.num_samples
        
        # Combined loss: original energy + NKAT regularization
        loss = energy_loss + self.nkat_weight * nkat_loss

        # Brier score is only calculated during evaluation
        if not self.training:
            return self.eval_brier(latent_predictions, input_ids[:, patch_size:], outputs, loss)

        return CustomCausalLMOutput(
            loss=loss,
        )
