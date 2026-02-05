#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STM Classification with Asym-Mixer-KAN
Asymmetric MLP-Mixer with Kolmogorov-Arnold Networks

This implementation follows Roadmap 1 from the research paper:
"Optimization of Spectrotemporal Modulation Analysis: Architectural Paradigms 
for Superior Audio Classification and Segregation"

Key Features:
1. Asymmetric 2-channel STM processing (Magnitude + Difference)
2. MLP-Mixer architecture with position-aware token mixing
3. KAN layers with learnable B-spline activation functions
4. LDAM loss with Deferred Reweighting (DRW)
5. CutMix augmentation (replaces Mixup to avoid ghosting)
6. DropBlock spatial regularization

Target: 0.89-0.90 Macro F1 Score
"""

import os
import sys
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import f1_score, classification_report

warnings.filterwarnings('ignore')


# ============================================================================
# Asymmetric STM Processing (2-Channel: Magnitude + Difference)
# ============================================================================

def process_asymmetric_stm(stm_data):
    """
    Process STM data to create 2-channel asymmetric representation.
    
    This addresses the paper's recommendation:
    "A standard signal processing approach is to 'fold' the spectrum... 
    This is a destructive operation for classification."
    
    Instead, we preserve both directional components:
    - Channel 0: S_up_flipped (upward sweeps, after alignment)
    - Channel 1: S_down (downward sweeps)
    
    Input: (batch, freq_bands=20, mod_rates=121)
    Output: (batch, channels=2, freq_bands=20, mod_rates=61)
    
    Steps:
    1. Separate negative rates [0:60] and positive rates [61:121]
    2. Flip negative chunk to align modulation rates
    3. Keep both channels separately:
       - Channel 0 = negative_flipped (upward sweeps)
       - Channel 1 = positive (downward sweeps)
    4. Concatenate DC (index 60) at position 0 for both channels
    
    Modulation rate mapping:
    - Input: -15 Hz (idx 0) ... 0 Hz (idx 60) ... +15 Hz (idx 120)
    - Output: 0 Hz (idx 0) ... +15 Hz (idx 60)
    
    Physical Interpretation:
    - Channel 0 (S_up): Upward frequency sweeps (rising pitch, question intonation)
    - Channel 1 (S_down): Downward frequency sweeps (falling pitch, statement intonation)
    - Preserves full directional information without lossy combination
    """
    # Step 1: Separate chunks along modulation rate dimension (last dim)
    negative_chunk = stm_data[:, :, 0:60]   # -15 Hz to -0.25 Hz (upward sweeps)
    dc_component = stm_data[:, :, 60:61]    # 0 Hz
    positive_chunk = stm_data[:, :, 61:121] # +0.25 Hz to +15 Hz (downward sweeps)
    
    # Step 2: Flip negative chunk (reverse modulation rate axis)
    # After flipping: -15Hz → position 0, -0.25Hz → position 59
    # This aligns with: +0.25Hz → position 0, +15Hz → position 59
    negative_flipped = torch.flip(negative_chunk, dims=[2])
    
    # Step 3: Keep both channels separately (no averaging or difference)
    # Channel 0: Upward sweeps (negative rates, after flipping)
    s_up_channel = negative_flipped
    
    # Channel 1: Downward sweeps (positive rates)
    s_down_channel = positive_chunk
    
    # Step 4: Concatenate DC at the beginning [DC, 0.25Hz...15Hz]
    s_up_out = torch.cat([dc_component, s_up_channel], dim=2)  # (batch, 20, 61)
    s_down_out = torch.cat([dc_component, s_down_channel], dim=2)  # (batch, 20, 61)
    
    # Stack into 2-channel tensor
    output = torch.stack([s_up_out, s_down_out], dim=1)  # (batch, 2, 20, 61)
    
    return output


class AsymmetricSTMDataset(Dataset):
    """Wrapper dataset that applies asymmetric 2-channel STM processing"""
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        data, label = self.base_dataset[idx]
        # data shape: (2420,) - flattened from (20, 121)
        # Reshape to (20, 121) for asymmetric processing
        data = data.reshape(20, 121)
        # Add batch dimension for processing
        data = data.unsqueeze(0)  # (1, 20, 121)
        data = process_asymmetric_stm(data)
        data = data.squeeze(0)    # (2, 20, 61)
        return data, label


# ============================================================================
# KAN Layer: Kolmogorov-Arnold Networks with B-spline Basis
# ============================================================================

class KANLayer(nn.Module):
    """
    Kolmogorov-Arnold Network Layer using B-spline basis functions.
    
    From the paper:
    "KAN layers use B-spline basis functions for learnable non-linearities"
    
    Unlike traditional MLPs with fixed activations (ReLU, GELU), KAN learns
    the activation function itself through a linear combination of B-spline
    basis functions. This allows the network to discover optimal non-linearities
    for the specific task.
    
    Architecture:
        output = Σ_{i=0}^{grid_size} w_i * B_i(x)
    
    where B_i are cubic B-spline basis functions and w_i are learnable weights.
    """
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3):
        super(KANLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        # Learnable spline coefficients
        # Shape: (out_features, in_features, grid_size + spline_order)
        self.spline_weight = nn.Parameter(
            torch.randn(out_features, in_features, grid_size + spline_order) * 0.1
        )
        
        # Grid points for B-spline knots (uniform spacing in [-1, 1])
        grid = torch.linspace(-1, 1, grid_size + 2 * spline_order + 1)
        self.register_buffer('grid', grid)
        
        # Base linear transformation (residual connection)
        self.base_linear = nn.Linear(in_features, out_features)
        
    def b_splines(self, x):
        """
        Compute cubic B-spline basis functions.
        
        Args:
            x: Input tensor (batch, in_features)
        Returns:
            basis: B-spline basis values (batch, in_features, grid_size + spline_order)
        """
        # Clamp input to grid range
        x = torch.clamp(x, -1.0, 1.0)
        
        # Expand dimensions for broadcasting
        x = x.unsqueeze(-1)  # (batch, in_features, 1)
        
        # Compute basis functions using Cox-de Boor recursion
        # For simplicity, we use a direct cubic B-spline implementation
        grid = self.grid  # (grid_size + 2*spline_order + 1,)
        
        # Initialize basis with order 0 (piecewise constant)
        bases = []
        for i in range(len(grid) - 1):
            basis = ((x >= grid[i]) & (x < grid[i + 1])).float()
            bases.append(basis)
        bases = torch.stack(bases, dim=-1)  # (batch, in_features, 1, num_intervals)
        bases = bases.squeeze(2)  # (batch, in_features, num_intervals)
        
        # Recursively compute higher order basis functions
        for order in range(1, self.spline_order + 1):
            new_bases = []
            for i in range(len(grid) - order - 1):
                # Left term
                denom1 = grid[i + order] - grid[i]
                if denom1 > 1e-8:
                    left = (x.squeeze(-1) - grid[i]) / denom1 * bases[..., i]
                else:
                    left = torch.zeros_like(bases[..., i])
                
                # Right term
                denom2 = grid[i + order + 1] - grid[i + 1]
                if denom2 > 1e-8:
                    right = (grid[i + order + 1] - x.squeeze(-1)) / denom2 * bases[..., i + 1]
                else:
                    right = torch.zeros_like(bases[..., i + 1])
                
                new_bases.append(left + right)
            bases = torch.stack(new_bases, dim=-1)
        
        return bases
    
    def forward(self, x):
        """
        Args:
            x: (batch, in_features)
        Returns:
            out: (batch, out_features)
        """
        batch_size = x.size(0)
        
        # Normalize input to [-1, 1] for stable B-spline computation
        x_normalized = torch.tanh(x)
        
        # Compute B-spline basis functions
        basis = self.b_splines(x_normalized)  # (batch, in_features, num_basis)
        
        # Apply learnable spline weights
        # Einstein summation: batch, in_feat, basis * out_feat, in_feat, basis -> batch, out_feat
        spline_output = torch.einsum('bik,oik->bo', basis, self.spline_weight)
        
        # Add base linear transformation (residual)
        base_output = self.base_linear(x)
        
        # Combine spline and linear components
        output = spline_output + base_output
        
        return output


# ============================================================================
# MLP-Mixer Block with KAN
# ============================================================================

class MixerBlock(nn.Module):
    """
    MLP-Mixer block with KAN layers.
    
    From the paper:
    "MLP-Mixer: Captures global structure without attention's O(L²) complexity"
    
    Architecture:
    Input → LayerNorm → Token-Mixing KAN → Residual
                     ↓
                  LayerNorm → Channel-Mixing KAN → Residual → Output
    
    Token-Mixing: Mix information across spatial tokens (frequency bands)
    Channel-Mixing: Mix information across feature channels at each location
    """
    def __init__(self, num_tokens, d_model, mlp_ratio=4, dropout=0.1, 
                 drop_block_size=4, kan_grid_size=5):
        super(MixerBlock, self).__init__()
        
        self.num_tokens = num_tokens
        self.d_model = d_model
        
        # Token-mixing (spatial mixing across frequency bands)
        self.norm1 = nn.LayerNorm(d_model)
        self.token_mixing = nn.Sequential(
            KANLayer(num_tokens, num_tokens * mlp_ratio, grid_size=kan_grid_size),
            nn.Dropout(dropout),
            KANLayer(num_tokens * mlp_ratio, num_tokens, grid_size=kan_grid_size),
        )
        
        # Channel-mixing (feature mixing at each spatial location)
        self.norm2 = nn.LayerNorm(d_model)
        self.channel_mixing = nn.Sequential(
            KANLayer(d_model, d_model * mlp_ratio, grid_size=kan_grid_size),
            nn.Dropout(dropout),
            KANLayer(d_model * mlp_ratio, d_model, grid_size=kan_grid_size),
        )
        
        # DropBlock for spatial regularization
        self.drop_block = DropBlock2D(drop_prob=dropout, block_size=drop_block_size)
        
    def forward(self, x):
        """
        Args:
            x: (batch, num_tokens, d_model)
        Returns:
            out: (batch, num_tokens, d_model)
        """
        batch_size = x.size(0)
        
        # Token-mixing: Mix across tokens for each feature independently
        residual = x
        x = self.norm1(x)
        # Transpose to (batch, d_model, num_tokens)
        x = x.transpose(1, 2)  # (batch, d_model, num_tokens)
        # Reshape to (batch * d_model, num_tokens) for KAN processing
        x = x.reshape(batch_size * self.d_model, self.num_tokens)
        x = self.token_mixing(x)  # (batch * d_model, num_tokens)
        # Reshape back to (batch, d_model, num_tokens)
        x = x.reshape(batch_size, self.d_model, self.num_tokens)
        # Transpose back to (batch, num_tokens, d_model)
        x = x.transpose(1, 2)
        x = residual + x
        
        # Channel-mixing: Mix across features for each token
        residual = x
        x = self.norm2(x)
        # Reshape to (batch * num_tokens, d_model) for KAN processing
        x = x.reshape(batch_size * self.num_tokens, self.d_model)
        x = self.channel_mixing(x)  # (batch * num_tokens, d_model)
        # Reshape back to (batch, num_tokens, d_model)
        x = x.reshape(batch_size, self.num_tokens, self.d_model)
        x = residual + x
        
        return x


# ============================================================================
# DropBlock: Spatial Regularization
# ============================================================================

class DropBlock2D(nn.Module):
    """
    DropBlock: Spatial regularization for convolutional networks.
    
    From the paper:
    "Standard Dropout is weak for STM because adjacent rate/scale bins are 
    highly correlated. DropBlock drops contiguous blocks, forcing distributed 
    representations."
    
    Instead of dropping individual units, DropBlock drops contiguous regions,
    which is more effective for spatially correlated features.
    """
    def __init__(self, drop_prob=0.1, block_size=4):
        super(DropBlock2D, self).__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size
        
    def forward(self, x):
        """
        Args:
            x: (batch, channels, height, width) or (batch, tokens, features)
        """
        if not self.training or self.drop_prob == 0:
            return x
        
        # Handle both 2D and sequence inputs
        is_sequence = (x.dim() == 3)
        if is_sequence:
            # Convert to 4D for DropBlock
            batch, tokens, features = x.shape
            h = int(np.sqrt(tokens))
            w = tokens // h
            x = x.reshape(batch, features, h, w)
        
        # Compute gamma (drop probability adjustment)
        gamma = self.drop_prob / (self.block_size ** 2)
        
        # Sample mask
        mask = torch.bernoulli(torch.ones_like(x) * gamma)
        
        # Compute block mask using max pooling
        mask = F.max_pool2d(
            mask,
            kernel_size=self.block_size,
            stride=1,
            padding=self.block_size // 2
        )
        
        # Invert mask (1 = keep, 0 = drop)
        mask = 1 - mask
        
        # Normalize to maintain expected value
        normalize_factor = mask.numel() / (mask.sum() + 1e-8)
        x = x * mask * normalize_factor
        
        # Convert back to sequence if needed
        if is_sequence:
            x = x.reshape(batch, features, -1).transpose(1, 2)
        
        return x


# ============================================================================
# Asym-Mixer-KAN Model
# ============================================================================

class AsymMixerKAN(nn.Module):
    """
    Asymmetric MLP-Mixer with Kolmogorov-Arnold Networks.
    
    Architecture Philosophy (from paper):
    "The STM space is a semantic map, not a spatial scene. A feature at 4 Hz 
    encodes syllabic rhythm, while 40 Hz encodes roughness. These are distinct 
    auditory objects requiring position-aware processing."
    
    Architecture:
    1. 2-Channel Input Embedding with Coordinate Awareness
    2. Stack of Mixer-KAN Blocks
    3. Global Pooling
    4. Classification Head
    
    Input: (batch, 2, 20, 61)
        - 2 channels: S_up (upward sweeps) + S_down (downward sweeps)
        - 20 frequency bands
        - 61 modulation rates (0 Hz to 15 Hz)
    
    Output: (batch, 6) - logits for 6 classes
    """
    def __init__(self, num_classes=6, d_model=256, depth=12, mlp_ratio=4,
                 dropout=0.1, drop_block_size=4, kan_grid_size=5):
        super(AsymMixerKAN, self).__init__()
        
        self.num_classes = num_classes
        self.d_model = d_model
        self.num_freq_bands = 20
        self.num_mod_rates = 61
        self.num_tokens = self.num_freq_bands  # Each frequency band is a token
        
        # Input embedding: 2-channel 2D map -> token sequence
        # Each token represents one frequency band with all modulation rates
        # Input per token: (2 channels, 61 rates) = 122 features
        self.input_dim = 2 * self.num_mod_rates  # 2 * 61 = 122
        self.patch_embed = nn.Sequential(
            nn.Linear(self.input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
        # Coordinate embeddings (position-aware, not positional)
        # Encode the semantic meaning of grid positions
        self.freq_coord_embed = nn.Parameter(torch.zeros(1, self.num_tokens, d_model))
        nn.init.trunc_normal_(self.freq_coord_embed, std=0.02)
        
        # Rate coordinate encoding (embed into the token representation)
        # This adds awareness of which modulation rates are present
        self.rate_coord_mlp = nn.Sequential(
            nn.Linear(self.num_mod_rates, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, d_model)
        )
        
        # Stack of Mixer blocks
        self.blocks = nn.ModuleList([
            MixerBlock(
                num_tokens=self.num_tokens,
                d_model=d_model,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                drop_block_size=drop_block_size,
                kan_grid_size=kan_grid_size
            )
            for _ in range(depth)
        ])
        
        # Final normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Classification head
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: (batch, 2, 20, 61) - 2-channel asymmetric STM
        Returns:
            logits: (batch, num_classes)
        """
        batch_size = x.size(0)
        
        # Reshape: (batch, 2, 20, 61) -> (batch, 20, 2*61)
        # Each frequency band becomes a token with 122 features (2 channels × 61 rates)
        x = x.permute(0, 2, 1, 3)  # (batch, 20, 2, 61)
        x = x.reshape(batch_size, self.num_tokens, self.input_dim)  # (batch, 20, 122)
        
        # Patch embedding
        x = self.patch_embed(x)  # (batch, 20, d_model)
        
        # Add frequency coordinate embeddings
        x = x + self.freq_coord_embed
        
        # Add rate coordinate information
        # Create a coordinate map for modulation rates
        rate_coords = torch.linspace(0, 1, self.num_mod_rates, device=x.device)
        rate_coords = rate_coords.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_tokens, -1)
        rate_encoding = self.rate_coord_mlp(rate_coords)
        x = x + rate_encoding
        
        # Apply Mixer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final normalization
        x = self.norm(x)
        
        # Global average pooling across tokens
        x = x.mean(dim=1)  # (batch, d_model)
        
        # Classification head
        logits = self.head(x)  # (batch, num_classes)
        
        return logits


# ============================================================================
# LDAM Loss with Deferred Reweighting (DRW)
# ============================================================================

class LDAMLoss(nn.Module):
    """
    Label-Distribution-Aware Margin Loss with Deferred Reweighting.
    
    From the paper:
    "LDAM pushes the decision boundary away from minority classes by 
    introducing class-dependent margins in the softmax formulation."
    
    Loss formulation:
        L = -log( exp(z_y - m_y) / Σ_j exp(z_j - m_j) )
    
    where m_y is the margin for class y, computed as:
        m_y = C / n_y^(1/4)
    
    Deferred Reweighting (DRW):
    - Epochs 0-40: Pure LDAM (no reweighting) to learn representations
    - Epochs 40-50: Enable class reweighting to fine-tune boundaries
    """
    def __init__(self, class_freq, max_m=0.5, s=30):
        super(LDAMLoss, self).__init__()
        
        # Compute margins inversely proportional to class frequency
        # m_y = max_m * (1 / freq_y)^(1/4)
        m_list = max_m * (1.0 / class_freq) ** 0.25
        m_list = m_list / m_list.max()  # Normalize
        self.m_list = torch.FloatTensor(m_list)
        
        self.s = s  # Scale factor
        self.class_freq = class_freq
        
    def forward(self, x, target, epoch=0, drw_start_epoch=40):
        """
        Args:
            x: Logits (batch, num_classes)
            target: True labels (batch,)
            epoch: Current training epoch
            drw_start_epoch: When to start deferred reweighting
        """
        device = x.device
        self.m_list = self.m_list.to(device)
        
        # Create margin tensor
        batch_m = self.m_list[target]  # (batch,)
        
        # Subtract margin from target class logits
        index = torch.zeros_like(x, dtype=torch.bool)
        index.scatter_(1, target.unsqueeze(1), True)
        
        x_m = x.clone()
        x_m[index] = x_m[index] - batch_m
        
        # Scale logits
        x_m = x_m * self.s
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(x_m, target, reduction='none')
        
        # Apply deferred reweighting if past threshold
        if epoch >= drw_start_epoch:
            # Compute per-class weights inversely proportional to frequency
            weights = 1.0 / torch.FloatTensor(self.class_freq).to(device)
            weights = weights / weights.sum() * len(weights)  # Normalize
            sample_weights = weights[target]
            loss = loss * sample_weights
        
        return loss.mean()


# ============================================================================
# CutMix Augmentation
# ============================================================================

def cutmix_data(x, y, alpha=1.0):
    """
    CutMix augmentation: Cut and paste patches between samples.
    
    From the paper:
    "CutMix preserves local structure of modulation. A patch of 'speech rhythm' 
    is pasted onto 'music harmonics'. The model learns to recognize both 
    distinct objects occurring simultaneously, which perfectly mimics 'Vocal Music' 
    compositionality."
    
    Args:
        x: Input tensor (batch, channels, height, width)
        y: Labels (batch,)
        alpha: Beta distribution parameter
    
    Returns:
        mixed_x: Mixed input
        y_a: First label
        y_b: Second label
        lam: Mixing coefficient
    """
    batch_size = x.size(0)
    
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    # Random permutation
    index = torch.randperm(batch_size, device=x.device)
    
    # Get random box coordinates
    _, _, H, W = x.size()
    cut_rat = np.sqrt(1.0 - lam)
    cut_h = int(H * cut_rat)
    cut_w = int(W * cut_rat)
    
    # Random center point
    cx = np.random.randint(H)
    cy = np.random.randint(W)
    
    # Compute box coordinates
    bbx1 = np.clip(cx - cut_h // 2, 0, H)
    bby1 = np.clip(cy - cut_w // 2, 0, W)
    bbx2 = np.clip(cx + cut_h // 2, 0, H)
    bby2 = np.clip(cy + cut_w // 2, 0, W)
    
    # Mix the images
    mixed_x = x.clone()
    mixed_x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    # Adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (H * W))
    
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


# ============================================================================
# Data Preparation
# ============================================================================

class prepData_STM_MixerKAN:
    """
    Data preparation for Asym-Mixer-KAN model.
    Follows the same structure as STM_ViM.py but outputs 2-channel format.
    """
    def __init__(self, addAug=False, ds_nontonal_speech=False):
        self.addAug = addAug
        self.ds_nontonal_speech = ds_nontonal_speech
        
        # STM dimensions (before asymmetric processing)
        self.n_freq = 20
        self.n_time = 121
        
    def corpora_list(self, addAug=False):
        """Generate list of all corpora"""
        corpus_speech_list = ['BibleTTS/akuapem-twi',
            'BibleTTS/asante-twi', 'BibleTTS/ewe', 'BibleTTS/hausa',
            'BibleTTS/lingala', 'BibleTTS/yoruba', 'Buckeye', 'EUROM',
            'HiltonMoser2022_speech', 'LibriSpeech',
            'MediaSpeech/AR', 'MediaSpeech/ES', 'MediaSpeech/FR', 'MediaSpeech/TR',
            'MozillaCommonVoice/ab', 'MozillaCommonVoice/ar', 'MozillaCommonVoice/ba',
            'MozillaCommonVoice/be', 'MozillaCommonVoice/bg', 'MozillaCommonVoice/bn',
            'MozillaCommonVoice/br', 'MozillaCommonVoice/ca', 'MozillaCommonVoice/ckb',
            'MozillaCommonVoice/cnh', 'MozillaCommonVoice/cs', 'MozillaCommonVoice/cv',
            'MozillaCommonVoice/cy', 'MozillaCommonVoice/da', 'MozillaCommonVoice/de',
            'MozillaCommonVoice/dv', 'MozillaCommonVoice/el', 'MozillaCommonVoice/en',
            'MozillaCommonVoice/eo', 'MozillaCommonVoice/es', 'MozillaCommonVoice/et',
            'MozillaCommonVoice/eu', 'MozillaCommonVoice/fa', 'MozillaCommonVoice/fi',
            'MozillaCommonVoice/fr', 'MozillaCommonVoice/fy-NL', 'MozillaCommonVoice/ga-IE',
            'MozillaCommonVoice/gl', 'MozillaCommonVoice/gn', 'MozillaCommonVoice/hi',
            'MozillaCommonVoice/hu', 'MozillaCommonVoice/hy-AM', 'MozillaCommonVoice/id',
            'MozillaCommonVoice/ig', 'MozillaCommonVoice/it', 'MozillaCommonVoice/ja',
            'MozillaCommonVoice/ka', 'MozillaCommonVoice/kab', 'MozillaCommonVoice/kk',
            'MozillaCommonVoice/kmr', 'MozillaCommonVoice/ky', 'MozillaCommonVoice/lg',
            'MozillaCommonVoice/lt', 'MozillaCommonVoice/ltg', 'MozillaCommonVoice/lv',
            'MozillaCommonVoice/mhr', 'MozillaCommonVoice/ml', 'MozillaCommonVoice/mn',
            'MozillaCommonVoice/mt', 'MozillaCommonVoice/nan-tw', 'MozillaCommonVoice/nl',
            'MozillaCommonVoice/oc', 'MozillaCommonVoice/or', 'MozillaCommonVoice/pl',
            'MozillaCommonVoice/pt', 'MozillaCommonVoice/ro', 'MozillaCommonVoice/ru',
            'MozillaCommonVoice/rw', 'MozillaCommonVoice/sr', 'MozillaCommonVoice/sv-SE',
            'MozillaCommonVoice/sw', 'MozillaCommonVoice/ta', 'MozillaCommonVoice/th',
            'MozillaCommonVoice/tr', 'MozillaCommonVoice/tt', 'MozillaCommonVoice/ug',
            'MozillaCommonVoice/uk', 'MozillaCommonVoice/ur', 'MozillaCommonVoice/uz',
            'MozillaCommonVoice/vi', 'MozillaCommonVoice/yo', 'MozillaCommonVoice/yue',
            'MozillaCommonVoice/zh-CN', 'MozillaCommonVoice/zh-TW',
            'primewords_chinese', 'room_reader', 'SpeechClarity', 'TAT-Vol2',
            'thchs30', 'TIMIT', 'TTS_Javanese', 'zeroth_korean'
        ]
        
        corpus_music_list = [
            'IRMAS', 'Albouy2020Science', 'GarlandEncyclopedia', 'fma_large',
            'ismir04_genre', 'MTG-Jamendo', 'HiltonMoser2022_song', 'NHS2', 'MagnaTagATune'
        ]
        
        if addAug:
            corpus_env_list = ['SONYC', 'MacaulayLibrary', 'SONYC_augmented']
        else:
            corpus_env_list = ['SONYC', 'MacaulayLibrary']
        
        corpus_speech_list.sort()
        corpus_music_list.sort()
        corpus_env_list.sort()
        
        return corpus_speech_list + corpus_music_list + corpus_env_list
    
    def load_data(self):
        """Load and preprocess STM data"""
        corpus_list_all = self.corpora_list(self.addAug)
        root_folder = '/vast-ac8888/MusicSpeech-STM/'
        
        STM_all = None
        for corp in corpus_list_all:
            filename = root_folder + 'STM_output/corpSTMnpy/' + corp.replace('/', '-') + '_STMall.npy'
            if STM_all is None:
                STM_all = np.load(filename)
            else:
                STM_all = np.vstack((STM_all, np.load(filename)))
            print(f"Loaded: {filename}, shape: {np.load(filename).shape}")
        
        # Load metadata
        speech_corp_df1 = pd.read_csv(root_folder + 'train_test_split/speech1_10folds_speakerGroupFold.csv', index_col=0)
        speech_corp_df2 = pd.read_csv(root_folder + 'train_test_split/speech2_10folds_speakerGroupFold.csv', index_col=0)
        music_corp_df = pd.read_csv(root_folder + 'train_test_split/music_10folds_speakerGroupFold.csv', index_col=0)
        df_SONYC = pd.read_csv(root_folder + 'train_test_split/env_10folds_speakerGroupFold.csv', index_col=0)
        
        all_corp_df = pd.concat([speech_corp_df1, speech_corp_df2, music_corp_df, df_SONYC], ignore_index=True)
        
        # Handle augmented data
        if self.addAug:
            SONYC_aug_len = np.load(root_folder + 'STM_output/corpSTMnpy/SONYC_augmented_STMall.npy').shape[0]
            target = pd.concat([all_corp_df['corpus_type'], pd.Series(['env: urban'] * SONYC_aug_len)], ignore_index=True)
            data_split = pd.concat([all_corp_df['10fold_labels'], pd.Series([1] * SONYC_aug_len)], ignore_index=True)
        else:
            target = all_corp_df['corpus_type'].copy()
            data_split = all_corp_df['10fold_labels'].copy()
        
        target.replace({
            'speech: non-tonal': 0,
            'speech: tonal': 1,
            'music: vocal': 2,
            'music: non-vocal': 3,
            'env: urban': 4,
            'env: wildlife': 5,
        }, inplace=True)
        
        # Downsample non-tonal speech if requested
        if self.ds_nontonal_speech:
            num_samples = 100000
            indices_target_0 = target.index[target == 0].to_numpy()
            
            if len(indices_target_0) < num_samples:
                raise ValueError(f"Not enough rows with target == 0 to sample {num_samples} rows.")
            
            np.random.seed(23)
            sampled_indices = np.random.choice(indices_target_0, size=num_samples, replace=False)
            
            mask = np.ones(len(target), dtype=bool)
            mask[indices_target_0] = False
            mask[sampled_indices] = True
            
            STM_all = STM_all[mask, :]
            data_split = data_split[mask].reset_index(drop=True)
            target = target[mask].reset_index(drop=True)
        
        # Split data
        train_ind = (data_split < 8).values
        val_ind = (data_split == 8).values
        test_ind = (data_split == 9).values
        
        # Compute class frequencies for LDAM
        train_labels = target[train_ind].values.astype(np.int64)
        class_counts = np.bincount(train_labels, minlength=6)
        class_freq = class_counts / class_counts.sum()
        
        print(f"\nDataset Statistics:")
        print(f"Total samples: {len(STM_all)}")
        print(f"Train samples: {sum(train_ind)}")
        print(f"Val samples: {sum(val_ind)}")
        print(f"Test samples: {sum(test_ind)}")
        print(f"Input shape (before asymmetric processing): ({self.n_freq}, {self.n_time})")
        print(f"Output shape (after asymmetric processing): (2, {self.n_freq}, 61)")
        print(f"\nClass Distribution (Training):")
        class_names = ['speech:non-tonal', 'speech:tonal', 'music:vocal', 
                      'music:non-vocal', 'env:urban', 'env:wildlife']
        for i, count in enumerate(class_counts):
            print(f"  {class_names[i]}: {count} ({class_freq[i]*100:.2f}%)")
        
        return STM_all, target.values, train_ind, val_ind, test_ind, class_freq
    
    def prepare_datasets(self):
        """Prepare PyTorch datasets"""
        STM_all, target, train_ind, val_ind, test_ind, class_freq = self.load_data()
        
        # Normalize per sample (preserve relative patterns)
        means = STM_all.mean(axis=1, keepdims=True)
        stds = STM_all.std(axis=1, keepdims=True)
        STM_all_norm = (STM_all - means) / (stds + 1e-8)
        
        # Convert to PyTorch tensors
        X_train = torch.FloatTensor(STM_all_norm[train_ind])
        y_train = torch.LongTensor(target[train_ind].astype(np.int64))
        
        X_val = torch.FloatTensor(STM_all_norm[val_ind])
        y_val = torch.LongTensor(target[val_ind].astype(np.int64))
        
        X_test = torch.FloatTensor(STM_all_norm[test_ind])
        y_test = torch.LongTensor(target[test_ind].astype(np.int64))
        
        # Create base datasets
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)
        
        print(f"\nPyTorch Dataset Shapes (before asymmetric processing):")
        print(f"Train: {X_train.shape}")
        print(f"Val: {X_val.shape}")
        print(f"Test: {X_test.shape}")
        
        return train_dataset, val_dataset, test_dataset, class_freq


# ============================================================================
# Trainer
# ============================================================================

class Trainer:
    """Training manager for Asym-Mixer-KAN"""
    def __init__(self, model, train_loader, val_loader, test_loader, 
                 device, class_freq, lr=1e-3, weight_decay=1e-4, 
                 cutmix_prob=0.5, cutmix_alpha=1.0, drw_start_epoch=40):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        
        # Loss function
        self.criterion = LDAMLoss(class_freq, max_m=0.5, s=30)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Scheduler: CosineAnnealingWarmRestarts
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,  # Restart every 10 epochs
            T_mult=2,  # Double the restart period after each restart
            eta_min=1e-6
        )
        
        # CutMix parameters
        self.cutmix_prob = cutmix_prob
        self.cutmix_alpha = cutmix_alpha
        
        # DRW parameter
        self.drw_start_epoch = drw_start_epoch
        
        # Tracking
        self.current_epoch = 0
        self.best_val_f1 = 0.0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_f1': [],
            'val_f1_per_class': []
        }
        
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint and resume training"""
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_f1 = checkpoint['best_val_f1']
        self.history = checkpoint['history']
        
        print(f"Resumed from epoch {self.current_epoch}")
        print(f"Best validation F1: {self.best_val_f1:.4f}")
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Apply CutMix with probability
            if np.random.rand() < self.cutmix_prob:
                data, target_a, target_b, lam = cutmix_data(data, target, self.cutmix_alpha)
                
                # Forward pass
                logits = self.model(data)
                
                # Mixed loss
                loss_a = self.criterion(logits, target_a, self.current_epoch, self.drw_start_epoch)
                loss_b = self.criterion(logits, target_b, self.current_epoch, self.drw_start_epoch)
                loss = lam * loss_a + (1 - lam) * loss_b
            else:
                # Standard forward pass
                logits = self.model(data)
                loss = self.criterion(logits, target, self.current_epoch, self.drw_start_epoch)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def evaluate(self, data_loader):
        """Evaluate model"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                logits = self.model(data)
                loss = F.cross_entropy(logits, target)
                
                # Predictions
                preds = torch.argmax(logits, dim=1)
                
                total_loss += loss.item()
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        avg_loss = total_loss / len(data_loader)
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        # Compute metrics
        macro_f1 = f1_score(all_targets, all_preds, average='macro')
        per_class_f1 = f1_score(all_targets, all_preds, average=None)
        
        return avg_loss, macro_f1, per_class_f1, all_preds, all_targets
    
    def train(self, num_epochs, checkpoint_dir):
        """Main training loop"""
        print("\n" + "="*80)
        print("Starting Training")
        print("="*80)
        
        class_names = ['speech:non-tonal', 'speech:tonal', 'music:vocal', 
                      'music:non-vocal', 'env:urban', 'env:wildlife']
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss, val_f1, val_f1_per_class, _, _ = self.evaluate(self.val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_f1'].append(val_f1)
            self.history['val_f1_per_class'].append(val_f1_per_class)
            
            epoch_time = time.time() - epoch_start_time
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Print progress
            print(f"\nEpoch [{epoch+1}/{num_epochs}] - {epoch_time:.2f}s - LR: {current_lr:.6f}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val Macro F1: {val_f1:.4f}")
            print(f"  Val Per-Class F1:")
            for i, f1 in enumerate(val_f1_per_class):
                print(f"    {class_names[i]}: {f1:.4f}")
            
            # Save best model
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_val_f1': self.best_val_f1,
                    'history': self.history
                }, os.path.join(checkpoint_dir, 'best_model.pt'))
                print(f"  *** New best model saved! (F1: {val_f1:.4f}) ***")
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_val_f1': self.best_val_f1,
                    'history': self.history
                }, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt'))
        
        print("\n" + "="*80)
        print("Training Complete!")
        print(f"Best Validation F1: {self.best_val_f1:.4f}")
        print("="*80)


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python STMasm_mixer_kan.py <mode> [--resume <checkpoint_dir>]")
        print("  mode 0: Full dataset")
        print("  mode 1: Downsampled non-tonal speech")
        print("  --resume <dir>: Resume training from checkpoint directory")
        sys.exit(1)
    
    mode = int(sys.argv[1])
    
    # Check for resume flag
    resume_dir = None
    if '--resume' in sys.argv:
        resume_idx = sys.argv.index('--resume')
        resume_dir = sys.argv[resume_idx + 1]
        print(f"Resume mode: Loading from {resume_dir}")
    
    # Set parameters based on mode
    if mode == 0:
        ds_nontonal_speech = False
        mode_name = "full"
    elif mode == 1:
        ds_nontonal_speech = True
        mode_name = "balanced"
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)
    
    # Create checkpoint directory
    if resume_dir:
        checkpoint_dir = resume_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_dir = f"model/STM/Asym-Mixer-KAN_{mode_name}_{timestamp}"
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"Checkpoint directory: {checkpoint_dir}")
    
    # Prepare data
    print("\n" + "="*80)
    print("Loading and preparing data...")
    print("="*80)
    
    data_prep = prepData_STM_MixerKAN(ds_nontonal_speech=ds_nontonal_speech)
    train_dataset, val_dataset, test_dataset, class_freq = data_prep.prepare_datasets()
    
    # Apply asymmetric STM processing
    print(f"\nApplying asymmetric 2-channel STM processing...")
    print(f"Original: 20 freq × 121 rates = 2420 features")
    print(f"After processing: 2 channels × 20 freq × 61 rates")
    print(f"  Channel 0: S_up (upward frequency sweeps, flipped)")
    print(f"  Channel 1: S_down (downward frequency sweeps)")
    
    train_dataset = AsymmetricSTMDataset(train_dataset)
    val_dataset = AsymmetricSTMDataset(val_dataset)
    test_dataset = AsymmetricSTMDataset(test_dataset)
    
    # Create data loaders
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    
    print(f"\nDataLoaders created with batch_size={batch_size}")
    
    # Create model
    print("\n" + "="*80)
    print("Creating Asym-Mixer-KAN model...")
    print("="*80)
    
    num_classes = 6
    model = AsymMixerKAN(
        num_classes=num_classes,
        d_model=256,           # Feature dimension
        depth=12,              # Number of Mixer blocks
        mlp_ratio=4,           # MLP expansion ratio
        dropout=0.1,
        drop_block_size=4,     # DropBlock spatial size
        kan_grid_size=5        # B-spline grid size for KAN
    )
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        class_freq=class_freq,
        lr=1e-3,
        weight_decay=1e-4,
        cutmix_prob=0.5,
        cutmix_alpha=1.0,
        drw_start_epoch=40  # Start deferred reweighting at epoch 40
    )
    
    # Resume from checkpoint if specified
    if resume_dir:
        best_checkpoint = os.path.join(resume_dir, 'best_model.pt')
        if os.path.exists(best_checkpoint):
            trainer.load_checkpoint(best_checkpoint)
        else:
            # Try to find latest checkpoint
            checkpoints = [f for f in os.listdir(resume_dir) if f.startswith('checkpoint_epoch_')]
            if checkpoints:
                latest_checkpoint = sorted(checkpoints)[-1]
                trainer.load_checkpoint(os.path.join(resume_dir, latest_checkpoint))
            else:
                print("No checkpoint found in resume directory. Starting from scratch.")
    
    # Train model
    num_epochs = 50
    trainer.train(num_epochs=num_epochs, checkpoint_dir=checkpoint_dir)
    
    # Evaluate on test set
    print("\n" + "="*80)
    print("Evaluating on test set...")
    print("="*80)
    
    # Load best model
    checkpoint = torch.load(os.path.join(checkpoint_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_f1, test_f1_per_class, test_preds, test_targets = trainer.evaluate(test_loader)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Macro F1: {test_f1:.4f}")
    
    # Detailed classification report
    print("\nClassification Report:")
    target_names = ['speech:non-tonal', 'speech:tonal', 'music:vocal', 
                   'music:non-vocal', 'env:urban', 'env:wildlife']
    print(classification_report(test_targets, test_preds, target_names=target_names))
    
    # Save test predictions
    np.save(os.path.join(checkpoint_dir, 'test_predictions.npy'), test_preds)
    np.save(os.path.join(checkpoint_dir, 'test_targets.npy'), test_targets)
    
    print("\n" + "="*80)
    print("Done!")
    print("="*80)
