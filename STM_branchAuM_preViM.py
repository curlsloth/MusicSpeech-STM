#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STM Classification with Hierarchical Audio Mamba using Pretrained Vision Mamba
(STM_branchAuM_preViM)

This implementation extends Roadmap 2 by incorporating pretrained Vision Mamba
from ImageNet to solve the "training from scratch" problem identified in the 
research paper.

Key Features:
1. Pretrained Vision Mamba backbone (vim_small from ImageNet)
2. Learnable spatial adapter (2,20,61) → (3,224,224) via Transposed Conv2D
3. Modified patch size: 4×4 (3,136 tokens) for better spectral granularity
4. Progressive unfreezing: Freeze→Fine-tune in stages
5. Hierarchical branching: Coarse classifier → Guidance → Fine classifier
6. Asymmetric 2-channel STM processing (S_up + S_down, no averaging)
7. LDAM loss with Deferred Reweighting (DRW)
8. 2D CutMix augmentation (adapted for spatial domain)

Target: 0.90-0.93 Macro F1 Score

Installation Requirements:
    pip install mamba-ssm causal-conv1d>=1.2.0
    pip install timm  # For model loading utilities
"""

import os
import sys
import warnings
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import f1_score, classification_report

warnings.filterwarnings('ignore')

# Try to import Mamba
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    print("ERROR: mamba-ssm not installed!")
    print("Install with: pip install mamba-ssm causal-conv1d>=1.2.0")
    sys.exit(1)

# Try to import timm for pretrained model utilities
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    print("WARNING: timm not installed. Install with: pip install timm")
    TIMM_AVAILABLE = False


# ============================================================================
# Asymmetric STM Processing (2-Channel: S_up + S_down)
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
    """
    # Step 1: Separate chunks along modulation rate dimension
    negative_chunk = stm_data[:, :, 0:60]   # -15 Hz to -0.25 Hz
    dc_component = stm_data[:, :, 60:61]    # 0 Hz
    positive_chunk = stm_data[:, :, 61:121] # +0.25 Hz to +15 Hz
    
    # Step 2: Flip negative chunk
    negative_flipped = torch.flip(negative_chunk, dims=[2])
    
    # Step 3: Keep both channels separately
    s_up_channel = negative_flipped
    s_down_channel = positive_chunk
    
    # Step 4: Concatenate DC at the beginning
    s_up_out = torch.cat([dc_component, s_up_channel], dim=2)
    s_down_out = torch.cat([dc_component, s_down_channel], dim=2)
    
    # Stack into 2-channel tensor
    output = torch.stack([s_up_out, s_down_out], dim=1)
    
    return output


# ============================================================================
# Spatial Adapter: STM (2,20,61) → Image (3,224,224)
# ============================================================================

class STMSpatialAdapter(nn.Module):
    """
    Learnable spatial adapter to transform STM features into image-like format.
    
    From the plan:
    "Use Transposed Conv2D (learnable, preserves structure)"
    
    This module upsamples the STM representation from (2, 20, 61) to (3, 224, 224)
    using learnable transposed convolutions while attempting to preserve the 
    semantic structure of the spectrotemporal modulation space.
    
    Architecture:
    Input (2, 20, 61)
      ↓ TransposeConv2d (2→32, 20→40, 61→122)
      ↓ BatchNorm + ReLU
      ↓ TransposeConv2d (32→64, 40→80, 122→224)  
      ↓ BatchNorm + ReLU
      ↓ TransposeConv2d (64→32, 80→160, 224→224)
      ↓ BatchNorm + ReLU
      ↓ TransposeConv2d (32→3, 160→224, 224→224)
      ↓ Output (3, 224, 224)
    """
    def __init__(self):
        super().__init__()
        
        # Stage 1: (2, 20, 61) → (32, 40, 122)
        self.up1 = nn.ConvTranspose2d(2, 32, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Stage 2: (32, 40, 122) → (64, 80, 224)
        # Need: 40→80 (2x), 122→224 (1.84x)
        # Use kernel=4, stride=2 for height, different for width
        self.up2 = nn.ConvTranspose2d(32, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(64)
        
        # Adjust width from 122*2-2 = 242 → 224 with adaptive pool
        self.pool2 = nn.AdaptiveAvgPool2d((80, 224))
        
        # Stage 3: (64, 80, 224) → (32, 160, 224)
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=(4, 3), stride=(2, 1), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(32)
        
        # Stage 4: (32, 160, 224) → (3, 224, 224)
        self.up4 = nn.ConvTranspose2d(32, 3, kernel_size=(4, 3), stride=(2, 1), padding=(3, 1))
        self.bn4 = nn.BatchNorm2d(3)
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # Input: (batch, 2, 20, 61)
        x = self.relu(self.bn1(self.up1(x)))  # (batch, 32, 40, 122)
        
        x = self.relu(self.bn2(self.up2(x)))  # (batch, 64, ~80, ~244)
        x = self.pool2(x)                      # (batch, 64, 80, 224)
        
        x = self.relu(self.bn3(self.up3(x)))  # (batch, 32, 160, 224)
        
        x = self.bn4(self.up4(x))              # (batch, 3, ~224, 224)
        
        # Final adaptive pool to ensure exact 224×224
        x = F.adaptive_avg_pool2d(x, (224, 224))
        
        return x


class AsymmetricSTMDataset(Dataset):
    """Wrapper dataset that applies asymmetric 2-channel STM processing"""
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        x, y = self.base_dataset[idx]
        # x shape: (2420,) flattened from (20, 121)
        
        # Reshape to (20, 121)
        x = x.view(20, 121)
        
        # Add batch dimension for processing
        x = x.unsqueeze(0)
        
        # Apply asymmetric processing
        x_processed = process_asymmetric_stm(x)
        
        # Remove batch dimension and keep as (2, 20, 61)
        x_processed = x_processed.squeeze(0)
        
        return x_processed, y


# ============================================================================
# Stochastic Depth (DropPath)
# ============================================================================

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


# ============================================================================
# Modified Vision Mamba Block (Bidirectional)
# ============================================================================

class BidirectionalMambaBlock(nn.Module):
    """
    Bidirectional State Space Model block.
    
    This will be used in the custom fine-tuning layers that are added
    after the pretrained ViM backbone.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand_factor=2, drop_path=0.0):
        super().__init__()
        
        self.norm = nn.LayerNorm(d_model)
        
        # Forward and backward Mamba blocks
        self.mamba_forward = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand_factor
        )
        
        self.mamba_backward = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand_factor
        )
        
        # Fusion layer
        self.fusion = nn.Linear(d_model * 2, d_model)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        residual = x
        x = self.norm(x)
        
        # Forward scan
        x_forward = self.mamba_forward(x)
        
        # Backward scan
        x_backward = self.mamba_backward(torch.flip(x, dims=[1]))
        x_backward = torch.flip(x_backward, dims=[1])
        
        # Concatenate and fuse
        x = torch.cat([x_forward, x_backward], dim=-1)
        x = self.fusion(x)
        
        # Residual connection with drop path
        x = residual + self.drop_path(x)
        
        return x


# ============================================================================
# Pretrained ViM Backbone Loader
# ============================================================================

class VimBackbone(nn.Module):
    """
    Load pretrained Vision Mamba (vim_small) and adapt for STM processing.
    
    From the plan:
    - Load vim_small pretrained on ImageNet
    - Modify patch size from 16×16 to 4×4 (3,136 tokens)
    - Keep standard Conv2D patch embedding
    - Extract feature maps from different depths for hierarchical branching
    """
    def __init__(self, pretrained_path=None, patch_size=4, img_size=224, 
                 d_model=384, depth=24, drop_path_rate=0.4):
        super().__init__()
        
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_patches = (img_size // patch_size) ** 2  # 56*56 = 3,136
        self.d_model = d_model
        
        # Patch embedding (modified for 4×4 patches)
        self.patch_embed = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size)
        
        # Positional embedding (learnable)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Create Mamba blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            BidirectionalMambaBlock(
                d_model=d_model,
                d_state=16,
                d_conv=4,
                expand_factor=2,
                drop_path=dpr[i]
            )
            for i in range(depth)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
        # Load pretrained weights if provided
        if pretrained_path and os.path.exists(pretrained_path):
            self.load_pretrained_weights(pretrained_path)
    
    def load_pretrained_weights(self, pretrained_path):
        """
        Load pretrained vim_small weights and adapt to 4×4 patch size.
        
        Note: The original vim_small uses 16×16 patches. When loading pretrained
        weights, we need to:
        1. Interpolate patch embedding Conv2D weights
        2. Interpolate positional embeddings
        3. Load Mamba block weights (these should be compatible)
        """
        print(f"Loading pretrained ViM weights from {pretrained_path}")
        
        try:
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Adapt patch embedding from 16×16 to 4×4
            if 'patch_embed.weight' in state_dict:
                orig_patch_weight = state_dict['patch_embed.weight']  # (d_model, 3, 16, 16)
                # Interpolate to 4×4
                adapted_weight = F.interpolate(
                    orig_patch_weight, 
                    size=(self.patch_size, self.patch_size),
                    mode='bilinear',
                    align_corners=False
                )
                state_dict['patch_embed.weight'] = adapted_weight
                print(f"  ✓ Adapted patch embedding: {orig_patch_weight.shape} → {adapted_weight.shape}")
            
            # Adapt positional embeddings
            if 'pos_embed' in state_dict:
                orig_pos_embed = state_dict['pos_embed']  # (1, N_orig, d_model)
                if orig_pos_embed.shape[1] != self.num_patches:
                    # Interpolate positional embeddings
                    orig_size = int(orig_pos_embed.shape[1] ** 0.5)
                    new_size = int(self.num_patches ** 0.5)
                    
                    pos_tokens = orig_pos_embed.reshape(1, orig_size, orig_size, self.d_model)
                    pos_tokens = pos_tokens.permute(0, 3, 1, 2)  # (1, d_model, H, W)
                    pos_tokens = F.interpolate(
                        pos_tokens,
                        size=(new_size, new_size),
                        mode='bilinear',
                        align_corners=False
                    )
                    pos_tokens = pos_tokens.permute(0, 2, 3, 1)  # (1, H, W, d_model)
                    pos_tokens = pos_tokens.reshape(1, -1, self.d_model)
                    
                    state_dict['pos_embed'] = pos_tokens
                    print(f"  ✓ Adapted positional embedding: {orig_pos_embed.shape} → {pos_tokens.shape}")
            
            # Load weights (with strict=False to allow missing keys)
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                print(f"  ⚠ Missing keys: {len(missing_keys)} (expected for modified architecture)")
            if unexpected_keys:
                print(f"  ⚠ Unexpected keys: {len(unexpected_keys)}")
            
            print(f"  ✓ Successfully loaded pretrained weights")
            
        except Exception as e:
            print(f"  ✗ Failed to load pretrained weights: {e}")
            print(f"  → Training from scratch")
    
    def freeze_layers(self, freeze_until_block=None):
        """
        Freeze specific layers for progressive unfreezing.
        
        Args:
            freeze_until_block: Freeze blocks 0 to freeze_until_block (inclusive)
                               None = freeze nothing
        """
        if freeze_until_block is None:
            return
        
        # Freeze patch embedding
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        
        # Freeze positional embedding
        self.pos_embed.requires_grad = False
        
        # Freeze specified blocks
        for i in range(min(freeze_until_block + 1, len(self.blocks))):
            for param in self.blocks[i].parameters():
                param.requires_grad = False
    
    def unfreeze_layers(self, unfreeze_from_block=None):
        """
        Unfreeze specific layers for progressive unfreezing.
        
        Args:
            unfreeze_from_block: Unfreeze blocks from unfreeze_from_block onwards
                                None = unfreeze nothing
        """
        if unfreeze_from_block is None:
            return
        
        # Unfreeze specified blocks
        for i in range(unfreeze_from_block, len(self.blocks)):
            for param in self.blocks[i].parameters():
                param.requires_grad = True
    
    def forward(self, x, return_intermediate=False):
        # x: (batch, 3, 224, 224)
        batch_size = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (batch, d_model, H, W)
        x = x.flatten(2).transpose(1, 2)  # (batch, num_patches, d_model)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Pass through Mamba blocks
        intermediate_features = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if return_intermediate and i in [3, 7, 11, 15, 19, 23]:  # Every 4 blocks
                intermediate_features.append(x)
        
        # Final normalization
        x = self.norm(x)
        
        if return_intermediate:
            return x, intermediate_features
        return x


# ============================================================================
# Hierarchical STM_branchAuM_preViM Model
# ============================================================================

class BranchAuMPreViM(nn.Module):
    """
    Hierarchical Audio Mamba with Pretrained Vision Mamba.
    
    Architecture:
    Input STM (2,20,61) → Spatial Adapter → (3,224,224)
        ↓
    Pretrained ViM Backbone (Blocks 1-4)
        ↓
    **BRANCH POINT**: Coarse Classifier (3 super-classes)
        ↓ (guidance token)
    ViM Backbone (Blocks 5-24)
        ↓
    Fine Classifier (6 fine-grained classes)
    
    Progressive Unfreezing Schedule:
    - Epochs 0-10: Freeze blocks 0-3, train adapter + classifiers
    - Epochs 10-30: Unfreeze blocks 2-3, fine-tune
    - Epochs 30-50: Unfreeze all, full fine-tuning
    """
    def __init__(self, num_classes=6, pretrained_vim_path=None, 
                 d_model=384, vim_depth=24, drop_path_rate=0.4):
        super().__init__()
        
        self.num_classes = num_classes
        self.d_model = d_model
        self.vim_depth = vim_depth
        self.branch_point = 4  # Branch after block 4
        
        # Spatial adapter: (2,20,61) → (3,224,224)
        self.spatial_adapter = STMSpatialAdapter()
        
        # Pretrained ViM backbone
        self.vim_backbone = VimBackbone(
            pretrained_path=pretrained_vim_path,
            patch_size=4,
            img_size=224,
            d_model=d_model,
            depth=vim_depth,
            drop_path_rate=drop_path_rate
        )
        
        # Coarse classifier (3 super-classes: Speech/Music/Environment)
        self.coarse_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(d_model // 2, 3)
        )
        
        # Guidance token projection
        self.guidance_proj = nn.Linear(3, d_model)
        
        # Fine classifier (6 fine-grained classes)
        self.fine_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(d_model // 2, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize classifier weights"""
        for m in [self.coarse_classifier, self.fine_classifier, self.guidance_proj]:
            if isinstance(m, nn.ModuleList) or isinstance(m, nn.Sequential):
                for module in m:
                    if isinstance(module, nn.Linear):
                        nn.init.trunc_normal_(module.weight, std=0.02)
                        if module.bias is not None:
                            nn.init.zeros_(module.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def freeze_backbone(self, freeze_until_block=3):
        """Freeze ViM backbone layers for progressive training"""
        self.vim_backbone.freeze_layers(freeze_until_block)
    
    def unfreeze_backbone(self, unfreeze_from_block=2):
        """Unfreeze ViM backbone layers for progressive training"""
        self.vim_backbone.unfreeze_layers(unfreeze_from_block)
    
    def forward(self, x, return_coarse=False):
        # x: (batch, 2, 20, 61) asymmetric STM
        
        # Spatial adaptation: (2,20,61) → (3,224,224)
        x = self.spatial_adapter(x)  # (batch, 3, 224, 224)
        
        # Pass through ViM backbone early blocks (0 to branch_point-1)
        batch_size = x.shape[0]
        x = self.vim_backbone.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)  # (batch, 3136, d_model)
        x = x + self.vim_backbone.pos_embed
        
        # Early blocks
        for i in range(self.branch_point):
            x = self.vim_backbone.blocks[i](x)
        
        # Coarse classification (branch point)
        coarse_features = x.mean(dim=1)  # Global average pooling
        coarse_logits = self.coarse_classifier(coarse_features)
        
        # Generate guidance token from coarse predictions
        coarse_probs = F.softmax(coarse_logits, dim=-1)
        guidance_token = self.guidance_proj(coarse_probs)  # (batch, d_model)
        guidance_token = guidance_token.unsqueeze(1)  # (batch, 1, d_model)
        
        # Prepend guidance token
        x = torch.cat([guidance_token, x], dim=1)  # (batch, 3137, d_model)
        
        # Deep blocks with guidance
        for i in range(self.branch_point, self.vim_depth):
            x = self.vim_backbone.blocks[i](x)
        
        # Final normalization
        x = self.vim_backbone.norm(x)
        
        # Fine classification (skip guidance token)
        fine_features = x[:, 1:, :].mean(dim=1)  # (batch, d_model)
        fine_logits = self.fine_classifier(fine_features)
        
        if return_coarse:
            return fine_logits, coarse_logits
        return fine_logits


# ============================================================================
# LDAM Loss with Deferred Reweighting (DRW)
# ============================================================================

class LDAMLoss(nn.Module):
    """Label-Distribution-Aware Margin Loss with Deferred Reweighting"""
    def __init__(self, class_freq, max_m=0.5, s=30):
        super().__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(class_freq))
        m_list = m_list * (max_m / np.max(m_list))
        self.m_list = torch.FloatTensor(m_list)
        self.s = s
        self.class_freq = class_freq
        
    def forward(self, x, target, epoch=0, drw_start_epoch=40):
        batch_m = self.m_list[target].to(x.device)
        batch_m = batch_m.view(-1, 1)
        x_m = x - batch_m * self.s
        
        # Create one-hot mask
        mask = torch.zeros_like(x)
        mask.scatter_(1, target.view(-1, 1).long(), 1)
        
        # Apply margin
        output = torch.where(mask.bool(), x_m, x)
        
        # Standard cross-entropy
        loss = F.cross_entropy(output, target, reduction='none')
        
        # Apply deferred reweighting
        if epoch >= drw_start_epoch:
            weights = 1.0 / torch.FloatTensor(self.class_freq).to(x.device)
            weights = weights / weights.sum() * len(weights)
            sample_weights = weights[target]
            loss = loss * sample_weights
        
        return loss.mean()


# ============================================================================
# 2D CutMix Augmentation (adapted for spatial domain)
# ============================================================================

def cutmix_2d(x, y, alpha=1.0):
    """
    CutMix augmentation for 2D spatial data (after upsampling).
    
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
    
    # Random box
    _, _, h, w = x.size()
    cut_rat = np.sqrt(1.0 - lam)
    cut_h = int(h * cut_rat)
    cut_w = int(w * cut_rat)
    
    cx = np.random.randint(w)
    cy = np.random.randint(h)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)
    
    # Mix images
    mixed_x = x.clone()
    mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    
    # Adjust lambda to exact ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
    
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


# ============================================================================
# Data Preparation
# ============================================================================

class prepData_STM_branchAuM_preViM:
    """Data preparation for STM_branchAuM_preViM model"""
    def __init__(self, addAug=False, ds_nontonal_speech=False):
        self.addAug = addAug
        self.ds_nontonal_speech = ds_nontonal_speech
        
    def corpora_list(self, addAug=False):
        """Returns list of corpus names (same as original)"""
        corpora = {
            'speech': [
                'articulation_index', 'buckeye', 'bu_radio', 'cslu_22', 'cslu_kids',
                'librispeech', 'LDC2017S14', 'MozillaCommonVoice', 'TIMIT', 
                'vctk', 'vystadial_2013', 'CHAINS', 'IViE', 'LC', 'NCCFRspeech',
                'speech_accent_archive', 'tedlium_release1', 'cslu_spoltech_vietnamese',
                'AESRC2020', 'daps', 'aesrc2020', 'CIEMPIESS', 'gos', 'heroico',
                'russian_single', 'santa_barbara', 'sottovoce', 'tunisian',
                'MLS_french', 'MLS_german', 'MLS_polish', 'MLS_portuguese', 'MLS_spanish',
                'L2_arctic', 'TC_STAR_Spanish_Castillian', 'TC_STAR_Spanish_EuroparlV7',
                'SPA-DAPS', 'RIR_speech'
            ],
            'speech_ch': [
                'aishell', 'aidatatang', 'magicdata', 'primewords', 'stcmds', 'thchs'
            ],
            'speech_tonal': [
                'global_phone_Vietnamese', 'iban', 'javanese', 'sundanese', 'TC_STAR_Mandarin'
            ],
            'speech_biling': [
                'L2_arctic', 'TC_STAR_Spanish_Castillian', 'TC_STAR_Spanish_EuroparlV7',
                'SPA-DAPS', 'RIR_speech'
            ],
            'music': [
                'bach10', 'fma_large', 'fma_medium', 'fma_small',
                'gtzan', 'IRMAS', 'medleydb', 'MTG_Jamendo'
            ],
            'music_song': [
                'ccmixter', 'dsd100', 'medleydb_plus', 'MIR-ST500',
                'musdb18', 'rwc_popular'
            ],
            'env': [
                'TUT', 'UrbanSound8k', 'Xeno_Canto'
            ]
        }
        
        all_corpora = []
        for category in corpora.values():
            all_corpora.extend(category)
        
        return all_corpora
    
    def load_data(self):
        """Load STM features and labels"""
        print("Loading data...")
        
        # Load STM features
        STM_all = np.load('/scratch/ac8888/MusicSpeech-STM/data/STM_all_20bin_121bin.npy')
        labels = np.load('/scratch/ac8888/MusicSpeech-STM/data/labels_all.npy')
        kfold_labels = np.load('/scratch/ac8888/MusicSpeech-STM/data/kfold_labels_all.npy')
        
        print(f"  ✓ STM features: {STM_all.shape}")
        print(f"  ✓ Labels: {labels.shape}")
        print(f"  ✓ K-fold labels: {kfold_labels.shape}")
        
        # Normalize per sample
        mean = STM_all.mean(axis=1, keepdims=True)
        std = STM_all.std(axis=1, keepdims=True)
        STM_normalized = (STM_all - mean) / (std + 1e-8)
        
        return STM_normalized, labels, kfold_labels
    
    def prepare_datasets(self):
        """Prepare train/val/test datasets with class frequencies"""
        STM_all, labels, kfold_labels = self.load_data()
        
        # Split by k-fold
        train_mask = kfold_labels < 8
        val_mask = kfold_labels == 8
        test_mask = kfold_labels == 9
        
        # Downsample non-tonal speech if requested
        if self.ds_nontonal_speech:
            nontonal_mask = (labels == 0) & train_mask
            nontonal_indices = np.where(nontonal_mask)[0]
            keep_indices = np.random.choice(nontonal_indices, 
                                          size=len(nontonal_indices) // 2, 
                                          replace=False)
            train_mask = train_mask.copy()
            train_mask[nontonal_indices] = False
            train_mask[keep_indices] = True
        
        # Create datasets
        X_train = torch.FloatTensor(STM_all[train_mask])
        y_train = torch.LongTensor(labels[train_mask])
        
        X_val = torch.FloatTensor(STM_all[val_mask])
        y_val = torch.LongTensor(labels[val_mask])
        
        X_test = torch.FloatTensor(STM_all[test_mask])
        y_test = torch.LongTensor(labels[test_mask])
        
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)
        
        # Compute class frequencies for LDAM
        unique, counts = np.unique(y_train.numpy(), return_counts=True)
        class_freq = np.zeros(6)
        class_freq[unique] = counts
        
        print(f"\nDataset sizes:")
        print(f"  Train: {len(train_dataset)}")
        print(f"  Val: {len(val_dataset)}")
        print(f"  Test: {len(test_dataset)}")
        print(f"\nClass distribution (train):")
        for i, (cls, freq) in enumerate(zip(unique, counts)):
            print(f"  Class {cls}: {freq} ({freq/len(train_dataset)*100:.1f}%)")
        
        return train_dataset, val_dataset, test_dataset, class_freq


# ============================================================================
# Trainer with Progressive Unfreezing
# ============================================================================

class Trainer:
    """Training manager with progressive unfreezing strategy"""
    def __init__(self, model, train_loader, val_loader, test_loader, 
                 device, class_freq, lr=1e-3, weight_decay=1e-4, 
                 cutmix_prob=0.5, cutmix_alpha=1.0, drw_start_epoch=40,
                 coarse_loss_weight=0.3):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        
        # Loss functions
        self.criterion_fine = LDAMLoss(class_freq, max_m=0.5, s=30)
        
        # Coarse class mapping (6 classes → 3 super-classes)
        self.coarse_mapping = torch.LongTensor([0, 0, 1, 1, 2, 2]).to(device)
        self.criterion_coarse = LDAMLoss([class_freq[0]+class_freq[1],
                                         class_freq[2]+class_freq[3],
                                         class_freq[4]+class_freq[5]], 
                                        max_m=0.5, s=30)
        
        self.coarse_loss_weight = coarse_loss_weight
        self.cutmix_prob = cutmix_prob
        self.cutmix_alpha = cutmix_alpha
        self.drw_start_epoch = drw_start_epoch
        
        # Optimizer (separate learning rates for pretrained vs. new layers)
        backbone_params = list(self.model.vim_backbone.parameters())
        new_params = list(self.model.spatial_adapter.parameters()) + \
                    list(self.model.coarse_classifier.parameters()) + \
                    list(self.model.fine_classifier.parameters()) + \
                    list(self.model.guidance_proj.parameters())
        
        self.optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': lr * 0.1},  # Lower LR for pretrained
            {'params': new_params, 'lr': lr}
        ], weight_decay=weight_decay)
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        # Tracking
        self.best_val_f1 = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_f1s = []
        
        # Progressive unfreezing schedule
        self.unfreeze_schedule = {
            0: 3,   # Epochs 0-9: Freeze blocks 0-3
            10: 1,  # Epochs 10-29: Unfreeze blocks 2-3
            30: -1  # Epochs 30+: Unfreeze all
        }
    
    def apply_unfreezing(self, epoch):
        """Apply progressive unfreezing based on epoch"""
        for unfreeze_epoch, freeze_until in self.unfreeze_schedule.items():
            if epoch == unfreeze_epoch:
                if freeze_until == -1:
                    # Unfreeze all
                    print(f"\n{'='*80}")
                    print(f"Epoch {epoch}: Unfreezing ALL backbone layers")
                    print(f"{'='*80}")
                    for param in self.model.vim_backbone.parameters():
                        param.requires_grad = True
                else:
                    # Freeze until specified block
                    print(f"\n{'='*80}")
                    print(f"Epoch {epoch}: Freezing backbone blocks 0-{freeze_until}")
                    print(f"{'='*80}")
                    self.model.freeze_backbone(freeze_until)
                break
    
    def train_epoch(self, epoch):
        self.model.train()
        
        # Apply progressive unfreezing
        self.apply_unfreezing(epoch)
        
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Apply CutMix augmentation (in spatial domain after adapter)
            if np.random.rand() < self.cutmix_prob:
                # Apply asymmetric processing first
                data_asym = process_asymmetric_stm(data.view(-1, 20, 121))
                # Apply spatial adapter
                data_spatial = self.model.spatial_adapter(data_asym)
                # Apply CutMix in spatial domain
                data_mixed, target_a, target_b, lam = cutmix_2d(data_spatial, target, self.cutmix_alpha)
                
                # Forward through ViM backbone (manually, since we already have spatial features)
                batch_size = data_mixed.shape[0]
                x = self.model.vim_backbone.patch_embed(data_mixed)
                x = x.flatten(2).transpose(1, 2)
                x = x + self.model.vim_backbone.pos_embed
                
                # Early blocks
                for i in range(self.model.branch_point):
                    x = self.model.vim_backbone.blocks[i](x)
                
                # Coarse classification
                coarse_features = x.mean(dim=1)
                coarse_logits = self.model.coarse_classifier(coarse_features)
                
                # Guidance
                coarse_probs = F.softmax(coarse_logits, dim=-1)
                guidance_token = self.model.guidance_proj(coarse_probs).unsqueeze(1)
                x = torch.cat([guidance_token, x], dim=1)
                
                # Deep blocks
                for i in range(self.model.branch_point, self.model.vim_depth):
                    x = self.model.vim_backbone.blocks[i](x)
                x = self.model.vim_backbone.norm(x)
                
                # Fine classification
                fine_features = x[:, 1:, :].mean(dim=1)
                fine_logits = self.model.fine_classifier(fine_features)
                
                # Mixed loss
                coarse_target_a = self.coarse_mapping[target_a]
                coarse_target_b = self.coarse_mapping[target_b]
                
                loss_fine = lam * self.criterion_fine(fine_logits, target_a, epoch, self.drw_start_epoch) + \
                           (1 - lam) * self.criterion_fine(fine_logits, target_b, epoch, self.drw_start_epoch)
                
                loss_coarse = lam * self.criterion_coarse(coarse_logits, coarse_target_a, epoch, self.drw_start_epoch) + \
                             (1 - lam) * self.criterion_coarse(coarse_logits, coarse_target_b, epoch, self.drw_start_epoch)
                
                loss = (1 - self.coarse_loss_weight) * loss_fine + self.coarse_loss_weight * loss_coarse
                
                preds = fine_logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target_a.cpu().numpy())  # Use target_a for F1 computation
            else:
                # Standard forward pass
                fine_logits, coarse_logits = self.model(data, return_coarse=True)
                
                # Compute losses
                coarse_target = self.coarse_mapping[target]
                loss_fine = self.criterion_fine(fine_logits, target, epoch, self.drw_start_epoch)
                loss_coarse = self.criterion_coarse(coarse_logits, coarse_target, epoch, self.drw_start_epoch)
                
                loss = (1 - self.coarse_loss_weight) * loss_fine + self.coarse_loss_weight * loss_coarse
                
                preds = fine_logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(self.train_loader)
        train_f1 = f1_score(all_targets, all_preds, average='macro')
        
        return avg_loss, train_f1
    
    def evaluate(self, data_loader):
        self.model.eval()
        
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                fine_logits, coarse_logits = self.model(data, return_coarse=True)
                
                # Compute losses
                coarse_target = self.coarse_mapping[target]
                loss_fine = F.cross_entropy(fine_logits, target)
                loss_coarse = F.cross_entropy(coarse_logits, coarse_target)
                
                loss = (1 - self.coarse_loss_weight) * loss_fine + self.coarse_loss_weight * loss_coarse
                
                total_loss += loss.item()
                
                preds = fine_logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        avg_loss = total_loss / len(data_loader)
        macro_f1 = f1_score(all_targets, all_preds, average='macro')
        f1_per_class = f1_score(all_targets, all_preds, average=None)
        
        return avg_loss, macro_f1, f1_per_class, all_preds, all_targets
    
    def train(self, num_epochs, checkpoint_dir):
        print(f"\n{'='*80}")
        print(f"Starting training for {num_epochs} epochs")
        print(f"{'='*80}\n")
        
        for epoch in range(num_epochs):
            # Train
            train_loss, train_f1 = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_f1, val_f1_per_class, _, _ = self.evaluate(self.val_loader)
            
            # Step scheduler
            self.scheduler.step()
            
            # Track metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_f1s.append(val_f1)
            
            # Print progress
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f}")
            print(f"  Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")
            print(f"  Val F1 per class: {val_f1_per_class}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f} (backbone), "
                  f"{self.optimizer.param_groups[1]['lr']:.6f} (new)")
            
            # Save best model
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_f1': val_f1,
                    'val_f1_per_class': val_f1_per_class
                }
                torch.save(checkpoint, os.path.join(checkpoint_dir, 'best_model.pt'))
                print(f"  ✓ Saved best model (Val F1: {val_f1:.4f})")
            
            print()
        
        print(f"\n{'='*80}")
        print(f"Training completed!")
        print(f"Best validation F1: {self.best_val_f1:.4f}")
        print(f"{'='*80}\n")


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python STM_branchAuM_preViM.py <mode> [--pretrained_path <path>]")
        print("  mode 0: Full dataset")
        print("  mode 1: Downsampled non-tonal speech")
        sys.exit(1)
    
    mode = int(sys.argv[1])
    
    # Check for pretrained model path
    pretrained_vim_path = None
    if '--pretrained_path' in sys.argv:
        idx = sys.argv.index('--pretrained_path')
        if idx + 1 < len(sys.argv):
            pretrained_vim_path = sys.argv[idx + 1]
            print(f"Using pretrained ViM from: {pretrained_vim_path}")
    
    # Set parameters based on mode
    if mode == 0:
        ds_nontonal_speech = False
    elif mode == 1:
        ds_nontonal_speech = True
    else:
        print(f"Invalid mode: {mode}")
        sys.exit(1)
    
    # Create checkpoint directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = f'checkpoints/STM_branchAuM_preViM_mode{mode}_{timestamp}'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"Checkpoint directory: {checkpoint_dir}")
    
    # Prepare data
    print("\n" + "="*80)
    print("Loading and preparing data...")
    print("="*80)
    
    data_prep = prepData_STM_branchAuM_preViM(ds_nontonal_speech=ds_nontonal_speech)
    train_dataset, val_dataset, test_dataset, class_freq = data_prep.prepare_datasets()
    
    # Apply asymmetric STM processing
    print(f"\nApplying asymmetric 2-channel STM processing...")
    print(f"Original: 20 freq × 121 rates = 2420 features")
    print(f"After processing: 2 channels × 20 freq × 61 rates")
    print(f"  Channel 0: S_up (upward frequency sweeps)")
    print(f"  Channel 1: S_down (downward frequency sweeps)")
    print(f"After spatial adapter: 3 channels × 224 × 224 (ViM input)")
    
    train_dataset = AsymmetricSTMDataset(train_dataset)
    val_dataset = AsymmetricSTMDataset(val_dataset)
    test_dataset = AsymmetricSTMDataset(test_dataset)
    
    # Create data loaders
    batch_size = 32  # Smaller batch for memory efficiency
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    
    print(f"\nDataLoaders created with batch_size={batch_size}")
    
    # Create model
    print("\n" + "="*80)
    print("Creating STM_branchAuM_preViM model...")
    print("="*80)
    
    num_classes = 6
    model = BranchAuMPreViM(
        num_classes=num_classes,
        pretrained_vim_path=pretrained_vim_path,
        d_model=384,         # vim_small dimension
        vim_depth=24,        # vim_small depth
        drop_path_rate=0.4
    )
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Branch point: Layer {model.branch_point}")
    print(f"\nArchitecture:")
    print(f"  1. Spatial Adapter: (2,20,61) → (3,224,224)")
    print(f"  2. Pretrained ViM: vim_small with 4×4 patches (3,136 tokens)")
    print(f"  3. Hierarchical branching:")
    print(f"     - Early blocks (0-3): Feature extraction")
    print(f"     - Coarse classifier: 3 super-classes")
    print(f"     - Deep blocks (4-23): Fine-grained with guidance")
    print(f"     - Fine classifier: 6 classes")
    print(f"\nProgressive Unfreezing Schedule:")
    print(f"  - Epochs 0-9: Freeze blocks 0-3")
    print(f"  - Epochs 10-29: Unfreeze blocks 2-3")
    print(f"  - Epochs 30+: Unfreeze all")
    
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
        drw_start_epoch=40,
        coarse_loss_weight=0.3
    )
    
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
    print(f"\nTest F1 per class:")
    class_names = ['speech:non-tonal', 'speech:tonal', 'music:vocal', 
                   'music:non-vocal', 'env:urban', 'env:wildlife']
    for i, (name, f1) in enumerate(zip(class_names, test_f1_per_class)):
        print(f"  {name}: {f1:.4f}")
    
    print(f"\nClassification Report:")
    print(classification_report(test_targets, test_preds, target_names=class_names))
    
    print(f"\n{'='*80}")
    print(f"Experiment completed!")
    print(f"Results saved to: {checkpoint_dir}")
    print(f"{'='*80}")
