#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STM Classification with Hierarchical Audio Mamba (STM_branchAuM)
Bidirectional State Space Model with Hierarchical Branching

This implementation follows Roadmap 2 from the research paper:
"Optimization of Spectrotemporal Modulation Analysis: Architectural Paradigms 
for Superior Audio Classification and Segregation"

Key Features:
1. Asymmetric 2-channel STM processing (S_up + S_down, no averaging)
2. Full sequence processing (2,440 tokens) without destructive patching
3. Bidirectional Mamba blocks (forward + backward scanning)
4. Hierarchical branching: Coarse classifier (Layer 4) → Fine classifier (Layer 12)
5. Stochastic depth regularization (0.0 → 0.4)
6. LDAM loss with Deferred Reweighting (DRW)
7. Sequence-based CutMix augmentation

Target: 0.89-0.91 Macro F1 Score

Installation Requirements:
    pip install mamba-ssm causal-conv1d>=1.2.0
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

# Try to import Mamba
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    print("ERROR: mamba-ssm not installed!")
    print("Install with: pip install mamba-ssm causal-conv1d>=1.2.0")
    sys.exit(1)


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
    """Wrapper dataset that applies asymmetric 2-channel STM processing and flattens to sequence"""
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
        
        # Flatten to sequence for Mamba: (2, 20, 61) → (2440,)
        # Order: [S_up_freq0_rate0, ..., S_up_freq0_rate60, S_up_freq1_rate0, ..., 
        #         S_down_freq0_rate0, ..., S_down_freq19_rate60]
        data = data.reshape(-1)  # (2440,)
        
        return data, label


# ============================================================================
# Stochastic Depth (DropPath)
# ============================================================================

class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample.
    
    From "Deep Networks with Stochastic Depth" (Huang et al., ECCV 2016)
    
    Critical for preventing overfitting in deep Mamba models without ImageNet pretraining.
    The paper notes: "ViM models are notoriously data-hungry and prone to overfitting 
    on smaller or specialized datasets."
    """
    def __init__(self, drop_prob=0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        # Work with different tensor shapes
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # Binarize
        output = x.div(keep_prob) * random_tensor
        return output


# ============================================================================
# Bidirectional Mamba Block
# ============================================================================

class BidirectionalMambaBlock(nn.Module):
    """
    Bidirectional State Space Model block for STM sequence processing.
    
    From the paper:
    "Original Mamba is unidirectional. While ViM introduces bidirectional scanning,
    applying it to the non-causal STM grid requires careful tuning of the scan 
    directions to capture the relevant spectrotemporal dependencies."
    
    Architecture:
    Input → [Forward SSM, Backward SSM] → Concatenate → Fuse → Residual → Output
    
    Key improvements over standard ViM:
    1. No spatial patching - processes full 2440-token sequence
    2. Bidirectional scanning captures Rate-Scale correlations
    3. Stochastic depth prevents overfitting on 1M dataset
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand_factor=2, drop_path=0.0):
        super(BidirectionalMambaBlock, self).__init__()
        
        self.norm = nn.LayerNorm(d_model)
        
        # Forward and backward Mamba layers
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
        
        # Fusion layer to combine bidirectional features
        self.fusion = nn.Linear(2 * d_model, d_model)
        
        # Stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        # Normalize
        x_norm = self.norm(x)
        
        # Forward scan: left → right
        x_fwd = self.mamba_forward(x_norm)
        
        # Backward scan: right → left
        # Flip sequence, process, flip back
        x_bwd = torch.flip(x_norm, dims=[1])
        x_bwd = self.mamba_backward(x_bwd)
        x_bwd = torch.flip(x_bwd, dims=[1])
        
        # Concatenate and fuse
        x_bidir = torch.cat([x_fwd, x_bwd], dim=-1)  # (batch, seq_len, 2*d_model)
        x_fused = self.fusion(x_bidir)  # (batch, seq_len, d_model)
        
        # Residual connection with stochastic depth
        return x + self.drop_path(x_fused)


# ============================================================================
# Hierarchical STM_branchAuM Model
# ============================================================================

class BranchAuM(nn.Module):
    """
    Hierarchical Audio Mamba (STM_branchAuM) for STM-based audio classification.
    
    From the paper (Roadmap 2):
    "This fixes the ViM failures by using Audio Mamba specifics and 
    Hierarchical Branching."
    
    Architecture:
    Input (2440 tokens) → Embedding + Positional Encoding
        ↓
    [Blocks 1-4: Early Feature Extraction]
        ↓
    **Coarse Classifier** (Speech/Music/Environment)
        ↓ (guidance token)
    [Blocks 5-12: Fine Feature Refinement]
        ↓
    **Fine Classifier** (6 fine-grained classes)
    
    Key innovations:
    1. No patching: Preserves spectral continuum (20 freq bands)
    2. Hierarchical branching: Coarse-to-fine classification
    3. Guidance mechanism: Coarse predictions guide deep layers
    4. Stochastic depth: 0.0 → 0.4 (linear schedule)
    """
    def __init__(self, seq_len=2440, num_classes=6, d_model=256, depth=12,
                 d_state=16, d_conv=4, expand_factor=2, drop_path_rate=0.4):
        super(BranchAuM, self).__init__()
        
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.d_model = d_model
        self.depth = depth
        self.branch_point = 4  # Insert coarse classifier after layer 4
        
        # Input embedding: Project 2440 tokens to d_model dimensions
        self.input_proj = nn.Linear(1, d_model)
        
        # Learnable positional embeddings (absolute positions)
        # Critical: STM is a semantic coordinate system, not a natural image
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Stochastic depth decay rule (linear schedule)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        # Early blocks (1-4): Feature extraction
        self.early_blocks = nn.ModuleList([
            BidirectionalMambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand_factor=expand_factor,
                drop_path=dpr[i]
            )
            for i in range(self.branch_point)
        ])
        
        # Coarse classifier (3 super-classes)
        self.coarse_norm = nn.LayerNorm(d_model)
        self.coarse_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 3)
        )
        
        # Guidance token projection
        # Coarse probabilities → guidance embedding
        self.guidance_proj = nn.Linear(3, d_model)
        
        # Deep blocks (5-12): Fine-grained refinement with guidance
        self.deep_blocks = nn.ModuleList([
            BidirectionalMambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand_factor=expand_factor,
                drop_path=dpr[i]
            )
            for i in range(self.branch_point, depth)
        ])
        
        # Fine classifier (6 fine-grained classes)
        self.fine_norm = nn.LayerNorm(d_model)
        self.fine_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_classes)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights"""
        # Initialize linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x, return_coarse=False):
        """
        Args:
            x: (batch, seq_len) - flattened STM sequence
            return_coarse: If True, return both coarse and fine logits
        
        Returns:
            If return_coarse=False: fine_logits (batch, 6)
            If return_coarse=True: (coarse_logits, fine_logits)
        """
        batch_size = x.shape[0]
        
        # Reshape to (batch, seq_len, 1) for projection
        x = x.unsqueeze(-1)  # (batch, 2440, 1)
        
        # Input embedding + positional encoding
        x = self.input_proj(x)  # (batch, 2440, d_model)
        x = x + self.pos_embed  # Add positional embeddings
        
        # Early blocks (1-4)
        for block in self.early_blocks:
            x = block(x)  # (batch, 2440, d_model)
        
        # Coarse classification branch
        # Global average pooling over sequence
        coarse_features = x.mean(dim=1)  # (batch, d_model)
        coarse_features = self.coarse_norm(coarse_features)
        coarse_logits = self.coarse_classifier(coarse_features)  # (batch, 3)
        
        # Generate guidance token from coarse predictions
        coarse_probs = F.softmax(coarse_logits, dim=-1)  # (batch, 3)
        guidance_token = self.guidance_proj(coarse_probs)  # (batch, d_model)
        guidance_token = guidance_token.unsqueeze(1)  # (batch, 1, d_model)
        
        # Prepend guidance token to sequence
        x = torch.cat([guidance_token, x], dim=1)  # (batch, 2441, d_model)
        
        # Deep blocks (5-12) with guidance
        for block in self.deep_blocks:
            x = block(x)  # (batch, 2441, d_model)
        
        # Fine classification
        # Global average pooling (skip guidance token)
        fine_features = x[:, 1:, :].mean(dim=1)  # (batch, d_model)
        fine_features = self.fine_norm(fine_features)
        fine_logits = self.fine_classifier(fine_features)  # (batch, 6)
        
        if return_coarse:
            return coarse_logits, fine_logits
        else:
            return fine_logits


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
        m_list = 1.0 / np.sqrt(np.sqrt(class_freq))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.FloatTensor(m_list)
        self.m_list = m_list
        self.s = s
        
        # Class weights for DRW
        self.weight = torch.FloatTensor(1.0 / class_freq)
        self.weight = self.weight / self.weight.sum() * len(class_freq)
        
    def forward(self, x, target, epoch=0, drw_start_epoch=40):
        """
        Args:
            x: logits (batch, num_classes)
            target: labels (batch,)
            epoch: current epoch (for DRW schedule)
            drw_start_epoch: epoch to start reweighting
        """
        self.m_list = self.m_list.to(x.device)
        self.weight = self.weight.to(x.device)
        
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.float()
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
        
        output = torch.where(index, x_m, x)
        
        # Apply DRW after specified epoch
        if epoch >= drw_start_epoch:
            return F.cross_entropy(self.s * output, target, weight=self.weight)
        else:
            return F.cross_entropy(self.s * output, target)


# ============================================================================
# Sequence CutMix Augmentation
# ============================================================================

def cutmix_sequence(x, y, alpha=1.0):
    """
    CutMix augmentation adapted for sequence data.
    
    From the paper:
    "CutMix preserves local structure of modulation. A patch of 'speech rhythm' 
    is pasted onto 'music harmonics'. The model learns to recognize both 
    distinct objects occurring simultaneously."
    
    For sequences: Cut and paste contiguous subsequences instead of 2D patches.
    
    Args:
        x: Input tensor (batch, seq_len)
        y: Labels (batch,)
        alpha: Beta distribution parameter
    
    Returns:
        mixed_x: Mixed input
        y_a: First label
        y_b: Second label
        lam: Mixing coefficient
    """
    batch_size = x.size(0)
    seq_len = x.size(1)
    
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    # Random permutation
    index = torch.randperm(batch_size, device=x.device)
    
    # Compute cut length
    cut_len = int(seq_len * (1.0 - lam))
    
    # Random start position
    start = np.random.randint(0, seq_len - cut_len + 1)
    end = start + cut_len
    
    # Mix sequences
    mixed_x = x.clone()
    mixed_x[:, start:end] = x[index, start:end]
    
    # Adjust lambda to exact ratio
    lam = 1.0 - (cut_len / seq_len)
    
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


# ============================================================================
# Data Preparation
# ============================================================================

class prepData_STM_branchAuM:
    """
    Data preparation for STM_branchAuM model.
    Follows the same structure as STMasm_mixer_kan.py.
    """
    def __init__(self, addAug=False, ds_nontonal_speech=False):
        self.addAug = addAug
        self.ds_nontonal_speech = ds_nontonal_speech
        
        # STM dimensions
        self.n_freq = 20
        self.n_time = 121  # Original
        
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
        
        # Map categories
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
                print(f"Warning: Only {len(indices_target_0)} non-tonal speech samples available")
            
            np.random.seed(23)
            sampled_indices = np.random.choice(indices_target_0, size=min(num_samples, len(indices_target_0)), replace=False)
            
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
        print(f"Original feature dim: {STM_all.shape[1]}")
        print(f"After asymmetric processing: 2440 tokens")
        print(f"\nClass Distribution (Training):")
        for i, count in enumerate(class_counts):
            print(f"  Class {i}: {count} samples ({class_freq[i]:.4f})")
        
        return STM_all, target, train_ind, val_ind, test_ind, class_freq
    
    def prepare_datasets(self):
        """Prepare PyTorch datasets"""
        STM_all, target, train_ind, val_ind, test_ind, class_freq = self.load_data()
        
        # Normalize per sample (preserve relative patterns)
        means = STM_all.mean(axis=1, keepdims=True)
        stds = STM_all.std(axis=1, keepdims=True)
        STM_all_norm = (STM_all - means) / (stds + 1e-8)
        
        # Convert to PyTorch tensors
        X_train = torch.FloatTensor(STM_all_norm[train_ind])
        y_train = torch.LongTensor(target[train_ind].values)
        
        X_val = torch.FloatTensor(STM_all_norm[val_ind])
        y_val = torch.LongTensor(target[val_ind].values)
        
        X_test = torch.FloatTensor(STM_all_norm[test_ind])
        y_test = torch.LongTensor(target[test_ind].values)
        
        # Create datasets (before asymmetric processing)
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)
        
        return train_dataset, val_dataset, test_dataset, class_freq


# ============================================================================
# Trainer
# ============================================================================

class Trainer:
    """Training manager for Branch-AuM"""
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
        
        # Coarse targets: map 6 classes → 3 super-classes
        # 0,1 → 0 (Speech), 2,3 → 1 (Music), 4,5 → 2 (Environment)
        self.class_to_coarse = torch.LongTensor([0, 0, 1, 1, 2, 2]).to(device)
        
        # Compute coarse class frequencies
        coarse_freq = np.zeros(3)
        coarse_freq[0] = class_freq[0] + class_freq[1]  # Speech
        coarse_freq[1] = class_freq[2] + class_freq[3]  # Music
        coarse_freq[2] = class_freq[4] + class_freq[5]  # Environment
        
        self.criterion_coarse = LDAMLoss(coarse_freq, max_m=0.5, s=30)
        
        # Loss weights
        self.coarse_loss_weight = coarse_loss_weight
        self.fine_loss_weight = 1.0 - coarse_loss_weight
        
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
            'train_loss_coarse': [],
            'train_loss_fine': [],
            'val_loss': [],
            'val_f1': [],
            'val_f1_per_class': []
        }
        
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint and resume training"""
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
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
        total_loss_coarse = 0.0
        total_loss_fine = 0.0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Map fine targets to coarse targets
            coarse_target = self.class_to_coarse[target]
            
            # Apply CutMix with probability
            if np.random.rand() < self.cutmix_prob:
                data, target_a, target_b, lam = cutmix_sequence(data, target, self.cutmix_alpha)
                coarse_target_a = self.class_to_coarse[target_a]
                coarse_target_b = self.class_to_coarse[target_b]
                
                # Forward pass
                coarse_logits, fine_logits = self.model(data, return_coarse=True)
                
                # Mixed coarse loss
                loss_coarse_a = self.criterion_coarse(coarse_logits, coarse_target_a, 
                                                     self.current_epoch, self.drw_start_epoch)
                loss_coarse_b = self.criterion_coarse(coarse_logits, coarse_target_b, 
                                                     self.current_epoch, self.drw_start_epoch)
                loss_coarse = lam * loss_coarse_a + (1 - lam) * loss_coarse_b
                
                # Mixed fine loss
                loss_fine_a = self.criterion_fine(fine_logits, target_a, 
                                                 self.current_epoch, self.drw_start_epoch)
                loss_fine_b = self.criterion_fine(fine_logits, target_b, 
                                                 self.current_epoch, self.drw_start_epoch)
                loss_fine = lam * loss_fine_a + (1 - lam) * loss_fine_b
            else:
                # Standard forward pass
                coarse_logits, fine_logits = self.model(data, return_coarse=True)
                
                # Coarse loss
                loss_coarse = self.criterion_coarse(coarse_logits, coarse_target, 
                                                   self.current_epoch, self.drw_start_epoch)
                
                # Fine loss
                loss_fine = self.criterion_fine(fine_logits, target, 
                                               self.current_epoch, self.drw_start_epoch)
            
            # Combined loss
            loss = self.coarse_loss_weight * loss_coarse + self.fine_loss_weight * loss_fine
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            total_loss_coarse += loss_coarse.item()
            total_loss_fine += loss_fine.item()
        
        avg_loss = total_loss / len(self.train_loader)
        avg_loss_coarse = total_loss_coarse / len(self.train_loader)
        avg_loss_fine = total_loss_fine / len(self.train_loader)
        
        return avg_loss, avg_loss_coarse, avg_loss_fine
    
    def evaluate(self, data_loader):
        """Evaluate model"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass (fine logits only)
                logits = self.model(data, return_coarse=False)
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
        print("Starting Training - Hierarchical STM_branchAuM")
        print("="*80)
        
        class_names = ['speech:non-tonal', 'speech:tonal', 'music:vocal', 
                      'music:non-vocal', 'env:urban', 'env:wildlife']
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Train
            train_loss, train_loss_coarse, train_loss_fine = self.train_epoch()
            
            # Validate
            val_loss, val_f1, val_f1_per_class, _, _ = self.evaluate(self.val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_loss_coarse'].append(train_loss_coarse)
            self.history['train_loss_fine'].append(train_loss_fine)
            self.history['val_loss'].append(val_loss)
            self.history['val_f1'].append(val_f1)
            self.history['val_f1_per_class'].append(val_f1_per_class)
            
            epoch_time = time.time() - epoch_start_time
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Print progress
            print(f"\nEpoch [{epoch+1}/{num_epochs}] - {epoch_time:.2f}s - LR: {current_lr:.6f}")
            print(f"  Train Loss: {train_loss:.4f} (Coarse: {train_loss_coarse:.4f}, Fine: {train_loss_fine:.4f})")
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
        print("Usage: python STM_branchAuM.py <mode> [--resume <checkpoint_dir>]")
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
        checkpoint_dir = f"model/STM/STM_branchAuM_{mode_name}_{timestamp}"
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"Checkpoint directory: {checkpoint_dir}")
    
    # Prepare data
    print("\n" + "="*80)
    print("Loading and preparing data...")
    print("="*80)
    
    data_prep = prepData_STM_branchAuM(ds_nontonal_speech=ds_nontonal_speech)
    train_dataset, val_dataset, test_dataset, class_freq = data_prep.prepare_datasets()
    
    # Apply asymmetric STM processing
    print(f"\nApplying asymmetric 2-channel STM processing...")
    print(f"Original: 20 freq × 121 rates = 2420 features")
    print(f"After processing: 2 channels × 20 freq × 61 rates → 2440 tokens")
    print(f"  Channel 0: S_up (upward frequency sweeps, flipped)")
    print(f"  Channel 1: S_down (downward frequency sweeps)")
    print(f"  Flattened for sequence processing by Mamba")
    
    train_dataset = AsymmetricSTMDataset(train_dataset)
    val_dataset = AsymmetricSTMDataset(val_dataset)
    test_dataset = AsymmetricSTMDataset(test_dataset)
    
    # Create data loaders
    batch_size = 64  # Smaller batch size for Mamba (longer sequences)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    
    print(f"\nDataLoaders created with batch_size={batch_size}")
    
    # Create model
    print("\n" + "="*80)
    print("Creating STM_branchAuM model...")
    print("="*80)
    
    num_classes = 6
    model = BranchAuM(
        seq_len=2440,          # 2 channels × 20 freq × 61 rates
        num_classes=num_classes,
        d_model=256,           # Feature dimension
        depth=12,              # Number of Mamba blocks
        d_state=16,            # SSM state dimension
        d_conv=4,              # Convolution kernel size
        expand_factor=2,       # MLP expansion ratio
        drop_path_rate=0.4     # Stochastic depth (0.0 → 0.4)
    )
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Branch point: Layer {model.branch_point}")
    print(f"Hierarchical structure:")
    print(f"  - Early blocks (1-4): Feature extraction")
    print(f"  - Coarse classifier: 3 super-classes (Speech/Music/Environment)")
    print(f"  - Deep blocks (5-12): Fine-grained refinement with guidance")
    print(f"  - Fine classifier: 6 fine-grained classes")
    
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
        drw_start_epoch=40,    # Start deferred reweighting at epoch 40
        coarse_loss_weight=0.3 # Weight for coarse classification loss
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
    checkpoint = torch.load(os.path.join(checkpoint_dir, 'best_model.pt'), weights_only=False)
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
