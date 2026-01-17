#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kanformer (KAN-integrated Conformer) for STM Audio Classification

This implements a Conformer architecture where the standard Feed-Forward Networks (FFNs)
are replaced by Group-Rational Kolmogorov-Arnold Network (GR-KAN) layers.

Based on the theoretical framework that KANs can better approximate the complex, 
non-linear decision boundaries in the spectrotemporal modulation manifold.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import datetime
import os
import sys
import math
from sklearn.metrics import f1_score
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# Group-Rational KAN Implementation
# ============================================================================

class RationalFunction(nn.Module):
    """
    Learnable rational activation function: P(x) / Q(x)
    where P and Q are polynomials with learnable coefficients.
    
    This replaces fixed activations (ReLU, Swish) with a learnable curve
    that can adapt to the specific non-linearities of STM features.
    """
    def __init__(self, numerator_degree=5, denominator_degree=4, init_type='identity'):
        super(RationalFunction, self).__init__()
        
        self.numerator_degree = numerator_degree
        self.denominator_degree = denominator_degree
        
        # Initialize polynomial coefficients
        self.numerator = nn.Parameter(torch.zeros(numerator_degree + 1))
        self.denominator = nn.Parameter(torch.zeros(denominator_degree + 1))
        
        self._initialize_coefficients(init_type)
    
    def _initialize_coefficients(self, init_type):
        """Initialize to approximate common activations"""
        if init_type == 'identity':
            # P(x) = x, Q(x) = 1
            self.numerator.data[1] = 1.0
            self.denominator.data[0] = 1.0
        elif init_type == 'relu':
            # Approximate ReLU
            self.numerator.data[0] = 0.0
            self.numerator.data[1] = 0.5
            self.numerator.data[2] = 0.5
            self.denominator.data[0] = 1.0
        else:
            # Xavier-style initialization
            nn.init.normal_(self.numerator, mean=0.0, std=0.1)
            nn.init.normal_(self.denominator, mean=0.0, std=0.1)
            self.denominator.data[0] = 1.0  # Ensure non-zero denominator
    
    def forward(self, x):
        """
        Compute P(x) / Q(x)
        
        Args:
            x: Input tensor of any shape
        
        Returns:
            Rational function applied element-wise
        """
        # Compute numerator: sum_{i} a_i * x^i
        numerator = torch.zeros_like(x)
        for i, coeff in enumerate(self.numerator):
            numerator = numerator + coeff * torch.pow(x, i)
        
        # Compute denominator: sum_{i} b_i * x^i
        denominator = torch.zeros_like(x)
        for i, coeff in enumerate(self.denominator):
            denominator = denominator + coeff * torch.pow(x, i)
        
        # Avoid division by zero
        denominator = torch.clamp(denominator, min=1e-6)
        
        return numerator / denominator


class GroupRationalKANLayer(nn.Module):
    """
    Group-Rational KAN Layer
    
    Instead of fixed activation functions, this layer learns a custom
    rational function for each group of features. This allows the network
    to develop specialized non-linearities for different regions of the
    STM manifold (e.g., speech-like vs music-like modulation patterns).
    
    Architecture:
        x -> Linear -> [Group-wise Rational Functions] -> Linear
    """
    def __init__(self, in_features, out_features, num_groups=8, 
                 numerator_degree=5, denominator_degree=4, dropout=0.1):
        super(GroupRationalKANLayer, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        
        # Ensure divisibility
        assert in_features % num_groups == 0, \
            f"in_features ({in_features}) must be divisible by num_groups ({num_groups})"
        
        self.group_size = in_features // num_groups
        
        # Linear transformations
        self.linear_in = nn.Linear(in_features, in_features)
        self.linear_out = nn.Linear(in_features, out_features)
        
        # Group-wise rational functions
        self.rational_functions = nn.ModuleList([
            RationalFunction(numerator_degree, denominator_degree, init_type='relu')
            for _ in range(num_groups)
        ])
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(in_features)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Apply group-wise rational activations
        
        Args:
            x: (batch, seq_len, in_features) or (batch, in_features)
        
        Returns:
            Transformed features with same shape as input leading dims + out_features
        """
        # Store original shape
        original_shape = x.shape
        is_3d = len(original_shape) == 3
        
        # Flatten if 3D
        if is_3d:
            batch_size, seq_len, _ = x.shape
            x = x.reshape(batch_size * seq_len, -1)
        
        # Linear transformation
        x = self.linear_in(x)
        x = self.layer_norm(x)
        
        # Apply group-wise rational functions
        group_outputs = []
        for i, rational_fn in enumerate(self.rational_functions):
            start_idx = i * self.group_size
            end_idx = (i + 1) * self.group_size
            group_input = x[:, start_idx:end_idx]
            group_output = rational_fn(group_input)
            group_outputs.append(group_output)
        
        # Concatenate group outputs
        x = torch.cat(group_outputs, dim=1)
        
        # Dropout
        x = self.dropout(x)
        
        # Output projection
        x = self.linear_out(x)
        
        # Restore shape if needed
        if is_3d:
            x = x.reshape(batch_size, seq_len, -1)
        
        return x


# ============================================================================
# Kanformer Block (Conformer with KAN-FFN)
# ============================================================================

class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention with relative positional encoding"""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv_proj(x)  # (batch, seq_len, 3*d_model)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, num_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # (batch, num_heads, seq_len, head_dim)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.d_model)
        
        # Output projection
        output = self.out_proj(attn_output)
        
        return output


class ConvolutionModule(nn.Module):
    """Conformer convolution module with depthwise separable convolution"""
    def __init__(self, d_model, kernel_size=31, dropout=0.1):
        super(ConvolutionModule, self).__init__()
        
        assert kernel_size % 2 == 1, "kernel_size must be odd for 'same' padding"
        
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Pointwise expansion
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        
        # GLU activation
        self.glu = nn.GLU(dim=1)
        
        # Depthwise convolution
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2, groups=d_model
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = nn.SiLU()  # Swish
        
        # Pointwise compression
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = self.layer_norm(x)
        
        # Transpose for Conv1d: (batch, d_model, seq_len)
        x = x.transpose(1, 2)
        
        # Pointwise expansion
        x = self.pointwise_conv1(x)
        
        # GLU
        x = self.glu(x)
        
        # Depthwise conv
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        
        # Pointwise compression
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        
        # Transpose back: (batch, seq_len, d_model)
        x = x.transpose(1, 2)
        
        return x


class KanformerBlock(nn.Module):
    """
    Kanformer Block: Conformer block with KAN-FFN instead of standard FFN
    
    Structure (Half-Step Residual):
        x -> FFN1(KAN) -> MHSA -> Conv -> FFN2(KAN) -> output
    """
    def __init__(self, d_model, num_heads, ffn_dim, kernel_size=31, 
                 dropout=0.1, num_kan_groups=8):
        super(KanformerBlock, self).__init__()
        
        # First KAN-FFN (half-step)
        self.ffn1 = nn.Sequential(
            nn.LayerNorm(d_model),
            GroupRationalKANLayer(d_model, ffn_dim, num_groups=num_kan_groups, dropout=dropout),
            nn.Dropout(dropout),
            GroupRationalKANLayer(ffn_dim, d_model, num_groups=num_kan_groups, dropout=dropout),
        )
        
        # Multi-Head Self-Attention
        self.mhsa = nn.Sequential(
            nn.LayerNorm(d_model),
            MultiHeadSelfAttention(d_model, num_heads, dropout),
            nn.Dropout(dropout)
        )
        
        # Convolution Module
        self.conv = ConvolutionModule(d_model, kernel_size, dropout)
        
        # Second KAN-FFN (half-step)
        self.ffn2 = nn.Sequential(
            nn.LayerNorm(d_model),
            GroupRationalKANLayer(d_model, ffn_dim, num_groups=num_kan_groups, dropout=dropout),
            nn.Dropout(dropout),
            GroupRationalKANLayer(ffn_dim, d_model, num_groups=num_kan_groups, dropout=dropout),
        )
        
        # Scale residual connections
        self.ffn_scale = 0.5
        
    def forward(self, x):
        # First FFN with half-step residual
        x = x + self.ffn_scale * self.ffn1(x)
        
        # MHSA
        x = x + self.mhsa(x)
        
        # Convolution
        x = x + self.conv(x)
        
        # Second FFN with half-step residual
        x = x + self.ffn_scale * self.ffn2(x)
        
        return x


# ============================================================================
# Kanformer Classifier
# ============================================================================

class KanformerClassifier(nn.Module):
    """
    Kanformer-based classifier for STM audio classification.
    
    Architecture:
    1. Convolutional stem with 2D positional encoding
    2. Kanformer blocks (Conformer + KAN-FFN)
    3. Global average pooling
    4. KAN-based classification head
    """
    def __init__(self, input_dim, num_classes, d_model=128, num_heads=4, 
                 ffn_dim=512, num_layers=4, kernel_size=31, dropout=0.1, num_kan_groups=8):
        super(KanformerClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Convolutional stem: (batch, freq, time) -> (batch, time, d_model)
        self.conv_stem = nn.Sequential(
            nn.Conv1d(input_dim, d_model // 2, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.BatchNorm1d(d_model // 2),
            nn.Conv1d(d_model // 2, d_model, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.BatchNorm1d(d_model),
            nn.Dropout(dropout)
        )
        
        # Learnable 2D positional embeddings (anisotropic)
        # This respects that Rate and Scale axes have different meanings
        self.pos_embedding = nn.Parameter(torch.randn(1, 121, d_model) * 0.02)  # Max time dimension
        
        # Kanformer blocks
        self.kanformer_blocks = nn.ModuleList([
            KanformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                kernel_size=kernel_size,
                dropout=dropout,
                num_kan_groups=num_kan_groups
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # KAN-based classifier (uses learnable activations)
        self.classifier = nn.Sequential(
            GroupRationalKANLayer(d_model, d_model // 2, num_groups=4, dropout=dropout),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
    def forward(self, x):
        # x: (batch, freq, time)
        batch_size = x.size(0)
        
        # Convolutional stem: (batch, freq, time) -> (batch, d_model, time)
        x = self.conv_stem(x)
        
        # Transpose: (batch, d_model, time) -> (batch, time, d_model)
        x = x.transpose(1, 2)
        
        # Add positional embeddings (crop to sequence length)
        seq_len = x.size(1)
        x = x + self.pos_embedding[:, :seq_len, :]
        
        # Kanformer blocks
        for block in self.kanformer_blocks:
            x = block(x)
        
        # Final normalization
        x = self.final_norm(x)
        
        # Transpose for pooling: (batch, time, d_model) -> (batch, d_model, time)
        x = x.transpose(1, 2)
        
        # Global pooling: (batch, d_model, time) -> (batch, d_model, 1)
        x = self.global_pool(x)
        
        # Flatten: (batch, d_model, 1) -> (batch, d_model)
        x = x.squeeze(-1)
        
        # Classification
        x = self.classifier(x)
        
        return x


# ============================================================================
# Data Preparation (Same as Conformer)
# ============================================================================

class prepData_STM_Kanformer:
    """Data preparation for Kanformer model"""
    def __init__(self, addAug=False, ds_nontonal_speech=False):
        self.addAug = addAug
        self.ds_nontonal_speech = ds_nontonal_speech
        
        # STM preprocessing parameters
        self.xmin = 190
        self.xmax = 310
        self.ymin = 75
        self.ymax = 114
        self.x_ds_factor = 1
        self.y_ds_factor = 2
        
        # Calculate 2D dimensions
        self.n_freq = (self.ymax - self.ymin + 1) // self.y_ds_factor  # 20
        self.n_time = (self.xmax - self.xmin + 1) // self.x_ds_factor  # 121
    
    def corpora_list(self, addAug=False):
        # ...existing code from STMconformer_model.py...
        corpus_speech_list = ['BibleTTS/akuapem-twi', 'BibleTTS/asante-twi', 'BibleTTS/ewe', 
            'BibleTTS/hausa', 'BibleTTS/lingala', 'BibleTTS/yoruba', 'Buckeye', 'EUROM', 
            'HiltonMoser2022_speech', 'LibriSpeech', 'MediaSpeech/AR', 'MediaSpeech/ES', 
            'MediaSpeech/FR', 'MediaSpeech/TR'] + [f'MozillaCommonVoice/{lang}' for lang in [
            'ab', 'ar', 'ba', 'be', 'bg', 'bn', 'br', 'ca', 'ckb', 'cnh', 'cs', 'cv', 'cy', 
            'da', 'de', 'dv', 'el', 'en', 'eo', 'es', 'et', 'eu', 'fa', 'fi', 'fr', 'fy-NL', 
            'ga-IE', 'gl', 'gn', 'hi', 'hu', 'hy-AM', 'id', 'ig', 'it', 'ja', 'ka', 'kab', 
            'kk', 'kmr', 'ky', 'lg', 'lt', 'ltg', 'lv', 'mhr', 'ml', 'mn', 'mt', 'nan-tw', 
            'nl', 'oc', 'or', 'pl', 'pt', 'ro', 'ru', 'rw', 'sr', 'sv-SE', 'sw', 'ta', 'th', 
            'tr', 'tt', 'ug', 'uk', 'ur', 'uz', 'vi', 'yo', 'yue', 'zh-CN', 'zh-TW']] + [
            'primewords_chinese', 'room_reader', 'SpeechClarity', 'TAT-Vol2', 'thchs30', 
            'TIMIT', 'TTS_Javanese', 'zeroth_korean']
        
        corpus_music_list = ['IRMAS', 'Albouy2020Science', 'GarlandEncyclopedia', 'fma_large',
            'ismir04_genre', 'MTG-Jamendo', 'HiltonMoser2022_song', 'NHS2', 'MagnaTagATune']
        
        corpus_env_list = ['SONYC', 'MacaulayLibrary', 'SONYC_augmented'] if addAug else ['SONYC', 'MacaulayLibrary']
        
        corpus_speech_list.sort()
        corpus_music_list.sort()
        corpus_env_list.sort()
        
        return corpus_speech_list + corpus_music_list + corpus_env_list
    
    def load_data(self):
        """Load and preprocess STM data"""
        # ...existing code from STMconformer_model.py (lines 92-198)...
        corpus_list_all = self.corpora_list()
        root_folder = '/vast-ac8888/MusicSpeech-STM/'
        
        STM_all = None
        for corp in corpus_list_all:
            filename = root_folder + 'STM_output/corpSTMnpy/' + corp.replace('/', '-') + '_STMall.npy'
            if STM_all is None:
                STM_all = np.load(filename)
            else:
                STM_all = np.vstack((STM_all, np.load(filename)))
            print(f"Loaded: {filename}, shape: {np.load(filename).shape}")
        
        speech_corp_df1 = pd.read_csv(root_folder + 'train_test_split/speech1_10folds_speakerGroupFold.csv', index_col=0)
        speech_corp_df2 = pd.read_csv(root_folder + 'train_test_split/speech2_10folds_speakerGroupFold.csv', index_col=0)
        music_corp_df = pd.read_csv(root_folder + 'train_test_split/music_10folds_speakerGroupFold.csv', index_col=0)
        df_SONYC = pd.read_csv(root_folder + 'train_test_split/env_10folds_speakerGroupFold.csv', index_col=0)
        
        all_corp_df = pd.concat([speech_corp_df1, speech_corp_df2, music_corp_df, df_SONYC], ignore_index=True)
        
        if self.addAug:
            SONYC_aug_len = np.load(root_folder + 'STM_output/corpSTMnpy/SONYC_augmented_STMall.npy').shape[0]
            target = pd.concat([all_corp_df['corpus_type'], pd.Series(['env'] * SONYC_aug_len)], ignore_index=True)
            data_split = pd.concat([all_corp_df['10fold_labels'], pd.Series([1] * SONYC_aug_len)], ignore_index=True)
        else:
            target = all_corp_df['corpus_type'].copy()
            data_split = all_corp_df['10fold_labels'].copy()
        
        target.replace({'speech: non-tonal': 0, 'speech: tonal': 1, 'music: vocal': 2, 
                       'music: non-vocal': 3, 'env: urban': 4, 'env: wildlife': 5}, inplace=True)
        
        if self.ds_nontonal_speech:
            num_samples = 100000
            indices_target_0 = target.index[target == 0].to_numpy()
            np.random.seed(23)
            sampled_indices = np.random.choice(indices_target_0, size=num_samples, replace=False)
            mask = np.ones(len(target), dtype=bool)
            mask[indices_target_0] = False
            mask[sampled_indices] = True
            STM_all = STM_all[mask, :]
            data_split = data_split[mask].reset_index(drop=True)
            target = target[mask].reset_index(drop=True)
        
        train_ind = (data_split < 8).values
        val_ind = (data_split == 8).values
        test_ind = (data_split == 9).values
        
        print(f"Total samples: {len(STM_all)}, Train: {sum(train_ind)}, Val: {sum(val_ind)}, Test: {sum(test_ind)}")
        
        return STM_all, target.values, train_ind, val_ind, test_ind
    
    def prepare_datasets(self):
        """Prepare PyTorch datasets"""
        STM_all, target, train_ind, val_ind, test_ind = self.load_data()
        
        # Reshape to 2D
        STM_all_2d = STM_all.reshape(-1, self.n_freq, self.n_time)
        
        # Normalize
        means = STM_all_2d.mean(axis=(1, 2), keepdims=True)
        stds = STM_all_2d.std(axis=(1, 2), keepdims=True)
        STM_all_2d = (STM_all_2d - means) / (stds + 1e-8)
        
        # Convert to tensors
        X_train = torch.FloatTensor(STM_all_2d[train_ind])
        y_train = torch.LongTensor(target[train_ind])
        X_val = torch.FloatTensor(STM_all_2d[val_ind])
        y_val = torch.LongTensor(target[val_ind])
        X_test = torch.FloatTensor(STM_all_2d[test_ind])
        y_test = torch.LongTensor(target[test_ind])
        
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)
        
        return train_dataset, val_dataset, test_dataset, self.n_freq, self.n_time


# ============================================================================
# Focal Loss for Class Imbalance
# ============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = (1 - p_t) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# ============================================================================
# Trainer
# ============================================================================

class KanformerTrainer:
    """Training manager for Kanformer"""
    def __init__(self, model, train_loader, val_loader, test_loader, 
                 device, lr=1e-4, weight_decay=1e-4):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        
        # Use Focal Loss for class imbalance
        self.criterion = FocalLoss(gamma=2.0)
        
        # AdamW with warmup
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Learning rate scheduler with warmup
        self.warmup_epochs = 5
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50, eta_min=1e-6
        )
        
        self.best_val_f1 = 0.0
        self.start_epoch = 0  # For resuming
        self.train_losses = []
        self.val_losses = []
        self.val_f1_scores = []
    
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint to resume training"""
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            return False
        
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if available (backward compatibility)
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("✓ Loaded optimizer state")
        else:
            print("⚠ Optimizer state not found in checkpoint, using fresh optimizer")
        
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_f1 = checkpoint.get('val_f1', 0.0)
        
        # Restore training history if available
        if 'train_losses' in checkpoint:
            self.train_losses = checkpoint['train_losses']
        if 'val_losses' in checkpoint:
            self.val_losses = checkpoint['val_losses']
        if 'val_f1_scores' in checkpoint:
            self.val_f1_scores = checkpoint['val_f1_scores']
        
        # Update scheduler state if past warmup
        if self.start_epoch >= self.warmup_epochs:
            for _ in range(self.start_epoch - self.warmup_epochs):
                self.scheduler.step()
        
        print(f"✓ Resumed from epoch {self.start_epoch}, Best Val F1: {self.best_val_f1:.4f}")
        return True
        
    def train_epoch(self, epoch):
        """Train for one epoch with warmup"""
        self.model.train()
        total_loss = 0.0
        
        # Warmup learning rate
        if epoch < self.warmup_epochs:
            lr_scale = (epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr_scale * 1e-4
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            
            # Gradient clipping (important for KANs)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"  Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}")
        
        return total_loss / len(self.train_loader)
    
    def evaluate(self, data_loader):
        """Evaluate on validation or test set"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
                
                preds = torch.argmax(output, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        avg_loss = total_loss / len(data_loader)
        macro_f1 = f1_score(np.array(all_targets), np.array(all_preds), average='macro')
        
        return avg_loss, macro_f1, np.array(all_preds), np.array(all_targets)
    
    def train(self, num_epochs, checkpoint_dir):
        """Full training loop"""
        print(f"\nStarting Kanformer training for {num_epochs} epochs...")
        
        for epoch in range(self.start_epoch, num_epochs):
            print(f"\n{'='*60}\nEpoch {epoch+1}/{num_epochs}\n{'='*60}")
            
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            print(f"Train Loss: {train_loss:.4f}")
            
            val_loss, val_f1, _, _ = self.evaluate(self.val_loader)
            self.val_losses.append(val_loss)
            self.val_f1_scores.append(val_f1)
            print(f"Val Loss: {val_loss:.4f}, Val Macro F1: {val_f1:.4f}")
            
            # Step scheduler after warmup
            if epoch >= self.warmup_epochs:
                self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Learning rate: {current_lr:.6f}")
            
            # Save best model
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_f1': val_f1,
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                    'val_f1_scores': self.val_f1_scores,
                }, os.path.join(checkpoint_dir, 'best_model.pt'))
                print(f"✓ Saved best model with Val F1: {val_f1:.4f}")
            
            # Save periodic checkpoint with full state
            if (epoch + 1) % 5 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_f1': val_f1,
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                    'val_f1_scores': self.val_f1_scores,
                }, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt'))
            
            # Always save latest checkpoint for easy resumption
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_f1': val_f1,
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'val_f1_scores': self.val_f1_scores,
            }, os.path.join(checkpoint_dir, 'latest_checkpoint.pt'))
        
        print(f"\n{'='*60}\nTraining completed! Best Val F1: {self.best_val_f1:.4f}\n{'='*60}")


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    if len(sys.argv) < 2:
        print("Usage: python STMkanformer_model.py <mode> [--resume <checkpoint_dir>]")
        print("  0: Standard training")
        print("  1: Downsample non-tonal speech")
        print("  --resume: Resume from checkpoint directory")
        sys.exit(1)
    
    mode = int(sys.argv[1])
    ds_nontonal_speech = (mode == 1)
    
    # Check for resume flag
    resume_dir = None
    if '--resume' in sys.argv:
        resume_idx = sys.argv.index('--resume')
        if resume_idx + 1 < len(sys.argv):
            resume_dir = sys.argv[resume_idx + 1]
            print(f"Resume mode: Will attempt to load from {resume_dir}")
    
    # Set directory
    directory = "model/STM/Kanformer_corpora_categories/" + ("downsample" if mode == 1 else "standard")
    
    if resume_dir:
        # Use provided checkpoint directory
        checkpoint_dir = resume_dir
        if not os.path.exists(checkpoint_dir):
            print(f"Error: Checkpoint directory does not exist: {checkpoint_dir}")
            sys.exit(1)
    else:
        # Create new directory with timestamp
        time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        checkpoint_dir = os.path.join(directory, "ckpt", time_stamp)
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"Checkpoint directory: {checkpoint_dir}")
    
    # Prepare data
    data_prep = prepData_STM_Kanformer(ds_nontonal_speech=ds_nontonal_speech)
    train_dataset, val_dataset, test_dataset, n_freq, n_time = data_prep.prepare_datasets()
    
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Create Kanformer model
    model = KanformerClassifier(
        input_dim=n_freq,
        num_classes=6,
        d_model=128,
        num_heads=4,
        ffn_dim=512,
        num_layers=4,
        kernel_size=31,
        dropout=0.1,
        num_kan_groups=8
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Train
    trainer = KanformerTrainer(model, train_loader, val_loader, test_loader, device, lr=1e-4, weight_decay=1e-4)
    
    # Resume from checkpoint if specified
    if resume_dir:
        latest_ckpt = os.path.join(checkpoint_dir, 'latest_checkpoint.pt')
        if os.path.exists(latest_ckpt):
            trainer.load_checkpoint(latest_ckpt)
        else:
            # Try to find the most recent epoch checkpoint
            ckpt_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_')]
            if ckpt_files:
                epochs = [int(f.split('_')[-1].replace('.pt', '')) for f in ckpt_files]
                latest_epoch = max(epochs)
                latest_ckpt = os.path.join(checkpoint_dir, f'checkpoint_epoch_{latest_epoch}.pt')
                trainer.load_checkpoint(latest_ckpt)
            else:
                print("Warning: No checkpoint found to resume from, starting fresh")
    
    trainer.train(num_epochs=50, checkpoint_dir=checkpoint_dir)
    
    # Test
    checkpoint = torch.load(os.path.join(checkpoint_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_f1, test_preds, test_targets = trainer.evaluate(test_loader)
    print(f"Test Loss: {test_loss:.4f}, Test Macro F1: {test_f1:.4f}")
    
    np.save(os.path.join(checkpoint_dir, 'test_predictions.npy'), test_preds)
    np.save(os.path.join(checkpoint_dir, 'test_targets.npy'), test_targets)
