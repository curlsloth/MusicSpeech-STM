#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Kanformer for STM Audio Classification

Key improvements over baseline:
1. Class-balanced Focal Loss with computed alpha weights
2. Batch normalization in KAN layers for stability
3. Reduced KAN groups (8→4) to prevent overfitting on large dataset
4. Label smoothing for better generalization
5. ReduceLROnPlateau scheduler (matches Conformer baseline)
6. Gradient clipping with adaptive norm
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
from collections import Counter
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# Enhanced Group-Rational KAN Implementation
# ============================================================================

class RationalFunction(nn.Module):
    """Learnable rational activation with gradient clipping for stability"""
    def __init__(self, numerator_degree=5, denominator_degree=4, init_type='identity'):
        super(RationalFunction, self).__init__()
        
        self.numerator_degree = numerator_degree
        self.denominator_degree = denominator_degree
        
        self.numerator = nn.Parameter(torch.zeros(numerator_degree + 1))
        self.denominator = nn.Parameter(torch.zeros(denominator_degree + 1))
        
        self._initialize_coefficients(init_type)
    
    def _initialize_coefficients(self, init_type):
        if init_type == 'identity':
            self.numerator.data[1] = 1.0
            self.denominator.data[0] = 1.0
        elif init_type == 'relu':
            # Better initialization for ReLU approximation
            self.numerator.data[0] = 0.0
            self.numerator.data[1] = 1.0
            self.denominator.data[0] = 1.0
            self.denominator.data[1] = 0.1  # Small slope for smoothness
        else:
            nn.init.normal_(self.numerator, mean=0.0, std=0.05)  # Reduced std
            nn.init.normal_(self.denominator, mean=0.0, std=0.05)
            self.denominator.data[0] = 1.0
    
    def forward(self, x):
        # Clamp input to prevent extreme values
        x = torch.clamp(x, min=-10.0, max=10.0)
        
        numerator = torch.zeros_like(x)
        for i, coeff in enumerate(self.numerator):
            numerator = numerator + coeff * torch.pow(x, i)
        
        denominator = torch.zeros_like(x)
        for i, coeff in enumerate(self.denominator):
            denominator = denominator + coeff * torch.pow(x, i)
        
        # Stronger clamping for denominator
        denominator = torch.clamp(denominator, min=1e-4)
        
        result = numerator / denominator
        # Clamp output to prevent explosions
        return torch.clamp(result, min=-20.0, max=20.0)


class EnhancedGroupRationalKANLayer(nn.Module):
    """
    Enhanced KAN Layer with Batch Normalization and reduced groups
    
    Changes from baseline:
    - Added batch normalization after rational functions for stability
    - Reduced default groups from 8 to 4 (less overfitting on large dataset)
    - Residual connection for better gradient flow
    """
    def __init__(self, in_features, out_features, num_groups=4, 
                 numerator_degree=5, denominator_degree=4, dropout=0.1):
        super(EnhancedGroupRationalKANLayer, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        
        assert in_features % num_groups == 0, \
            f"in_features ({in_features}) must be divisible by num_groups ({num_groups})"
        
        self.group_size = in_features // num_groups
        
        self.linear_in = nn.Linear(in_features, in_features)
        self.linear_out = nn.Linear(in_features, out_features)
        
        # Group-wise rational functions
        self.rational_functions = nn.ModuleList([
            RationalFunction(numerator_degree, denominator_degree, init_type='relu')
            for _ in range(num_groups)
        ])
        
        # Batch normalization for stability (NEW)
        self.batch_norm = nn.BatchNorm1d(in_features)
        self.layer_norm = nn.LayerNorm(in_features)
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection flag
        self.use_residual = (in_features == out_features)
        
    def forward(self, x):
        original_shape = x.shape
        is_3d = len(original_shape) == 3
        
        if is_3d:
            batch_size, seq_len, _ = x.shape
            x = x.reshape(batch_size * seq_len, -1)
        
        identity = x if self.use_residual else None
        
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
        
        x = torch.cat(group_outputs, dim=1)
        
        # Batch normalization (NEW)
        x = self.batch_norm(x)
        
        x = self.dropout(x)
        x = self.linear_out(x)
        
        # Residual connection if dimensions match
        if self.use_residual and identity is not None:
            x = x + identity
        
        if is_3d:
            x = x.reshape(batch_size, seq_len, -1)
        
        return x


# ============================================================================
# Kanformer Architecture (unchanged from baseline)
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
        
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.d_model)
        
        output = self.out_proj(attn_output)
        
        return output


class ConvolutionModule(nn.Module):
    """Conformer convolution module"""
    def __init__(self, d_model, kernel_size=31, dropout=0.1):
        super(ConvolutionModule, self).__init__()
        
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2, groups=d_model
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.layer_norm(x)
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)
        return x


class EnhancedKanformerBlock(nn.Module):
    """
    Enhanced Kanformer Block with reduced KAN groups
    Uses num_kan_groups=4 instead of 8 for less overfitting
    """
    def __init__(self, d_model, num_heads, ffn_dim, kernel_size=31, 
                 dropout=0.1, num_kan_groups=4):  # Changed default from 8 to 4
        super(EnhancedKanformerBlock, self).__init__()
        
        self.ffn1 = nn.Sequential(
            nn.LayerNorm(d_model),
            EnhancedGroupRationalKANLayer(d_model, ffn_dim, num_groups=num_kan_groups, dropout=dropout),
            nn.Dropout(dropout),
            EnhancedGroupRationalKANLayer(ffn_dim, d_model, num_groups=num_kan_groups, dropout=dropout),
        )
        
        self.mhsa = nn.Sequential(
            nn.LayerNorm(d_model),
            MultiHeadSelfAttention(d_model, num_heads, dropout),
            nn.Dropout(dropout)
        )
        
        self.conv = ConvolutionModule(d_model, kernel_size, dropout)
        
        self.ffn2 = nn.Sequential(
            nn.LayerNorm(d_model),
            EnhancedGroupRationalKANLayer(d_model, ffn_dim, num_groups=num_kan_groups, dropout=dropout),
            nn.Dropout(dropout),
            EnhancedGroupRationalKANLayer(ffn_dim, d_model, num_groups=num_kan_groups, dropout=dropout),
        )
        
        self.ffn_scale = 0.5
        
    def forward(self, x):
        x = x + self.ffn_scale * self.ffn1(x)
        x = x + self.mhsa(x)
        x = x + self.conv(x)
        x = x + self.ffn_scale * self.ffn2(x)
        return x


class EnhancedKanformerClassifier(nn.Module):
    """
    Enhanced Kanformer with reduced KAN groups and better stability
    """
    def __init__(self, input_dim, num_classes, d_model=128, num_heads=4, 
                 ffn_dim=512, num_layers=4, kernel_size=31, dropout=0.1, num_kan_groups=4):
        super(EnhancedKanformerClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        self.conv_stem = nn.Sequential(
            nn.Conv1d(input_dim, d_model // 2, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.BatchNorm1d(d_model // 2),
            nn.Conv1d(d_model // 2, d_model, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.BatchNorm1d(d_model),
            nn.Dropout(dropout)
        )
        
        self.pos_embedding = nn.Parameter(torch.randn(1, 121, d_model) * 0.02)
        
        self.kanformer_blocks = nn.ModuleList([
            EnhancedKanformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                kernel_size=kernel_size,
                dropout=dropout,
                num_kan_groups=num_kan_groups
            )
            for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(d_model)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Simpler classifier with label smoothing compatibility
        self.classifier = nn.Sequential(
            EnhancedGroupRationalKANLayer(d_model, d_model // 2, num_groups=2, dropout=dropout),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        x = self.conv_stem(x)
        x = x.transpose(1, 2)
        
        seq_len = x.size(1)
        x = x + self.pos_embedding[:, :seq_len, :]
        
        for block in self.kanformer_blocks:
            x = block(x)
        
        x = self.final_norm(x)
        x = x.transpose(1, 2)
        x = self.global_pool(x)
        x = x.squeeze(-1)
        x = self.classifier(x)
        
        return x


# ============================================================================
# Data Preparation (same as baseline)
# ============================================================================

class prepData_STM_Kanformer:
    """Data preparation for Kanformer model"""
    def __init__(self, addAug=False, ds_nontonal_speech=False):
        self.addAug = addAug
        self.ds_nontonal_speech = ds_nontonal_speech
        
        self.xmin = 190
        self.xmax = 310
        self.ymin = 75
        self.ymax = 114
        self.x_ds_factor = 1
        self.y_ds_factor = 2
        
        self.n_freq = (self.ymax - self.ymin + 1) // self.y_ds_factor  # 20
        self.n_time = (self.xmax - self.xmin + 1) // self.x_ds_factor  # 121
    
    def corpora_list(self, addAug=False):
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
        
        STM_all_2d = STM_all.reshape(-1, self.n_freq, self.n_time)
        
        means = STM_all_2d.mean(axis=(1, 2), keepdims=True)
        stds = STM_all_2d.std(axis=(1, 2), keepdims=True)
        STM_all_2d = (STM_all_2d - means) / (stds + 1e-8)
        
        X_train = torch.FloatTensor(STM_all_2d[train_ind])
        y_train = torch.LongTensor(target[train_ind])
        X_val = torch.FloatTensor(STM_all_2d[val_ind])
        y_val = torch.LongTensor(target[val_ind])
        X_test = torch.FloatTensor(STM_all_2d[test_ind])
        y_test = torch.LongTensor(target[test_ind])
        
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)
        
        return train_dataset, val_dataset, test_dataset, self.n_freq, self.n_time, y_train.numpy()


# ============================================================================
# Enhanced Focal Loss with Class Balancing
# ============================================================================

class BalancedFocalLoss(nn.Module):
    """
    Class-Balanced Focal Loss with Label Smoothing
    
    Improvements:
    1. Computes alpha weights from training data (inverse frequency)
    2. Supports label smoothing for better generalization
    3. More stable implementation
    """
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.1, reduction='mean'):
        super(BalancedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        # Apply label smoothing
        if self.label_smoothing > 0:
            num_classes = inputs.size(-1)
            targets_one_hot = F.one_hot(targets, num_classes).float()
            targets_smoothed = targets_one_hot * (1 - self.label_smoothing) + \
                             (1 - targets_one_hot) * self.label_smoothing / (num_classes - 1)
            ce_loss = -(targets_smoothed * F.log_softmax(inputs, dim=-1)).sum(dim=-1)
        else:
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


def compute_class_weights(y_train, num_classes=6):
    """
    Compute inverse frequency weights for class balancing
    
    Returns alpha weights that sum to num_classes
    """
    class_counts = Counter(y_train)
    total = len(y_train)
    
    weights = []
    for i in range(num_classes):
        count = class_counts.get(i, 1)
        weight = total / (num_classes * count)
        weights.append(weight)
    
    weights = torch.FloatTensor(weights)
    # Normalize so sum = num_classes (standard practice)
    weights = weights / weights.sum() * num_classes
    
    print(f"\nClass distribution:")
    for i in range(num_classes):
        count = class_counts.get(i, 0)
        print(f"  Class {i}: {count:7d} samples (weight: {weights[i]:.4f})")
    
    return weights


# ============================================================================
# Enhanced Trainer
# ============================================================================

class EnhancedKanformerTrainer:
    """
    Enhanced trainer with:
    1. Class-balanced Focal Loss
    2. ReduceLROnPlateau scheduler (matches Conformer)
    3. Adaptive gradient clipping
    4. Better checkpointing
    """
    def __init__(self, model, train_loader, val_loader, test_loader, 
                 device, class_weights, lr=1e-4, weight_decay=1e-4):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        
        # Balanced Focal Loss with label smoothing
        self.criterion = BalancedFocalLoss(
            alpha=class_weights.to(device),
            gamma=2.0,
            label_smoothing=0.05  # Small smoothing for generalization
        )
        
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Use ReduceLROnPlateau like Conformer baseline (removed verbose parameter)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=3
        )
        
        self.best_val_f1 = 0.0
        self.start_epoch = 0
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
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'optimizer_state_dict' in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("✓ Loaded optimizer state")
            except Exception as e:
                print(f"⚠ Warning: Could not load optimizer state: {e}")
        else:
            print("⚠ Optimizer state not found in checkpoint")
        
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_f1 = checkpoint.get('val_f1', 0.0)
        
        if 'train_losses' in checkpoint:
            self.train_losses = checkpoint['train_losses']
        if 'val_losses' in checkpoint:
            self.val_losses = checkpoint['val_losses']
        if 'val_f1_scores' in checkpoint:
            self.val_f1_scores = checkpoint['val_f1_scores']
        
        print(f"✓ Resumed from epoch {self.start_epoch} (best Val F1: {self.best_val_f1:.4f})")
        return True
        
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Check for NaN
            if torch.isnan(loss):
                print(f"Warning: NaN loss detected at batch {batch_idx}, skipping")
                continue
            
            loss.backward()
            
            # Adaptive gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"  Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}, Grad Norm: {grad_norm:.4f}")
        
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
        print(f"\nStarting Enhanced Kanformer training for {num_epochs} epochs...")
        
        for epoch in range(self.start_epoch, num_epochs):
            print(f"\n{'='*60}\nEpoch {epoch+1}/{num_epochs}\n{'='*60}")
            
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            print(f"Train Loss: {train_loss:.4f}")
            
            val_loss, val_f1, _, _ = self.evaluate(self.val_loader)
            self.val_losses.append(val_loss)
            self.val_f1_scores.append(val_f1)
            print(f"Val Loss: {val_loss:.4f}, Val Macro F1: {val_f1:.4f}")
            
            # ReduceLROnPlateau scheduler
            self.scheduler.step(val_f1)
            
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
            
            # Save periodic checkpoints
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
                print(f"✓ Saved checkpoint at epoch {epoch+1}")
            
            # Always save latest
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
        print("Usage: python STMkanformer_enhanced.py <mode> [--resume <checkpoint_dir>]")
        print("  0: Standard training")
        print("  1: Downsample non-tonal speech")
        print("  --resume: Resume from checkpoint directory")
        sys.exit(1)
    
    mode = int(sys.argv[1])
    ds_nontonal_speech = (mode == 1)
    
    resume_dir = None
    if '--resume' in sys.argv:
        resume_idx = sys.argv.index('--resume')
        if resume_idx + 1 < len(sys.argv):
            resume_dir = sys.argv[resume_idx + 1]
            print(f"Resume mode: Will attempt to load from {resume_dir}")
    
    directory = "model/STM/Kanformer_enhanced_corpora_categories/" + ("downsample" if mode == 1 else "standard")
    
    if resume_dir:
        checkpoint_dir = resume_dir
        if not os.path.exists(checkpoint_dir):
            print(f"Error: Checkpoint directory does not exist: {checkpoint_dir}")
            sys.exit(1)
    else:
        time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        checkpoint_dir = os.path.join(directory, "ckpt", time_stamp)
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"Checkpoint directory: {checkpoint_dir}")
    
    # Prepare data (returns y_train for class weight computation)
    data_prep = prepData_STM_Kanformer(ds_nontonal_speech=ds_nontonal_speech)
    train_dataset, val_dataset, test_dataset, n_freq, n_time, y_train = data_prep.prepare_datasets()
    
    # Compute class weights
    class_weights = compute_class_weights(y_train, num_classes=6)
    
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Create Enhanced Kanformer with reduced groups (4 instead of 8)
    model = EnhancedKanformerClassifier(
        input_dim=n_freq,
        num_classes=6,
        d_model=128,
        num_heads=4,
        ffn_dim=512,
        num_layers=4,
        kernel_size=31,
        dropout=0.1,
        num_kan_groups=4  # Reduced from 8
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Train
    trainer = EnhancedKanformerTrainer(
        model, train_loader, val_loader, test_loader, device, 
        class_weights=class_weights, lr=1e-4, weight_decay=1e-4
    )
    
    # Resume if specified
    if resume_dir:
        latest_ckpt = os.path.join(checkpoint_dir, 'latest_checkpoint.pt')
        if os.path.exists(latest_ckpt):
            trainer.load_checkpoint(latest_ckpt)
        else:
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
    print(f"\n{'='*60}")
    print(f"FINAL TEST RESULTS")
    print(f"{'='*60}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Macro F1: {test_f1:.4f}")
    print(f"{'='*60}")
    
    np.save(os.path.join(checkpoint_dir, 'test_predictions.npy'), test_preds)
    np.save(os.path.join(checkpoint_dir, 'test_targets.npy'), test_targets)
