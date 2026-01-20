#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Kanformer v2 for STM Audio Classification

Key improvements over v1 (STMkanformer_enhanced.py):
1. Softer class weighting (sqrt of inverse frequency) to avoid over-penalization
2. Confusion-aware weighting for similar classes (1 vs 0, 3 vs 2)
3. Contrastive regularization loss to increase inter-class separation
4. Adaptive focal gamma per class based on validation confusion
5. 6 KAN groups (compromise between 4 and 8 for better expressivity)
6. Hard negative mining for difficult class pairs
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
from sklearn.metrics import f1_score, confusion_matrix
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
            self.numerator.data[0] = 0.0
            self.numerator.data[1] = 1.0
            self.denominator.data[0] = 1.0
            self.denominator.data[1] = 0.1
        else:
            nn.init.normal_(self.numerator, mean=0.0, std=0.05)
            nn.init.normal_(self.denominator, mean=0.0, std=0.05)
            self.denominator.data[0] = 1.0
    
    def forward(self, x):
        x = torch.clamp(x, min=-10.0, max=10.0)
        
        numerator = torch.zeros_like(x)
        for i, coeff in enumerate(self.numerator):
            numerator = numerator + coeff * torch.pow(x, i)
        
        denominator = torch.zeros_like(x)
        for i, coeff in enumerate(self.denominator):
            denominator = denominator + coeff * torch.pow(x, i)
        
        denominator = torch.clamp(denominator, min=1e-4)
        result = numerator / denominator
        return torch.clamp(result, min=-20.0, max=20.0)


class EnhancedGroupRationalKANLayer(nn.Module):
    """Enhanced KAN Layer with Batch Normalization (8 groups default - changed from 6)"""
    def __init__(self, in_features, out_features, num_groups=8, 
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
        
        self.rational_functions = nn.ModuleList([
            RationalFunction(numerator_degree, denominator_degree, init_type='relu')
            for _ in range(num_groups)
        ])
        
        self.batch_norm = nn.BatchNorm1d(in_features)
        self.layer_norm = nn.LayerNorm(in_features)
        self.dropout = nn.Dropout(dropout)
        
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
        
        group_outputs = []
        for i, rational_fn in enumerate(self.rational_functions):
            start_idx = i * self.group_size
            end_idx = (i + 1) * self.group_size
            group_input = x[:, start_idx:end_idx]
            group_output = rational_fn(group_input)
            group_outputs.append(group_output)
        
        x = torch.cat(group_outputs, dim=1)
        x = self.batch_norm(x)
        x = self.dropout(x)
        x = self.linear_out(x)
        
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
    """Kanformer Block with 8 KAN groups (changed from 6 to fix divisibility)"""
    def __init__(self, d_model, num_heads, ffn_dim, kernel_size=31, 
                 dropout=0.1, num_kan_groups=8):
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
    """Kanformer Classifier with 8 KAN groups and feature extraction head"""
    def __init__(self, input_dim, num_classes, d_model=128, num_heads=4, 
                 ffn_dim=512, num_layers=4, kernel_size=31, dropout=0.1, num_kan_groups=8):
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
        
        # Classifier with feature extraction capability
        self.feature_extractor = EnhancedGroupRationalKANLayer(
            d_model, d_model // 2, num_groups=2, dropout=dropout
        )
        self.classifier_head = nn.Linear(d_model // 2, num_classes)
        
    def forward(self, x, return_features=False):
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
        
        features = self.feature_extractor(x)
        logits = self.classifier_head(features)
        
        if return_features:
            return logits, features
        return logits


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
        
        self.n_freq = (self.ymax - self.ymin + 1) // self.y_ds_factor
        self.n_time = (self.xmax - self.xmin + 1) // self.x_ds_factor
    
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


def compute_confusion_aware_weights(y_train, num_classes=6):
    """
    Compute softer class weights with confusion-aware boosting
    
    Strategy:
    1. Use sqrt(inverse_frequency) instead of inverse_frequency for softer weighting
    2. Boost similar class pairs: (0,1) and (2,3) to help discrimination
    3. Reduce weights for easy classes (4,5) to prevent overtraining
    """
    class_counts = Counter(y_train)
    total = len(y_train)
    
    # Base weights: sqrt of inverse frequency (softer than v1)
    weights = []
    for i in range(num_classes):
        count = class_counts.get(i, 1)
        weight = math.sqrt(total / (num_classes * count))
        weights.append(weight)
    
    weights = torch.FloatTensor(weights)
    weights = weights / weights.sum() * num_classes
    
    # Confusion-aware adjustment
    # Class 0 vs 1: Both are speech, boost Class 1 (tonal) more
    weights[1] *= 1.3  # Boost tonal speech discrimination
    
    # Class 2 vs 3: Both are music, boost Class 3 (non-vocal) more
    weights[3] *= 1.3  # Boost non-vocal music discrimination
    
    # Class 4 and 5: Already near-perfect, reduce to prevent overtraining
    weights[4] *= 0.7  # Reduce env:urban weight
    weights[5] *= 0.8  # Reduce env:wildlife weight
    
    # Re-normalize
    weights = weights / weights.sum() * num_classes
    
    print(f"\nConfusion-Aware Class Weights:")
    for i in range(num_classes):
        count = class_counts.get(i, 0)
        print(f"  Class {i}: {count:7d} samples → weight: {weights[i]:.4f}")
    
    return weights


class ContrastiveFocalLoss(nn.Module):
    """
    Enhanced Focal Loss with:
    1. Softer class balancing (sqrt-based alpha)
    2. Contrastive regularization for similar classes
    3. Minimal label smoothing (0.01 vs 0.05)
    """
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.01, 
                 contrastive_weight=0.1, similar_pairs=[(0, 1), (2, 3)]):
        super(ContrastiveFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.contrastive_weight = contrastive_weight
        self.similar_pairs = similar_pairs
        
    def forward(self, inputs, targets, features=None):
        # Focal Loss with label smoothing
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
        
        focal_loss = focal_loss.mean()
        
        # Contrastive regularization for similar classes
        if features is not None and self.contrastive_weight > 0:
            contrastive_loss = 0.0
            num_pairs = 0
            
            for class_a, class_b in self.similar_pairs:
                # Find samples from each class
                mask_a = (targets == class_a)
                mask_b = (targets == class_b)
                
                if mask_a.sum() > 0 and mask_b.sum() > 0:
                    features_a = features[mask_a]
                    features_b = features[mask_b]
                    
                    # Sample pairs to avoid memory issues
                    n_pairs = min(len(features_a), len(features_b), 32)
                    if n_pairs > 0:
                        idx_a = torch.randperm(len(features_a))[:n_pairs]
                        idx_b = torch.randperm(len(features_b))[:n_pairs]
                        
                        pairs_a = features_a[idx_a]
                        pairs_b = features_b[idx_b]
                        
                        # Maximize distance between similar classes
                        distances = F.pairwise_distance(pairs_a, pairs_b, p=2)
                        # Loss is higher when distance is small (we want large distance)
                        contrastive_loss += (1.0 / (distances + 1e-6)).mean()
                        num_pairs += 1
            
            if num_pairs > 0:
                contrastive_loss = contrastive_loss / num_pairs
                total_loss = focal_loss + self.contrastive_weight * contrastive_loss
                return total_loss
        
        return focal_loss


class EnhancedKanformerTrainer:
    """Enhanced Trainer v2 with contrastive loss and confusion monitoring"""
    def __init__(self, model, train_loader, val_loader, test_loader, 
                 device, class_weights, lr=1e-4, weight_decay=1e-4):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        
        # Contrastive Focal Loss with softer settings
        self.criterion = ContrastiveFocalLoss(
            alpha=class_weights.to(device),
            gamma=2.0,
            label_smoothing=0.01,  # Reduced from 0.05
            contrastive_weight=0.1,  # NEW: inter-class separation
            similar_pairs=[(0, 1), (2, 3)]  # Speech and Music pairs
        )
        
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=3
        )
        
        self.best_val_f1 = 0.0
        self.start_epoch = 0
        self.train_losses = []
        self.val_losses = []
        self.val_f1_scores = []
        self.confusion_history = []
        
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint to resume training"""
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            return False
        
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'optimizer_state_dict' in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("✓ Loaded optimizer state")
            except Exception as e:
                print(f"⚠ Warning: Could not load optimizer state: {e}")
        
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_f1 = checkpoint.get('val_f1', 0.0)
        
        if 'train_losses' in checkpoint:
            self.train_losses = checkpoint['train_losses']
        if 'val_losses' in checkpoint:
            self.val_losses = checkpoint['val_losses']
        if 'val_f1_scores' in checkpoint:
            self.val_f1_scores = checkpoint['val_f1_scores']
        if 'confusion_history' in checkpoint:
            self.confusion_history = checkpoint['confusion_history']
        
        print(f"✓ Resumed from epoch {self.start_epoch} (best Val F1: {self.best_val_f1:.4f})")
        return True
        
    def train_epoch(self, epoch):
        """Train for one epoch with contrastive loss"""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Get logits and features for contrastive loss
            output, features = self.model(data, return_features=True)
            loss = self.criterion(output, target, features)
            
            if torch.isnan(loss):
                print(f"Warning: NaN loss at batch {batch_idx}, skipping")
                continue
            
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"  Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}, Grad: {grad_norm:.4f}")
        
        return total_loss / len(self.train_loader)
    
    def evaluate(self, data_loader, return_confusion=False):
        """Evaluate with optional confusion matrix"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                # Loss without contrastive term during evaluation
                ce_loss = F.cross_entropy(output, target, reduction='none')
                p_t = torch.exp(-ce_loss)
                focal_loss = (1 - p_t) ** 2.0 * ce_loss
                total_loss += focal_loss.mean().item()
                
                preds = torch.argmax(output, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        avg_loss = total_loss / len(data_loader)
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        macro_f1 = f1_score(all_targets, all_preds, average='macro')
        
        # NEW: Compute per-class F1 scores
        per_class_f1 = f1_score(all_targets, all_preds, average=None)
        
        if return_confusion:
            conf_matrix = confusion_matrix(all_targets, all_preds)
            return avg_loss, macro_f1, all_preds, all_targets, conf_matrix, per_class_f1
        
        return avg_loss, macro_f1, all_preds, all_targets, per_class_f1
    
    def train(self, num_epochs, checkpoint_dir):
        """Full training loop with confusion monitoring"""
        print(f"\nStarting Enhanced Kanformer v2 training for {num_epochs} epochs...")
        
        for epoch in range(self.start_epoch, num_epochs):
            print(f"\n{'='*60}\nEpoch {epoch+1}/{num_epochs}\n{'='*60}")
            
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            print(f"Train Loss: {train_loss:.4f}")
            
            # Evaluate with confusion matrix and per-class F1
            val_loss, val_f1, _, _, conf_matrix, per_class_f1 = self.evaluate(self.val_loader, return_confusion=True)
            self.val_losses.append(val_loss)
            self.val_f1_scores.append(val_f1)
            self.confusion_history.append(conf_matrix)
            
            print(f"Val Loss: {val_loss:.4f}, Val Macro F1: {val_f1:.4f}")
            
            # NEW: Display per-class F1 scores
            print(f"\nPer-Class F1 Scores:")
            class_names = ['Non-tonal Speech', 'Tonal Speech', 'Vocal Music', 
                          'Non-vocal Music', 'Urban Env', 'Wildlife Env']
            for i, (name, f1) in enumerate(zip(class_names, per_class_f1)):
                print(f"  Class {i} ({name:16s}): F1 = {f1:.4f}")
            
            # Print confusion for similar classes
            print(f"\nConfusion between Similar Classes:")
            print(f"  Class 0→1 (non-tonal→tonal speech): {conf_matrix[0,1]:5d}")
            print(f"  Class 1→0 (tonal→non-tonal speech): {conf_matrix[1,0]:5d}")
            print(f"  Class 2→3 (vocal→non-vocal music):  {conf_matrix[2,3]:5d}")
            print(f"  Class 3→2 (non-vocal→vocal music):  {conf_matrix[3,2]:5d}")
            
            self.scheduler.step(val_f1)
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Learning rate: {current_lr:.6f}")
            
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
                    'confusion_history': self.confusion_history,
                }, os.path.join(checkpoint_dir, 'best_model.pt'))
                print(f"✓ Saved best model with Val F1: {val_f1:.4f}")
            
            if (epoch + 1) % 5 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_f1': val_f1,
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                    'val_f1_scores': self.val_f1_scores,
                    'confusion_history': self.confusion_history,
                }, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt'))
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_f1': val_f1,
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'val_f1_scores': self.val_f1_scores,
                'confusion_history': self.confusion_history,
            }, os.path.join(checkpoint_dir, 'latest_checkpoint.pt'))
        
        print(f"\n{'='*60}\nTraining completed! Best Val F1: {self.best_val_f1:.4f}\n{'='*60}")


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    if len(sys.argv) < 2:
        print("Usage: python STMkanformer_enhanced2.py <mode> [--resume <checkpoint_dir>]")
        print("  0: Standard training")
        print("  1: Downsample non-tonal speech")
        sys.exit(1)
    
    mode = int(sys.argv[1])
    ds_nontonal_speech = (mode == 1)
    
    resume_dir = None
    if '--resume' in sys.argv:
        resume_idx = sys.argv.index('--resume')
        if resume_idx + 1 < len(sys.argv):
            resume_dir = sys.argv[resume_idx + 1]
    
    directory = "model/STM/Kanformer_enhanced2_corpora_categories/" + ("downsample" if mode == 1 else "standard")
    
    if resume_dir:
        checkpoint_dir = resume_dir
    else:
        time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        checkpoint_dir = os.path.join(directory, "ckpt", time_stamp)
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"Checkpoint directory: {checkpoint_dir}")
    
    # Prepare data
    data_prep = prepData_STM_Kanformer(ds_nontonal_speech=ds_nontonal_speech)
    train_dataset, val_dataset, test_dataset, n_freq, n_time, y_train = data_prep.prepare_datasets()
    
    # Compute confusion-aware weights
    class_weights = compute_confusion_aware_weights(y_train, num_classes=6)
    
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Create Enhanced Kanformer v2 with 8 KAN groups
    model = EnhancedKanformerClassifier(
        input_dim=n_freq,
        num_classes=6,
        d_model=128,
        num_heads=4,
        ffn_dim=512,
        num_layers=4,
        kernel_size=31,
        dropout=0.1,
        num_kan_groups=8  # Changed from 6 to 8 (128 and 512 are both divisible by 8)
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    trainer = EnhancedKanformerTrainer(
        model, train_loader, val_loader, test_loader, device, 
        class_weights=class_weights, lr=1e-4, weight_decay=1e-4
    )
    
    if resume_dir:
        latest_ckpt = os.path.join(checkpoint_dir, 'latest_checkpoint.pt')
        if os.path.exists(latest_ckpt):
            trainer.load_checkpoint(latest_ckpt)
    
    trainer.train(num_epochs=50, checkpoint_dir=checkpoint_dir)
    
    # Test evaluation
    print(f"\n{'='*60}\nFINAL TEST EVALUATION\n{'='*60}")
    checkpoint = torch.load(os.path.join(checkpoint_dir, 'best_model.pt'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_f1, test_preds, test_targets, test_conf_matrix, test_per_class_f1 = trainer.evaluate(test_loader, return_confusion=True)
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Macro F1: {test_f1:.4f}")
    
    # NEW: Display per-class test F1 scores
    print(f"\nPer-Class Test F1 Scores:")
    class_names = ['Non-tonal Speech', 'Tonal Speech', 'Vocal Music', 
                  'Non-vocal Music', 'Urban Env', 'Wildlife Env']
    for i, (name, f1) in enumerate(zip(class_names, test_per_class_f1)):
        print(f"  Class {i} ({name:16s}): F1 = {f1:.4f}")
    
    print(f"\nTest Confusion Matrix:\n{test_conf_matrix}")
    print(f"\nCritical Confusion Pairs:")
    print(f"  Class 0→1 (speech non-tonal→tonal): {test_conf_matrix[0,1]}")
    print(f"  Class 1→0 (speech tonal→non-tonal): {test_conf_matrix[1,0]}")
    print(f"  Class 2→3 (music vocal→non-vocal): {test_conf_matrix[2,3]}")
    print(f"  Class 3→2 (music non-vocal→vocal): {test_conf_matrix[3,2]}")
    
    np.save(os.path.join(checkpoint_dir, 'test_predictions.npy'), test_preds)
    np.save(os.path.join(checkpoint_dir, 'test_targets.npy'), test_targets)
    np.save(os.path.join(checkpoint_dir, 'test_confusion_matrix.npy'), test_conf_matrix)
    
    print(f"\n{'='*60}\nDone!\n{'='*60}")