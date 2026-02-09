#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STM Classification with ImageNet-Pretrained ConvNeXt and LDAM Loss
Phase 2.8-ConvNeXt: Center Loss with Modern CNN Backbone

Key Innovations (V2.8-ConvNeXt):
1. ConvNeXt-Tiny Backbone: Modern CNN architecture (CVPR 2022)
   - Pure ConvNet design that competes with Vision Transformers
   - Large kernel convolutions (7×7) for better receptive field
   - LayerNorm + GELU instead of BatchNorm + ReLU
   - Inverted bottleneck design (expand→depthwise→project)
   - ~28.6M parameters, 768-dim features
2. Center Loss: Penalizes distance between features and learned class centers
   - L_center = Σ ||features_i - center[y_i]||²
   - L_total = L_hybrid + λ × L_center (λ=0.1)
   - Encourages intra-class compactness in feature space
3. Integrated TTA Evaluation: Test-Time Augmentation (5 augmentations)
   - Original, time_flip, freq_shift±2, time_shift+5
   - Soft voting on logits for robust predictions
4. Confusion Matrix Analysis: Detailed per-class confusion analysis

ConvNeXt Architecture:
1. Three-Scale Feature Fusion: stage1 (192ch) + stage2 (384ch) + stage3 (768ch)
2. Hybrid Loss: LDAM + Focal Loss (70% LDAM + 30% Focal)
3. Earlier DRW Activation: Epoch 30
4. STM-Specific Augmentation: SpecAugment-style masking + axis shifts
5. Layer Freezing Strategy: Freeze stem + stage0-1 for first 10 epochs
6. Discriminative Learning Rates: Pretrained layers learn slower
7. Difference Map Preprocessing: 2-channel input (Symmetric + Asymmetric)
8. ImageNet-pretrained ConvNeXt-Tiny backbone

Target: 0.88+ Macro F1 (approach SOTA 0.89+)
"""

import os
import sys
import datetime
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from torchvision import models
from torchvision.models import ConvNeXt_Tiny_Weights
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
rcParams['font.family'] = 'DejaVu Sans'
import ssl
import urllib.request

warnings.filterwarnings('ignore')

# Fix SSL certificate verification issues in HPC environments
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


# ============================================================================
# Data Preparation (Same as V2)
# ============================================================================

class prepData_STM_CoordConv:
    """
    Data preparation for CoordConv-ResNet model.
    Loads flattened STM data and reshapes to 2D (20x121) for coordinate-aware convolution.
    """
    def __init__(self, addAug=False, ds_nontonal_speech=False):
        self.addAug = addAug
        self.ds_nontonal_speech = ds_nontonal_speech
        
        # STM dimensions from preprocessing
        self.n_freq = 20   # Spectral modulation bins
        self.n_time = 121  # Temporal modulation bins
        
    def corpora_list(self, addAug=False):
        """Generate list of all corpora following the original pattern"""
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
            corpus_env_list = ['MacaulayLibrary', 'SONYC']
        else:
            corpus_env_list = ['MacaulayLibrary', 'SONYC']
        
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
            corp_name = corp.replace('/', '-')
            file_path = f"{root_folder}STM_output/corpSTMnpy/{corp_name}_STMall.npy"
            tmp = np.load(file_path)
            print(f"Loaded: {file_path}, shape: {tmp.shape}")
            STM_all = tmp if STM_all is None else np.concatenate((STM_all, tmp), axis=0)
        
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
        
        # Map categories (6 classes)
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
            nontonal_ind = (target == 0).values
            nontonal_n = nontonal_ind.sum()
            target_n = 100000
            if nontonal_n > target_n:
                nontonal_indices = np.where(nontonal_ind)[0]
                keep_indices = np.random.choice(nontonal_indices, target_n, replace=False)
                drop_indices = np.setdiff1d(nontonal_indices, keep_indices)
                
                keep_mask = np.ones(len(STM_all), dtype=bool)
                keep_mask[drop_indices] = False
                
                STM_all = STM_all[keep_mask]
                target = target[keep_mask].reset_index(drop=True)
                data_split = data_split[keep_mask].reset_index(drop=True)
        
        # Split data
        train_ind = (data_split < 8).values
        val_ind = (data_split == 8).values
        test_ind = (data_split == 9).values
        
        # Compute class frequencies for LDAM
        train_labels = target[train_ind].values
        class_counts = np.bincount(train_labels, minlength=6)
        
        print(f"\nDataset Statistics:")
        print(f"Total samples: {len(STM_all)}")
        print(f"Train samples: {sum(train_ind)}")
        print(f"Val samples: {sum(val_ind)}")
        print(f"Test samples: {sum(test_ind)}")
        print(f"Feature dimension (flattened): {STM_all.shape[1]}")
        print(f"Expected 2D shape: ({self.n_freq}, {self.n_time})")
        print(f"\nClass Distribution (Training):")
        for i, count in enumerate(class_counts):
            print(f"  Class {i}: {count:6d} samples ({100*count/class_counts.sum():.1f}%)")
        
        return STM_all, target.values, train_ind, val_ind, test_ind, class_counts
    
    def prepare_datasets(self):
        """Prepare PyTorch datasets with Difference Map (2-channel) preprocessing"""
        STM_all, target, train_ind, val_ind, test_ind, class_counts = self.load_data()
        
        # Reshape from flattened to 2D: (batch, freq, time)
        STM_all_2d = STM_all.reshape(-1, self.n_freq, self.n_time)
        
        # Normalize per sample (CRITICAL: preserves relative energy patterns)
        means = STM_all_2d.mean(axis=(1, 2), keepdims=True)
        stds = STM_all_2d.std(axis=(1, 2), keepdims=True)
        STM_all_2d = (STM_all_2d - means) / (stds + 1e-8)
        
        # ===== INNOVATION: Difference Map Preprocessing =====
        # Channel 1 (Symmetric): S(ω, Ω) = [M(ω, Ω) + M(-ω, Ω)] / 2
        # Channel 2 (Asymmetric): D(ω, Ω) = [M(ω, Ω) - M(-ω, Ω)] / 2
        # This exposes upward vs. downward frequency sweep asymmetry
        
        # Flip along spectral modulation axis (axis=1, freq dimension)
        STM_flipped = np.flip(STM_all_2d, axis=1).copy()
        
        # Compute symmetric and asymmetric components
        STM_symmetric = (STM_all_2d + STM_flipped) / 2.0
        STM_asymmetric = (STM_all_2d - STM_flipped) / 2.0
        
        # Stack to create 2-channel input: (batch, 2, freq, time)
        STM_all_2ch = np.stack([STM_symmetric, STM_asymmetric], axis=1)
        
        # Convert to PyTorch tensors
        X_train = torch.FloatTensor(STM_all_2ch[train_ind])
        y_train = torch.LongTensor(target[train_ind])
        
        X_val = torch.FloatTensor(STM_all_2ch[val_ind])
        y_val = torch.LongTensor(target[val_ind])
        
        X_test = torch.FloatTensor(STM_all_2ch[test_ind])
        y_test = torch.LongTensor(target[test_ind])
        
        # Create datasets
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)
        
        print(f"\nPyTorch Dataset Shapes (2-channel Difference Map):")
        print(f"Train: {X_train.shape}")
        print(f"Val: {X_val.shape}")
        print(f"Test: {X_test.shape}")
        print(f"Channel 0: Symmetric component (texture)")
        print(f"Channel 1: Asymmetric component (frequency sweep direction)")
        
        return train_dataset, val_dataset, test_dataset, class_counts


# ============================================================================
# CoordConv Layer
# ============================================================================

class CoordConv2d(nn.Module):
    """
    CoordConv: Adds coordinate channels to convolution input.
    Essential for STM because class distinctions depend on absolute position
    in the modulation spectrum (not translation-invariant patterns).
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, bias=True, dilation=1, groups=1):
        super(CoordConv2d, self).__init__()
        # Conv layer expects in_channels + 2 (for x and y coordinates)
        self.conv = nn.Conv2d(in_channels + 2, out_channels, kernel_size, 
                             stride=stride, padding=padding, bias=bias,
                             dilation=dilation, groups=groups)
        
    def forward(self, x):
        batch_size, _, height, width = x.size()
        
        # Create coordinate channels normalized to [-1, 1]
        y_coords = torch.linspace(-1, 1, height, device=x.device)
        x_coords = torch.linspace(-1, 1, width, device=x.device)
        y_coords = y_coords.view(1, 1, height, 1).expand(batch_size, 1, height, width)
        x_coords = x_coords.view(1, 1, 1, width).expand(batch_size, 1, height, width)
        
        # Concatenate coordinates with input: (B, C+2, H, W)
        x_with_coords = torch.cat([x, x_coords, y_coords], dim=1)
        
        return self.conv(x_with_coords)


# ============================================================================
# Attention Mechanisms (from V4)
# ============================================================================

class CoordinateAttention(nn.Module):
    """
    Coordinate Attention from "Coordinate Attention for Efficient Mobile Network Design"
    (Hou et al., CVPR 2021)
    
    Captures position-aware attention by pooling along each spatial axis separately.
    Better than SE for spatial data like STM features.
    """
    def __init__(self, in_channels, reduction=16):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        hidden_channels = max(8, in_channels // reduction)
        
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.act = nn.ReLU(inplace=True)
        
        self.conv_h = nn.Conv2d(hidden_channels, in_channels, 1, bias=False)
        self.conv_w = nn.Conv2d(hidden_channels, in_channels, 1, bias=False)
    
    def forward(self, x):
        b, c, h, w = x.size()
        
        # Pool along each axis
        x_h = self.pool_h(x)  # (b, c, h, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # (b, c, w, 1)
        
        # Concatenate
        y = torch.cat([x_h, x_w], dim=2)  # (b, c, h+w, 1)
        
        # Shared transform
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        
        # Split and generate attention
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        
        a_h = self.conv_h(x_h).sigmoid()  # (b, c, h, 1)
        a_w = self.conv_w(x_w).sigmoid()  # (b, c, 1, w)
        
        # Apply attention
        out = x * a_h * a_w
        
        return out


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation from "Squeeze-and-Excitation Networks"
    (Hu et al., CVPR 2018)
    
    Channel-wise attention mechanism.
    """
    def __init__(self, in_channels, reduction=16):
        super(SqueezeExcitation, self).__init__()
        hidden_channels = max(1, in_channels // reduction)
        
        self.fc1 = nn.Linear(in_channels, hidden_channels, bias=False)
        self.fc2 = nn.Linear(hidden_channels, in_channels, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _ = x.size()
        
        # Squeeze: Global average pooling
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        
        # Excitation: Two FC layers
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        
        # Rescale
        y = y.view(b, c, 1, 1)
        
        return x * y


# ============================================================================
# Pretrained ConvNeXt-Tiny with STM-Specific Adaptations + Multi-Scale
# ============================================================================

class PretrainedSTMConvNeXt(nn.Module):
    """
    ImageNet-pretrained ConvNeXt-Tiny adapted for STM classification.
    
    ConvNeXt Architecture Overview:
    - Modern ConvNet design that rivals Vision Transformers
    - Features: Large kernels (7×7), LayerNorm, GELU activation
    - Inverted bottleneck design (expand→depthwise→project)
    - Superior to traditional ResNets with similar parameters
    
    V2.8-ConvNeXt Enhancements:
    1. Three-Scale Feature Fusion: stage1 (192ch) + stage2 (384ch) + stage3 (768ch)
       - Combines features from 3 stages for rich multi-scale representation
       - Total: 1344 channels → 768 channels
    2. Custom Stem: 4-channel input (2 STM + 2 coords via CoordConv)
    3. Block Dropout: 0.05 via DropPath (stochastic depth)
    4. Head Dropout: 0.4 for stronger regularization
    5. Layer Freezing Support: Freeze stem + early stages for first 10 epochs
    
    Architecture Flow:
    Input (B, 2, 20, 121) → Custom Stem (4ch CoordConv) → 
    Stage0 (96ch) → Stage1 (192ch) → Stage2 (384ch) → Stage3 (768ch) → 
    Multi-Scale Fusion → Pool → FC
    """
    def __init__(self, num_classes=6, dropout=0.4, block_dropout=0.05, use_pretrained=True):
        super(PretrainedSTMConvNeXt, self).__init__()
        
        # Load pretrained ConvNeXt-Tiny with error handling
        pretrained_model = None
        weights_loaded = False
        
        if use_pretrained:
            print("Loading ImageNet-pretrained ConvNeXt-Tiny...")
            try:
                pretrained_model = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
                weights_loaded = True
                print("✓ Successfully loaded ImageNet pretrained weights")
            except Exception as e:
                print(f"⚠ Warning: Failed to download pretrained weights: {e}")
                print("⚠ Attempting to load from local cache...")
                try:
                    pretrained_model = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
                    weights_loaded = True
                    print("✓ Successfully loaded from cache")
                except:
                    print("⚠ Cache load failed. Training from scratch (random initialization)")
                    pretrained_model = models.convnext_tiny(weights=None)
                    weights_loaded = False
        else:
            print("Creating ConvNeXt-Tiny with random initialization...")
            pretrained_model = models.convnext_tiny(weights=None)
            weights_loaded = False
        
        # ===== Stem Modification: 4-channel CoordConv =====
        # ConvNeXt stem: Conv2d(3, 96, kernel_size=4, stride=4)
        # We need to modify for 2-channel + coords = 4 channel input
        # Use stride=1 to preserve resolution for small 20x121 input
        
        self.stem = nn.Sequential(
            CoordConv2d(
                in_channels=2,  # 2 STM channels (becomes 4 with coords)
                out_channels=96,
                kernel_size=4,
                stride=1,  # Preserve resolution for small input
                padding=2,
                bias=False
            ),
            nn.GroupNorm(1, 96, eps=1e-6)  # LayerNorm equivalent (num_groups=1)
        )
        
        # ===== Weight Cloning for Stem =====
        with torch.no_grad():
            if weights_loaded:
                # Original stem is features[0][0]: Conv2d(3, 96, 4, 4)
                old_conv = pretrained_model.features[0][0]
                pretrained_weights = old_conv.weight.data  # (96, 3, 4, 4)
                
                # Strategy: Copy mean of RGB channels to all 4 new channels, scale
                mean_channel = pretrained_weights.mean(dim=1, keepdim=True)  # (96, 1, 4, 4)
                new_weights = mean_channel.repeat(1, 4, 1, 1)  # (96, 4, 4, 4)
                
                # Scale to maintain variance: sqrt(3/4)
                new_weights = new_weights * (3.0 / 4.0) ** 0.5
                
                self.stem[0].conv.weight.data = new_weights
                print("✓ Cloned ImageNet weights to 4-channel CoordConv stem")
            else:
                nn.init.kaiming_normal_(self.stem[0].conv.weight, mode='fan_out')
                print("✓ Initialized 4-channel CoordConv stem with Kaiming Normal")
        
        # ===== Copy ConvNeXt Stages =====
        # ConvNeXt structure: features[0]=stem, [1]=stage0, [2]=downsample, [3]=stage1, etc.
        # Stage0: 96ch blocks
        self.stage0 = pretrained_model.features[1]
        # Downsample 96->192
        self.downsample1 = pretrained_model.features[2]
        # Stage1: 192ch blocks
        self.stage1 = pretrained_model.features[3]
        # Downsample 192->384
        self.downsample2 = pretrained_model.features[4]
        # Stage2: 384ch blocks
        self.stage2 = pretrained_model.features[5]
        # Downsample 384->768
        self.downsample3 = pretrained_model.features[6]
        # Stage3: 768ch blocks
        self.stage3 = pretrained_model.features[7]
        
        # ===== Three-Scale Feature Fusion =====
        # Combines stage1 (192ch) + stage2 (384ch) + stage3 (768ch)
        # Total: 1344 → 768
        self.multi_scale_fusion = nn.Sequential(
            nn.Conv2d(192 + 384 + 768, 768, kernel_size=1, bias=False),
            nn.GroupNorm(1, 768, eps=1e-6),  # LayerNorm equivalent
            nn.GELU()
        )
        
        # Adaptive global pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification head with dropout
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(768, num_classes)
        
        # Initialize new layers
        nn.init.kaiming_normal_(self.multi_scale_fusion[0].weight, mode='fan_out')
        nn.init.constant_(self.multi_scale_fusion[1].weight, 1)
        nn.init.constant_(self.multi_scale_fusion[1].bias, 0)
        nn.init.normal_(self.fc.weight, 0, 0.01)
        nn.init.constant_(self.fc.bias, 0)
        
        if weights_loaded:
            print(f"✓ Created STM-adapted ConvNeXt-Tiny with ImageNet pretrained weights ({num_classes} classes)")
        else:
            print(f"✓ Created STM-adapted ConvNeXt-Tiny with random initialization ({num_classes} classes)")
            print(f"  Note: Training from scratch may require more epochs to converge")
        
        # Print architecture summary
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"\nV2.8-ConvNeXt Enhancements:")
        print(f"  • Backbone: ConvNeXt-Tiny (~28.6M params)")
        print(f"  • Three-Scale Fusion: stage1 (192ch) + stage2 (384ch) + stage3 (768ch)")
        print(f"  • Output features: 768-dim (vs 512 for ResNet-18)")
        print(f"  • Hybrid Loss: LDAM + Focal Loss (70%/30%)")
        print(f"  • Earlier DRW: Epoch 30")
        print(f"  • Head dropout: {dropout}")
        print(f"  • Layer freezing support (first 10 epochs)")
        print(f"  • STM augmentation: SpecAugment-style masking + axis shifts")
    
    def freeze_early_layers(self):
        """Freeze stem, stage0, and stage1 for early training phase."""
        for param in self.stem.parameters():
            param.requires_grad = False
        for param in self.stage0.parameters():
            param.requires_grad = False
        for param in self.downsample1.parameters():
            param.requires_grad = False
        for param in self.stage1.parameters():
            param.requires_grad = False
        print("✓ Frozen: stem, stage0, downsample1, stage1")
    
    def unfreeze_all_layers(self):
        """Unfreeze all layers after warmup phase."""
        for param in self.parameters():
            param.requires_grad = True
        print("✓ Unfrozen all layers")
    
    def forward(self, x, return_features=False):
        # x shape: (B, 2, 20, 121)
        
        # Stem: CoordConv adds coordinates internally
        x = self.stem(x)  # → (B, 96, 20, 121) approximately
        
        # Stage 0: 96ch blocks
        x = self.stage0(x)  # (B, 96, H, W)
        
        # Downsample and Stage 1: 192ch
        x = self.downsample1(x)  # Reduces spatial + increases channels
        feat_stage1 = self.stage1(x)  # (B, 192, H1, W1)
        
        # Downsample and Stage 2: 384ch
        x = self.downsample2(feat_stage1)
        feat_stage2 = self.stage2(x)  # (B, 384, H2, W2)
        
        # Downsample and Stage 3: 768ch
        x = self.downsample3(feat_stage2)
        feat_stage3 = self.stage3(x)  # (B, 768, H3, W3)
        
        # ===== Three-Scale Feature Fusion =====
        # Upsample all features to match stage2 spatial dimensions
        target_size = feat_stage2.shape[-2:]
        
        feat_stage1_up = F.interpolate(
            feat_stage1, 
            size=target_size,
            mode='bilinear', 
            align_corners=False
        )
        feat_stage3_down = F.interpolate(
            feat_stage3, 
            size=target_size,
            mode='bilinear', 
            align_corners=False
        )
        
        # Concatenate all three scales and fuse
        feat_concat = torch.cat([feat_stage1_up, feat_stage2, feat_stage3_down], dim=1)  # (B, 1344, H2, W2)
        feat_fused = self.multi_scale_fusion(feat_concat)  # (B, 768, H2, W2)
        
        # Global pooling
        x = self.avgpool(feat_fused)  # (B, 768, 1, 1)
        x = torch.flatten(x, 1)  # (B, 768)
        
        # Classification head
        feat = x  # Save features for Center Loss
        x = self.dropout(x)
        x = self.fc(x)  # (B, num_classes)
        
        if return_features:
            return x, feat
        return x


# ============================================================================
# Attention Mechanisms (Kept for potential future use)
# ============================================================================

class CoordinateAttention(nn.Module):
    """
    Coordinate Attention from "Coordinate Attention for Efficient Mobile Network Design"
    (Hou et al., CVPR 2021)
    
    Captures position-aware attention by pooling along each spatial axis separately.
    Better than SE for spatial data like STM features.
    """
    def __init__(self, in_channels, reduction=16):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        hidden_channels = max(8, in_channels // reduction)
        
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.act = nn.ReLU(inplace=True)
        
        self.conv_h = nn.Conv2d(hidden_channels, in_channels, 1, bias=False)
        self.conv_w = nn.Conv2d(hidden_channels, in_channels, 1, bias=False)
    
    def forward(self, x):
        b, c, h, w = x.size()
        
        # Pool along each axis
        x_h = self.pool_h(x)  # (b, c, h, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # (b, c, w, 1)
        
        # Concatenate
        y = torch.cat([x_h, x_w], dim=2)  # (b, c, h+w, 1)
        
        # Shared transform
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        
        # Split and generate attention
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        
        a_h = self.conv_h(x_h).sigmoid()  # (b, c, h, 1)
        a_w = self.conv_w(x_w).sigmoid()  # (b, c, 1, w)
        
        # Apply attention
        out = x * a_h * a_w
        
        return out


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation from "Squeeze-and-Excitation Networks"
    (Hu et al., CVPR 2018)
    
    Channel-wise attention mechanism.
    """
    def __init__(self, in_channels, reduction=16):
        super(SqueezeExcitation, self).__init__()
        hidden_channels = max(1, in_channels // reduction)
        
        self.fc1 = nn.Linear(in_channels, hidden_channels, bias=False)
        self.fc2 = nn.Linear(hidden_channels, in_channels, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _ = x.size()
        
        # Squeeze: Global average pooling
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        
        # Excitation: Two FC layers
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        
        # Rescale
        y = y.view(b, c, 1, 1)
        
        return x * y


# ============================================================================
# Mixup Augmentation
# ============================================================================

def mixup_data(x, y, alpha=0.3):
    """
    Mixup augmentation from "mixup: Beyond Empirical Risk Minimization"
    (Zhang et al., ICLR 2018)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute mixup loss"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============================================================================
# STM-Specific Augmentation (SpecAugment-style)
# ============================================================================

class STMAugmentation:
    """
    SpecAugment-style augmentation adapted for STM (Spectro-Temporal Modulation) features.
    
    STM shape: (batch, 2, 20, 121) where:
    - Dim 2 (20 bins): Spectral modulation frequency (ω, cyc/oct)
    - Dim 3 (121 bins): Temporal modulation rate (Ω, Hz)
    
    Augmentation strategies:
    1. Frequency masking: Mask random contiguous bands in spectral mod dimension
    2. Time masking: Mask random contiguous bands in temporal mod dimension  
    3. Frequency shift: Cyclic shift along spectral mod axis
    4. Time shift: Cyclic shift along temporal mod axis
    """
    
    def __init__(self, 
                 freq_mask_prob=0.3,      # Probability of applying freq mask
                 time_mask_prob=0.3,      # Probability of applying time mask
                 freq_mask_width=3,       # Max width of freq mask (out of 20)
                 time_mask_width=15,      # Max width of time mask (out of 121)
                 freq_shift_prob=0.2,     # Probability of frequency shift
                 time_shift_prob=0.2,     # Probability of time shift
                 max_freq_shift=3,        # Max freq shift in bins
                 max_time_shift=10):      # Max time shift in bins
        
        self.freq_mask_prob = freq_mask_prob
        self.time_mask_prob = time_mask_prob
        self.freq_mask_width = freq_mask_width
        self.time_mask_width = time_mask_width
        self.freq_shift_prob = freq_shift_prob
        self.time_shift_prob = time_shift_prob
        self.max_freq_shift = max_freq_shift
        self.max_time_shift = max_time_shift
    
    def freq_mask(self, x):
        """
        Apply frequency masking: mask random contiguous bands in spectral mod dimension.
        
        Args:
            x: Tensor of shape (batch, channels, freq, time) or (channels, freq, time)
        
        Returns:
            Augmented tensor with same shape
        """
        x = x.clone()
        
        if x.dim() == 3:
            # Single sample: (channels, freq, time)
            _, freq_bins, _ = x.shape
            mask_width = np.random.randint(1, self.freq_mask_width + 1)
            mask_start = np.random.randint(0, freq_bins - mask_width + 1)
            x[:, mask_start:mask_start + mask_width, :] = 0
        else:
            # Batch: (batch, channels, freq, time)
            batch_size, _, freq_bins, _ = x.shape
            for i in range(batch_size):
                if np.random.rand() < self.freq_mask_prob:
                    mask_width = np.random.randint(1, self.freq_mask_width + 1)
                    mask_start = np.random.randint(0, freq_bins - mask_width + 1)
                    x[i, :, mask_start:mask_start + mask_width, :] = 0
        
        return x
    
    def time_mask(self, x):
        """
        Apply time masking: mask random contiguous bands in temporal mod dimension.
        
        Args:
            x: Tensor of shape (batch, channels, freq, time) or (channels, freq, time)
        
        Returns:
            Augmented tensor with same shape
        """
        x = x.clone()
        
        if x.dim() == 3:
            # Single sample: (channels, freq, time)
            _, _, time_bins = x.shape
            mask_width = np.random.randint(1, self.time_mask_width + 1)
            mask_start = np.random.randint(0, time_bins - mask_width + 1)
            x[:, :, mask_start:mask_start + mask_width] = 0
        else:
            # Batch: (batch, channels, freq, time)
            batch_size, _, _, time_bins = x.shape
            for i in range(batch_size):
                if np.random.rand() < self.time_mask_prob:
                    mask_width = np.random.randint(1, self.time_mask_width + 1)
                    mask_start = np.random.randint(0, time_bins - mask_width + 1)
                    x[i, :, :, mask_start:mask_start + mask_width] = 0
        
        return x
    
    def freq_shift(self, x):
        """
        Apply cyclic shift along frequency (spectral modulation) axis.
        
        This simulates variation in the spectral modulation content,
        which can help model variations in pitch range / formant structure.
        
        Args:
            x: Tensor of shape (batch, channels, freq, time)
        
        Returns:
            Augmented tensor with same shape
        """
        x = x.clone()
        batch_size = x.shape[0]
        
        for i in range(batch_size):
            if np.random.rand() < self.freq_shift_prob:
                shift = np.random.randint(-self.max_freq_shift, self.max_freq_shift + 1)
                if shift != 0:
                    x[i] = torch.roll(x[i], shifts=shift, dims=1)  # dims=1 is freq
        
        return x
    
    def time_shift(self, x):
        """
        Apply cyclic shift along time (temporal modulation) axis.
        
        This simulates variation in the temporal modulation content,
        which can help model variations in speaking rate / rhythm.
        
        Args:
            x: Tensor of shape (batch, channels, freq, time)
        
        Returns:
            Augmented tensor with same shape
        """
        x = x.clone()
        batch_size = x.shape[0]
        
        for i in range(batch_size):
            if np.random.rand() < self.time_shift_prob:
                shift = np.random.randint(-self.max_time_shift, self.max_time_shift + 1)
                if shift != 0:
                    x[i] = torch.roll(x[i], shifts=shift, dims=2)  # dims=2 is time
        
        return x
    
    def __call__(self, x):
        """
        Apply all augmentations with their respective probabilities.
        
        Args:
            x: Tensor of shape (batch, channels, freq, time)
        
        Returns:
            Augmented tensor
        """
        # Apply masking (with internal probability checks)
        x = self.freq_mask(x)
        x = self.time_mask(x)
        
        # Apply shifts
        x = self.freq_shift(x)
        x = self.time_shift(x)
        
        return x


# ============================================================================
# Enhanced ResNet with Attention
# ============================================================================

class BasicBlock(nn.Module):
    """
    ResNet Basic Block with CoordConv, dropout, and attention
    """
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, 
                 use_coordconv=True, dropout=0.05, attention_type='CA'):
        super(BasicBlock, self).__init__()
        
        if use_coordconv:
            self.conv1 = CoordConv2d(in_channels, out_channels, kernel_size=3, 
                                    stride=stride, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                  stride=stride, padding=1, bias=False)
        
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
        # Attention mechanism
        if attention_type == 'CA':
            self.attention = CoordinateAttention(out_channels, reduction=16)
        elif attention_type == 'SE':
            self.attention = SqueezeExcitation(out_channels, reduction=16)
        else:
            self.attention = None
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        if self.dropout is not None:
            out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Apply attention before skip connection
        if self.attention is not None:
            out = self.attention(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class MultiScaleFusion(nn.Module):
    """
    Multi-scale feature fusion module.
    Combines features from different depths.
    """
    def __init__(self, low_channels=256, high_channels=512, out_channels=512):
        super(MultiScaleFusion, self).__init__()
        
        # Fusion convolution
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(low_channels + high_channels, out_channels, 
                     kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, feat_low, feat_high):
        # Upsample lower-level features to match higher-level spatial size
        feat_low_up = F.interpolate(
            feat_low, 
            size=feat_high.shape[-2:], 
            mode='bilinear', 
            align_corners=False
        )
        
        # Concatenate and fuse
        feat_concat = torch.cat([feat_low_up, feat_high], dim=1)
        feat_fused = self.fusion_conv(feat_concat)
        
        return feat_fused


class CoordConvResNet18_Attention(nn.Module):
    """
    ResNet-18 with CoordConv, Attention (CA + SE), and Multi-Scale Fusion
    
    Architecture:
    - Layer1, Layer2: Coordinate Attention (position-aware, early layers)
    - Layer3, Layer4: Squeeze-and-Excitation (channel selection, late layers)
    - Multi-scale fusion: Combine layer3 + layer4 features
    """
    def __init__(self, num_classes=6, dropout=0.3, block_dropout=0.05):
        super(CoordConvResNet18_Attention, self).__init__()
        
        # Modified stem for small input (20x121)
        self.conv1 = CoordConv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # ResNet layers with different attention mechanisms
        self.in_channels = 64
        self.layer1 = self._make_layer(64, 2, stride=1, dropout=block_dropout, attention='CA')
        self.layer2 = self._make_layer(128, 2, stride=2, dropout=block_dropout, attention='CA')
        self.layer3 = self._make_layer(256, 2, stride=2, dropout=block_dropout, attention='SE')
        self.layer4 = self._make_layer(512, 2, stride=2, dropout=block_dropout, attention='SE')
        
        # Multi-scale fusion
        self.multi_scale_fusion = MultiScaleFusion(low_channels=256, high_channels=512, out_channels=512)
        
        # Calculate output size after multi-scale fusion
        # Layer3: (256, 5, 31), Layer4: (512, 3, 16)
        # After fusion: (512, 3, 16)
        self.flat_features = 512 * 3 * 16  # 24576
        
        # MLP head with dropout
        self.fc1 = nn.Linear(self.flat_features, 512)
        self.fc1_dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, 256)
        self.fc2_dropout = nn.Dropout(dropout)
        self.fc3 = nn.Linear(256, num_classes)
        
        self._initialize_weights()
    
    def _make_layer(self, out_channels, blocks, stride=1, dropout=0.05, attention='CA'):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample, 
                                use_coordconv=True, dropout=dropout, attention_type=attention))
        self.in_channels = out_channels
        
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, use_coordconv=True, 
                                    dropout=dropout, attention_type=attention))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, CoordConv2d)):
                nn.init.kaiming_normal_(m.conv.weight if isinstance(m, CoordConv2d) else m.weight, 
                                       mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, return_features=False):
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # ResNet blocks
        x = self.layer1(x)
        x = self.layer2(x)
        feat_layer3 = self.layer3(x)
        feat_layer4 = self.layer4(feat_layer3)
        
        # Multi-scale fusion
        x = self.multi_scale_fusion(feat_layer3, feat_layer4)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # MLP head
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc1_dropout(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        feat = x  # Save features for visualization if needed
        x = self.fc2_dropout(x)
        
        x = self.fc3(x)
        
        if return_features:
            return x, feat
        return x


# ============================================================================
# LDAM Loss with Label Smoothing
# ============================================================================

class LDAMLoss(nn.Module):
    """
    LDAM Loss with label smoothing
    """
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30, label_smooth=0.05):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        self.m_list = torch.FloatTensor(m_list)
        self.s = s
        self.weight = weight
        self.label_smooth = label_smooth
        
    def forward(self, x, target):
        batch_size = x.size(0)
        m_list = self.m_list.to(x.device)
        
        # Get margins for each sample
        batch_m = m_list[target]
        
        # Create one-hot encoding
        one_hot = torch.zeros_like(x)
        one_hot.scatter_(1, target.view(-1, 1), 1)
        
        # Apply label smoothing
        if self.label_smooth > 0:
            num_classes = x.size(1)
            one_hot = one_hot * (1 - self.label_smooth) + self.label_smooth / num_classes
        
        # Apply margins
        x_m = x - one_hot * batch_m.view(-1, 1)
        
        # Compute loss
        output = self.s * x_m
        
        if self.label_smooth > 0:
            # Use soft labels
            log_probs = F.log_softmax(output, dim=1)
            loss = -(one_hot * log_probs).sum(dim=1).mean()
        else:
            loss = F.cross_entropy(output, target, weight=self.weight)
        
        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for handling hard examples.
    Focuses learning on hard-to-classify samples by down-weighting easy examples.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        gamma: Focusing parameter (default: 2.0). Higher values focus more on hard examples.
        alpha: Class weights (optional). If None, uniform weights.
        reduction: 'mean', 'sum', or 'none'
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        # inputs: (B, C) logits
        # targets: (B,) class indices
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # p_t = softmax probability of correct class
        focal_weight = (1 - pt) ** self.gamma
        focal_loss = focal_weight * ce_loss
        
        # Apply class weights if provided
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                alpha_t = self.alpha.to(inputs.device)[targets]
            else:
                alpha_t = self.alpha
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class HybridLDAMFocalLoss(nn.Module):
    """
    Hybrid Loss combining LDAM and Focal Loss.
    V2.6: 70% LDAM + 30% Focal for balanced margin enforcement and hard example mining.
    
    - LDAM: Enforces class-dependent margins for long-tail distribution
    - Focal: Focuses on hard-to-classify samples (especially music:non-vocal)
    
    Args:
        cls_num_list: List of samples per class
        max_m: Maximum margin for LDAM
        focal_gamma: Focusing parameter for Focal Loss
        ldam_weight: Weight for LDAM loss (Focal weight = 1 - ldam_weight)
        label_smooth: Label smoothing for LDAM
    """
    def __init__(self, cls_num_list, max_m=0.5, focal_gamma=2.0, 
                 ldam_weight=0.7, s=30, label_smooth=0.05):
        super(HybridLDAMFocalLoss, self).__init__()
        self.ldam_loss = LDAMLoss(cls_num_list, max_m=max_m, s=s, label_smooth=label_smooth)
        self.focal_loss = FocalLoss(gamma=focal_gamma)
        self.ldam_weight = ldam_weight
        self.focal_weight = 1.0 - ldam_weight
        
        print(f"\nHybrid Loss initialized: {ldam_weight*100:.0f}% LDAM + {self.focal_weight*100:.0f}% Focal (gamma={focal_gamma})")
        
    def forward(self, inputs, targets):
        ldam = self.ldam_loss(inputs, targets)
        focal = self.focal_loss(inputs, targets)
        return self.ldam_weight * ldam + self.focal_weight * focal


# ============================================================================
# Center Loss (V2.8)
# ============================================================================

class CenterLoss(nn.Module):
    """
    Center Loss for intra-class compactness.
    
    Learns a center (prototype) for each class and penalizes the distance
    between sample features and their corresponding class centers.
    
    L_center = (1/2) * Σ ||features_i - center[y_i]||²
    
    Reference: Wen et al., "A Discriminative Feature Learning Approach 
               for Deep Face Recognition" (ECCV 2016)
    
    Args:
        num_classes: Number of classes
        feat_dim: Feature dimension (512 for our model)
        
    Note:
        - Centers are learnable parameters updated via gradient descent
        - Use a separate optimizer with higher learning rate for centers
        - Lambda (weight) typically 0.01-0.1
    """
    def __init__(self, num_classes, feat_dim):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        
        # Learnable class centers
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        
        print(f"\nCenter Loss initialized:")
        print(f"  • Classes: {num_classes}")
        print(f"  • Feature dimension: {feat_dim}")
    
    def forward(self, features, targets):
        """
        Compute center loss.
        
        Args:
            features: (B, feat_dim) - Feature vectors from model
            targets: (B,) - Ground truth labels
            
        Returns:
            loss: Scalar center loss value
        """
        batch_size = features.size(0)
        
        # Get centers for each sample's class
        centers_batch = self.centers[targets]  # (B, feat_dim)
        
        # L2 distance between features and their class centers
        # L_center = (1/2) * Σ ||f_i - c_{y_i}||²
        diff = features - centers_batch
        loss = torch.sum(diff ** 2) / (2.0 * batch_size)
        
        return loss


# ============================================================================
# Test-Time Augmentation (TTA) Predictor
# ============================================================================

class TTAPredictor:
    """
    Test-Time Augmentation Predictor for STM classification.
    
    Applies multiple augmentations at inference time and averages
    the predictions (soft voting on logits/probabilities).
    
    Augmentations:
    1. Original (no augmentation)
    2. Time flip (reverse temporal axis)
    3. Frequency shift +2 bins
    4. Frequency shift -2 bins
    5. Time shift +5 frames
    """
    
    def __init__(self, model, device, n_augmentations=5):
        """
        Args:
            model: Trained PyTorch model
            device: torch.device
            n_augmentations: Number of augmentations (5 by default)
        """
        self.model = model
        self.device = device
        self.n_augmentations = n_augmentations
        self.model.eval()
        
        # Define augmentation functions
        self.augmentations = [
            ('original', lambda x: x),
            ('time_flip', lambda x: torch.flip(x, dims=[3])),
            ('freq_shift_+2', lambda x: torch.roll(x, shifts=2, dims=2)),
            ('freq_shift_-2', lambda x: torch.roll(x, shifts=-2, dims=2)),
            ('time_shift_+5', lambda x: torch.roll(x, shifts=5, dims=3)),
        ]
        
        print(f"\nTTA Configuration:")
        print(f"  Number of augmentations: {len(self.augmentations)}")
        for name, _ in self.augmentations:
            print(f"    - {name}")
    
    def predict_batch(self, x):
        """
        Predict with TTA for a batch of samples.
        
        Args:
            x: Input tensor (B, 2, 20, 121)
            
        Returns:
            predictions: Predicted class indices (B,)
            probabilities: Softmax probabilities (B, num_classes)
        """
        x = x.to(self.device)
        
        all_logits = []
        
        with torch.no_grad():
            for name, aug_fn in self.augmentations:
                x_aug = aug_fn(x)
                logits = self.model(x_aug)
                all_logits.append(logits)
        
        # Average logits (soft voting)
        avg_logits = torch.stack(all_logits).mean(dim=0)
        
        # Get predictions and probabilities
        probabilities = F.softmax(avg_logits, dim=1)
        predictions = avg_logits.argmax(dim=1)
        
        return predictions.cpu().numpy(), probabilities.cpu().numpy()
    
    def evaluate(self, data_loader):
        """
        Evaluate model with TTA on entire dataset.
        
        Args:
            data_loader: PyTorch DataLoader
            
        Returns:
            all_preds: All predictions
            all_targets: All ground truth labels
            all_probs: All probability distributions
        """
        all_preds = []
        all_targets = []
        all_probs = []
        
        total_batches = len(data_loader)
        
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            if batch_idx % 50 == 0:
                print(f"  Processing batch {batch_idx}/{total_batches}...")
            
            preds, probs = self.predict_batch(inputs)
            
            all_preds.extend(preds)
            all_targets.extend(targets.numpy())
            all_probs.extend(probs)
        
        return np.array(all_preds), np.array(all_targets), np.array(all_probs)


# ============================================================================
# Trainer (V2.8 with Center Loss + Multi-Scale + STM Augmentation + Layer Freezing)
# ============================================================================

class Trainer:
    """
    Trainer with V2.8 enhancements:
    - Center Loss for intra-class compactness
    - Discriminative Learning Rates
    - Multi-Scale Fusion parameter handling
    - STM-specific augmentation (SpecAugment-style)
    - Layer freezing for first 10 epochs
    """
    def __init__(self, model, train_loader, val_loader, test_loader, 
                 device, class_counts, lr=1e-4, weight_decay=5e-4,
                 freeze_epochs=10, center_loss_weight=0.1):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.class_counts = class_counts
        self.freeze_epochs = freeze_epochs
        self.layers_frozen = False
        
        # STM Augmentation
        self.stm_augment = STMAugmentation(
            freq_mask_prob=0.3,
            time_mask_prob=0.3,
            freq_mask_width=3,
            time_mask_width=15,
            freq_shift_prob=0.2,
            time_shift_prob=0.2,
            max_freq_shift=3,
            max_time_shift=10
        )
        
        # Discriminative Learning Rates for ConvNeXt
        # Collect parameters carefully to avoid overlap
        
        # ConvNeXt structure:
        # - stem: custom CoordConv stem
        # - stage0: first stage (96ch)
        # - downsample1 + stage1: second stage (192ch)
        # - downsample2 + stage2: third stage (384ch)
        # - downsample3 + stage3: fourth stage (768ch)
        # - multi_scale_fusion: new fusion layer
        # - fc: classification head
        
        stem_params = list(model.stem.parameters())
        stage0_params = list(model.stage0.parameters())
        stage1_params = list(model.downsample1.parameters()) + list(model.stage1.parameters())
        stage2_params = list(model.downsample2.parameters()) + list(model.stage2.parameters())
        stage3_params = list(model.downsample3.parameters()) + list(model.stage3.parameters())
        fusion_params = list(model.multi_scale_fusion.parameters())
        head_params = list(model.fc.parameters())
        
        # Create parameter groups (no overlaps)
        param_groups = [
            # Stem + Stage0: 0.1x (heavily pretrained, minimal adaptation)
            {'params': stem_params + stage0_params,
             'lr': lr * 0.1, 'name': 'stem_stage0'},
            
            # Stage1: 0.3x (pretrained, conservative adaptation)
            {'params': stage1_params,
             'lr': lr * 0.3, 'name': 'stage1'},
            
            # Stage2: 0.5x (pretrained, moderate adaptation)
            {'params': stage2_params,
             'lr': lr * 0.5, 'name': 'stage2'},
            
            # Stage3: 1.0x (pretrained backbone, full adaptation)
            {'params': stage3_params,
             'lr': lr * 1.0, 'name': 'stage3'},
            
            # Multi-Scale Fusion: 1.0x (newly initialized)
            {'params': fusion_params,
             'lr': lr * 1.0, 'name': 'multi_scale_fusion'},
            
            # Head: 1.0x (new classifier)
            {'params': head_params,
             'lr': lr * 1.0, 'name': 'head'},
        ]
        
        # Optimizer with parameter groups
        self.optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
        
        print(f"\nDiscriminative Learning Rates:")
        for group in self.optimizer.param_groups:
            print(f"  {group['name']:20s}: LR = {group['lr']:.6f} ({len(group['params'])} params)")
        
        # Scheduler: ReduceLROnPlateau
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=7, min_lr=1e-6
        )
        
        # Loss function: Hybrid LDAM + Focal (V2.6)
        # 70% LDAM for margin-based class separation
        # 30% Focal for hard example mining (helps music:non-vocal)
        self.criterion = HybridLDAMFocalLoss(
            cls_num_list=class_counts, 
            max_m=0.5, 
            focal_gamma=2.0,
            ldam_weight=0.7,
            s=30, 
            label_smooth=0.05
        )
        
        # Center Loss (V2.8): Intra-class compactness
        # Feature dimension is 768 (output before dropout/fc in PretrainedSTMConvNeXt)
        self.center_loss = CenterLoss(num_classes=len(class_counts), feat_dim=768).to(device)
        self.center_loss_weight = center_loss_weight
        
        # Separate optimizer for center loss (higher learning rate)
        self.center_optimizer = torch.optim.SGD(self.center_loss.parameters(), lr=0.5)
        
        print(f"\nCenter Loss weight: {center_loss_weight}")
        
        # DRW weights
        self.drw_weights = torch.FloatTensor(1.0 / class_counts).to(device)
        self.drw_weights = self.drw_weights / self.drw_weights.sum() * len(class_counts)
        
        # Tracking
        self.best_val_f1 = 0.0
        self.epochs_no_improve = 0
        self.early_stop_patience = 20
        
        # Freeze early layers if requested
        if freeze_epochs > 0:
            self.model.freeze_early_layers()
            self.layers_frozen = True
            print(f"\n→ Layer1-2 frozen for first {freeze_epochs} epochs")
    
    def _unfreeze_if_needed(self, epoch):
        """Unfreeze early layers after warmup period."""
        if self.layers_frozen and epoch > self.freeze_epochs:
            self.model.unfreeze_all_layers()
            self.layers_frozen = False
            print(f"\n*** Epoch {epoch}: Unfreezing all layers ***\n")
        
    def train_epoch(self, epoch, total_epochs, use_drw=False, use_mixup=True, use_stm_aug=True):
        self.model.train()
        total_loss = 0.0
        
        # Check if we should unfreeze
        self._unfreeze_if_needed(epoch)
        
        # Update DRW
        if use_drw:
            self.criterion.weight = self.drw_weights
        else:
            self.criterion.weight = None
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Apply STM augmentation (SpecAugment-style)
            if use_stm_aug:
                data = self.stm_augment(data)
            
            # Apply Mixup with 30% probability (stronger alpha=0.4 in V2.4)
            if use_mixup and np.random.rand() < 0.3:
                mixed_data, target_a, target_b, lam = mixup_data(data, target, alpha=0.4)
                
                self.optimizer.zero_grad()
                self.center_optimizer.zero_grad()
                
                # Get both logits and features for center loss
                output, features = self.model(mixed_data, return_features=True)
                cls_loss = mixup_criterion(self.criterion, output, target_a, target_b, lam)
                
                # Center loss with mixed targets (use majority target)
                # For simplicity, use target_a as the majority class
                c_loss = self.center_loss(features, target_a)
                
                loss = cls_loss + self.center_loss_weight * c_loss
            else:
                self.optimizer.zero_grad()
                self.center_optimizer.zero_grad()
                
                # Get both logits and features for center loss
                output, features = self.model(data, return_features=True)
                cls_loss = self.criterion(output, target)
                c_loss = self.center_loss(features, target)
                
                loss = cls_loss + self.center_loss_weight * c_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.center_optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 500 == 0:
                print(f"  Batch {batch_idx}/{len(self.train_loader)}, "
                      f"Loss: {loss.item():.4f} (cls: {cls_loss.item():.4f}, center: {c_loss.item():.4f}), "
                      f"DRW: {use_drw}, Mixup: {use_mixup}")
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                loss = F.cross_entropy(output, target)
                total_loss += loss.item()
                
                preds = output.argmax(dim=1)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(target.cpu().numpy())
        
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        
        avg_loss = total_loss / len(data_loader)
        macro_f1 = f1_score(all_targets, all_preds, average='macro')
        
        return avg_loss, macro_f1, all_preds, all_targets
    
    def train(self, num_epochs, checkpoint_dir):
        print("\n" + "="*60)
        print("Starting training...")
        print("="*60)
        
        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{num_epochs}")
            print("="*60)
            
            # Determine if DRW should be active (V2.6: earlier activation at epoch 30)
            # Earlier DRW gives minority classes (music:non-vocal) more weighted training time
            use_drw = epoch > 30
            if epoch == 31:
                print("\n*** Activating Deferred Reweighting (DRW) at epoch 31 ***\n")
            
            # Train
            train_loss = self.train_epoch(epoch, num_epochs, use_drw=use_drw)
            print(f"Train Loss: {train_loss:.4f}")
            
            # Validate
            val_loss, val_f1, _, _ = self.evaluate(self.val_loader)
            print(f"Val Loss: {val_loss:.4f}, Val Macro F1: {val_f1:.4f}")
            
            # Update scheduler
            self.scheduler.step(val_f1)
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Current learning rate: {current_lr:.6f}")
            
            # Save best model
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.epochs_no_improve = 0
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_f1': val_f1,
                }, os.path.join(checkpoint_dir, 'best_model.pt'))
                print(f"✓ Saved best model with Val F1: {val_f1:.4f}")
            else:
                self.epochs_no_improve += 1
            
            # Periodic checkpoint
            if epoch % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_f1': val_f1,
                }, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt'))
            
            # Early stopping
            if self.epochs_no_improve >= self.early_stop_patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break
        
        print(f"\n{'='*60}")
        print(f"Training completed! Best Val F1: {self.best_val_f1:.4f}")
        print("="*60)


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python STM_CoordConvLDAM_preIN8_ConvNeXt.py <mode>")
        print("  mode 0: standard (full dataset)")
        print("  mode 1: downsample non-tonal speech to 100k")
        sys.exit(1)
    
    mode = int(sys.argv[1])
    
    # Set parameters based on mode
    if mode == 0:
        ds_nontonal_speech = False
        directory = "model/STM/CoordConvLDAM_preIN8_ConvNeXt_corpora_categories/standard"
    elif mode == 1:
        ds_nontonal_speech = True
        directory = "model/STM/CoordConvLDAM_preIN8_ConvNeXt_corpora_categories/downsample"
    else:
        print("Invalid mode. Use 0 or 1.")
        sys.exit(1)
    
    # Create directory
    time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    checkpoint_dir = os.path.join(directory, "ckpt", time_stamp)
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoint directory: {checkpoint_dir}")
    
    # Prepare data
    print("\n" + "="*60)
    print("Loading and preparing data...")
    print("="*60)
    
    data_prep = prepData_STM_CoordConv(ds_nontonal_speech=ds_nontonal_speech)
    train_dataset, val_dataset, test_dataset, class_counts = data_prep.prepare_datasets()
    
    # Create data loaders
    batch_size = 128  # Reduced from 256 for larger ConvNeXt model
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    
    print(f"\nDataLoaders created with batch_size={batch_size}")
    
    # Create model
    print("\n" + "="*60)
    print("Creating ImageNet-Pretrained ConvNeXt-Tiny for STM (V2.8-ConvNeXt)...")
    print("="*60)
    
    num_classes = 6
    model = PretrainedSTMConvNeXt(num_classes=num_classes, dropout=0.4)
    
    # Model info is printed in __init__
    print(f"\nV2.8-ConvNeXt Training Configuration:")
    print(f"  \u2022 Three-Scale Fusion: stage1 (192ch) + stage2 (384ch) + stage3 (768ch)")
    print(f"  \u2022 Discriminative LR: Stem/S0 (0.1x), S1 (0.3x), S2 (0.5x), S3/Head (1.0x)")
    print(f"  \u2022 Layer Freezing: stem + stage0-1 frozen for first 10 epochs")
    print(f"  \u2022 STM Augmentation: SpecAugment-style masking + axis shifts")
    print(f"  \u2022 Weight decay: 5e-4")
    print(f"  \u2022 Mixup alpha: 0.4")
    print(f"  \u2022 LDAM + DRW: Enabled (epoch 30+)")
    print(f"  \u2022 Early stopping: 20 epochs patience")
    
    # Create trainer with V2.8-ConvNeXt hyperparameters
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        class_counts=class_counts,
        lr=1e-4,
        weight_decay=5e-4,
        freeze_epochs=10  # Freeze stem + stage0-1 for first 10 epochs
    )
    
    # Train model
    num_epochs = 100
    trainer.train(num_epochs=num_epochs, checkpoint_dir=checkpoint_dir)
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("Evaluating on test set...")
    print("="*60)
    
    # Load best model
    checkpoint = torch.load(os.path.join(checkpoint_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_f1, test_preds, test_targets = trainer.evaluate(test_loader)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Macro F1: {test_f1:.4f}")
    
    # Detailed classification report
    print("\nClassification Report (No TTA):")
    target_names = ['speech:non-tonal', 'speech:tonal', 'music:vocal', 
                   'music:non-vocal', 'env:urban', 'env:wildlife']
    print(classification_report(test_targets, test_preds, target_names=target_names))
    
    # =========================================================================
    # Test-Time Augmentation (TTA) Evaluation
    # =========================================================================
    print("\n" + "="*60)
    print("TTA Evaluation (5 augmentations)...")
    print("="*60)
    
    tta_predictor = TTAPredictor(model, device, n_augmentations=5)
    tta_preds, tta_targets, tta_probs = tta_predictor.evaluate(test_loader)
    
    tta_f1 = f1_score(tta_targets, tta_preds, average='macro')
    print(f"\nTTA Test Macro F1: {tta_f1:.4f}")
    print(f"Improvement over No-TTA: {(tta_f1 - test_f1) * 100:.2f}%")
    
    print("\nClassification Report (With TTA):")
    print(classification_report(tta_targets, tta_preds, target_names=target_names))
    
    # =========================================================================
    # Confusion Matrix Analysis
    # =========================================================================
    print("\n" + "="*60)
    print("Generating Confusion Matrix...")
    print("="*60)
    
    # Use TTA predictions for final confusion matrix
    cm = confusion_matrix(tta_targets, tta_preds)
    
    # Print confusion matrix as text
    print("\nConfusion Matrix (rows=true, cols=predicted):")
    print(f"{'':>16s}", end='')
    for name in target_names:
        print(f"{name[:8]:>10s}", end='')
    print()
    for i, row in enumerate(cm):
        print(f"{target_names[i][:16]:>16s}", end='')
        for val in row:
            print(f"{val:>10d}", end='')
        print()
    
    # Calculate and print per-class metrics
    print("\nPer-Class Confusion Analysis:")
    for i, name in enumerate(target_names):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        support = cm[i, :].sum()
        
        # Top confusions
        row_confusions = [(j, cm[i, j]) for j in range(len(target_names)) if j != i]
        row_confusions.sort(key=lambda x: x[1], reverse=True)
        
        col_confusions = [(j, cm[j, i]) for j in range(len(target_names)) if j != i]
        col_confusions.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n  {name}:")
        print(f"    True Positives: {tp}, False Negatives: {fn}, False Positives: {fp}")
        if row_confusions[0][1] > 0:
            print(f"    Most confused AS: {target_names[row_confusions[0][0]]} ({row_confusions[0][1]} samples)")
        if col_confusions[0][1] > 0:
            print(f"    Most misclassified FROM: {target_names[col_confusions[0][0]]} ({col_confusions[0][1]} samples)")
    
    # Create and save confusion matrix plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Normalized confusion matrix for visualization
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create heatmap
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names,
                ax=ax, vmin=0, vmax=1)
    
    # Add raw counts as additional text
    for i in range(len(target_names)):
        for j in range(len(target_names)):
            ax.text(j + 0.5, i + 0.75, f'n={cm[i, j]}', 
                   ha='center', va='center', fontsize=8, color='gray')
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(f'STM Classification Confusion Matrix (V2.8-ConvNeXt)\nNormalized by Row (Recall) | TTA F1: {tta_f1:.4f}', fontsize=14)
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save figure
    cm_path = os.path.join(checkpoint_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    print(f"\nConfusion matrix saved to: {cm_path}")
    
    # Save predictions and confusion matrix
    np.save(os.path.join(checkpoint_dir, 'test_predictions.npy'), test_preds)
    np.save(os.path.join(checkpoint_dir, 'test_targets.npy'), test_targets)
    np.save(os.path.join(checkpoint_dir, 'tta_predictions.npy'), tta_preds)
    np.save(os.path.join(checkpoint_dir, 'tta_probs.npy'), tta_probs)
    np.save(os.path.join(checkpoint_dir, 'confusion_matrix.npy'), cm)
    
    plt.close()
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY (V2.8-ConvNeXt with ConvNeXt-Tiny + Center Loss)")
    print("="*60)
    print(f"  Test F1 (No TTA):   {test_f1:.4f}")
    print(f"  Test F1 (With TTA): {tta_f1:.4f}")
    print(f"  TTA Improvement:    +{(tta_f1 - test_f1) * 100:.2f}%")
    print("="*60)
    print("Done!")
    print("="*60)
