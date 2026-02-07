#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STM Classification with ImageNet-Pretrained ResNet and LDAM Loss
Phase 2.4: Multi-Scale Fusion + Advanced Augmentation + Layer Freezing

Key Innovations (V2.4 over V2.1):
1. Multi-Scale Feature Fusion: Combines layer3 (256ch, 5×31) + layer4 (512ch, 3×16)
   - Captures both mid-level patterns and high-level semantics
   - Uses bilinear upsampling + concat + 1×1 conv fusion
2. STM-Specific Augmentation:
   - SpecAugment-style masking: Random frequency/time band masking
   - Axis shifts: Random cyclic shifts along freq/time dimensions
   - Increased augmentation diversity for better generalization
3. Layer Freezing Strategy:
   - Freeze layer1-2 for first 10 epochs (preserve ImageNet knowledge)
   - Gradual unfreezing allows attention modules to stabilize first
   - Prevents early corruption of pretrained features

Preserved from V2.1:
1. Attention Mechanisms: CA (layer1-2) + SE (layer3-4) for adaptive features
2. Discriminative Learning Rates: Pretrained layers learn slower (0.1x-0.5x)
3. Block Dropout: 0.05 in residual blocks to reduce overfitting
4. Difference Map Preprocessing: 2-channel input (Symmetric + Asymmetric)
5. ImageNet-pretrained ResNet-18 backbone (texture bias for STM ripples)
6. LDAM-DRW training: Proven long-tail handling strategy

Target: 0.88-0.90 Macro F1 (surpass V2.1's 0.8618, approach SOTA 0.89+)
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
from sklearn.metrics import f1_score, classification_report
from torchvision import models
from torchvision.models import ResNet18_Weights
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
# BasicBlock with Attention and Dropout
# ============================================================================

class BasicBlockWithAttention(nn.Module):
    """
    ResNet BasicBlock with optional attention and dropout.
    Used to add attention to pretrained ResNet layers.
    """
    expansion = 1
    
    def __init__(self, block, attention_type='CA', dropout=0.05):
        """
        Wraps an existing BasicBlock and adds attention + dropout.
        
        Args:
            block: Existing nn.Module (BasicBlock from pretrained ResNet)
            attention_type: 'CA' (Coordinate Attention), 'SE' (Squeeze-Excitation), or None
            dropout: Dropout probability for regularization
        """
        super(BasicBlockWithAttention, self).__init__()
        self.block = block
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None
        
        # Get number of output channels from the block
        # Assuming block has conv2 as the last conv layer
        out_channels = block.conv2.out_channels
        
        # Add attention
        if attention_type == 'CA':
            self.attention = CoordinateAttention(out_channels)
        elif attention_type == 'SE':
            self.attention = SqueezeExcitation(out_channels)
        else:
            self.attention = None
    
    def forward(self, x):
        # Original block forward
        out = self.block(x)
        
        # Apply attention before final activation
        if self.attention is not None:
            out = self.attention(out)
        
        # Apply dropout
        if self.dropout is not None:
            out = self.dropout(out)
        
        return out


# ============================================================================
# Pretrained ResNet-18 with STM-Specific Adaptations + Attention + Multi-Scale
# ============================================================================

class PretrainedSTMResNet18(nn.Module):
    """
    ImageNet-pretrained ResNet-18 adapted for STM classification.
    
    V2.4 Enhancements:
    1. Multi-Scale Feature Fusion: Combines layer3 (256ch) + layer4 (512ch)
       - Captures both mid-level patterns and high-level semantics
       - Better for classes with mixed spectro-temporal characteristics
    2. Attention: CA for layer1-2 (spatial), SE for layer3-4 (channel)
    3. Block Dropout: 0.05 in all blocks to reduce overfitting
    4. Head Dropout: Increased to 0.4 for stronger regularization
    5. Layer Freezing Support: Methods to freeze/unfreeze early layers
    
    Key Modifications (from V2.0):
    1. Stem: 4-channel CoordConv (2 STM channels + 2 coordinate channels)
    2. Weight Cloning: Initialize with ImageNet weights (texture bias)
    3. Resolution Preservation: stride=1 in conv1, remove maxpool
    4. LDAM-compatible head: Direct classification output
    
    Architecture Flow:
    Input (B, 2, 20, 121) → CoordConv+Coords (B, 4, 20, 121) → 
    Layer1 (CA) → Layer2 (CA) → Layer3 (SE) → Layer4 (SE) → 
    Multi-Scale Fusion → Pool → FC
    """
    def __init__(self, num_classes=6, dropout=0.4, block_dropout=0.05, use_pretrained=True):
        super(PretrainedSTMResNet18, self).__init__()
        
        # Load pretrained ResNet-18 with error handling
        pretrained_model = None
        weights_loaded = False
        
        if use_pretrained:
            print("Loading ImageNet-pretrained ResNet-18...")
            try:
                pretrained_model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
                weights_loaded = True
                print("✓ Successfully loaded ImageNet pretrained weights")
            except Exception as e:
                print(f"⚠ Warning: Failed to download pretrained weights: {e}")
                print("⚠ Attempting to load from local cache...")
                try:
                    # Try loading from torch hub cache
                    pretrained_model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
                    weights_loaded = True
                    print("✓ Successfully loaded from cache")
                except:
                    print("⚠ Cache load failed. Training from scratch (random initialization)")
                    pretrained_model = models.resnet18(weights=None)
                    weights_loaded = False
        else:
            print("Creating ResNet-18 with random initialization...")
            pretrained_model = models.resnet18(weights=None)
            weights_loaded = False
        
        # ===== Stem Modification: 4-channel CoordConv =====
        # Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # New: CoordConv(2, 64, ...) which internally handles 4 channels (2+2 coords)
        
        old_conv1 = pretrained_model.conv1
        self.conv1 = CoordConv2d(
            in_channels=2,  # 2 STM channels (will become 4 with coords)
            out_channels=64,
            kernel_size=7,
            stride=1,  # CRITICAL: Preserve resolution for 20-bin input
            padding=3,
            bias=False
        )
        
        # ===== Weight Cloning: Preserve ImageNet Knowledge =====
        with torch.no_grad():
            if weights_loaded:
                # Pretrained weights available - clone them
                # old_conv1.weight shape: (64, 3, 7, 7)
                # self.conv1.conv.weight shape: (64, 4, 7, 7)  [2 STM + 2 coords]
                
                # Strategy: Copy Red channel to all 4 new channels, scale by sqrt(3/4)
                # This maintains the expected magnitude of activations
                pretrained_weights = old_conv1.weight.data  # (64, 3, 7, 7)
                red_channel = pretrained_weights[:, 0:1, :, :]  # (64, 1, 7, 7)
                
                # Initialize all 4 channels with red channel weights
                new_weights = red_channel.repeat(1, 4, 1, 1)  # (64, 4, 7, 7)
                
                # Scale to maintain variance: sqrt(3/4) ≈ 0.866
                new_weights = new_weights * (3.0 / 4.0) ** 0.5
                
                self.conv1.conv.weight.data = new_weights
                print("✓ Cloned ImageNet weights to 4-channel CoordConv stem")
            else:
                # Random initialization with proper scaling
                nn.init.kaiming_normal_(self.conv1.conv.weight, mode='fan_out', nonlinearity='relu')
                print("✓ Initialized 4-channel CoordConv stem with Kaiming Normal")
        
        # Copy other stem components
        self.bn1 = pretrained_model.bn1
        self.relu = pretrained_model.relu
        
        # CRITICAL: Remove aggressive maxpool (would reduce 20 → 10 → 5)
        # Replace with identity to preserve spatial resolution
        self.maxpool = nn.Identity()
        
        # Copy ResNet blocks and add attention
        # Layer1: 2 BasicBlocks, add CA (position-aware, early stage)
        self.layer1 = nn.Sequential(
            *[BasicBlockWithAttention(block, attention_type='CA', dropout=block_dropout) 
              for block in pretrained_model.layer1]
        )
        
        # Layer2: 2 BasicBlocks, add CA (position still important)
        self.layer2 = nn.Sequential(
            *[BasicBlockWithAttention(block, attention_type='CA', dropout=block_dropout) 
              for block in pretrained_model.layer2]
        )
        
        # Layer3: 2 BasicBlocks, add SE (channel selection, semantic features)
        self.layer3 = nn.Sequential(
            *[BasicBlockWithAttention(block, attention_type='SE', dropout=block_dropout) 
              for block in pretrained_model.layer3]
        )
        
        # Layer4: 2 BasicBlocks, add SE (channel selection, abstract features)
        self.layer4 = nn.Sequential(
            *[BasicBlockWithAttention(block, attention_type='SE', dropout=block_dropout) 
              for block in pretrained_model.layer4]
        )
        
        # ===== Multi-Scale Feature Fusion =====
        # Combines layer3 (256ch, 5×31) + layer4 (512ch, 3×16) features
        # This captures both mid-level patterns and high-level semantics
        self.multi_scale_fusion = nn.Sequential(
            nn.Conv2d(256 + 512, 512, kernel_size=1, bias=False),  # Channel reduction
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Adaptive global pooling (preserves coordinate information better than GAP)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification head with dropout
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512, num_classes)  # Multi-scale output is 512 features
        
        # Initialize new layers
        nn.init.kaiming_normal_(self.multi_scale_fusion[0].weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.multi_scale_fusion[1].weight, 1)
        nn.init.constant_(self.multi_scale_fusion[1].bias, 0)
        nn.init.normal_(self.fc.weight, 0, 0.01)
        nn.init.constant_(self.fc.bias, 0)
        
        if weights_loaded:
            print(f"✓ Created STM-adapted ResNet-18 V2.4 with ImageNet pretrained weights ({num_classes} classes)")
        else:
            print(f"✓ Created STM-adapted ResNet-18 V2.4 with random initialization ({num_classes} classes)")
            print(f"  Note: Training from scratch may require more epochs to converge")
        
        # Print architecture summary
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"\nV2.4 Enhancements:")
        print(f"  • Multi-Scale Fusion: layer3 (256ch) + layer4 (512ch)")
        print(f"  • Attention: CA (layer1-2), SE (layer3-4)")
        print(f"  • Block dropout: {block_dropout}")
        print(f"  • Head dropout: {dropout}")
        print(f"  • Layer freezing support (first 10 epochs)")
        print(f"  • STM augmentation: SpecAugment-style masking + axis shifts")
    
    def freeze_early_layers(self):
        """Freeze stem, layer1, and layer2 for early training phase."""
        for param in self.conv1.parameters():
            param.requires_grad = False
        for param in self.bn1.parameters():
            param.requires_grad = False
        for param in self.layer1.parameters():
            param.requires_grad = False
        for param in self.layer2.parameters():
            param.requires_grad = False
        print("✓ Frozen: conv1, bn1, layer1, layer2")
    
    def unfreeze_all_layers(self):
        """Unfreeze all layers after warmup phase."""
        for param in self.parameters():
            param.requires_grad = True
        print("✓ Unfrozen all layers")
    
    def forward(self, x, return_features=False):
        # x shape: (B, 2, 20, 121)
        
        # Stem: CoordConv adds coordinates internally
        x = self.conv1(x)  # → (B, 64, 20, 121)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # Identity, no change
        
        # ResNet blocks
        x = self.layer1(x)  # (B, 64, 20, 121)
        x = self.layer2(x)  # (B, 128, 10, 61) [stride=2 in layer2]
        feat_layer3 = self.layer3(x)  # (B, 256, 5, 31)  [stride=2 in layer3]
        feat_layer4 = self.layer4(feat_layer3)  # (B, 512, 3, 16)  [stride=2 in layer4]
        
        # ===== Multi-Scale Feature Fusion =====
        # Upsample layer4 to match layer3 spatial dimensions
        feat_layer4_up = F.interpolate(
            feat_layer4, 
            size=feat_layer3.shape[-2:],  # (5, 31)
            mode='bilinear', 
            align_corners=False
        )
        # Concatenate and fuse
        feat_concat = torch.cat([feat_layer3, feat_layer4_up], dim=1)  # (B, 768, 5, 31)
        feat_fused = self.multi_scale_fusion(feat_concat)  # (B, 512, 5, 31)
        
        # Global pooling
        x = self.avgpool(feat_fused)  # (B, 512, 1, 1)
        x = torch.flatten(x, 1)  # (B, 512)
        
        # Classification head
        feat = x  # Save features for analysis
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


# ============================================================================
# Trainer (V2.4 with Multi-Scale + STM Augmentation + Layer Freezing)
# ============================================================================

class Trainer:
    """
    Trainer with V2.4 enhancements:
    - Discriminative Learning Rates
    - Multi-Scale Fusion parameter handling
    - STM-specific augmentation (SpecAugment-style)
    - Layer freezing for first 10 epochs
    """
    def __init__(self, model, train_loader, val_loader, test_loader, 
                 device, class_counts, lr=1e-4, weight_decay=5e-4,
                 freeze_epochs=10):
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
        
        # Discriminative Learning Rates
        # Collect parameters carefully to avoid overlap
        
        # Get all parameter IDs to track what's been assigned
        stem_params = []
        layer1_params = []
        layer2_params = []
        layer3_params = []
        layer4_params = []
        attention_params = []
        fusion_params = []
        head_params = []
        bn_params = []
        
        # Stem (conv1 only, bn1 handled separately)
        stem_params = list(model.conv1.conv.parameters())
        
        # BatchNorm1 (stem)
        bn_params.extend(list(model.bn1.parameters()))
        
        # Layers: separate conv parameters from attention and batchnorm
        for block in model.layer1:
            # Get conv parameters (excluding bn and attention)
            for name, param in block.block.named_parameters():
                if 'bn' not in name:  # Exclude batchnorm
                    layer1_params.append(param)
                else:
                    bn_params.append(param)
            # Get attention parameters
            if block.attention is not None:
                attention_params.extend(list(block.attention.parameters()))
        
        for block in model.layer2:
            for name, param in block.block.named_parameters():
                if 'bn' not in name:
                    layer2_params.append(param)
                else:
                    bn_params.append(param)
            if block.attention is not None:
                attention_params.extend(list(block.attention.parameters()))
        
        for block in model.layer3:
            for name, param in block.block.named_parameters():
                if 'bn' not in name:
                    layer3_params.append(param)
                else:
                    bn_params.append(param)
            if block.attention is not None:
                attention_params.extend(list(block.attention.parameters()))
        
        for block in model.layer4:
            for name, param in block.block.named_parameters():
                if 'bn' not in name:
                    layer4_params.append(param)
                else:
                    bn_params.append(param)
            if block.attention is not None:
                attention_params.extend(list(block.attention.parameters()))
        
        # Multi-scale fusion parameters
        fusion_params = list(model.multi_scale_fusion.parameters())
        
        # Head (fc and dropout)
        head_params = list(model.fc.parameters())
        # Note: Dropout has no parameters
        
        # Create parameter groups (no overlaps now)
        param_groups = [
            # Stem + Layer1: 0.1x (heavily pretrained, minimal adaptation)
            {'params': stem_params + layer1_params,
             'lr': lr * 0.1, 'name': 'stem_layer1'},
            
            # Layer2-3: 0.5x (pretrained, moderate adaptation)
            {'params': layer2_params + layer3_params,
             'lr': lr * 0.5, 'name': 'layer2_layer3'},
            
            # Layer4: 1.0x (pretrained backbone, full adaptation)
            {'params': layer4_params,
             'lr': lr * 1.0, 'name': 'layer4'},
            
            # Multi-Scale Fusion: 1.0x (newly initialized)
            {'params': fusion_params,
             'lr': lr * 1.0, 'name': 'multi_scale_fusion'},
            
            # Head: 1.0x (new classifier)
            {'params': head_params,
             'lr': lr * 1.0, 'name': 'head'},
            
            # Attention modules: 1.0x (newly initialized, need full LR)
            {'params': attention_params,
             'lr': lr * 1.0, 'name': 'attention'},
            
            # BatchNorm layers: 1.0x (always adapt to new data distribution)
            {'params': bn_params,
             'lr': lr * 1.0, 'name': 'batchnorm'}
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
        
        # Loss function
        self.criterion = LDAMLoss(
            cls_num_list=class_counts, max_m=0.5, s=30, label_smooth=0.05
        )
        
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
                output = self.model(mixed_data)
                loss = mixup_criterion(self.criterion, output, target_a, target_b, lam)
            else:
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 500 == 0:
                print(f"  Batch {batch_idx}/{len(self.train_loader)}, "
                      f"Loss: {loss.item():.4f}, DRW: {use_drw}, Mixup: {use_mixup}")
        
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
            
            # Determine if DRW should be active (after 50% of training)
            use_drw = epoch > (num_epochs // 2)
            if epoch == (num_epochs // 2) + 1:
                print("\n*** Activating Deferred Reweighting (DRW) ***\n")
            
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
        print("Usage: python STM_CoordConvLDAM_preIN4.py <mode>")
        print("  mode 0: standard (full dataset)")
        print("  mode 1: downsample non-tonal speech to 100k")
        sys.exit(1)
    
    mode = int(sys.argv[1])
    
    # Set parameters based on mode
    if mode == 0:
        ds_nontonal_speech = False
        directory = "model/STM/CoordConvLDAM_preIN4_corpora_categories/standard"
    elif mode == 1:
        ds_nontonal_speech = True
        directory = "model/STM/CoordConvLDAM_preIN4_corpora_categories/downsample"
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
    batch_size = 256
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    
    print(f"\nDataLoaders created with batch_size={batch_size}")
    
    # Create model
    print("\n" + "="*60)
    print("Creating ImageNet-Pretrained ResNet-18 for STM (V2.4)...")
    print("="*60)
    
    num_classes = 6
    model = PretrainedSTMResNet18(num_classes=num_classes, dropout=0.4, block_dropout=0.05)
    
    # Model info is printed in __init__
    print(f"\nV2.4 Training Configuration:")
    print(f"  \u2022 Multi-Scale Fusion: layer3 (256ch) + layer4 (512ch)")
    print(f"  \u2022 Discriminative LR: Stem/L1 (0.1x), L2-3 (0.5x), L4/Head (1.0x)")
    print(f"  \u2022 Layer Freezing: layer1-2 frozen for first 10 epochs")
    print(f"  \u2022 STM Augmentation: SpecAugment-style masking + axis shifts")
    print(f"  \u2022 Weight decay: 5e-4")
    print(f"  \u2022 Mixup alpha: 0.4")
    print(f"  \u2022 LDAM + DRW: Enabled (epoch 50+)")
    print(f"  \u2022 Early stopping: 20 epochs patience")
    
    # Create trainer with V2.4 hyperparameters
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        class_counts=class_counts,
        lr=1e-4,
        weight_decay=5e-4,
        freeze_epochs=10  # Freeze layer1-2 for first 10 epochs
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
    print("\nClassification Report:")
    target_names = ['speech:non-tonal', 'speech:tonal', 'music:vocal', 
                   'music:non-vocal', 'env:urban', 'env:wildlife']
    print(classification_report(test_targets, test_preds, target_names=target_names))
    
    # Save test predictions
    np.save(os.path.join(checkpoint_dir, 'test_predictions.npy'), test_preds)
    np.save(os.path.join(checkpoint_dir, 'test_targets.npy'), test_targets)
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)
