#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STM Classification with CoordConv-ResNet and LDAM Loss
Phase 1.6: Advanced Regularization - DropBlock + CutMix + Stochastic Depth

Improvements over STM_CoordConvLDAM4.py:
1. DropBlock2d: Spatial dropout that drops contiguous blocks (vs individual pixels)
2. Stochastic Depth: Random skip connections in residual blocks
3. CutMix augmentation: Replace mixup with stronger cutout-based mixing
4. Cosine annealing LR scheduler: Replace ReduceLROnPlateau for smoother convergence
5. Longer training with stronger regularization: 150 epochs, higher weight decay
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

warnings.filterwarnings('ignore')


# ============================================================================
# Data Preparation (Same as V4)
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
        """Prepare PyTorch datasets with 2D reshaping"""
        STM_all, target, train_ind, val_ind, test_ind, class_counts = self.load_data()
        
        # Reshape from flattened to 2D: (batch, freq, time)
        STM_all_2d = STM_all.reshape(-1, self.n_freq, self.n_time)
        
        # Normalize per sample (CRITICAL: preserves relative energy patterns)
        means = STM_all_2d.mean(axis=(1, 2), keepdims=True)
        stds = STM_all_2d.std(axis=(1, 2), keepdims=True)
        STM_all_2d = (STM_all_2d - means) / (stds + 1e-8)
        
        # Add channel dimension: (batch, 1, freq, time)
        STM_all_2d = STM_all_2d[:, np.newaxis, :, :]
        
        # Convert to PyTorch tensors
        X_train = torch.FloatTensor(STM_all_2d[train_ind])
        y_train = torch.LongTensor(target[train_ind])
        
        X_val = torch.FloatTensor(STM_all_2d[val_ind])
        y_val = torch.LongTensor(target[val_ind])
        
        X_test = torch.FloatTensor(STM_all_2d[test_ind])
        y_test = torch.LongTensor(target[test_ind])
        
        # Create datasets
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)
        
        print(f"\nPyTorch Dataset Shapes:")
        print(f"Train: {X_train.shape}")
        print(f"Val: {X_val.shape}")
        print(f"Test: {X_test.shape}")
        
        return train_dataset, val_dataset, test_dataset, class_counts


# ============================================================================
# CoordConv Layer
# ============================================================================

class CoordConv2d(nn.Module):
    """
    CoordConv: Adds coordinate channels to convolution input.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, bias=True):
        super(CoordConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels + 2, out_channels, kernel_size, 
                             stride=stride, padding=padding, bias=bias)
        
    def forward(self, x):
        batch_size, _, height, width = x.size()
        
        # Create coordinate channels
        y_coords = torch.linspace(-1, 1, height, device=x.device)
        x_coords = torch.linspace(-1, 1, width, device=x.device)
        y_coords = y_coords.view(1, 1, height, 1).expand(batch_size, 1, height, width)
        x_coords = x_coords.view(1, 1, 1, width).expand(batch_size, 1, height, width)
        
        # Concatenate coordinates with input
        x_with_coords = torch.cat([x, x_coords, y_coords], dim=1)
        
        return self.conv(x_with_coords)


# ============================================================================
# DropBlock2d Module (NEW in V6)
# ============================================================================

class DropBlock2d(nn.Module):
    """
    DropBlock: Spatial dropout that drops contiguous blocks instead of individual pixels.
    From "DropBlock: A regularization method for convolutional networks" (Ghiasi et al., NeurIPS 2018)
    
    More effective than standard dropout for convolutional layers because:
    - Removes spatially correlated information
    - Forces network to learn more diverse features
    """
    def __init__(self, drop_prob=0.1, block_size=7):
        super(DropBlock2d, self).__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size
        
    def _compute_gamma(self, x):
        """
        Compute gamma parameter for mask sampling.
        gamma adjusts the sampling probability to account for block expansion.
        """
        return self.drop_prob / (self.block_size ** 2)
    
    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            return x
        
        # Compute gamma
        gamma = self._compute_gamma(x)
        
        # Sample mask for center of blocks
        batch_size, channels, height, width = x.size()
        mask = torch.bernoulli(
            torch.ones(batch_size, channels, height, width, device=x.device) * gamma
        )
        
        # Expand mask to blocks using max pooling
        # This creates contiguous rectangular regions
        block_mask = 1 - F.max_pool2d(
            mask,
            kernel_size=self.block_size,
            stride=1,
            padding=self.block_size // 2
        )
        
        # Normalize to preserve expected sum
        normalize_factor = block_mask.numel() / (block_mask.sum() + 1e-6)
        
        return x * block_mask * normalize_factor


# ============================================================================
# Attention Mechanisms (Same as V4)
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
        self.act = nn.ReLU(inplace=False)
        
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
        self.relu = nn.ReLU(inplace=False)
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
# CutMix Augmentation (NEW in V6 - Replaces Mixup)
# ============================================================================

def cutmix_data(x, y, alpha=1.0):
    """
    CutMix augmentation from "CutMix: Regularization Strategy to Train Strong Classifiers 
    with Localizable Features" (Yun et al., ICCV 2019)
    
    Instead of linearly interpolating images (mixup), CutMix cuts and pastes patches.
    This is more effective because:
    - Encourages model to focus on diverse spatial regions
    - Maintains local spatial structure
    - Better generalization than mixup for vision tasks
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    # Get dimensions
    _, _, H, W = x.size()
    
    # Generate random box
    cut_rat = np.sqrt(1. - lam)
    cut_h = int(H * cut_rat)
    cut_w = int(W * cut_rat)
    
    # Uniform sampling of box center
    cx = np.random.randint(H)
    cy = np.random.randint(W)
    
    # Clip box coordinates
    bbx1 = np.clip(cx - cut_h // 2, 0, H)
    bby1 = np.clip(cy - cut_w // 2, 0, W)
    bbx2 = np.clip(cx + cut_h // 2, 0, H)
    bby2 = np.clip(cy + cut_w // 2, 0, W)
    
    # Apply CutMix
    mixed_x = x.clone()
    mixed_x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    # Adjust lambda to exactly match pixel ratio
    adjusted_lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (H * W))
    
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, adjusted_lam


def cutmix_criterion(criterion, pred, y_a, y_b, lam):
    """Compute CutMix loss"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============================================================================
# Enhanced ResNet with DropBlock and Stochastic Depth
# ============================================================================

class BasicBlock(nn.Module):
    """
    ResNet Basic Block with:
    - CoordConv
    - DropBlock (spatial dropout)
    - Stochastic Depth (skip connections)
    - Attention (CA or SE)
    """
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, 
                 use_coordconv=True, attention_type='CA',
                 dropblock_prob=0.0, dropblock_size=5,
                 stochastic_depth_prob=0.0):
        super(BasicBlock, self).__init__()
        
        if use_coordconv:
            self.conv1 = CoordConv2d(in_channels, out_channels, kernel_size=3, 
                                    stride=stride, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                  stride=stride, padding=1, bias=False)
        
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # DropBlock after bn2 (NEW in V6)
        self.dropblock = DropBlock2d(drop_prob=dropblock_prob, block_size=dropblock_size)
        
        # Stochastic Depth (NEW in V6)
        self.stochastic_depth_prob = stochastic_depth_prob
        
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
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Apply DropBlock before attention
        out = self.dropblock(out)
        
        # Apply attention
        if self.attention is not None:
            out = self.attention(out)
        
        # Stochastic Depth: randomly skip block during training
        if self.training and self.stochastic_depth_prob > 0:
            if torch.rand(1).item() < self.stochastic_depth_prob:
                # Skip the entire residual branch
                if self.downsample is not None:
                    identity = self.downsample(x)
                return self.relu(identity)
        
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
            nn.ReLU(inplace=False)
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
    ResNet-18 with CoordConv, Attention (CA + SE), Multi-Scale Fusion,
    DropBlock, and Stochastic Depth
    
    Architecture:
    - Layer1, Layer2: Coordinate Attention (position-aware, early layers)
    - Layer3, Layer4: Squeeze-and-Excitation (channel selection, late layers)
    - Multi-scale fusion: Combine layer3 + layer4 features
    - DropBlock: Progressive spatial dropout
    - Stochastic Depth: Progressive drop probability per layer
    """
    def __init__(self, num_classes=6, dropout=0.3, 
                 dropblock_sizes=(0, 5, 5, 3),
                 stochastic_depth_probs=(0.0, 0.05, 0.10, 0.20)):
        super(CoordConvResNet18_Attention, self).__init__()
        
        # Modified stem for small input (20x121)
        self.conv1 = CoordConv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)
        
        # ResNet layers with different attention mechanisms, DropBlock, and Stochastic Depth
        self.in_channels = 64
        self.layer1 = self._make_layer(64, 2, stride=1, attention='CA',
                                       dropblock_size=dropblock_sizes[0],
                                       stochastic_depth_prob=stochastic_depth_probs[0])
        self.layer2 = self._make_layer(128, 2, stride=2, attention='CA',
                                       dropblock_size=dropblock_sizes[1],
                                       stochastic_depth_prob=stochastic_depth_probs[1])
        self.layer3 = self._make_layer(256, 2, stride=2, attention='SE',
                                       dropblock_size=dropblock_sizes[2],
                                       stochastic_depth_prob=stochastic_depth_probs[2])
        self.layer4 = self._make_layer(512, 2, stride=2, attention='SE',
                                       dropblock_size=dropblock_sizes[3],
                                       stochastic_depth_prob=stochastic_depth_probs[3])
        
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
    
    def _make_layer(self, out_channels, blocks, stride=1, attention='CA',
                   dropblock_size=5, stochastic_depth_prob=0.0):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample, 
                                use_coordconv=True, attention_type=attention,
                                dropblock_prob=0.0,  # Start with 0, will be updated during training
                                dropblock_size=dropblock_size,
                                stochastic_depth_prob=stochastic_depth_prob))
        self.in_channels = out_channels
        
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, use_coordconv=True, 
                                    attention_type=attention,
                                    dropblock_prob=0.0,  # Start with 0, will be updated during training
                                    dropblock_size=dropblock_size,
                                    stochastic_depth_prob=stochastic_depth_prob))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, CoordConv2d)):
                nn.init.kaiming_normal_(m.conv.weight if isinstance(m, CoordConv2d) else m.weight, 
                                       mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
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
# Trainer with V6 Enhancements
# ============================================================================

class Trainer:
    """
    Trainer with V6 enhancements:
    - CosineAnnealingLR scheduler (replaces ReduceLROnPlateau)
    - CutMix augmentation (replaces Mixup)
    - Progressive DropBlock ramping
    - Longer training (150 epochs)
    - Higher weight decay (5e-4)
    - Later DRW start (epoch 75)
    - Longer patience (30)
    """
    def __init__(self, model, train_loader, val_loader, test_loader, 
                 device, class_counts, lr=1e-4, weight_decay=5e-4, num_epochs=150):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.class_counts = class_counts
        
        # Optimizer with higher weight decay
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Scheduler: CosineAnnealingLR (NEW in V6)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_epochs, eta_min=1e-6
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
        self.early_stop_patience = 30  # Longer patience for V6
        
    def update_dropblock(self, epoch, model, start_epoch=20, end_epoch=60, max_prob=0.15):
        """
        Progressively increase DropBlock probability from 0 to max_prob
        Linear ramp from start_epoch to end_epoch
        """
        if epoch < start_epoch:
            drop_prob = 0.0
        elif epoch >= end_epoch:
            drop_prob = max_prob
        else:
            # Linear ramp
            drop_prob = max_prob * (epoch - start_epoch) / (end_epoch - start_epoch)
        
        # Update all DropBlock modules
        for module in model.modules():
            if isinstance(module, DropBlock2d):
                module.drop_prob = drop_prob
        
        return drop_prob
    
    def train_epoch(self, epoch, total_epochs, use_drw=False, use_cutmix=True):
        self.model.train()
        total_loss = 0.0
        
        # Update DropBlock probability
        drop_prob = self.update_dropblock(epoch, self.model, start_epoch=20, end_epoch=60, max_prob=0.15)
        
        # Update DRW
        if use_drw:
            self.criterion.weight = self.drw_weights
        else:
            self.criterion.weight = None
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Apply CutMix with 80% probability (NEW in V6)
            if use_cutmix and np.random.rand() < 0.8:
                mixed_data, target_a, target_b, lam = cutmix_data(data, target, alpha=1.0)
                
                self.optimizer.zero_grad()
                output = self.model(mixed_data)
                loss = cutmix_criterion(self.criterion, output, target_a, target_b, lam)
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
                      f"Loss: {loss.item():.4f}, DRW: {use_drw}, CutMix: {use_cutmix}, "
                      f"DropBlock: {drop_prob:.3f}")
        
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
        print("Starting training with V6 enhancements...")
        print("="*60)
        
        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{num_epochs}")
            print("="*60)
            
            # Determine if DRW should be active (starts at epoch 75 for 150 epochs)
            use_drw = epoch > 75
            if epoch == 76:
                print("\n*** Activating Deferred Reweighting (DRW) ***\n")
            
            # Train
            train_loss = self.train_epoch(epoch, num_epochs, use_drw=use_drw)
            print(f"Train Loss: {train_loss:.4f}")
            
            # Validate
            val_loss, val_f1, _, _ = self.evaluate(self.val_loader)
            print(f"Val Loss: {val_loss:.4f}, Val Macro F1: {val_f1:.4f}")
            
            # Update scheduler (cosine annealing updates every epoch)
            self.scheduler.step()
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
        print("Usage: python STM_CoordConvLDAM6.py <mode>")
        print("  mode 0: standard (full dataset)")
        print("  mode 1: downsample non-tonal speech to 100k")
        sys.exit(1)
    
    mode = int(sys.argv[1])
    
    # Set parameters based on mode
    if mode == 0:
        ds_nontonal_speech = False
        directory = "model/STM/CoordConvLDAM6_corpora_categories/standard"
    elif mode == 1:
        ds_nontonal_speech = True
        directory = "model/STM/CoordConvLDAM6_corpora_categories/downsample"
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
    
    # Create model with V6 enhancements
    print("\n" + "="*60)
    print("Creating CoordConv-ResNet18 with V6 Regularization...")
    print("="*60)
    
    num_classes = 6
    model = CoordConvResNet18_Attention(
        num_classes=num_classes, 
        dropout=0.3,
        dropblock_sizes=(0, 5, 5, 3),  # Layer1: none, Layer2-3: 5x5, Layer4: 3x3
        stochastic_depth_probs=(0.0, 0.05, 0.10, 0.20)  # Progressive drop probability
    )
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"\nV6 Enhancements:")
    print("  - DropBlock: Spatial dropout (progressive ramp 0→0.15)")
    print("  - Stochastic Depth: Random skip connections (0.0→0.20)")
    print("  - CutMix: Region-based augmentation (80% probability)")
    print("  - CosineAnnealingLR: Smooth learning rate decay")
    print("  - Extended training: 150 epochs")
    print("  - Higher weight decay: 5e-4")
    
    # Create trainer
    num_epochs = 150
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        class_counts=class_counts,
        lr=1e-4,
        weight_decay=5e-4,
        num_epochs=num_epochs
    )
    
    # Train model
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
