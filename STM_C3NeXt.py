#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STM Classification with CoordConv-ConvNeXt (C3NeXt)

Integrates CoordConv spatial awareness with ConvNeXt's modern architecture:
1. CoordConv stem: Solves translation variance problem
2. ConvNeXt blocks: Large kernels (7×7), depthwise convolutions, LayerNorm
3. LDAM + DRW + Mixup: Proven training dynamics from V4
4. Optimized for STM features (20×121) with position-critical patterns
"""

import os
import sys
import warnings
import datetime
import numpy as np
import pandas as pd
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import f1_score, classification_report

warnings.filterwarnings('ignore')


# ============================================================================
# Data Preparation (Same as CoordConvLDAM4)
# ============================================================================

class prepData_STM_CoordConv:
    """
    Data preparation for CoordConv-based models.
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
            SONYC_aug_STM = np.load(root_folder + 'STM_output/corpSTMnpy/SONYC_augmented_STMall.npy')
            STM_all = np.concatenate((STM_all, SONYC_aug_STM), axis=0)
            SONYC_aug_len = SONYC_aug_STM.shape[0]
            print(f"Loaded: {root_folder}STM_output/corpSTMnpy/SONYC_augmented_STMall.npy, shape: {SONYC_aug_STM.shape}")
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
# CoordConv Layer (Same as CoordConvLDAM4)
# ============================================================================

class CoordConv2d(nn.Module):
    """
    CoordConv: Adds coordinate channels to convolution input.
    Solves translation variance by making position explicit.
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
# ConvNeXt Components
# ============================================================================

class LayerNorm(nn.Module):
    """
    LayerNorm that supports two data formats: channels_last (default) or channels_first.
    channels_last corresponds to inputs with shape (batch_size, height, width, channels)
    channels_first corresponds to inputs with shape (batch_size, channels, height, width)
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # Normalize over spatial dimensions (H, W) for each channel
            # Input: (B, C, H, W)
            # Mean and std over H, W dimensions for each channel separately
            u = x.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
            s = (x - u).pow(2).mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
            x = (x - u) / torch.sqrt(s + self.eps)
            # Apply per-channel affine transform
            x = self.weight.view(1, -1, 1, 1) * x + self.bias.view(1, -1, 1, 1)
            return x


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
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


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt Block: Modernized inverted bottleneck design.
    
    Structure:
    1. Depthwise Conv 7×7 (spatial mixing)
    2. LayerNorm
    3. Pointwise Conv 1×1 (expansion to 4×channels)
    4. GELU activation
    5. Pointwise Conv 1×1 (projection back)
    6. Layer Scale
    7. Drop Path + Residual connection
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # Depthwise
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # Pointwise/1×1 conv, expansion
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)  # Pointwise/1×1 conv, projection
        
        # Layer Scale (learnable per-channel scaling)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


# ============================================================================
# C3NeXt Model: CoordConv + ConvNeXt
# ============================================================================

class C3NeXt(nn.Module):
    """
    CoordConv-ConvNeXt for STM Classification
    
    Architecture:
    - Stem: CoordConv 4×4, stride 4 (downsample 20×121 -> 5×31)
    - Stage 1: 3 ConvNeXt blocks (channels=96)
    - Downsample 1: LayerNorm + Conv 2×2, stride 2 (5×31 -> 3×16)
    - Stage 2: 3 ConvNeXt blocks (channels=192)
    - Downsample 2: LayerNorm + Conv 2×2, stride 2 (3×16 -> 2×8)
    - Stage 3: 9 ConvNeXt blocks (channels=384)
    - Downsample 3: LayerNorm + Conv 2×2, stride 2 (2×8 -> 1×4)
    - Stage 4: 3 ConvNeXt blocks (channels=768)
    - Head: Global average pooling + LayerNorm + Linear
    
    Total: 18 ConvNeXt blocks (similar depth to ResNet-18)
    """
    def __init__(self, num_classes=6, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768],
                 drop_path_rate=0.1, layer_scale_init_value=1e-6, head_dropout=0.3):
        super().__init__()
        
        # CoordConv Stem: 1 -> 96 channels, stride 4
        self.stem = nn.Sequential(
            CoordConv2d(1, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        
        # Build 4 stages
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        
        for i in range(4):
            stage = nn.Sequential(
                *[ConvNeXtBlock(dim=dims[i], drop_path=dp_rates[cur + j], 
                                layer_scale_init_value=layer_scale_init_value) 
                  for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]
        
        # Downsampling layers between stages
        self.downsample_layers = nn.ModuleList()
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)
        
        # Classification head
        self.norm = LayerNorm(dims[-1], eps=1e-6, data_format="channels_first")
        self.head_dropout = nn.Dropout(head_dropout)
        self.head = nn.Linear(dims[-1], num_classes)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, return_features=False):
        # Stem
        x = self.stem(x)
        
        # 4 stages with downsampling
        for i in range(4):
            x = self.stages[i](x)
            if i < 3:
                x = self.downsample_layers[i](x)
        
        # Global average pooling
        x = self.norm(x)
        x = x.mean([-2, -1])  # (N, C)
        
        features = x
        
        # Classification
        x = self.head_dropout(x)
        x = self.head(x)
        
        if return_features:
            return x, features
        return x


# ============================================================================
# Mixup Augmentation (Same as CoordConvLDAM4)
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
# LDAM Loss with Label Smoothing (Same as CoordConvLDAM4)
# ============================================================================

class LDAMLoss(nn.Module):
    """
    LDAM Loss with label smoothing
    From "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss"
    (Cao et al., NeurIPS 2019)
    """
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30, label_smooth=0.05):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.FloatTensor(m_list)
        self.m_list = m_list
        self.s = s
        self.weight = weight
        self.label_smooth = label_smooth
        
    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.type(torch.FloatTensor).to(x.device)
        batch_m = torch.matmul(self.m_list.to(x.device)[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        
        # Label smoothing
        if self.label_smooth > 0:
            num_classes = x.size(1)
            one_hot = torch.zeros_like(x).scatter_(1, target.view(-1, 1), 1)
            one_hot = one_hot * (1 - self.label_smooth) + (1 - one_hot) * self.label_smooth / (num_classes - 1)
            log_prb = F.log_softmax(self.s * output, dim=1)
            loss = -(one_hot * log_prb).sum(dim=1)
            if self.weight is not None:
                loss = loss * self.weight[target]
            return loss.mean()
        else:
            return F.cross_entropy(self.s * output, target, weight=self.weight)


# ============================================================================
# Trainer (Same structure as CoordConvLDAM4)
# ============================================================================

class Trainer:
    """
    Trainer with proven V4 dynamics: LDAM, DRW, Mixup, ReduceLROnPlateau, Early Stopping
    """
    def __init__(self, model, train_loader, val_loader, test_loader, 
                 device, class_counts, lr=1e-4, weight_decay=2e-4, resume_checkpoint=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.class_counts = class_counts
        
        # Loss function: LDAM with label smoothing
        self.criterion = LDAMLoss(
            cls_num_list=class_counts,
            max_m=0.5,
            s=30,
            label_smooth=0.05
        )
        
        # Optimizer: AdamW with weight decay
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Scheduler: ReduceLROnPlateau
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=10,
            min_lr=1e-6
        )
        
        # Early stopping
        self.best_val_f1 = 0
        self.patience_counter = 0
        self.max_patience = 20
        self.start_epoch = 1
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_f1s = []
        
        # Resume from checkpoint if provided
        if resume_checkpoint is not None:
            self.load_checkpoint(resume_checkpoint)
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load checkpoint to resume training
        """
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            return False
        
        print(f"\nLoading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("✓ Loaded optimizer state")
        
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("✓ Loaded scheduler state")
        
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_f1 = checkpoint.get('val_f1', 0.0)
        
        if 'patience_counter' in checkpoint:
            self.patience_counter = checkpoint['patience_counter']
        
        if 'train_losses' in checkpoint:
            self.train_losses = checkpoint['train_losses']
        if 'val_losses' in checkpoint:
            self.val_losses = checkpoint['val_losses']
        if 'val_f1s' in checkpoint:
            self.val_f1s = checkpoint['val_f1s']
        
        print(f"✓ Resumed from epoch {self.start_epoch}, Best Val F1: {self.best_val_f1:.4f}")
        return True
        
    def train_epoch(self, epoch, total_epochs, use_drw=False, use_mixup=True):
        self.model.train()
        total_loss = 0
        
        # DRW: Adjust loss weights based on class imbalance
        if use_drw:
            beta = 0.9999
            effective_num = 1.0 - np.power(beta, self.class_counts)
            per_cls_weights = (1.0 - beta) / effective_num
            per_cls_weights = per_cls_weights / per_cls_weights.sum() * len(self.class_counts)
            per_cls_weights = torch.FloatTensor(per_cls_weights).to(self.device)
            self.criterion.weight = per_cls_weights
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Mixup augmentation (30% of batches)
            if use_mixup and np.random.rand() < 0.3:
                data, targets_a, targets_b, lam = mixup_data(data, target, alpha=0.3)
                outputs = self.model(data)
                loss = mixup_criterion(self.criterion, outputs, targets_a, targets_b, lam)
            else:
                outputs = self.model(data)
                loss = self.criterion(outputs, target)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Progress logging
            if (batch_idx + 1) % 500 == 0:
                print(f"  Batch {batch_idx+1}/{len(self.train_loader)}, "
                      f"Loss: {loss.item():.4f}, DRW: {use_drw}, Mixup: {use_mixup}")
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, target)
                
                total_loss += loss.item()
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(target.cpu().numpy())
        
        avg_loss = total_loss / len(data_loader)
        f1 = f1_score(all_targets, all_preds, average='macro')
        
        return avg_loss, f1, np.array(all_preds), np.array(all_targets)
    
    def train(self, num_epochs, checkpoint_dir):
        print("\n" + "="*60)
        if self.start_epoch > 1:
            print(f"Resuming Training from Epoch {self.start_epoch}...")
        else:
            print("Starting Training...")
        print("="*60)
        print(f"Total epochs: {num_epochs}")
        print(f"Starting from epoch {self.start_epoch}")
        
        drw_start_epoch = 50  # Start DRW at epoch 50
        
        for epoch in range(self.start_epoch, num_epochs + 1):
            # Activate DRW after specified epoch
            use_drw = (epoch >= drw_start_epoch)
            if epoch == drw_start_epoch:
                print("\n*** Activating Deferred Reweighting (DRW) ***")
            
            # Train
            train_loss = self.train_epoch(epoch, num_epochs, use_drw=use_drw, use_mixup=True)
            
            # Validate
            val_loss, val_f1, _, _ = self.evaluate(self.val_loader)
            
            # Track history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_f1s.append(val_f1)
            
            # Update scheduler
            self.scheduler.step(val_f1)
            
            # Print progress
            if (epoch) % 5 == 0 or epoch == self.start_epoch:
                print(f"\nEpoch {epoch}/{num_epochs}")
                print("="*60)
                print(f"Train Loss: {train_loss:.4f}")
                print(f"Val Loss: {val_loss:.4f}, Val Macro F1: {val_f1:.4f}")
                print(f"Current learning rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save checkpoint every 10 epochs
            if epoch % 10 == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_f1': val_f1,
                    'patience_counter': self.patience_counter,
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                    'val_f1s': self.val_f1s,
                }, checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path}")
            
            # Save best model
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.patience_counter = 0
                
                best_model_path = os.path.join(checkpoint_dir, 'best_model.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_f1': val_f1,
                    'patience_counter': self.patience_counter,
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                    'val_f1s': self.val_f1s,
                }, best_model_path)
                print(f"*** New best model saved! Val F1: {val_f1:.4f} ***")
            else:
                self.patience_counter += 1
                print(f"No improvement. Patience: {self.patience_counter}/{self.max_patience}")
                
                if self.patience_counter >= self.max_patience:
                    print("Early stopping triggered!")
                    break
            
            # Always save latest checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'val_f1': val_f1,
                'patience_counter': self.patience_counter,
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'val_f1s': self.val_f1s,
            }, os.path.join(checkpoint_dir, 'latest_checkpoint.pt'))
        
        print("\n" + "="*60)
        print("Training completed!")
        print(f"Best validation F1: {self.best_val_f1:.4f}")
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
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python STM_C3NeXt.py <mode> [--resume <checkpoint_dir>]")
        print("  mode 0: Standard training")
        print("  mode 1: Downsample non-tonal speech")
        print("Options:")
        print("  --resume <checkpoint_dir>: Resume from checkpoint directory")
        sys.exit(1)
    
    mode = int(sys.argv[1])
    
    # Check for resume flag
    resume_dir = None
    if '--resume' in sys.argv:
        resume_idx = sys.argv.index('--resume')
        if resume_idx + 1 < len(sys.argv):
            resume_dir = sys.argv[resume_idx + 1]
            print(f"Resume mode: Will attempt to load from {resume_dir}")
    
    # Set parameters based on mode
    if mode == 0:
        ds_nontonal_speech = False
        directory = 'model/STM/C3NeXt_corpora_categories/standard'
    elif mode == 1:
        ds_nontonal_speech = True
        directory = 'model/STM/C3NeXt_corpora_categories/downsample'
    else:
        print("Invalid mode! Use 0 or 1.")
        sys.exit(1)
    
    # Create directory
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
    print("Creating C3NeXt (CoordConv + ConvNeXt)...")
    print("="*60)
    
    num_classes = 6
    model = C3NeXt(
        num_classes=num_classes,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        drop_path_rate=0.1,
        layer_scale_init_value=1e-6,
        head_dropout=0.3
    )
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Architecture: ConvNeXt-Tiny with CoordConv stem")
    print(f"Blocks: 18 ConvNeXt blocks [3, 3, 9, 3]")
    print(f"Channels: [96, 192, 384, 768]")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        class_counts=class_counts,
        lr=1e-4,
        weight_decay=2e-4,
        resume_checkpoint=None  # Will be loaded below
    )
    
    # Resume from checkpoint if specified
    if resume_dir:
        latest_ckpt = os.path.join(checkpoint_dir, 'latest_checkpoint.pt')
        if os.path.exists(latest_ckpt):
            trainer.load_checkpoint(latest_ckpt)
        else:
            ckpt_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_')]
            if ckpt_files:
                latest_ckpt = max(ckpt_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
                latest_ckpt_path = os.path.join(checkpoint_dir, latest_ckpt)
                trainer.load_checkpoint(latest_ckpt_path)
            else:
                print(f"Warning: No checkpoint found in {checkpoint_dir}, starting from scratch")
    
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
