#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STM Classification with CoordConv-ResNet and LDAM Loss
Phase 1.2: Enhanced Training Dynamics

Improvements over STM_CoordConvLDAM.py:
1. Stronger regularization (dropout 0.5, weight decay 5e-4)
2. Dropout in residual blocks (not just head)
3. ReduceLROnPlateau scheduler (adaptive learning rate)
4. Early DRW activation (epoch 20 instead of 40)
5. Mixup augmentation for better generalization
6. Label smoothing to prevent overconfidence
7. Early stopping with patience
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
# Data Preparation
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
    
    From "An Intriguing Failing of Convolutional Neural Networks 
    and the CoordConv Solution" (Liu et al., NeurIPS 2018)
    
    Allows filters to learn position-dependent features:
    - Channel 0: Original feature map (STM energy)
    - Channel 1: x-coordinate (temporal modulation bins, -1 to 1)
    - Channel 2: y-coordinate (spectral modulation bins, -1 to 1)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, bias=True):
        super(CoordConv2d, self).__init__()
        
        # Add 2 channels for x,y coordinates
        self.conv = nn.Conv2d(in_channels + 2, out_channels, kernel_size, 
                             stride=stride, padding=padding, bias=bias)
        
    def forward(self, x):
        batch_size, _, height, width = x.size()
        
        # Create coordinate channels
        y_coords = torch.linspace(-1, 1, height, device=x.device)
        x_coords = torch.linspace(-1, 1, width, device=x.device)
        
        y_coords = y_coords.view(1, 1, height, 1).expand(batch_size, 1, height, width)
        x_coords = x_coords.view(1, 1, 1, width).expand(batch_size, 1, height, width)
        
        # Concatenate coordinate channels with input
        x_with_coords = torch.cat([x, x_coords, y_coords], dim=1)
        
        return self.conv(x_with_coords)


# ============================================================================
# Mixup Augmentation
# ============================================================================

def mixup_data(x, y, alpha=0.2):
    """
    Mixup augmentation from "mixup: Beyond Empirical Risk Minimization"
    (Zhang et al., ICLR 2018)
    
    Interpolates between random pairs of training examples
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
# CoordConv-ResNet Architecture (Enhanced)
# ============================================================================

class BasicBlock(nn.Module):
    """
    ResNet Basic Block with CoordConv and dropout
    """
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, 
                 use_coordconv=True, dropout=0.1):
        super(BasicBlock, self).__init__()
        
        # First convolution (CoordConv or standard)
        if use_coordconv:
            self.conv1 = CoordConv2d(in_channels, out_channels, kernel_size=3, 
                                     stride=stride, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                                  stride=stride, padding=1, bias=False)
        
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        
        # Second convolution (always standard)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class CoordConvResNet18(nn.Module):
    """
    ResNet-18 adapted for STM classification with CoordConv
    Enhanced with dropout in residual blocks
    """
    def __init__(self, num_classes=6, dropout=0.5, block_dropout=0.1):
        super(CoordConvResNet18, self).__init__()
        
        # CRITICAL: Initialize in_channels FIRST before calling _make_layer
        self.in_channels = 64
        
        # Modified stem for small input (20x121)
        self.conv1 = CoordConv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # ResNet layers
        self.layer1 = self._make_layer(64, 2, stride=1, dropout=block_dropout)
        self.layer2 = self._make_layer(128, 2, stride=2, dropout=block_dropout)
        self.layer3 = self._make_layer(256, 2, stride=2, dropout=block_dropout)
        self.layer4 = self._make_layer(512, 2, stride=2, dropout=block_dropout)
        
        # Calculate flattened size after conv layers
        # Input: (1, 20, 121)
        # After layer4: (512, 3, 16) = 24576
        self.fc1 = nn.Linear(512 * 3 * 16, 512)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(256, num_classes)
        
        self._initialize_weights()
    
    def _make_layer(self, out_channels, blocks, stride=1, dropout=0.1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        
        layers = []
        # First block (with potential downsampling and CoordConv)
        layers.append(BasicBlock(self.in_channels, out_channels, stride, 
                                downsample, use_coordconv=True, dropout=dropout))
        self.in_channels = out_channels
        
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, 
                                    use_coordconv=False, dropout=dropout))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        return x


# ============================================================================
# LDAM Loss with Label Smoothing
# ============================================================================

class LDAMLoss(nn.Module):
    """
    Label-Distribution-Aware Margin Loss with Label Smoothing
    
    From "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss"
    (Cao et al., NeurIPS 2019)
    
    Enhanced with label smoothing to prevent overconfidence
    """
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30, label_smooth=0.1):
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
        batch_m = torch.matmul(self.m_list[None, :].to(x.device), index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        
        # Apply label smoothing
        if self.label_smooth > 0:
            n_classes = x.size(1)
            log_probs = F.log_softmax(self.s * output, dim=1)
            
            # Smooth labels
            with torch.no_grad():
                true_dist = torch.zeros_like(log_probs)
                true_dist.fill_(self.label_smooth / (n_classes - 1))
                true_dist.scatter_(1, target.data.unsqueeze(1), 1.0 - self.label_smooth)
            
            loss = torch.mean(torch.sum(-true_dist * log_probs, dim=1))
        else:
            loss = F.cross_entropy(self.s * output, target, weight=self.weight)
        
        return loss


# ============================================================================
# Trainer with Enhanced Training Dynamics
# ============================================================================

class Trainer:
    """
    Enhanced trainer with:
    - Mixup augmentation
    - ReduceLROnPlateau scheduler
    - Early DRW (epoch 20)
    - Early stopping with patience
    - Label smoothing
    """
    def __init__(self, model, train_loader, val_loader, test_loader, 
                 device, class_counts, lr=1e-4, weight_decay=5e-4):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        
        # Optimizer with higher weight decay
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        
        # ReduceLROnPlateau scheduler (more adaptive)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='max',  # Monitor validation F1
            factor=0.5,  # Reduce LR by half
            patience=7,  # Wait 7 epochs before reducing
            min_lr=1e-6
        )
        
        # LDAM Loss with label smoothing
        self.criterion = LDAMLoss(
            cls_num_list=class_counts, 
            max_m=0.5, 
            s=30,
            label_smooth=0.05
        )
        
        # Class weights for DRW
        effective_num = 1.0 - np.power(0.9999, class_counts)
        per_cls_weights = (1.0 - 0.9999) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(class_counts)
        self.per_cls_weights = torch.FloatTensor(per_cls_weights).to(device)
        
        # Training state
        self.best_val_f1 = 0
        self.patience_counter = 0
        self.patience = 20  # Early stopping patience
        
    def train_epoch(self, epoch, total_epochs, use_drw=False, use_mixup=True):
        self.model.train()
        total_loss = 0
        
        # Update loss criterion for DRW
        if use_drw:
            self.criterion.weight = self.per_cls_weights
        else:
            self.criterion.weight = None
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Apply mixup augmentation
            if use_mixup and np.random.rand() < 0.3:  # 30% chance
                data, target_a, target_b, lam = mixup_data(data, target, alpha=0.3)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = mixup_criterion(self.criterion, output, target_a, target_b, lam)
            else:
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Print progress
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
        
        # Temporarily disable DRW for evaluation
        temp_weight = self.criterion.weight
        self.criterion.weight = None
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                
                preds = output.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        # Restore criterion weight
        self.criterion.weight = temp_weight
        
        avg_loss = total_loss / len(data_loader)
        f1 = f1_score(all_targets, all_preds, average='macro')
        
        return avg_loss, f1, np.array(all_preds), np.array(all_targets)
    
    def train(self, num_epochs, checkpoint_dir):
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"DRW will start at epoch {num_epochs // 2:.0f}")  # ~50% of training
        print(f"Early stopping patience: {self.patience} epochs")
        
        drw_start_epoch = int(num_epochs / 2)  # Start DRW at 50%
        
        for epoch in range(num_epochs):
            use_drw = epoch >= drw_start_epoch
            
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"{'='*60}")
            
            # Train
            train_loss = self.train_epoch(epoch, num_epochs, use_drw=use_drw, use_mixup=True)
            
            # Validate
            val_loss, val_f1, _, _ = self.evaluate(self.val_loader)
            
            # Update scheduler based on validation F1
            self.scheduler.step(val_f1)
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Macro F1: {val_f1:.4f}")
            print(f"Current learning rate: {current_lr:.6f}")
            
            # Save best model
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.patience_counter = 0
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_f1': val_f1,
                    'val_loss': val_loss,
                }
                torch.save(checkpoint, os.path.join(checkpoint_dir, 'best_model.pt'))
                print(f"✓ Saved best model with Val F1: {val_f1:.4f}")
            else:
                self.patience_counter += 1
                print(f"No improvement for {self.patience_counter} epoch(s)")
            
            # Early stopping
            if self.patience_counter >= self.patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                print(f"Best Val F1: {self.best_val_f1:.4f}")
                break
            
            # Save periodic checkpoints
            if (epoch + 1) % 10 == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }
                torch.save(checkpoint, 
                          os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt'))


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
        print("Usage: python STM_CoordConvLDAM2.py <mode>")
        print("  mode 0: Standard training (full dataset)")
        print("  mode 1: Downsampled non-tonal speech (100k samples)")
        sys.exit(1)
    
    mode = int(sys.argv[1])
    
    # Set parameters based on mode
    if mode == 0:
        ds_nontonal_speech = False
        directory = "model/STM/CoordConvLDAM2_corpora_categories/standard"
    elif mode == 1:
        ds_nontonal_speech = True
        directory = "model/STM/CoordConvLDAM2_corpora_categories/downsample"
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
    
    # Create model with enhanced regularization
    print("\n" + "="*60)
    print("Creating Enhanced CoordConv-ResNet18 model...")
    print("="*60)
    
    num_classes = 6
    model = CoordConvResNet18(num_classes=num_classes, dropout=0.3, block_dropout=0.05)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Improvements: Moderate dropout (0.3), block dropout (0.05), label smoothing (0.05)")
    
    # Create trainer with enhanced settings
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        class_counts=class_counts,
        lr=1e-4,
        weight_decay=2e-4  # Moderate regularization
    )
    
    # Train model
    num_epochs = 100  # Increased, but early stopping will kick in
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
