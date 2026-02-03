#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STM Classification with C3NeXt2: Focal-LDAM + Deeper Network

Improvements over C3NeXt:
1. Focal-LDAM Loss: Combines focal loss with LDAM for hard example mining
2. Deeper network: [4, 12, 8] blocks (24 total) for better capacity
3. Earlier DRW: Starts at epoch 40 instead of 50
4. Stronger regularization: Higher drop path (0.2), stronger mixup (alpha=0.4)

Target: Improve speech:tonal (0.63→0.70+) and music:non-vocal (0.60→0.68+)
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import f1_score, classification_report

warnings.filterwarnings('ignore')


# ============================================================================
# Data Preparation (Same as C3NeXt)
# ============================================================================

class prepData_STM_CoordConv:
    """Data preparation for CoordConv-based models"""
    def __init__(self, addAug=False, ds_nontonal_speech=False):
        self.addAug = addAug
        self.ds_nontonal_speech = ds_nontonal_speech
        self.n_freq = 20
        self.n_time = 121
        
    def corpora_list(self, addAug=False):
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
        
        corpus_env_list = ['MacaulayLibrary', 'SONYC']
        
        corpus_speech_list.sort()
        corpus_music_list.sort()
        corpus_env_list.sort()
        
        return corpus_speech_list + corpus_music_list + corpus_env_list
    
    def load_data(self):
        corpus_list_all = self.corpora_list(self.addAug)
        root_folder = '/vast-ac8888/MusicSpeech-STM/'
        
        STM_all = None
        for corp in corpus_list_all:
            corp_name = corp.replace('/', '-')
            file_path = f"{root_folder}STM_output/corpSTMnpy/{corp_name}_STMall.npy"
            tmp = np.load(file_path)
            print(f"Loaded: {file_path}, shape: {tmp.shape}")
            STM_all = tmp if STM_all is None else np.concatenate((STM_all, tmp), axis=0)
        
        speech_corp_df1 = pd.read_csv(root_folder + 'train_test_split/speech1_10folds_speakerGroupFold.csv', index_col=0)
        speech_corp_df2 = pd.read_csv(root_folder + 'train_test_split/speech2_10folds_speakerGroupFold.csv', index_col=0)
        music_corp_df = pd.read_csv(root_folder + 'train_test_split/music_10folds_speakerGroupFold.csv', index_col=0)
        df_SONYC = pd.read_csv(root_folder + 'train_test_split/env_10folds_speakerGroupFold.csv', index_col=0)
        
        all_corp_df = pd.concat([speech_corp_df1, speech_corp_df2, music_corp_df, df_SONYC], ignore_index=True)
        
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
        
        target.replace({
            'speech: non-tonal': 0,
            'speech: tonal': 1,
            'music: vocal': 2,
            'music: non-vocal': 3,
            'env: urban': 4,
            'env: wildlife': 5,
        }, inplace=True)
        
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
        
        train_ind = (data_split < 8).values
        val_ind = (data_split == 8).values
        test_ind = (data_split == 9).values
        
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
        STM_all, target, train_ind, val_ind, test_ind, class_counts = self.load_data()
        
        STM_all_2d = STM_all.reshape(-1, self.n_freq, self.n_time)
        means = STM_all_2d.mean(axis=(1, 2), keepdims=True)
        stds = STM_all_2d.std(axis=(1, 2), keepdims=True)
        STM_all_2d = (STM_all_2d - means) / (stds + 1e-8)
        STM_all_2d = STM_all_2d[:, np.newaxis, :, :]
        
        X_train = torch.FloatTensor(STM_all_2d[train_ind])
        y_train = torch.LongTensor(target[train_ind])
        X_val = torch.FloatTensor(STM_all_2d[val_ind])
        y_val = torch.LongTensor(target[val_ind])
        X_test = torch.FloatTensor(STM_all_2d[test_ind])
        y_test = torch.LongTensor(target[test_ind])
        
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)
        
        print(f"\nPyTorch Dataset Shapes:")
        print(f"Train: {X_train.shape}")
        print(f"Val: {X_val.shape}")
        print(f"Test: {X_test.shape}")
        
        return train_dataset, val_dataset, test_dataset, class_counts


# ============================================================================
# Architecture Components (Same as C3NeXt)
# ============================================================================

class CoordConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(CoordConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels + 2, out_channels, kernel_size, 
                             stride=stride, padding=padding, bias=bias)
        
    def forward(self, x):
        batch_size, _, height, width = x.size()
        y_coords = torch.linspace(-1, 1, height, device=x.device)
        x_coords = torch.linspace(-1, 1, width, device=x.device)
        y_coords = y_coords.view(1, 1, height, 1).expand(batch_size, 1, height, width)
        x_coords = x_coords.view(1, 1, 1, width).expand(batch_size, 1, height, width)
        x_with_coords = torch.cat([x, x_coords, y_coords], dim=1)
        return self.conv(x_with_coords)


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(dim=(2, 3), keepdim=True)
            s = (x - u).pow(2).mean(dim=(2, 3), keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight.view(1, -1, 1, 1) * x + self.bias.view(1, -1, 1, 1)
            return x


class DropPath(nn.Module):
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
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = input + self.drop_path(x)
        return x


# ============================================================================
# C3NeXt2 Model: Deeper Network (24 blocks)
# ============================================================================

class C3NeXt2(nn.Module):
    """
    C3NeXt2: Deeper variant with [4, 12, 8] blocks (24 total)
    
    Architecture:
    - Stem: CoordConv 4×4, stride 4 (20×121 -> 5×30)
    - Stage 1: 4 ConvNeXt blocks (channels=96)
    - Downsample 1: LayerNorm + Conv 2×2, stride 2 (5×30 -> 2×15)
    - Stage 2: 12 ConvNeXt blocks (channels=192) [Main capacity]
    - Downsample 2: LayerNorm + Conv 2×2, stride 2 (2×15 -> 1×7)
    - Stage 3: 8 ConvNeXt blocks (channels=384)
    - Head: Global average pooling + LayerNorm + Linear
    
    Total: 24 ConvNeXt blocks (+33% vs C3NeXt)
    """
    def __init__(self, num_classes=6, depths=[4, 12, 8], dims=[96, 192, 384],
                 drop_path_rate=0.2, layer_scale_init_value=1e-6, head_dropout=0.4):
        super().__init__()
        
        self.stem = nn.Sequential(
            CoordConv2d(1, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        
        for i in range(3):
            stage = nn.Sequential(
                *[ConvNeXtBlock(dim=dims[i], drop_path=dp_rates[cur + j], 
                                layer_scale_init_value=layer_scale_init_value) 
                  for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]
        
        self.downsample_layers = nn.ModuleList()
        for i in range(2):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)
        
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
        x = self.stem(x)
        
        for i in range(3):
            x = self.stages[i](x)
            if i < 2:
                x = self.downsample_layers[i](x)
        
        x = self.norm(x)
        x = x.mean([-2, -1])
        features = x
        x = self.head_dropout(x)
        x = self.head(x)
        
        if return_features:
            return x, features
        return x


# ============================================================================
# Focal-LDAM Loss: Combines Focal Loss with LDAM
# ============================================================================

class FocalLDAMLoss(nn.Module):
    """
    Focal-LDAM Loss: Combines focal loss focus on hard examples with LDAM margins
    
    Focal component: Down-weights easy examples (high confidence)
    LDAM component: Larger margins for minority classes
    """
    def __init__(self, cls_num_list, max_m=0.5, gamma=2.0, s=30, label_smooth=0.05):
        super(FocalLDAMLoss, self).__init__()
        self.gamma = gamma  # Focal loss focusing parameter
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        self.m_list = torch.FloatTensor(m_list)
        self.s = s
        self.label_smooth = label_smooth
        self.num_classes = len(cls_num_list)
        
    def forward(self, x, target):
        # Move margin list to same device
        if self.m_list.device != x.device:
            self.m_list = self.m_list.to(x.device)
        
        batch_size = x.size(0)
        
        # Apply LDAM margin
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.float()
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        
        # Apply scaling
        output = output * self.s
        
        # Compute softmax probabilities
        log_probs = F.log_softmax(output, dim=1)
        probs = torch.exp(log_probs)
        
        # Get target probabilities for focal weight
        target_probs = probs.gather(1, target.view(-1, 1))
        
        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - target_probs).pow(self.gamma)
        
        # Label smoothing
        if self.label_smooth > 0:
            one_hot = torch.zeros_like(output).scatter_(1, target.view(-1, 1), 1)
            one_hot = one_hot * (1 - self.label_smooth) + (1 - one_hot) * self.label_smooth / (self.num_classes - 1)
            log_probs = F.log_softmax(output, dim=1)
            loss = -(one_hot * log_probs).sum(dim=1)
        else:
            loss = F.cross_entropy(output, target, reduction='none')
        
        # Apply focal weight
        loss = focal_weight.squeeze() * loss
        
        return loss.mean()


# ============================================================================
# Mixup and Training
# ============================================================================

def mixup_data(x, y, alpha=0.4):
    """Stronger mixup with alpha=0.4"""
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
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class Trainer:
    """Trainer with Focal-LDAM and earlier DRW"""
    def __init__(self, model, train_loader, val_loader, test_loader, 
                 device, class_counts, lr=1e-4, weight_decay=2e-4, resume_checkpoint=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        
        # Focal-LDAM loss
        self.criterion = FocalLDAMLoss(
            cls_num_list=class_counts,
            max_m=0.5,
            gamma=2.0,  # Focal loss gamma
            s=30,
            label_smooth=0.05
        ).to(device)
        
        # DRW weights (earlier start at epoch 40)
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, class_counts)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * len(weights)
        self.drw_weights = torch.FloatTensor(weights).to(device)
        
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=10,
            min_lr=1e-6
        )
        
        self.best_val_f1 = 0
        self.patience_counter = 0
        self.patience = 20
        
        self.train_losses = []
        self.val_losses = []
        self.val_f1s = []
        
        if resume_checkpoint:
            self.load_checkpoint(resume_checkpoint)
    
    def load_checkpoint(self, checkpoint_path):
        print(f"\nLoading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'train_losses' in checkpoint:
            self.train_losses = checkpoint['train_losses']
            self.val_losses = checkpoint['val_losses']
            self.val_f1s = checkpoint['val_f1s']
            self.best_val_f1 = max(self.val_f1s) if self.val_f1s else 0
        
        print(f"Resumed from checkpoint. Best val F1: {self.best_val_f1:.4f}")
        
    def train_epoch(self, epoch, total_epochs, use_drw=False, use_mixup=True):
        self.model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Stronger mixup (40% of batches, alpha=0.4)
            if use_mixup and np.random.rand() < 0.4:
                data, targets_a, targets_b, lam = mixup_data(data, target, alpha=0.4)
                
                self.optimizer.zero_grad()
                outputs = self.model(data)
                
                if use_drw:
                    loss_a = F.cross_entropy(outputs, targets_a, weight=self.drw_weights)
                    loss_b = F.cross_entropy(outputs, targets_b, weight=self.drw_weights)
                    loss = lam * loss_a + (1 - lam) * loss_b
                else:
                    loss = mixup_criterion(self.criterion, outputs, targets_a, targets_b, lam)
            else:
                self.optimizer.zero_grad()
                outputs = self.model(data)
                
                if use_drw:
                    loss = F.cross_entropy(outputs, target, weight=self.drw_weights)
                else:
                    loss = self.criterion(outputs, target)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % 500 == 0:
                print(f"  Batch {batch_idx+1}/{len(self.train_loader)}, Loss: {loss.item():.4f}, "
                      f"DRW: {use_drw}, Mixup: {use_mixup}")
        
        return total_loss / len(self.train_loader)
    
    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                loss = F.cross_entropy(outputs, target)
                
                total_loss += loss.item()
                preds = outputs.argmax(dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        avg_loss = total_loss / len(data_loader)
        macro_f1 = f1_score(all_targets, all_preds, average='macro')
        
        return avg_loss, macro_f1, np.array(all_preds), np.array(all_targets)
    
    def train(self, num_epochs, checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        start_epoch = len(self.train_losses) + 1
        
        print("\n" + "="*60)
        print("Starting Training...")
        print("="*60)
        print(f"Total epochs: {num_epochs}")
        print(f"Starting from epoch {start_epoch}")
        print(f"DRW starts at epoch 40")  # Earlier DRW
        print(f"Stronger mixup: alpha=0.4, 40% batches")
        print(f"Higher regularization: drop_path=0.2, dropout=0.4")
        
        for epoch in range(start_epoch, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("="*60)
            
            # Earlier DRW start (epoch 40 instead of 50)
            use_drw = epoch >= 40
            
            train_loss = self.train_epoch(epoch, num_epochs, use_drw=use_drw, use_mixup=True)
            val_loss, val_f1, _, _ = self.evaluate(self.val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_f1s.append(val_f1)
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Macro F1: {val_f1:.4f}")
            print(f"Current learning rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save checkpoints
            if epoch % 10 == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                    'val_f1s': self.val_f1s,
                }, checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path}")
            
            # Best model
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.patience_counter = 0
                
                best_model_path = os.path.join(checkpoint_dir, 'best_model.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_f1': val_f1,
                }, best_model_path)
                print(f"*** New best model saved! Val F1: {val_f1:.4f} ***")
            else:
                self.patience_counter += 1
                print(f"No improvement. Patience: {self.patience_counter}/{self.patience}")
            
            # Latest checkpoint
            latest_checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'val_f1s': self.val_f1s,
            }, latest_checkpoint_path)
            
            # Learning rate scheduling
            self.scheduler.step(val_f1)
            
            # Early stopping
            if self.patience_counter >= self.patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break
        
        print("\n" + "="*60)
        print("Training completed!")
        print(f"Best validation F1: {self.best_val_f1:.4f}")
        print("="*60)


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    if len(sys.argv) < 2:
        print("Usage: python STM_C3NeXt2.py <mode>")
        print("  mode 0: standard (full dataset)")
        print("  mode 1: downsample non-tonal speech to 100k")
        sys.exit(1)
    
    mode = int(sys.argv[1])
    
    resume_dir = None
    if '--resume' in sys.argv:
        resume_idx = sys.argv.index('--resume')
        if resume_idx + 1 < len(sys.argv):
            resume_dir = sys.argv[resume_idx + 1]
    
    if mode == 0:
        ds_nontonal_speech = False
        directory = "model/STM/C3NeXt2_corpora_categories/standard"
    elif mode == 1:
        ds_nontonal_speech = True
        directory = "model/STM/C3NeXt2_corpora_categories/downsample"
    else:
        print("Invalid mode. Use 0 or 1.")
        sys.exit(1)
    
    if resume_dir:
        checkpoint_dir = resume_dir
        print(f"Resuming training from: {checkpoint_dir}")
    else:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        checkpoint_dir = os.path.join(directory, f"ckpt/{timestamp}")
    
    print(f"Checkpoint directory: {checkpoint_dir}")
    
    print("\n" + "="*60)
    print("Loading and preparing data...")
    print("="*60)
    
    data_prep = prepData_STM_CoordConv(ds_nontonal_speech=ds_nontonal_speech)
    train_dataset, val_dataset, test_dataset, class_counts = data_prep.prepare_datasets()
    
    batch_size = 256
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    
    print(f"\nDataLoaders created with batch_size={batch_size}")
    
    print("\n" + "="*60)
    print("Creating C3NeXt2 (Focal-LDAM + Deeper)...")
    print("="*60)
    
    num_classes = 6
    model = C3NeXt2(
        num_classes=num_classes,
        depths=[4, 12, 8],
        dims=[96, 192, 384],
        drop_path_rate=0.2,
        layer_scale_init_value=1e-6,
        head_dropout=0.4
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Architecture: C3NeXt2 - Deeper variant")
    print(f"Blocks: 24 ConvNeXt blocks [4, 12, 8]")
    print(f"Channels: [96, 192, 384]")
    print(f"Loss: Focal-LDAM (gamma=2.0)")
    print(f"DRW start: Epoch 40 (earlier)")
    print(f"Mixup: alpha=0.4 (stronger)")
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        class_counts=class_counts,
        lr=1e-4,
        weight_decay=2e-4,
        resume_checkpoint=None
    )
    
    if resume_dir:
        resume_checkpoint = os.path.join(resume_dir, 'latest_checkpoint.pt')
        if os.path.exists(resume_checkpoint):
            trainer.load_checkpoint(resume_checkpoint)
        else:
            print(f"Warning: Resume checkpoint not found at {resume_checkpoint}")
            print("Starting training from scratch")
    
    num_epochs = 100
    trainer.train(num_epochs=num_epochs, checkpoint_dir=checkpoint_dir)
    
    print("\n" + "="*60)
    print("Evaluating on test set...")
    print("="*60)
    
    checkpoint = torch.load(os.path.join(checkpoint_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_f1, test_preds, test_targets = trainer.evaluate(test_loader)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Macro F1: {test_f1:.4f}")
    
    print("\nClassification Report:")
    target_names = ['speech:non-tonal', 'speech:tonal', 'music:vocal', 
                   'music:non-vocal', 'env:urban', 'env:wildlife']
    print(classification_report(test_targets, test_preds, target_names=target_names))
    
    np.save(os.path.join(checkpoint_dir, 'test_predictions.npy'), test_preds)
    np.save(os.path.join(checkpoint_dir, 'test_targets.npy'), test_targets)
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)
