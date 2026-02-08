#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STM Classification - Test-Time Augmentation (TTA) Evaluation Script
For V2.6 Model (STM_CoordConvLDAM_preIN6)

This script applies Test-Time Augmentation to improve inference accuracy
by averaging predictions from multiple augmented versions of each test sample.

TTA Augmentations:
1. Original (no augmentation)
2. Time flip: Reverse along temporal axis
3. Frequency shift +2: Cyclic shift along spectral mod axis
4. Frequency shift -2: Cyclic shift in opposite direction
5. Time shift +5: Cyclic shift along temporal axis

Expected improvement: +0.5-1.5% Macro F1 over single-inference baseline.

Usage:
    python STM_CoordConvLDAM_preIN6_TTA.py [checkpoint_path]
    
Example:
    python STM_CoordConvLDAM_preIN6_TTA.py model/STM/CoordConvLDAM_preIN6_corpora_categories/standard/ckpt/2026-02-07_11-08
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import warnings
import ssl

warnings.filterwarnings('ignore')

# Fix SSL certificate verification issues in HPC environments
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


# ============================================================================
# Data Preparation (Same as V2.6)
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
            corpus_env_list = ['MacaulayLibrary', 'MacaulayLibrary_aug', 'SONYC', 'SONYC_aug']
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
            fname = root_folder + 'STM_output/corpSTMnpy/' + corp.replace('/', '-') + '_STMall.npy'
            temp_STM = np.load(fname)
            if STM_all is None:
                STM_all = temp_STM
            else:
                STM_all = np.vstack((STM_all, temp_STM))
        
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
        
        # Split data
        train_ind = (data_split < 8).values
        val_ind = (data_split == 8).values
        test_ind = (data_split == 9).values
        
        # Compute class frequencies for LDAM
        train_labels = target[train_ind].values
        class_counts = np.bincount(train_labels, minlength=6)
        
        return STM_all, target.values, train_ind, val_ind, test_ind, class_counts
    
    def prepare_test_dataset(self):
        """Prepare PyTorch test dataset with Difference Map (2-channel) preprocessing"""
        STM_all, target, train_ind, val_ind, test_ind, class_counts = self.load_data()
        
        # Reshape from flattened to 2D: (batch, freq, time)
        STM_all_2d = STM_all.reshape(-1, self.n_freq, self.n_time)
        
        # Normalize per sample (CRITICAL: preserves relative energy patterns)
        means = STM_all_2d.mean(axis=(1, 2), keepdims=True)
        stds = STM_all_2d.std(axis=(1, 2), keepdims=True)
        STM_all_2d = (STM_all_2d - means) / (stds + 1e-8)
        
        # ===== Difference Map Preprocessing =====
        STM_flipped = np.flip(STM_all_2d, axis=1).copy()
        STM_symmetric = (STM_all_2d + STM_flipped) / 2.0
        STM_asymmetric = (STM_all_2d - STM_flipped) / 2.0
        STM_all_2ch = np.stack([STM_symmetric, STM_asymmetric], axis=1)
        
        # Convert to PyTorch tensors (test only)
        X_test = torch.FloatTensor(STM_all_2ch[test_ind])
        y_test = torch.LongTensor(target[test_ind])
        
        test_dataset = TensorDataset(X_test, y_test)
        
        print(f"\nTest Dataset Shape: {X_test.shape}")
        print(f"Test samples: {len(X_test)}")
        
        return test_dataset, class_counts


# ============================================================================
# CoordConv Layer
# ============================================================================

class CoordConv2d(nn.Module):
    """CoordConv: Adds coordinate channels to convolution input."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, bias=True, dilation=1, groups=1):
        super(CoordConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels + 2, out_channels, kernel_size, 
                             stride=stride, padding=padding, bias=bias,
                             dilation=dilation, groups=groups)
        
    def forward(self, x):
        batch_size, _, height, width = x.size()
        
        y_coords = torch.linspace(-1, 1, height, device=x.device)
        x_coords = torch.linspace(-1, 1, width, device=x.device)
        y_coords = y_coords.view(1, 1, height, 1).expand(batch_size, 1, height, width)
        x_coords = x_coords.view(1, 1, 1, width).expand(batch_size, 1, height, width)
        
        x_with_coords = torch.cat([x, x_coords, y_coords], dim=1)
        return self.conv(x_with_coords)


# ============================================================================
# Attention Mechanisms
# ============================================================================

class CoordinateAttention(nn.Module):
    """Coordinate Attention (Hou et al., CVPR 2021)"""
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
        
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        
        return x * a_h * a_w


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation (Hu et al., CVPR 2018)"""
    def __init__(self, in_channels, reduction=16):
        super(SqueezeExcitation, self).__init__()
        hidden_channels = max(1, in_channels // reduction)
        
        self.fc1 = nn.Linear(in_channels, hidden_channels, bias=False)
        self.fc2 = nn.Linear(hidden_channels, in_channels, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        y = y.view(b, c, 1, 1)
        return x * y


# ============================================================================
# BasicBlock with Attention
# ============================================================================

class BasicBlockWithAttention(nn.Module):
    """ResNet BasicBlock with optional attention and dropout."""
    expansion = 1
    
    def __init__(self, block, attention_type='CA', dropout=0.05):
        super(BasicBlockWithAttention, self).__init__()
        self.block = block
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None
        
        out_channels = block.conv2.out_channels
        
        if attention_type == 'CA':
            self.attention = CoordinateAttention(out_channels)
        elif attention_type == 'SE':
            self.attention = SqueezeExcitation(out_channels)
        else:
            self.attention = None
    
    def forward(self, x):
        out = self.block(x)
        
        if self.attention is not None:
            out = self.attention(out)
        
        if self.dropout is not None:
            out = self.dropout(out)
        
        return out


# ============================================================================
# Pretrained ResNet-18 with STM Adaptations (V2.6)
# ============================================================================

class PretrainedSTMResNet18(nn.Module):
    """ImageNet-pretrained ResNet-18 adapted for STM classification (V2.6)"""
    
    def __init__(self, num_classes=6, dropout=0.4, block_dropout=0.05, use_pretrained=True):
        super(PretrainedSTMResNet18, self).__init__()
        
        from torchvision import models
        from torchvision.models import ResNet18_Weights
        
        # Load pretrained ResNet-18
        pretrained_model = None
        weights_loaded = False
        
        if use_pretrained:
            try:
                pretrained_model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
                weights_loaded = True
            except Exception:
                pretrained_model = models.resnet18(weights=None)
        else:
            pretrained_model = models.resnet18(weights=None)
        
        # Stem: CoordConv
        self.conv1 = CoordConv2d(
            in_channels=2,
            out_channels=64,
            kernel_size=7,
            stride=1,
            padding=3,
            bias=False
        )
        
        # Clone weights if available
        if weights_loaded:
            with torch.no_grad():
                old_weight = pretrained_model.conv1.weight.data
                new_weight = torch.zeros(64, 4, 7, 7)
                rgb_mean = old_weight.mean(dim=1, keepdim=True)
                new_weight[:, 0:1, :, :] = rgb_mean
                new_weight[:, 1:2, :, :] = rgb_mean
                new_weight[:, 2:4, :, :] = 0.01 * torch.randn(64, 2, 7, 7)
                self.conv1.conv.weight.data = new_weight
        
        self.bn1 = pretrained_model.bn1
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.Identity()  # Preserve resolution
        
        # ResNet layers with attention
        self.layer1 = nn.Sequential(*[
            BasicBlockWithAttention(block, attention_type='CA', dropout=block_dropout)
            for block in pretrained_model.layer1
        ])
        self.layer2 = nn.Sequential(*[
            BasicBlockWithAttention(block, attention_type='CA', dropout=block_dropout)
            for block in pretrained_model.layer2
        ])
        self.layer3 = nn.Sequential(*[
            BasicBlockWithAttention(block, attention_type='SE', dropout=block_dropout)
            for block in pretrained_model.layer3
        ])
        self.layer4 = nn.Sequential(*[
            BasicBlockWithAttention(block, attention_type='SE', dropout=block_dropout)
            for block in pretrained_model.layer4
        ])
        
        # Three-Scale Feature Fusion (V2.6)
        self.multi_scale_fusion = nn.Sequential(
            nn.Conv2d(128 + 256 + 512, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Global pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification head with dropout
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512, num_classes)  # Multi-scale output is 512 features
        
        print(f"✓ Loaded PretrainedSTMResNet18 V2.6 ({num_classes} classes)")
    
    def forward(self, x, return_features=False):
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # ResNet blocks
        x = self.layer1(x)
        feat_layer2 = self.layer2(x)
        feat_layer3 = self.layer3(feat_layer2)
        feat_layer4 = self.layer4(feat_layer3)
        
        # Three-Scale Feature Fusion
        feat_layer2_down = F.interpolate(
            feat_layer2, 
            size=feat_layer3.shape[-2:],
            mode='bilinear', 
            align_corners=False
        )
        feat_layer4_up = F.interpolate(
            feat_layer4, 
            size=feat_layer3.shape[-2:],
            mode='bilinear', 
            align_corners=False
        )
        feat_concat = torch.cat([feat_layer2_down, feat_layer3, feat_layer4_up], dim=1)
        feat_fused = self.multi_scale_fusion(feat_concat)
        
        # Pool and classify
        x = self.avgpool(feat_fused)
        x = torch.flatten(x, 1)
        
        # Classification head
        feat = x  # Save features for analysis
        x = self.dropout(x)
        x = self.fc(x)
        
        if return_features:
            return x, feat
        return x


# ============================================================================
# Test-Time Augmentation (TTA)
# ============================================================================

class TTAPredictor:
    """
    Test-Time Augmentation Predictor for STM classification.
    
    Applies multiple augmentations at inference time and averages
    the predictions (soft voting on logits/probabilities).
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
            print(f"    • {name}")
    
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
# Main Execution
# ============================================================================

if __name__ == "__main__":
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Default checkpoint path
    default_checkpoint = "model/STM/CoordConvLDAM_preIN6_corpora_categories/standard/ckpt/2026-02-07_11-08"
    
    # Parse command line arguments
    # Ignore numeric arguments (mode flags from SLURM scripts)
    if len(sys.argv) > 1 and not sys.argv[1].isdigit():
        checkpoint_dir = sys.argv[1]
    else:
        checkpoint_dir = default_checkpoint
        print(f"\nUsing default checkpoint: {checkpoint_dir}")
    
    checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
    
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        print("Please specify a valid checkpoint directory.")
        sys.exit(1)
    
    print(f"\nCheckpoint: {checkpoint_path}")
    
    # Prepare data
    print("\n" + "="*60)
    print("Loading test data...")
    print("="*60)
    
    data_prep = prepData_STM_CoordConv(ds_nontonal_speech=False)
    test_dataset, class_counts = data_prep.prepare_test_dataset()
    
    batch_size = 256
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    
    # Create model
    print("\n" + "="*60)
    print("Creating model and loading weights...")
    print("="*60)
    
    num_classes = 6
    model = PretrainedSTMResNet18(num_classes=num_classes, dropout=0.4, block_dropout=0.05)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✓ Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"  Best validation F1: {checkpoint.get('val_f1', checkpoint.get('best_val_f1', 'unknown')):.4f}")
    
    # ========== Baseline Evaluation (No TTA) ==========
    print("\n" + "="*60)
    print("Baseline Evaluation (No TTA)...")
    print("="*60)
    
    all_preds_baseline = []
    all_targets_baseline = []
    
    model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds_baseline.extend(preds)
            all_targets_baseline.extend(targets.numpy())
    
    all_preds_baseline = np.array(all_preds_baseline)
    all_targets_baseline = np.array(all_targets_baseline)
    
    baseline_f1 = f1_score(all_targets_baseline, all_preds_baseline, average='macro')
    print(f"\nBaseline Test Macro F1: {baseline_f1:.4f}")
    
    # ========== TTA Evaluation ==========
    print("\n" + "="*60)
    print("TTA Evaluation (5 augmentations)...")
    print("="*60)
    
    tta_predictor = TTAPredictor(model, device, n_augmentations=5)
    all_preds_tta, all_targets_tta, all_probs_tta = tta_predictor.evaluate(test_loader)
    
    tta_f1 = f1_score(all_targets_tta, all_preds_tta, average='macro')
    
    print(f"\nTTA Test Macro F1: {tta_f1:.4f}")
    print(f"Improvement over baseline: {(tta_f1 - baseline_f1)*100:.2f}%")
    
    # Detailed classification report
    print("\n" + "="*60)
    print("Classification Report (TTA)")
    print("="*60)
    
    target_names = ['speech:non-tonal', 'speech:tonal', 'music:vocal', 
                   'music:non-vocal', 'env:urban', 'env:wildlife']
    print(classification_report(all_targets_tta, all_preds_tta, target_names=target_names))
    
    # Confusion matrix
    print("\nConfusion Matrix (TTA):")
    cm = confusion_matrix(all_targets_tta, all_preds_tta)
    print(cm)
    
    # Per-class comparison
    print("\n" + "="*60)
    print("Per-Class F1 Comparison")
    print("="*60)
    
    from sklearn.metrics import f1_score as f1_per_class
    baseline_per_class = f1_per_class(all_targets_baseline, all_preds_baseline, average=None)
    tta_per_class = f1_per_class(all_targets_tta, all_preds_tta, average=None)
    
    print(f"{'Class':<20} {'Baseline':>10} {'TTA':>10} {'Δ':>10}")
    print("-" * 50)
    for i, name in enumerate(target_names):
        delta = tta_per_class[i] - baseline_per_class[i]
        sign = "+" if delta >= 0 else ""
        print(f"{name:<20} {baseline_per_class[i]:>10.4f} {tta_per_class[i]:>10.4f} {sign}{delta:>9.4f}")
    print("-" * 50)
    print(f"{'MACRO AVG':<20} {baseline_f1:>10.4f} {tta_f1:>10.4f} {'+' if tta_f1 >= baseline_f1 else ''}{tta_f1 - baseline_f1:>9.4f}")
    
    # Save results
    output_dir = checkpoint_dir
    np.save(os.path.join(output_dir, 'test_predictions_tta.npy'), all_preds_tta)
    np.save(os.path.join(output_dir, 'test_probabilities_tta.npy'), all_probs_tta)
    
    print(f"\n✓ Results saved to {output_dir}")
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)
