#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STM Classification with Vision Mamba (Vim) - Variant 2: Lighter/Faster Model
Phase 2: The Modern Sequence Model

This is a lighter variant optimized for faster training and reduced overfitting:
- Reduced model dimension: d_model=128 (vs 192 in baseline)
- Fewer layers: depth=8 (vs 12 in baseline)
- Smaller SSM state: d_state=12 (vs 16 in baseline)
- Same batch size: 64
- Expected training time: ~20-25 min/epoch (vs 30-35 min for baseline)
- Total parameters: ~3.5M (vs ~8M in baseline)

Goal: Establish a fast baseline with reduced capacity for comparison
"""

import os
import sys
import warnings
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import f1_score, classification_report
from pathlib import Path

warnings.filterwarnings('ignore')


# ============================================================================
# Symmetric STM Processing (from STMasm_enhanced5.py)
# ============================================================================

def process_symmetric_stm(stm_data):
    """
    Process STM data to exploit up/down-sweep symmetry.
    
    Input: (batch, freq_bands, mod_rates=121)
    Output: (batch, freq_bands, mod_rates=61)
    
    Steps:
    A. Separate negative rates [0:60] and positive rates [61:121] along modulation axis
    B. Flip negative chunk to align modulation rates
    C. Average flipped negative and positive chunks
    D. Concatenate DC (index 60) at position 0
    
    Modulation rate mapping:
    - Original: -15 Hz (idx 0) ... 0 Hz (idx 60) ... +15 Hz (idx 120)
    - Output: 0 Hz (idx 0) ... +15 Hz (idx 60)
    
    Speed benefit: 2420 tokens → 1220 tokens = 2× faster (O(L) complexity)
    """
    # Step A: Separate chunks along modulation rate dimension (last dim)
    negative_chunk = stm_data[:, :, 0:60]   # -15 Hz to -0.25 Hz
    dc_component = stm_data[:, :, 60:61]    # 0 Hz
    positive_chunk = stm_data[:, :, 61:121] # +0.25 Hz to +15 Hz
    
    # Step B: Flip negative chunk (reverse modulation rate axis)
    negative_flipped = torch.flip(negative_chunk, dims=[2])
    
    # Step C: Average aligned chunks
    averaged_chunk = (negative_flipped + positive_chunk) / 2.0
    
    # Step D: Concatenate DC at the beginning
    output = torch.cat([dc_component, averaged_chunk], dim=2)
    
    return output


class SymmetricSTMDataset(Dataset):
    """Wrapper dataset that applies symmetric STM processing"""
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        data, label = self.base_dataset[idx]
        # data shape: (2420,) - flattened from (20, 121)
        # Reshape to (20, 121) for symmetric processing
        data = data.reshape(20, 121)
        # Add batch dimension for processing
        data = data.unsqueeze(0)  # (1, 20, 121)
        data = process_symmetric_stm(data)
        data = data.squeeze(0)    # (20, 61)
        # Flatten for sequence processing
        data = data.reshape(-1)   # (1220,)
        return data, label


# Try to import Mamba - provide helpful error if not available
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    print("WARNING: mamba-ssm not installed. Install with:")
    print("  pip install mamba-ssm causal-conv1d>=1.2.0")
    print("\nUsing placeholder Mamba implementation for structure demonstration.")
    MAMBA_AVAILABLE = False


# ============================================================================
# Data Preparation
# ============================================================================

class prepData_STM_Mamba:
    """
    Data preparation for Vision Mamba model.
    Keeps STM features flattened as 1220-dimensional vectors (after symmetric processing).
    """
    def __init__(self, addAug=False, ds_nontonal_speech=False):
        self.addAug = addAug
        self.ds_nontonal_speech = ds_nontonal_speech
        
        # STM dimensions (after symmetric processing)
        self.n_freq = 20
        self.n_time = 61  # Reduced from 121 via symmetric processing
        self.seq_len = self.n_freq * self.n_time  # 1220
        
    def corpora_list(self, addAug=False):
        """Generate list of all corpora"""
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
            corpus_env_list = ['SONYC', 'MacaulayLibrary', 'SONYC_augmented']
        else:
            corpus_env_list = ['SONYC', 'MacaulayLibrary']
        
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
            filename = root_folder + 'STM_output/corpSTMnpy/' + corp.replace('/', '-') + '_STMall.npy'
            if STM_all is None:
                STM_all = np.load(filename)
            else:
                STM_all = np.vstack((STM_all, np.load(filename)))
            print(f"Loaded: {filename}, shape: {np.load(filename).shape}")
        
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
        
        # Map categories
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
            num_samples = 100000
            indices_target_0 = target.index[target == 0].to_numpy()
            
            if len(indices_target_0) < num_samples:
                raise ValueError(f"Not enough rows with target == 0 to sample {num_samples} rows.")
            
            np.random.seed(23)
            sampled_indices = np.random.choice(indices_target_0, size=num_samples, replace=False)
            
            mask = np.ones(len(target), dtype=bool)
            mask[indices_target_0] = False
            mask[sampled_indices] = True
            
            STM_all = STM_all[mask, :]
            data_split = data_split[mask].reset_index(drop=True)
            target = target[mask].reset_index(drop=True)
        
        # Split data
        train_ind = (data_split < 8).values
        val_ind = (data_split == 8).values
        test_ind = (data_split == 9).values
        
        # Compute class frequencies for Balanced Softmax
        train_labels = target[train_ind].values.astype(np.int64)
        class_counts = np.bincount(train_labels, minlength=6)
        class_freq = class_counts / class_counts.sum()
        
        print(f"\nDataset Statistics:")
        print(f"Total samples: {len(STM_all)}")
        print(f"Train samples: {sum(train_ind)}")
        print(f"Val samples: {sum(val_ind)}")
        print(f"Test samples: {sum(test_ind)}")
        print(f"Sequence length: {self.seq_len}")
        print(f"\nClass Distribution (Training):")
        for i, count in enumerate(class_counts):
            print(f"  Class {i}: {count} samples ({100*count/len(train_labels):.2f}%)")
        
        return STM_all, target.values, train_ind, val_ind, test_ind, class_freq
    
    def prepare_datasets(self):
        """Prepare PyTorch datasets - keep flattened for sequence processing"""
        STM_all, target, train_ind, val_ind, test_ind, class_freq = self.load_data()
        
        # Normalize per sample (preserve relative patterns)
        means = STM_all.mean(axis=1, keepdims=True)
        stds = STM_all.std(axis=1, keepdims=True)
        STM_all_norm = (STM_all - means) / (stds + 1e-8)
        
        # Convert to PyTorch tensors (batch, seq_len)
        X_train = torch.FloatTensor(STM_all_norm[train_ind])
        y_train = torch.LongTensor(target[train_ind].astype(np.int64))
        
        X_val = torch.FloatTensor(STM_all_norm[val_ind])
        y_val = torch.LongTensor(target[val_ind].astype(np.int64))
        
        X_test = torch.FloatTensor(STM_all_norm[test_ind])
        y_test = torch.LongTensor(target[test_ind].astype(np.int64))
        
        # Create datasets
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)
        
        print(f"\nPyTorch Dataset Shapes:")
        print(f"Train: {X_train.shape}")
        print(f"Val: {X_val.shape}")
        print(f"Test: {X_test.shape}")
        
        return train_dataset, val_dataset, test_dataset, class_freq


# ============================================================================
# Vision Mamba Block
# ============================================================================

class VimBlock(nn.Module):
    """
    Vision Mamba Block with bidirectional scanning
    
    From "Vision Mamba: Efficient Visual Representation Learning with 
    Bidirectional State Space Model" (Zhu et al., ICML 2024)
    
    Key features:
    - Bidirectional SSM scanning (forward + backward)
    - Layer normalization
    - Residual connections
    - DropPath regularization
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, drop_path=0.0):
        super(VimBlock, self).__init__()
        
        self.norm = nn.LayerNorm(d_model)
        
        if MAMBA_AVAILABLE:
            self.mamba_forward = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            self.mamba_backward = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        else:
            # Placeholder MLP for structure demonstration
            self.mamba_forward = nn.Sequential(
                nn.Linear(d_model, d_model * expand),
                nn.GELU(),
                nn.Linear(d_model * expand, d_model)
            )
            self.mamba_backward = nn.Sequential(
                nn.Linear(d_model, d_model * expand),
                nn.GELU(),
                nn.Linear(d_model * expand, d_model)
            )
        
        self.drop_path = nn.Identity() if drop_path == 0 else nn.Dropout(drop_path)
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            out: (batch, seq_len, d_model)
        """
        # Residual connection
        residual = x
        
        # Normalize
        x = self.norm(x)
        
        # Forward scan
        x_forward = self.mamba_forward(x)
        
        # Backward scan (flip sequence)
        x_backward = torch.flip(x, dims=[1])
        x_backward = self.mamba_backward(x_backward)
        x_backward = torch.flip(x_backward, dims=[1])
        
        # Combine bidirectional features
        x = x_forward + x_backward
        
        # Apply drop path and residual
        x = residual + self.drop_path(x)
        
        return x


# ============================================================================
# Vision Mamba Model - Lighter Variant
# ============================================================================

class VisionMamba(nn.Module):
    """
    Vision Mamba for STM Classification - Lighter/Faster Variant
    
    Architecture:
    1. Patch Embedding (1x1 patches = each bin is a token)
    2. Positional Embedding (learnable, absolute positions)
    3. Stack of Vim Blocks (bidirectional SSM) - REDUCED TO 8 LAYERS
    4. Global Average Pooling
    5. Classification Head
    
    Changes from baseline:
    - d_model: 128 (vs 192)
    - depth: 8 (vs 12)
    - d_state: 12 (vs 16)
    - Total params: ~3.5M (vs ~8M)
    """
    def __init__(self, seq_len=1220, num_classes=6, d_model=128, depth=8,
                 d_state=12, d_conv=4, expand=2, drop_path_rate=0.1, dropout=0.1):
        super(VisionMamba, self).__init__()
        
        self.seq_len = seq_len
        self.d_model = d_model
        
        # Patch embedding: each modulation bin becomes a token
        # Input: (batch, seq_len=1220) -> (batch, seq_len, d_model)
        self.patch_embed = nn.Linear(1, d_model)
        
        # Learnable positional embeddings (CRITICAL for position awareness)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        # Stack of Vim blocks
        self.blocks = nn.ModuleList([
            VimBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                drop_path=dpr[i]
            )
            for i in range(depth)
        ])
        
        # Final normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Classification head
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len=1220) - after symmetric STM processing
        Returns:
            logits: (batch, num_classes)
        """
        batch_size = x.size(0)
        
        # Reshape for embedding: (batch, seq_len) -> (batch, seq_len, 1)
        x = x.unsqueeze(-1)
        
        # Patch embedding: (batch, seq_len, 1) -> (batch, seq_len, d_model)
        x = self.patch_embed(x)
        
        # Add positional embeddings
        x = x + self.pos_embed
        
        # Apply Vim blocks
        for block in self.blocks:
            x = block(x)
        
        # Final norm
        x = self.norm(x)
        
        # Global average pooling
        x = x.mean(dim=1)  # (batch, d_model)
        
        # Classification
        logits = self.head(x)
        
        return logits


# ============================================================================
# Balanced Softmax Loss
# ============================================================================

class BalancedSoftmaxLoss(nn.Module):
    """
    Balanced Softmax Loss for long-tailed recognition
    
    From "Balanced Meta-Softmax for Long-Tailed Visual Recognition"
    (Ren et al., NeurIPS 2020)
    
    Adjusts logits by log class frequency to compensate for imbalance:
        L_balanced = CrossEntropy(logits + log(class_freq), target)
    """
    def __init__(self, class_freq):
        super(BalancedSoftmaxLoss, self).__init__()
        self.class_freq = torch.FloatTensor(class_freq)
        
    def forward(self, logits, target):
        """Compute balanced softmax loss"""
        self.class_freq = self.class_freq.to(logits.device)
        adjusted_logits = logits + torch.log(self.class_freq + 1e-8)
        return nn.CrossEntropyLoss()(adjusted_logits, target)


# ============================================================================
# Trainer
# ============================================================================

class Trainer:
    """Training manager for Vision Mamba with Balanced Softmax"""
    def __init__(self, model, train_loader, val_loader, test_loader, 
                 device, class_freq, lr=1e-4, weight_decay=1e-4):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        
        # Loss and optimizer
        self.criterion = BalancedSoftmaxLoss(class_freq)
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50)
        
        # Tracking
        self.best_val_f1 = 0.0
        self.start_epoch = 0
        self.train_losses = []
        self.val_losses = []
        self.val_f1s = []
        
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint to resume training"""
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            return False
        
        print(f"Loading checkpoint from: {checkpoint_path}")
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
        
        if 'train_losses' in checkpoint:
            self.train_losses = checkpoint['train_losses']
        if 'val_losses' in checkpoint:
            self.val_losses = checkpoint['val_losses']
        if 'val_f1s' in checkpoint:
            self.val_f1s = checkpoint['val_f1s']
        
        print(f"✓ Resumed from epoch {self.start_epoch}, Best Val F1: {self.best_val_f1:.4f}")
        return True
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Collect predictions for F1 score
            preds = output.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
        
        avg_loss = total_loss / len(self.train_loader)
        macro_f1 = f1_score(all_targets, all_preds, average='macro')
        
        return avg_loss, macro_f1
    
    def evaluate(self, data_loader):
        """Evaluate model"""
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
                
                preds = output.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        avg_loss = total_loss / len(data_loader)
        macro_f1 = f1_score(all_targets, all_preds, average='macro')
        
        return avg_loss, macro_f1, np.array(all_preds), np.array(all_targets)
    
    def train(self, num_epochs, checkpoint_dir):
        """Full training loop"""
        print("\nStarting training...")
        print(f"Total epochs: {num_epochs}")
        print(f"Starting from epoch {self.start_epoch + 1}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        
        for epoch in range(self.start_epoch, num_epochs):
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss, val_f1, _, _ = self.evaluate(self.val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Save best model
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_f1': val_f1,
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                    'val_f1s': self.val_f1s,
                }
                torch.save(checkpoint, os.path.join(checkpoint_dir, 'best_model.pt'))
            
            # Save periodic checkpoints
            if (epoch + 1) % 5 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_f1': val_f1,
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                    'val_f1s': self.val_f1s,
                }, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt'))
                print(f"✓ Saved checkpoint at epoch {epoch+1}")
            
            # Always save latest
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'val_f1': val_f1,
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'val_f1s': self.val_f1s,
            }, os.path.join(checkpoint_dir, 'latest_checkpoint.pt'))
            
            # Print progress
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{num_epochs} - "
                      f"Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, "
                      f"Val F1: {val_f1:.4f}, "
                      f"Best F1: {self.best_val_f1:.4f}")
        
        print(f"\nTraining complete! Best Val F1: {self.best_val_f1:.4f}")


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    
    # Check Mamba availability
    if not MAMBA_AVAILABLE:
        print("\n" + "="*60)
        print("ERROR: mamba-ssm not available")
        print("Install with: pip install mamba-ssm causal-conv1d>=1.2.0")
        print("="*60)
        sys.exit(1)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python STM_ViM2.py <mode> [--resume <checkpoint_dir>]")
        print("  mode 0: standard (no downsampling)")
        print("  mode 1: downsample non-tonal speech")
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
        directory = 'model/STM/ViM2_corpora_categories/standard'
    elif mode == 1:
        ds_nontonal_speech = True
        directory = 'model/STM/ViM2_corpora_categories/downsample'
    else:
        print("Invalid mode. Use 0 or 1.")
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
    
    data_prep = prepData_STM_Mamba(ds_nontonal_speech=ds_nontonal_speech)
    train_dataset, val_dataset, test_dataset, class_freq = data_prep.prepare_datasets()
    
    # Apply symmetric STM processing
    print(f"\nApplying symmetric STM processing...")
    print(f"Original: 20 freq × 121 rates = 2420 tokens")
    print(f"After symmetric processing: 20 freq × 61 rates = 1220 tokens")
    print(f"Speed improvement: 2× faster (O(L) complexity)")
    
    train_dataset = SymmetricSTMDataset(train_dataset)
    val_dataset = SymmetricSTMDataset(val_dataset)
    test_dataset = SymmetricSTMDataset(test_dataset)
    
    # Create data loaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    
    print(f"\nDataLoaders created with batch_size={batch_size}")
    
    # Create model
    print("\n" + "="*60)
    print("Creating Vision Mamba model (Lighter Variant)...")
    print("="*60)
    
    num_classes = 6
    model = VisionMamba(
        seq_len=1220,       # After symmetric STM processing (20×61)
        num_classes=num_classes,
        d_model=128,        # REDUCED from 192
        depth=8,            # REDUCED from 12
        d_state=12,         # REDUCED from 16
        d_conv=4,           # Same as baseline
        expand=2,           # Same as baseline
        drop_path_rate=0.1, # Same as baseline
        dropout=0.1
    )
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size reduction: ~{8000000 / total_params:.1f}× smaller than baseline")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        class_freq=class_freq,
        lr=1e-4,
        weight_decay=1e-4
    )
    
    # Resume from checkpoint if specified
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
    
    # Train model
    num_epochs = 50
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
