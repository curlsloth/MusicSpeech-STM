#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STM Classification with FT-Transformer
Phase 3: The Tabular Approach

This implementation follows the Audio Classification Model Improvement document:
- FT-Transformer (Feature Tokenizer + Transformer) for tabular data
- Symmetric STM processing: exploits up/down-sweep symmetry (121 → 61 bins)
- Per-feature learnable embeddings (1210 unique embeddings)
- Feature-wise self-attention (learns modulation bin co-occurrence)
- Balanced Softmax loss for imbalanced data

Installation Requirements:
    Standard PyTorch only (no additional dependencies)

Key Design Principles:
1. Symmetric processing: Average negative and positive modulation rates (4× speedup)
2. Treat STM as tabular data: 1220 continuous features (20×61)
3. Each feature gets a unique learnable embedding
4. Attention discovers which bins co-occur for each class
5. Avoids spatial inductive biases of CNNs/Mamba

Conceptual Example:
    Feature 250 (4Hz temporal) attends to Feature 800 (2cyc/oct spectral)
    → Model learns "4Hz + 2cyc = Speech" association
"""

import os
import sys
import datetime
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import f1_score, classification_report
import torch.nn.functional as F
import math

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
    
    Speed benefit: 2420 features → 1220 features = 4× faster (O(L²) complexity)
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


# Symmetric processing now applied during data preparation (removed wrapper class)


# ============================================================================
# Data Preparation (Keep flattened - treat as tabular features)
# ============================================================================

class prepData_STM_FTTransformer:
    """
    Data preparation for FT-Transformer model.
    Keeps STM features flattened as 1210-dimensional feature vectors (after symmetric processing).
    """
    def __init__(self, addAug=False, ds_nontonal_speech=False):
        self.addAug = addAug
        self.ds_nontonal_speech = ds_nontonal_speech
        
        # STM dimensions (after symmetric processing)
        self.n_features = 1220  # Treat as 1220 separate features (20×61)
        
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
        train_labels = target[train_ind].values
        class_counts = np.bincount(train_labels, minlength=6)
        class_freq = class_counts / class_counts.sum()
        
        print(f"\nDataset Statistics:")
        print(f"Total samples: {len(STM_all)}")
        print(f"Train samples: {sum(train_ind)}")
        print(f"Val samples: {sum(val_ind)}")
        print(f"Test samples: {sum(test_ind)}")
        print(f"Number of features: {self.n_features}")
        print(f"\nClass Distribution (Training):")
        for i, count in enumerate(class_counts):
            print(f"  Class {i}: {count} samples ({100*count/len(train_labels):.2f}%)")
        
        return STM_all, target.values, train_ind, val_ind, test_ind, class_freq
    
    def prepare_datasets(self):
        """Prepare PyTorch datasets - apply symmetric processing, then flatten for tabular processing"""
        STM_all, target, train_ind, val_ind, test_ind, class_freq = self.load_data()
        
        # Reshape from (n_samples, 2420) to (n_samples, 20, 121) for symmetric processing
        print(f"\nOriginal STM shape: {STM_all.shape}")
        n_samples = STM_all.shape[0]
        STM_all = STM_all.reshape(n_samples, 20, 121)
        
        # Apply symmetric STM processing: (n_samples, 20, 121) -> (n_samples, 20, 61)
        STM_all_tensor = torch.FloatTensor(STM_all)
        STM_all_symmetric = process_symmetric_stm(STM_all_tensor)  # (n_samples, 20, 61)
        STM_all = STM_all_symmetric.numpy()
        print(f"After symmetric processing: {STM_all.shape}")
        
        # Flatten to (n_samples, 1220) for tabular processing
        STM_all = STM_all.reshape(n_samples, -1)
        print(f"After flattening: {STM_all.shape}")
        
        # Normalize per sample
        means = STM_all.mean(axis=1, keepdims=True)
        stds = STM_all.std(axis=1, keepdims=True)
        STM_all_norm = (STM_all - means) / (stds + 1e-8)
        
        # Convert to PyTorch tensors (batch, n_features=1220)
        X_train = torch.FloatTensor(STM_all_norm[train_ind])
        y_train = torch.LongTensor(target[train_ind])
        
        X_val = torch.FloatTensor(STM_all_norm[val_ind])
        y_val = torch.LongTensor(target[val_ind])
        
        X_test = torch.FloatTensor(STM_all_norm[test_ind])
        y_test = torch.LongTensor(target[test_ind])
        
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
# Feature Tokenizer
# ============================================================================

class FeatureTokenizer(nn.Module):
    """
    Feature Tokenizer: Maps each continuous feature to an embedding
    
    From "Revisiting Deep Learning Models for Tabular Data" 
    (Gorishniy et al., NeurIPS 2021)
    
    Key idea: Instead of directly feeding features to MLP (which mixes them
    immediately), assign each feature a learnable embedding that captures
    its identity. Then use attention to discover feature interactions.
    
    For STM (after symmetric processing):
        - Feature 0 (freq=0, time=0) → Embedding vector e_0
        - Feature 1 (freq=0, time=1) → Embedding vector e_1
        - ...
        - Feature 1219 (freq=19, time=60) → Embedding vector e_1219
    
    Each embedding learns "what kind of information does this modulation bin contain?"
    """
    def __init__(self, n_features, d_model):
        super(FeatureTokenizer, self).__init__()
        
        # Per-feature embeddings (CRITICAL: one per feature!)
        self.feature_embeddings = nn.Parameter(torch.randn(n_features, d_model))
        nn.init.trunc_normal_(self.feature_embeddings, std=0.02)
        
        # Linear projection for feature values
        self.value_projection = nn.Linear(1, d_model, bias=False)
        
    def forward(self, x):
        """
        Args:
            x: (batch, n_features=1220) - after symmetric STM processing (20×61)
        Returns:
            tokens: (batch, n_features, d_model)
        """
        batch_size = x.size(0)
        
        # Project feature values: (batch, n_features) -> (batch, n_features, d_model)
        x_proj = self.value_projection(x.unsqueeze(-1))
        
        # Add per-feature embeddings
        # feature_embeddings: (n_features, d_model)
        # Broadcast to (batch, n_features, d_model)
        tokens = x_proj + self.feature_embeddings.unsqueeze(0)
        
        return tokens


# ============================================================================
# FT-Transformer Architecture
# ============================================================================

class TransformerBlock(nn.Module):
    """
    Standard Transformer block with multi-head attention
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """
        Args:
            x: (batch, n_features, d_model)
        Returns:
            out: (batch, n_features, d_model)
        """
        # Multi-head attention with residual
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Feed-forward with residual
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        return x


class FTTransformer(nn.Module):
    """
    FT-Transformer (Feature Tokenizer + Transformer) for STM Classification
    
    Architecture:
    1. Feature Tokenizer: 1220 features → 1220 tokens (each with d_model dim)
    2. Stack of Transformer blocks (feature-wise attention)
    3. CLS token aggregation (or mean pooling)
    4. Classification head
    
    Key insight: Attention discovers which modulation bins co-occur
    Example: Attention weight from Feature_125 (4Hz) to Feature_400 (2cyc/oct)
             reveals their correlation for "Speech" class
    
    Note: n_features=1220 after symmetric STM processing (20 freq × 61 rates)
    """
    def __init__(self, n_features=1220, num_classes=6, d_model=192, n_heads=8,
                 depth=6, d_ff=512, dropout=0.1, use_gradient_checkpointing=False):
        super(FTTransformer, self).__init__()
        
        self.n_features = n_features
        self.d_model = d_model
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Feature tokenizer
        self.tokenizer = FeatureTokenizer(n_features, d_model)
        
        # Optional: CLS token (like BERT)
        self.use_cls = True
        if self.use_cls:
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(depth)
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
            x: (batch, n_features=1220) - after symmetric STM processing (20×61)
        Returns:
            logits: (batch, num_classes)
        """
        # Feature tokenization: (batch, 1220) -> (batch, 1220, d_model)
        x = self.tokenizer(x)
        
        # Add CLS token
        if self.use_cls:
            batch_size = x.size(0)
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)  # (batch, 1221, d_model)
        
        # Apply Transformer blocks with optional gradient checkpointing
        if self.use_gradient_checkpointing and self.training:
            for block in self.blocks:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
        else:
            for block in self.blocks:
                x = block(x)
        
        # Final normalization
        x = self.norm(x)
        
        # Extract representation
        if self.use_cls:
            # Use CLS token
            x = x[:, 0, :]  # (batch, d_model)
        else:
            # Mean pooling
            x = x.mean(dim=1)  # (batch, d_model)
        
        # Classification
        x = self.head(x)
        
        return x


# ============================================================================
# Balanced Softmax Loss (Same as Vision Mamba)
# ============================================================================

class BalancedSoftmaxLoss(nn.Module):
    """Balanced Softmax Loss for long-tailed recognition"""
    def __init__(self, class_freq):
        super(BalancedSoftmaxLoss, self).__init__()
        self.log_class_freq = torch.log(torch.FloatTensor(class_freq) + 1e-8)
        
    def forward(self, logits, target):
        log_freq = self.log_class_freq.to(logits.device)
        adjusted_logits = logits + log_freq
        return F.cross_entropy(adjusted_logits, target)


# ============================================================================
# Trainer
# ============================================================================

class Trainer:
    """Training manager for FT-Transformer"""
    def __init__(self, model, train_loader, val_loader, test_loader, 
                 device, class_freq, lr=1e-4, weight_decay=1e-5):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        
        # Balanced Softmax loss
        self.criterion = BalancedSoftmaxLoss(class_freq)
        
        # Optimizer and scheduler
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50)
        
        # Tracking
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
        if 'val_f1_scores' in checkpoint:
            self.val_f1_scores = checkpoint['val_f1_scores']
        
        print(f"✓ Resumed from epoch {self.start_epoch}, Best Val F1: {self.best_val_f1:.4f}")
        return True
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
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
            
            if batch_idx % 100 == 0:
                print(f"  Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
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
                loss = F.cross_entropy(output, target)
                total_loss += loss.item()
                
                preds = torch.argmax(output, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        avg_loss = total_loss / len(data_loader)
        
        # Calculate metrics
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        macro_f1 = f1_score(all_targets, all_preds, average='macro')
        
        return avg_loss, macro_f1, all_preds, all_targets
    
    def train(self, num_epochs, checkpoint_dir):
        """Full training loop"""
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Starting from epoch {self.start_epoch + 1}")
        
        for epoch in range(self.start_epoch, num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"{'='*60}")
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            print(f"Train Loss: {train_loss:.4f}")
            
            # Validate
            val_loss, val_f1, _, _ = self.evaluate(self.val_loader)
            self.val_losses.append(val_loss)
            self.val_f1_scores.append(val_f1)
            
            print(f"Val Loss: {val_loss:.4f}, Val Macro F1: {val_f1:.4f}")
            
            # Learning rate scheduling
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Current learning rate: {current_lr:.6f}")
            
            # Save best model
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_f1': val_f1,
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                    'val_f1_scores': self.val_f1_scores,
                }, checkpoint_path)
                print(f"✓ Saved best model with Val F1: {val_f1:.4f}")
            
            # Save periodic checkpoints (every 5 epochs)
            if (epoch + 1) % 5 == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_f1': val_f1,
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                    'val_f1_scores': self.val_f1_scores,
                }, checkpoint_path)
                print(f"✓ Saved checkpoint at epoch {epoch+1}")
            
            # Always save latest checkpoint for resume
            latest_checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'val_f1': val_f1,
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'val_f1_scores': self.val_f1_scores,
            }, latest_checkpoint_path)
        
        print(f"\n{'='*60}")
        print(f"Training completed! Best Val F1: {self.best_val_f1:.4f}")
        print(f"{'='*60}")


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
        print("Usage: python STM_FTtransformer.py <mode> [--resume <checkpoint_dir>]")
        print("Modes:")
        print("  0: Standard training")
        print("  1: Downsample non-tonal speech")
        print("Options:")
        print("  --resume <checkpoint_dir>: Resume from checkpoint directory")
        sys.exit(1)
    
    # Check for resume flag
    resume_dir = None
    if '--resume' in sys.argv:
        resume_idx = sys.argv.index('--resume')
        if resume_idx + 1 < len(sys.argv):
            resume_dir = sys.argv[resume_idx + 1]
            print(f"Resume mode: Will attempt to load from {resume_dir}")
    
    mode = int(sys.argv[1])
    
    # Set parameters based on mode
    if mode == 0:
        print("\nMode 0: Standard training (full dataset)")
        ds_nontonal_speech = False
        directory = "model/STM/FTTransformer_corpora_categories/standard"
    elif mode == 1:
        print("\nMode 1: Downsample non-tonal speech to 100k")
        ds_nontonal_speech = True
        directory = "model/STM/FTTransformer_corpora_categories/downsample"
    else:
        print(f"Unknown mode: {mode}")
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
    
    data_prep = prepData_STM_FTTransformer(ds_nontonal_speech=ds_nontonal_speech)
    train_dataset, val_dataset, test_dataset, class_freq = data_prep.prepare_datasets()
    
    # Symmetric STM processing already applied during data preparation
    # Data is now (n_samples, 1220) instead of (n_samples, 2420)
    print(f"\nSymmetric processing applied: 20 freq × 61 rates = 1220 features")
    print(f"Speed improvement vs original (2420): 4× faster (O(L²) complexity)")
    
    # Create data loaders
    # Batch size optimized for 1220 tokens with O(L^2) attention
    batch_size = 64  # Can increase from 32 since we're actually using 1220 tokens now
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    
    print(f"\nDataLoaders created with batch_size={batch_size}")
    
    # Create model
    print("\n" + "="*60)
    print("Creating FT-Transformer model...")
    print("="*60)
    
    num_classes = 6
    model = FTTransformer(
        n_features=1220,  # After symmetric STM processing (20×61)
        num_classes=num_classes,
        d_model=192,       # Embedding dimension per feature
        n_heads=8,         # Multi-head attention
        depth=6,           # Number of Transformer blocks
        d_ff=512,          # Feed-forward hidden dimension
        dropout=0.1,
        use_gradient_checkpointing=True  # Enable to reduce GPU memory usage
    )
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        class_freq=class_freq,
        lr=1e-4,
        weight_decay=1e-5
    )
    
    # Resume from checkpoint if specified
    if resume_dir:
        latest_ckpt = os.path.join(checkpoint_dir, 'latest_checkpoint.pt')
        if os.path.exists(latest_ckpt):
            trainer.load_checkpoint(latest_ckpt)
        else:
            # Try to find the most recent epoch checkpoint
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
