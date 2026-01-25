#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Audio Spectrogram Mixer v4 (ASM-RH) for STM Classification

Builds on v3 with symmetric STM processing:
1. Exploits up/down-sweep symmetry to reduce dimensionality
2. Averages negative rates (up-sweeps) with positive rates (down-sweeps)
3. Reduces frequency dimension from 121 to 61 (0 to +15 Hz only)
4. Increased model capacity to compensate for reduced input size

Signal Processing Pipeline (Applied to each STM input):
Step A: Separate negative rates [0:60] and positive rates [61:121]
Step B: Flip negative chunk to align with positive chunk
Step C: Average flipped negative and positive chunks
Step D: Concatenate DC (index 60) back, yielding 61 frequency bins

Model Enhancements from v3:
- Increased depth: 4 → 6 blocks
- Increased dimension: 128 → 160
- Maintains v3's confusion-aware loss and contrastive regularization
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
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import warnings

warnings.filterwarnings('ignore')

# Import data preparation
import importlib.util
spec = importlib.util.spec_from_file_location(
    "stm_conformer", 
    "/vast/ac8888/MusicSpeech-STM/STMconformer_model.py"
)
stm_conformer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stm_conformer)

prepData_STM_Conformer = stm_conformer.prepData_STM_Conformer


# ============================================================================
# NEW v4: Symmetric STM Processing
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
        # Apply symmetric processing
        processed_data = process_symmetric_stm(data.unsqueeze(0)).squeeze(0)
        return processed_data, label


# ============================================================================
# Data Augmentation (unchanged from v3)
# ============================================================================

class SpecAugment(nn.Module):
    """SpecAugment-style augmentation for STM features"""
    def __init__(self, freq_mask_param=4, time_mask_param=20, n_freq_masks=1, n_time_masks=2):
        super(SpecAugment, self).__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
        
    def forward(self, x):
        if not self.training:
            return x
            
        batch, freq, time = x.shape
        x = x.clone()
        
        for _ in range(self.n_freq_masks):
            f = np.random.randint(0, self.freq_mask_param)
            f0 = np.random.randint(0, max(1, freq - f))
            x[:, f0:f0+f, :] = 0
        
        for _ in range(self.n_time_masks):
            t = np.random.randint(0, self.time_mask_param)
            t0 = np.random.randint(0, max(1, time - t))
            x[:, :, t0:t0+t] = 0
            
        return x


# ============================================================================
# Enhanced Positional Encoding (unchanged from v3)
# ============================================================================

class Enhanced2DPositionalEncoding(nn.Module):
    """2D positional encoding for anisotropic time/frequency axes"""
    def __init__(self, time_steps, freq_steps, dim):
        super(Enhanced2DPositionalEncoding, self).__init__()
        self.time_embed = nn.Parameter(torch.randn(1, time_steps, 1, dim // 2) * 0.02)
        self.freq_embed = nn.Parameter(torch.randn(1, 1, freq_steps, dim // 2) * 0.02)
        
    def forward(self, x):
        batch, time, freq, dim = x.shape
        time_pos = self.time_embed.expand(batch, time, freq, -1)
        freq_pos = self.freq_embed.expand(batch, time, freq, -1)
        pos_encoding = torch.cat([time_pos, freq_pos], dim=-1)
        return x + pos_encoding


# ============================================================================
# Core ASM Components (unchanged from v3)
# ============================================================================

class RollTimeMixing(nn.Module):
    def __init__(self, dim, shift_range=2):
        super(RollTimeMixing, self).__init__()
        self.shift_range = shift_range
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim)
        )
        
    def forward(self, x):
        batch, time, freq, channels = x.shape
        x = self.norm(x)
        accumulated = torch.zeros_like(x)
        num_shifts = 2 * self.shift_range + 1
        
        for shift in range(-self.shift_range, self.shift_range + 1):
            shifted = torch.roll(x, shifts=shift, dims=1)
            accumulated = accumulated + shifted
        
        output = accumulated / num_shifts
        output = self.mlp(output)
        return output


class HermitFFTMixing(nn.Module):
    def __init__(self, dim):
        super(HermitFFTMixing, self).__init__()
        self.dim = dim
        self.scale = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        batch, time, freq, channels = x.shape
        x = self.norm(x)
        x_reshaped = x.reshape(batch * time, freq, channels)
        x_fft = torch.fft.rfft(x_reshaped, dim=1, norm='ortho')
        x_fft = x_fft * self.scale.view(1, 1, -1)
        x_ifft = torch.fft.irfft(x_fft, n=freq, dim=1, norm='ortho')
        x_ifft = x_ifft + self.bias.view(1, 1, -1)
        output = x_ifft.reshape(batch, time, freq, channels)
        return output


class TokenMixing(nn.Module):
    def __init__(self, seq_len, dim, expansion_factor=2):
        super(TokenMixing, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.channel_mix = nn.Sequential(
            nn.Linear(dim, dim * expansion_factor),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * expansion_factor, dim)
        )
        
    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.channel_mix(x)
        return x + residual


class ChannelMixing(nn.Module):
    def __init__(self, dim, expansion_factor=2):
        super(ChannelMixing, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * expansion_factor),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * expansion_factor, dim)
        )
        
    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.mlp(x)
        return x + residual


class ASM_RH_Block(nn.Module):
    def __init__(self, time_steps, freq_steps, dim, shift_range=2, expansion_factor=2):
        super(ASM_RH_Block, self).__init__()
        self.time_steps = time_steps
        self.freq_steps = freq_steps
        self.pos_encoding = Enhanced2DPositionalEncoding(time_steps, freq_steps, dim)
        self.roll_time = RollTimeMixing(dim, shift_range)
        self.hermit_fft = HermitFFTMixing(dim)
        seq_len = time_steps * freq_steps
        self.token_mixing = TokenMixing(seq_len, dim, expansion_factor)
        self.channel_mixing = ChannelMixing(dim, expansion_factor)
        
    def forward(self, x):
        batch, seq_len, dim = x.shape
        x_2d = x.reshape(batch, self.time_steps, self.freq_steps, dim)
        x_2d = self.pos_encoding(x_2d)
        x_2d = x_2d + self.roll_time(x_2d)
        x_2d = x_2d + self.hermit_fft(x_2d)
        x = x_2d.reshape(batch, seq_len, dim)
        x = self.token_mixing(x)
        x = self.channel_mixing(x)
        return x


class EnhancedASM_RH_Classifier(nn.Module):
    """
    Enhanced ASM-RH v4 with symmetric STM processing.
    
    NEW in v4:
    - Increased depth: 6 blocks (was 4 in v3)
    - Increased dimension: 160 (was 128 in v3)
    - Input frequency dimension: 61 (was 121 in v3)
    """
    def __init__(self, time_steps, freq_steps, num_classes, 
                 dim=160, num_blocks=6, shift_range=2, expansion_factor=2, dropout=0.1):
        super(EnhancedASM_RH_Classifier, self).__init__()
        
        self.time_steps = time_steps
        self.freq_steps = freq_steps
        self.dim = dim
        
        self.spec_augment = SpecAugment(
            freq_mask_param=4, time_mask_param=20, 
            n_freq_masks=1, n_time_masks=2
        )
        
        self.input_proj = nn.Sequential(
            nn.Conv2d(1, dim // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim // 4),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        
        self.blocks = nn.ModuleList([
            ASM_RH_Block(time_steps, freq_steps, dim, shift_range, expansion_factor)
            for _ in range(num_blocks)
        ])
        
        self.norm = nn.LayerNorm(dim)
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Feature extractor before classification (for contrastive loss)
        self.feature_extractor = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.classifier = nn.Linear(dim // 2, num_classes)
        
    def forward(self, x, return_features=False):
        batch_size = x.size(0)
        x = self.spec_augment(x)
        x = x.unsqueeze(1)
        x = self.input_proj(x)
        x = x.permute(0, 1, 3, 2)
        x = x.flatten(2).transpose(1, 2)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        x = x.transpose(1, 2)
        pooled = self.pool(x).squeeze(-1)
        
        # Extract features
        features = self.feature_extractor(pooled)
        logits = self.classifier(features)
        
        if return_features:
            return logits, features
        
        return logits


# ============================================================================
# Confusion-Aware Loss Functions (unchanged from v3)
# ============================================================================

def compute_confusion_aware_weights(y_train, num_classes=6):
    """
    Compute softer class weights with confusion-aware boosting.
    Same strategy as v3 (from Kanformer v2).
    """
    class_counts = Counter(y_train)
    total = len(y_train)
    
    # Base weights: sqrt of inverse frequency (softer than v1/v2)
    weights = []
    for i in range(num_classes):
        count = class_counts.get(i, 1)
        weight = math.sqrt(total / (num_classes * count))
        weights.append(weight)
    
    weights = torch.FloatTensor(weights)
    weights = weights / weights.sum() * num_classes
    
    # Confusion-aware adjustment
    weights[1] *= 1.3  # Boost tonal speech discrimination
    weights[3] *= 1.3  # Boost non-vocal music discrimination
    weights[4] *= 0.7  # Reduce env:urban weight
    weights[5] *= 0.8  # Reduce env:wildlife weight
    
    # Re-normalize
    weights = weights / weights.sum() * num_classes
    
    print(f"\nConfusion-Aware Class Weights (v4):")
    for i in range(num_classes):
        count = class_counts.get(i, 0)
        print(f"  Class {i}: {count:7d} samples → weight: {weights[i]:.4f}")
    
    return weights


class ContrastiveFocalLoss(nn.Module):
    """Enhanced Focal Loss with contrastive regularization (from v3)"""
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.01, 
                 contrastive_weight=0.1, similar_pairs=[(0, 1), (2, 3)]):
        super(ContrastiveFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.contrastive_weight = contrastive_weight
        self.similar_pairs = similar_pairs
        
    def forward(self, inputs, targets, features=None):
        # Focal Loss with minimal label smoothing
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
                        contrastive_loss += (1.0 / (distances + 1e-6)).mean()
                        num_pairs += 1
            
            if num_pairs > 0:
                contrastive_loss = contrastive_loss / num_pairs
                total_loss = focal_loss + self.contrastive_weight * contrastive_loss
                return total_loss
        
        return focal_loss


# ============================================================================
# Enhanced Trainer v4 (same as v3)
# ============================================================================

class EnhancedTrainerV4:
    """Enhanced ASM trainer v4 with symmetric STM processing"""
    def __init__(self, model, train_loader, val_loader, test_loader, 
                 device, class_weights, lr=1e-3, weight_decay=1e-4, warmup_epochs=5):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.warmup_epochs = warmup_epochs
        
        # Contrastive Focal Loss with softer settings
        self.criterion = ContrastiveFocalLoss(
            alpha=class_weights.to(device),
            gamma=2.0,
            label_smoothing=0.01,
            contrastive_weight=0.1,
            similar_pairs=[(0, 1), (2, 3)]
        )
        
        # Cross-entropy for evaluation
        self.ce_criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
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
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
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
        
        print(f"✓ Resumed from epoch {self.start_epoch}, Best Val F1: {self.best_val_f1:.4f}")
        return True
        
    def train_epoch(self, epoch):
        """Train for one epoch with contrastive loss"""
        self.model.train()
        total_loss = 0.0
        
        # Warmup
        if epoch < self.warmup_epochs:
            lr_scale = (epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr_scale * 1e-3
        
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
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"  Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}")
        
        if epoch >= self.warmup_epochs:
            self.scheduler.step(epoch - self.warmup_epochs)
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
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
                loss = self.ce_criterion(output, target)
                total_loss += loss.item()
                
                preds = torch.argmax(output, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        avg_loss = total_loss / len(data_loader)
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        macro_f1 = f1_score(all_targets, all_preds, average='macro')
        per_class_f1 = f1_score(all_targets, all_preds, average=None)
        
        if return_confusion:
            cm = confusion_matrix(all_targets, all_preds)
            return avg_loss, macro_f1, all_preds, all_targets, cm, per_class_f1
        
        return avg_loss, macro_f1, all_preds, all_targets, per_class_f1
    
    def train(self, num_epochs, checkpoint_dir):
        """Full training loop with confusion monitoring"""
        print(f"\nStarting Enhanced ASM v4 training for {num_epochs} epochs...")
        print(f"Starting from epoch {self.start_epoch + 1}")
        print(f"Strategy: Symmetric STM + v3 confusion-aware loss")
        
        for epoch in range(self.start_epoch, num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"{'='*60}")
            
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            print(f"Train Loss: {train_loss:.4f}")
            
            val_loss, val_f1, _, _, conf_matrix, per_class_f1 = self.evaluate(
                self.val_loader, return_confusion=True
            )
            self.val_losses.append(val_loss)
            self.val_f1_scores.append(val_f1)
            self.confusion_history.append(conf_matrix)
            
            print(f"Val Loss: {val_loss:.4f}, Val Macro F1: {val_f1:.4f}")
            
            # Display per-class F1 scores
            print(f"Per-class F1 scores:")
            for i, f1 in enumerate(per_class_f1):
                marker = ""
                if i == 1:
                    marker = " (tonal speech)"
                elif i == 3:
                    marker = " (env/music)"
                print(f"  Class {i}: {f1:.4f}{marker}")
            
            # Print confusion for similar classes
            print(f"Confusion between Similar Classes:")
            print(f"  Class 0→1: {conf_matrix[0,1]:5d} | Class 1→0: {conf_matrix[1,0]:5d}")
            print(f"  Class 2→3: {conf_matrix[2,3]:5d} | Class 3→2: {conf_matrix[3,2]:5d}")
            
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Current learning rate: {current_lr:.6f}")
            
            # Save best model
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_f1': val_f1,
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                    'val_f1_scores': self.val_f1_scores,
                    'confusion_history': self.confusion_history,
                }, os.path.join(checkpoint_dir, 'best_model.pt'))
                print(f"✓ Saved best model with Val F1: {val_f1:.4f}")
            
            # Periodic checkpoints
            if (epoch + 1) % 5 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_f1': val_f1,
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                    'val_f1_scores': self.val_f1_scores,
                    'confusion_history': self.confusion_history,
                }, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt'))
            
            # Always save latest
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'val_f1': val_f1,
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'val_f1_scores': self.val_f1_scores,
                'confusion_history': self.confusion_history,
            }, os.path.join(checkpoint_dir, 'latest_checkpoint.pt'))
        
        print(f"\n{'='*60}")
        print(f"Training completed! Best Val F1: {self.best_val_f1:.4f}")
        print(f"{'='*60}")


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if len(sys.argv) < 2:
        print("Usage: python STMasm_enhanced4.py <mode> [--resume <checkpoint_dir>]")
        sys.exit(1)
    
    mode = int(sys.argv[1])
    resume_dir = None
    if '--resume' in sys.argv:
        resume_idx = sys.argv.index('--resume')
        if resume_idx + 1 < len(sys.argv):
            resume_dir = sys.argv[resume_idx + 1]
    
    if mode == 0:
        print("Mode 0: Standard training with symmetric STM processing")
        ds_nontonal_speech = False
        directory = "model/STM/ASM_Enhanced4_corpora_categories/standard"
    elif mode == 1:
        print("Mode 1: Downsample non-tonal speech + symmetric STM")
        ds_nontonal_speech = True
        directory = "model/STM/ASM_Enhanced4_corpora_categories/downsample"
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)
    
    if resume_dir:
        checkpoint_dir = resume_dir
    else:
        time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        checkpoint_dir = os.path.join(directory, "ckpt", time_stamp)
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"Checkpoint directory: {checkpoint_dir}")
    
    # Prepare data
    print("\n" + "="*60)
    print("Loading and preparing data...")
    print("="*60)
    
    data_prep = prepData_STM_Conformer(ds_nontonal_speech=ds_nontonal_speech)
    train_dataset, val_dataset, test_dataset, n_freq_original, n_time = data_prep.prepare_datasets()
    
    print(f"Original STM dimensions: Freq_bands={n_freq_original}, Mod_rates={n_time}")
    print(f"Applying symmetric STM processing to modulation rate dimension...")
    
    # Apply symmetric processing to all datasets
    train_dataset = SymmetricSTMDataset(train_dataset)
    val_dataset = SymmetricSTMDataset(val_dataset)
    test_dataset = SymmetricSTMDataset(test_dataset)
    
    # New modulation rate dimension after symmetric processing
    # Original: n_time=121 mod rates → New: 61 mod rates (DC + 60 positive rates)
    n_time_processed = 61  # 0 Hz + 60 bins (0.25 Hz to 15 Hz)
    print(f"Processed STM dimensions: Freq_bands={n_freq_original}, Mod_rates={n_time_processed}")
    
    # Extract labels for class weight computation
    train_labels = []
    for i in range(len(train_dataset.base_dataset)):
        _, label = train_dataset.base_dataset[i]
        train_labels.append(label.item())
    train_labels = np.array(train_labels)
    
    # Compute confusion-aware weights
    class_weights = compute_confusion_aware_weights(train_labels, num_classes=6)
    
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    
    print(f"DataLoaders created with symmetric STM processing applied")
    
    # Create model with increased capacity
    print("\n" + "="*60)
    print("Creating Enhanced ASM-RH v4 model...")
    print("="*60)
    print(f"Architecture changes from v3:")
    print(f"  - Modulation rate dimension: 121 → {n_time_processed}")
    print(f"  - Model dimension: 128 → 160")
    print(f"  - Number of blocks: 4 → 6")
    
    num_classes = 6
    model = EnhancedASM_RH_Classifier(
        time_steps=n_time_processed,  # Use processed mod rates (61)
        freq_steps=n_freq_original,   # Use original freq bands (20)
        num_classes=num_classes,
        dim=160,          # Increased from 128
        num_blocks=6,     # Increased from 4
        shift_range=2,
        expansion_factor=2,
        dropout=0.1
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Create trainer v4
    trainer = EnhancedTrainerV4(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        class_weights=class_weights,
        lr=1e-3,
        weight_decay=1e-4,
        warmup_epochs=5
    )
    
    # Resume if specified
    if resume_dir:
        latest_ckpt = os.path.join(checkpoint_dir, 'latest_checkpoint.pt')
        if not os.path.exists(latest_ckpt):
            ckpt_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_')]
            if ckpt_files:
                epochs = [int(f.split('_')[-1].replace('.pt', '')) for f in ckpt_files]
                latest_ckpt = os.path.join(checkpoint_dir, f'checkpoint_epoch_{max(epochs)}.pt')
        trainer.load_checkpoint(latest_ckpt)
    
    # Train
    num_epochs = 50
    trainer.train(num_epochs=num_epochs, checkpoint_dir=checkpoint_dir)
    
    # Final evaluation
    print("\n" + "="*60)
    print("Final evaluation on test set...")
    print("="*60)
    
    checkpoint = torch.load(os.path.join(checkpoint_dir, 'best_model.pt'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_f1, test_preds, test_targets, cm, per_class_f1 = trainer.evaluate(
        test_loader, return_confusion=True
    )
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Macro F1: {test_f1:.4f}")
    print(f"\nPer-class F1 scores:")
    for i, f1 in enumerate(per_class_f1):
        print(f"  Class {i}: {f1:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(cm)
    
    print(f"\nConfusion analysis for target pairs:")
    for i, j in [(0, 1), (2, 3)]:
        conf_ij = cm[i, j]
        conf_ji = cm[j, i]
        total_i = cm[i, :].sum()
        total_j = cm[j, :].sum()
        print(f"  Class {i} → Class {j}: {conf_ij}/{total_i} ({100*conf_ij/total_i:.1f}%)")
        print(f"  Class {j} → Class {i}: {conf_ji}/{total_j} ({100*conf_ji/total_j:.1f}%)")
    
    # Save results
    np.save(os.path.join(checkpoint_dir, 'test_predictions.npy'), test_preds)
    np.save(os.path.join(checkpoint_dir, 'test_targets.npy'), test_targets)
    np.save(os.path.join(checkpoint_dir, 'confusion_matrix.npy'), cm)
    
    print("\n" + "="*60)
    print("Training and evaluation completed!")
    print("="*60)
