#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Audio Spectrogram Mixer v2 (ASM-RH) for STM Classification

Key improvements over v1:
1. Confusion-aware class weighting (reduces weights for similar class pairs)
2. Asymmetric focal loss (higher gamma for confusable classes)
3. Inter-class margin regularization (pushes similar classes apart)
4. Adaptive label smoothing (more smoothing between similar classes)

Addresses v1 issues:
- Class 1 vs 0 confusion (both speech, tonal vs non-tonal)
- Class 3 vs 2 confusion (environment vs music)
- Over-weighting of truly distinct minorities (4, 5)
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
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
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
# Data Augmentation (unchanged from v1)
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
# Enhanced Positional Encoding (unchanged from v1)
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
# Core ASM Components (unchanged from v1)
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
    """Enhanced ASM-RH with improved feature extraction"""
    def __init__(self, time_steps, freq_steps, num_classes, 
                 dim=128, num_blocks=4, shift_range=2, expansion_factor=2, dropout=0.1):
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
        
        # Improved classifier with auxiliary head for confusion reduction
        self.main_classifier = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, num_classes)
        )
        
        # Auxiliary binary classifiers for confusable pairs
        self.binary_classifier_01 = nn.Linear(dim, 2)  # Class 0 vs 1
        self.binary_classifier_23 = nn.Linear(dim, 2)  # Class 2 vs 3
        
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
        features = self.pool(x).squeeze(-1)
        
        # Main classification
        logits = self.main_classifier(features)
        
        if return_features:
            # Binary classifiers for auxiliary loss
            binary_01 = self.binary_classifier_01(features)
            binary_23 = self.binary_classifier_23(features)
            return logits, features, binary_01, binary_23
        
        return logits


# ============================================================================
# Confusion-Aware Loss Functions (NEW)
# ============================================================================

class ConfusionAwareFocalLoss(nn.Module):
    """
    Focal loss with confusion-aware class weighting.
    
    Key improvements:
    1. Adjusts class weights based on confusion patterns
    2. Uses asymmetric gamma for confusable vs distinct classes
    3. Adaptive label smoothing (more between similar classes)
    """
    def __init__(self, class_weights, confusion_pairs=None, gamma_base=2.0, 
                 gamma_confusable=2.5, label_smoothing=0.1):
        super(ConfusionAwareFocalLoss, self).__init__()
        
        # Adjust weights for confusable pairs (reduce to avoid over-weighting)
        self.class_weights = class_weights.clone()
        if confusion_pairs:
            for i, j in confusion_pairs:
                # Reduce weight for confusable classes by 30%
                self.class_weights[i] *= 0.7
                self.class_weights[j] *= 0.7
        
        self.confusion_pairs = confusion_pairs or []
        self.gamma_base = gamma_base
        self.gamma_confusable = gamma_confusable
        self.label_smoothing = label_smoothing
        
        print(f"Adjusted class weights: {self.class_weights}")
        
    def forward(self, inputs, targets):
        n_classes = inputs.size(1)
        
        # Adaptive label smoothing (more for confusable pairs)
        smoothed_targets = torch.zeros_like(inputs)
        smoothed_targets.fill_(self.label_smoothing / (n_classes - 1))
        
        # Extra smoothing for confusable pairs
        for i, j in self.confusion_pairs:
            mask_i = (targets == i)
            mask_j = (targets == j)
            if mask_i.any():
                smoothed_targets[mask_i, j] = self.label_smoothing * 2  # Double smoothing
            if mask_j.any():
                smoothed_targets[mask_j, i] = self.label_smoothing * 2
        
        smoothed_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        
        # Compute probabilities
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        
        # Asymmetric gamma (higher for confusable classes)
        gamma = torch.ones_like(targets, dtype=torch.float) * self.gamma_base
        for i, j in self.confusion_pairs:
            gamma[(targets == i) | (targets == j)] = self.gamma_confusable
        
        # Focal loss with asymmetric gamma
        focal_weight = (1 - probs.gather(1, targets.unsqueeze(1))).squeeze() ** gamma
        loss = -focal_weight.unsqueeze(1) * log_probs * smoothed_targets
        
        # Apply adjusted class weights
        weight_tensor = self.class_weights[targets]
        loss = loss.sum(dim=1) * weight_tensor
        
        return loss.mean()


class MarginRankingLoss(nn.Module):
    """
    Encourages larger margins between confusable class pairs.
    Pushes class 1 away from class 0, class 3 away from class 2.
    """
    def __init__(self, margin=0.5):
        super(MarginRankingLoss, self).__init__()
        self.margin = margin
        
    def forward(self, features, targets, confusion_pairs):
        """
        features: (batch, feature_dim)
        targets: (batch,)
        confusion_pairs: list of (i, j) tuples
        """
        if len(confusion_pairs) == 0:
            return torch.tensor(0.0, device=features.device)
        
        total_loss = 0.0
        num_pairs = 0
        
        for class_i, class_j in confusion_pairs:
            # Find samples from each class
            mask_i = (targets == class_i)
            mask_j = (targets == class_j)
            
            if mask_i.sum() > 0 and mask_j.sum() > 0:
                features_i = features[mask_i]
                features_j = features[mask_j]
                
                # Compute pairwise distances
                for feat_i in features_i:
                    for feat_j in features_j:
                        dist = F.pairwise_distance(feat_i.unsqueeze(0), feat_j.unsqueeze(0))
                        # Penalize if distance < margin
                        loss = F.relu(self.margin - dist)
                        total_loss += loss
                        num_pairs += 1
        
        if num_pairs > 0:
            return total_loss / num_pairs
        return torch.tensor(0.0, device=features.device)


# ============================================================================
# Enhanced Trainer v2 (NEW)
# ============================================================================

class EnhancedTrainerV2:
    """
    Enhanced trainer with confusion-aware strategies.
    """
    def __init__(self, model, train_loader, val_loader, test_loader, 
                 device, train_labels, lr=1e-3, weight_decay=1e-4, warmup_epochs=5):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.warmup_epochs = warmup_epochs
        
        # Compute class weights
        unique_classes = np.unique(train_labels)
        class_weights = compute_class_weight('balanced', classes=unique_classes, y=train_labels)
        self.base_class_weights = torch.FloatTensor(class_weights).to(device)
        
        print(f"Base class weights: {class_weights}")
        print(f"Class distribution: {np.bincount(train_labels)}")
        
        # Define confusable pairs based on v1 results
        # Class 1 (Speech: Tonal) confused with Class 0 (Speech: Non-tonal)
        # Class 3 (Environment) confused with Class 2 (Music)
        self.confusion_pairs = [(0, 1), (2, 3)]
        
        # Confusion-aware focal loss
        self.criterion = ConfusionAwareFocalLoss(
            class_weights=self.base_class_weights,
            confusion_pairs=self.confusion_pairs,
            gamma_base=2.0,
            gamma_confusable=2.5,  # Higher gamma for confusable pairs
            label_smoothing=0.1
        )
        
        # Margin ranking loss for confusable pairs
        self.margin_loss = MarginRankingLoss(margin=0.5)
        self.margin_weight = 0.1  # Weight for margin loss
        
        # Cross-entropy for evaluation
        self.ce_criterion = nn.CrossEntropyLoss()
        
        # Binary classification loss for auxiliary heads
        self.binary_criterion = nn.CrossEntropyLoss()
        self.binary_weight = 0.2  # Weight for binary auxiliary loss
        
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
        
        print(f"✓ Resumed from epoch {self.start_epoch}, Best Val F1: {self.best_val_f1:.4f}")
        return True
        
    def train_epoch(self, epoch):
        """Train for one epoch with multi-task learning"""
        self.model.train()
        total_loss = 0.0
        total_main_loss = 0.0
        total_margin_loss = 0.0
        total_binary_loss = 0.0
        
        # Warmup
        if epoch < self.warmup_epochs:
            lr_scale = (epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr_scale * 1e-3
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with features and auxiliary outputs
            logits, features, binary_01, binary_23 = self.model(data, return_features=True)
            
            # Main classification loss (confusion-aware focal)
            main_loss = self.criterion(logits, target)
            
            # Margin ranking loss for confusable pairs
            margin_loss = self.margin_loss(features, target, self.confusion_pairs)
            
            # Binary auxiliary losses
            binary_loss = torch.tensor(0.0, device=self.device)
            
            # Binary classifier 0 vs 1
            mask_01 = (target == 0) | (target == 1)
            if mask_01.sum() > 0:
                binary_target_01 = (target[mask_01] == 1).long()
                binary_loss += self.binary_criterion(binary_01[mask_01], binary_target_01)
            
            # Binary classifier 2 vs 3
            mask_23 = (target == 2) | (target == 3)
            if mask_23.sum() > 0:
                binary_target_23 = (target[mask_23] == 3).long()
                binary_loss += self.binary_criterion(binary_23[mask_23], binary_target_23)
            
            # Combined loss
            loss = main_loss + self.margin_weight * margin_loss + self.binary_weight * binary_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_main_loss += main_loss.item()
            total_margin_loss += margin_loss.item()
            total_binary_loss += binary_loss.item()
            
            if batch_idx % 100 == 0:
                print(f"  Batch {batch_idx}/{len(self.train_loader)}, "
                      f"Loss: {loss.item():.4f} (Main: {main_loss.item():.4f}, "
                      f"Margin: {margin_loss.item():.4f}, Binary: {binary_loss.item():.4f})")
        
        if epoch >= self.warmup_epochs:
            self.scheduler.step(epoch - self.warmup_epochs)
        
        avg_loss = total_loss / len(self.train_loader)
        print(f"  Avg Losses - Total: {avg_loss:.4f}, Main: {total_main_loss/len(self.train_loader):.4f}, "
              f"Margin: {total_margin_loss/len(self.train_loader):.4f}, "
              f"Binary: {total_binary_loss/len(self.train_loader):.4f}")
        
        return avg_loss
    
    def evaluate(self, data_loader, return_confusion=False):
        """Evaluate on validation or test set"""
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
        """Full training loop"""
        print(f"\nStarting enhanced ASM v2 training for {num_epochs} epochs...")
        print(f"Starting from epoch {self.start_epoch + 1}")
        print(f"Confusion-aware strategies enabled for pairs: {self.confusion_pairs}")
        print(f"\nFixed class weights: {self.criterion.class_weights.cpu().numpy()}")
        
        for epoch in range(self.start_epoch, num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"{'='*60}")
            
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            print(f"Train Loss: {train_loss:.4f}")
            
            val_loss, val_f1, _, _, per_class_f1 = self.evaluate(self.val_loader)
            self.val_losses.append(val_loss)
            self.val_f1_scores.append(val_f1)
            print(f"Val Loss: {val_loss:.4f}, Val Macro F1: {val_f1:.4f}")
            
            # Display per-class F1 scores
            print(f"Per-class F1 scores:")
            for i, f1 in enumerate(per_class_f1):
                print(f"  Class {i}: {f1:.4f}", end="")
                # Highlight confusable pairs
                if any(i in pair for pair in self.confusion_pairs):
                    confusable_with = [j for pair in self.confusion_pairs for j in pair if j != i and i in pair]
                    if confusable_with:
                        print(f" (confusable with {confusable_with[0]})", end="")
                print()
            
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
    
    # ...existing argument parsing...
    if len(sys.argv) < 2:
        print("Usage: python STMasm_enhanced2.py <mode> [--resume <checkpoint_dir>]")
        sys.exit(1)
    
    mode = int(sys.argv[1])
    resume_dir = None
    if '--resume' in sys.argv:
        resume_idx = sys.argv.index('--resume')
        if resume_idx + 1 < len(sys.argv):
            resume_dir = sys.argv[resume_idx + 1]
    
    # ...existing mode configuration...
    if mode == 0:
        print("Mode 0: Standard training with confusion-aware improvements")
        ds_nontonal_speech = False
        directory = "model/STM/ASM_Enhanced2_corpora_categories/standard"
    elif mode == 1:
        print("Mode 1: Downsample non-tonal speech with confusion-aware improvements")
        ds_nontonal_speech = True
        directory = "model/STM/ASM_Enhanced2_corpora_categories/downsample"
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)
    
    # ...existing directory setup...
    if resume_dir:
        checkpoint_dir = resume_dir
    else:
        time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        checkpoint_dir = os.path.join(directory, "ckpt", time_stamp)
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"Checkpoint directory: {checkpoint_dir}")
    
    # ...existing data preparation...
    print("\n" + "="*60)
    print("Loading and preparing data...")
    print("="*60)
    
    data_prep = prepData_STM_Conformer(ds_nontonal_speech=ds_nontonal_speech)
    train_dataset, val_dataset, test_dataset, n_freq, n_time = data_prep.prepare_datasets()
    train_labels = train_dataset.tensors[1].numpy()
    
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    
    print(f"DataLoaders created, STM dimensions: Time={n_time}, Freq={n_freq}")
    
    # Create model
    print("\n" + "="*60)
    print("Creating Enhanced ASM-RH v2 model...")
    print("="*60)
    
    num_classes = 6
    model = EnhancedASM_RH_Classifier(
        time_steps=n_time,
        freq_steps=n_freq,
        num_classes=num_classes,
        dim=128,
        num_blocks=4,
        shift_range=2,
        expansion_factor=2,
        dropout=0.1
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Create trainer v2
    trainer = EnhancedTrainerV2(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        train_labels=train_labels,
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
    
    # Final evaluation with detailed metrics
    print("\n" + "="*60)
    print("Final evaluation on test set...")
    print("="*60)
    
    checkpoint = torch.load(os.path.join(checkpoint_dir, 'best_model.pt'))
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
    
    # Analyze confusable pairs
    print(f"\nConfusion analysis for target pairs:")
    for i, j in trainer.confusion_pairs:
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
