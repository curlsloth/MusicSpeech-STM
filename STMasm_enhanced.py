#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Audio Spectrogram Mixer (ASM-RH) for STM Classification

Improvements over base ASM:
1. Class-weighted focal loss for severe imbalance
2. Enhanced 2D positional encoding (time and frequency aware)
3. SpecAugment-style augmentation during training
4. Label smoothing for better generalization
5. Warmup schedule like Kanformer
6. Improved normalization strategy

Architecture optimized for 121×20 STM grids (Rate × Scale dimensions).
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
from sklearn.metrics import f1_score
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
# Data Augmentation
# ============================================================================

class SpecAugment(nn.Module):
    """
    SpecAugment-style augmentation for STM features.
    Randomly masks time and frequency regions during training.
    """
    def __init__(self, freq_mask_param=4, time_mask_param=20, n_freq_masks=1, n_time_masks=2):
        super(SpecAugment, self).__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
        
    def forward(self, x):
        # x: (batch, freq, time)
        if not self.training:
            return x
            
        batch, freq, time = x.shape
        x = x.clone()
        
        # Frequency masking
        for _ in range(self.n_freq_masks):
            f = np.random.randint(0, self.freq_mask_param)
            f0 = np.random.randint(0, max(1, freq - f))
            x[:, f0:f0+f, :] = 0
        
        # Time masking
        for _ in range(self.n_time_masks):
            t = np.random.randint(0, self.time_mask_param)
            t0 = np.random.randint(0, max(1, time - t))
            x[:, :, t0:t0+t] = 0
            
        return x


# ============================================================================
# Enhanced Positional Encoding
# ============================================================================

class Enhanced2DPositionalEncoding(nn.Module):
    """
    2D positional encoding that respects anisotropic axes.
    Separate encodings for time (Rate) and frequency (Scale) dimensions.
    """
    def __init__(self, time_steps, freq_steps, dim):
        super(Enhanced2DPositionalEncoding, self).__init__()
        
        # Separate embeddings for time and frequency
        self.time_embed = nn.Parameter(torch.randn(1, time_steps, 1, dim // 2) * 0.02)
        self.freq_embed = nn.Parameter(torch.randn(1, 1, freq_steps, dim // 2) * 0.02)
        
    def forward(self, x):
        # x: (batch, time, freq, dim)
        batch, time, freq, dim = x.shape
        
        # Expand and broadcast
        time_pos = self.time_embed.expand(batch, time, freq, -1)
        freq_pos = self.freq_embed.expand(batch, time, freq, -1)
        
        # Concatenate time and frequency positional information
        pos_encoding = torch.cat([time_pos, freq_pos], dim=-1)
        
        return x + pos_encoding


# ============================================================================
# Core ASM Components (from base model)
# ============================================================================

class RollTimeMixing(nn.Module):
    """Memory-efficient roll-time mixing with enhanced normalization"""
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
        # x: (batch, time, freq, channels)
        batch, time, freq, channels = x.shape
        
        # Pre-normalization
        x = self.norm(x)
        
        # Memory-efficient accumulation
        accumulated = torch.zeros_like(x)
        num_shifts = 2 * self.shift_range + 1
        
        for shift in range(-self.shift_range, self.shift_range + 1):
            shifted = torch.roll(x, shifts=shift, dims=1)
            accumulated = accumulated + shifted
        
        output = accumulated / num_shifts
        output = self.mlp(output)
        
        return output


class HermitFFTMixing(nn.Module):
    """FFT-based mixing with enhanced scaling"""
    def __init__(self, dim):
        super(HermitFFTMixing, self).__init__()
        self.dim = dim
        self.scale = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        # x: (batch, time, freq, channels)
        batch, time, freq, channels = x.shape
        
        # Pre-normalization
        x = self.norm(x)
        
        x_reshaped = x.reshape(batch * time, freq, channels)
        x_fft = torch.fft.rfft(x_reshaped, dim=1, norm='ortho')
        x_fft = x_fft * self.scale.view(1, 1, -1)
        x_ifft = torch.fft.irfft(x_fft, n=freq, dim=1, norm='ortho')
        x_ifft = x_ifft + self.bias.view(1, 1, -1)
        output = x_ifft.reshape(batch, time, freq, channels)
        
        return output


class TokenMixing(nn.Module):
    """Optimized channel-wise token mixing"""
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
    """Channel mixing with pre-normalization"""
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
    """Enhanced ASM-RH block with better normalization"""
    def __init__(self, time_steps, freq_steps, dim, shift_range=2, expansion_factor=2):
        super(ASM_RH_Block, self).__init__()
        
        self.time_steps = time_steps
        self.freq_steps = freq_steps
        
        # Enhanced positional encoding
        self.pos_encoding = Enhanced2DPositionalEncoding(time_steps, freq_steps, dim)
        
        self.roll_time = RollTimeMixing(dim, shift_range)
        self.hermit_fft = HermitFFTMixing(dim)
        
        seq_len = time_steps * freq_steps
        self.token_mixing = TokenMixing(seq_len, dim, expansion_factor)
        self.channel_mixing = ChannelMixing(dim, expansion_factor)
        
    def forward(self, x):
        # x: (batch, seq_len, dim)
        batch, seq_len, dim = x.shape
        
        # Reshape to 2D
        x_2d = x.reshape(batch, self.time_steps, self.freq_steps, dim)
        
        # Apply enhanced positional encoding
        x_2d = self.pos_encoding(x_2d)
        
        # Spatial mixing
        x_2d = x_2d + self.roll_time(x_2d)  # Residual connection
        x_2d = x_2d + self.hermit_fft(x_2d)  # Residual connection
        
        # Flatten back
        x = x_2d.reshape(batch, seq_len, dim)
        
        # Token and channel mixing (already have residuals inside)
        x = self.token_mixing(x)
        x = self.channel_mixing(x)
        
        return x


class EnhancedASM_RH_Classifier(nn.Module):
    """
    Enhanced ASM-RH Classifier with:
    - SpecAugment augmentation
    - Better positional encoding
    - Improved normalization
    """
    def __init__(self, time_steps, freq_steps, num_classes, 
                 dim=128, num_blocks=4, shift_range=2, expansion_factor=2, dropout=0.1):
        super(EnhancedASM_RH_Classifier, self).__init__()
        
        self.time_steps = time_steps
        self.freq_steps = freq_steps
        self.dim = dim
        
        # SpecAugment for training
        self.spec_augment = SpecAugment(
            freq_mask_param=4, time_mask_param=20, 
            n_freq_masks=1, n_time_masks=2
        )
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Conv2d(1, dim // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim // 4),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        
        # ASM-RH blocks (now with enhanced positional encoding inside)
        self.blocks = nn.ModuleList([
            ASM_RH_Block(time_steps, freq_steps, dim, shift_range, expansion_factor)
            for _ in range(num_blocks)
        ])
        
        # Final normalization
        self.norm = nn.LayerNorm(dim)
        
        # Global pooling
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, num_classes)
        )
        
    def forward(self, x):
        # x: (batch, freq, time)
        batch_size = x.size(0)
        
        # Apply SpecAugment during training
        x = self.spec_augment(x)
        
        # Add channel dimension
        x = x.unsqueeze(1)
        
        # Input projection
        x = self.input_proj(x)
        
        # Permute to (batch, dim, time, freq)
        x = x.permute(0, 1, 3, 2)
        
        # Flatten and transpose
        x = x.flatten(2).transpose(1, 2)
        
        # Apply ASM-RH blocks
        for block in self.blocks:
            x = block(x)
        
        # Final normalization
        x = self.norm(x)
        
        # Global pooling
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)
        
        # Classification
        x = self.classifier(x)
        
        return x


# ============================================================================
# Enhanced Loss Function
# ============================================================================

class WeightedFocalLoss(nn.Module):
    """
    Focal loss with class weights computed from training data.
    Handles severe class imbalance better than standard focal loss.
    """
    def __init__(self, class_weights=None, gamma=2.0, label_smoothing=0.1):
        super(WeightedFocalLoss, self).__init__()
        self.class_weights = class_weights
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        
    def forward(self, inputs, targets):
        # Apply label smoothing
        n_classes = inputs.size(1)
        smoothed_targets = torch.zeros_like(inputs)
        smoothed_targets.fill_(self.label_smoothing / (n_classes - 1))
        smoothed_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        
        # Compute log probabilities
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        
        # Focal loss computation
        focal_weight = (1 - probs) ** self.gamma
        loss = -focal_weight * log_probs * smoothed_targets
        
        # Apply class weights
        if self.class_weights is not None:
            weight_tensor = self.class_weights[targets]
            loss = loss.sum(dim=1) * weight_tensor
        else:
            loss = loss.sum(dim=1)
        
        return loss.mean()


# ============================================================================
# Enhanced Trainer
# ============================================================================

class EnhancedTrainer:
    """Enhanced training manager with warmup and class weighting"""
    def __init__(self, model, train_loader, val_loader, test_loader, 
                 device, train_labels, lr=1e-3, weight_decay=1e-4, warmup_epochs=5):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.warmup_epochs = warmup_epochs
        
        # Compute class weights from training labels
        unique_classes = np.unique(train_labels)
        class_weights = compute_class_weight('balanced', classes=unique_classes, y=train_labels)
        self.class_weights = torch.FloatTensor(class_weights).to(device)
        
        print(f"Class weights: {class_weights}")
        print(f"Class distribution: {np.bincount(train_labels)}")
        
        # Weighted focal loss with label smoothing
        self.criterion = WeightedFocalLoss(
            class_weights=self.class_weights,
            gamma=2.0,
            label_smoothing=0.1
        )
        self.ce_criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Scheduler with warmup
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
        
    def train_epoch(self, epoch):
        """Train for one epoch with warmup"""
        self.model.train()
        total_loss = 0.0
        
        # Warmup learning rate
        if epoch < self.warmup_epochs:
            lr_scale = (epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr_scale * 1e-3
        
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
        
        # Step scheduler after warmup
        if epoch >= self.warmup_epochs:
            self.scheduler.step(epoch - self.warmup_epochs)
        
        return total_loss / len(self.train_loader)
    
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
                loss = self.ce_criterion(output, target)
                total_loss += loss.item()
                
                preds = torch.argmax(output, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        avg_loss = total_loss / len(data_loader)
        macro_f1 = f1_score(np.array(all_targets), np.array(all_preds), average='macro')
        
        return avg_loss, macro_f1, np.array(all_preds), np.array(all_targets)
    
    def train(self, num_epochs, checkpoint_dir):
        """Full training loop"""
        print(f"\nStarting enhanced ASM training for {num_epochs} epochs...")
        print(f"Starting from epoch {self.start_epoch + 1}")
        
        for epoch in range(self.start_epoch, num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"{'='*60}")
            
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            print(f"Train Loss: {train_loss:.4f}")
            
            val_loss, val_f1, _, _ = self.evaluate(self.val_loader)
            self.val_losses.append(val_loss)
            self.val_f1_scores.append(val_f1)
            print(f"Val Loss: {val_loss:.4f}, Val Macro F1: {val_f1:.4f}")
            
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
                    'val_f1_scores': self.val_f1_scores,
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
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    if len(sys.argv) < 2:
        print("Usage: python STMasm_enhanced.py <mode> [--resume <checkpoint_dir>]")
        print("Modes:")
        print("  0: Standard training")
        print("  1: Downsample non-tonal speech")
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
    
    # Set parameters
    if mode == 0:
        print("Mode 0: Standard training")
        ds_nontonal_speech = False
        directory = "model/STM/ASM_Enhanced_corpora_categories/standard"
    elif mode == 1:
        print("Mode 1: Downsample non-tonal speech")
        ds_nontonal_speech = True
        directory = "model/STM/ASM_Enhanced_corpora_categories/downsample"
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
    
    data_prep = prepData_STM_Conformer(ds_nontonal_speech=ds_nontonal_speech)
    train_dataset, val_dataset, test_dataset, n_freq, n_time = data_prep.prepare_datasets()
    
    # Extract training labels for class weights
    train_labels = train_dataset.tensors[1].numpy()
    
    # Create data loaders
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    
    print(f"\nDataLoaders created with batch_size={batch_size}")
    print(f"STM grid dimensions: Time={n_time}, Freq={n_freq}")
    
    # Create enhanced model
    print("\n" + "="*60)
    print("Creating Enhanced ASM-RH model...")
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
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create enhanced trainer
    trainer = EnhancedTrainer(
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
    
    checkpoint = torch.load(os.path.join(checkpoint_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_f1, test_preds, test_targets = trainer.evaluate(test_loader)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Macro F1: {test_f1:.4f}")
    
    # Save test predictions
    np.save(os.path.join(checkpoint_dir, 'test_predictions.npy'), test_preds)
    np.save(os.path.join(checkpoint_dir, 'test_targets.npy'), test_targets)
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)
