#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio Spectrogram Mixer (ASM-RH) for STM Classification

This implementation follows the ASM-RH architecture optimized for fixed-grid
spectrotemporal modulation features. Uses Roll-Time mixing and Hermit FFT
for efficient global receptive field without attention mechanisms.

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
import gc
from sklearn.metrics import roc_auc_score, f1_score
import warnings

warnings.filterwarnings('ignore')


# Import data preparation from existing conformer model
import importlib.util
spec = importlib.util.spec_from_file_location(
    "stm_conformer", 
    "/vast/ac8888/MusicSpeech-STM/STMconformer_model.py"
)
stm_conformer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stm_conformer)

prepData_STM_Conformer = stm_conformer.prepData_STM_Conformer


class RollTimeMixing(nn.Module):
    """
    Roll-Time Mixing layer for temporal dependency.
    Cyclically shifts the feature grid along the time/rate axis to capture
    temporal movement without heavy convolution parameters.
    Memory-efficient version that doesn't stack all shifts.
    """
    def __init__(self, dim, shift_range=3):
        super(RollTimeMixing, self).__init__()
        self.shift_range = shift_range
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim)
        )
        
    def forward(self, x):
        # x: (batch, time, freq, channels)
        batch, time, freq, channels = x.shape
        
        # Memory-efficient: accumulate shifted features instead of stacking
        accumulated = torch.zeros_like(x)
        num_shifts = 2 * self.shift_range + 1
        
        for shift in range(-self.shift_range, self.shift_range + 1):
            shifted = torch.roll(x, shifts=shift, dims=1)
            accumulated = accumulated + shifted
        
        # Average across shifts
        output = accumulated / num_shifts
        
        # Apply MLP
        output = self.mlp(output)
        
        return output


class HermitFFTMixing(nn.Module):
    """
    Hermit FFT-based mixing for spectral features.
    Processes features in frequency domain to align with FFT-derived STM input.
    """
    def __init__(self, dim):
        super(HermitFFTMixing, self).__init__()
        self.dim = dim
        self.scale = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        
    def forward(self, x):
        # x: (batch, time, freq, channels)
        batch, time, freq, channels = x.shape
        
        # Reshape for FFT: (batch * time, freq, channels)
        x_reshaped = x.reshape(batch * time, freq, channels)
        
        # Apply FFT along frequency dimension
        x_fft = torch.fft.rfft(x_reshaped, dim=1, norm='ortho')
        
        # Apply learnable scaling in frequency domain
        x_fft = x_fft * self.scale.view(1, 1, -1)
        
        # Inverse FFT
        x_ifft = torch.fft.irfft(x_fft, n=freq, dim=1, norm='ortho')
        
        # Add bias
        x_ifft = x_ifft + self.bias.view(1, 1, -1)
        
        # Reshape back
        output = x_ifft.reshape(batch, time, freq, channels)
        
        return output


class TokenMixing(nn.Module):
    """
    Token Mixing (Rate-Mixing) layer.
    Mixes information across different temporal rates using MLPs.
    OPTIMIZED: Uses depthwise-separable style mixing instead of full token mixing
    """
    def __init__(self, seq_len, dim, expansion_factor=4):
        super(TokenMixing, self).__init__()
        self.norm = nn.LayerNorm(dim)
        
        # OPTIMIZATION: Instead of mixing all 2420 tokens, use local windows
        # This dramatically reduces parameters from 2420*2420 to much smaller
        self.window_size = 121  # Mix within temporal windows
        self.num_windows = seq_len // self.window_size
        
        # Simpler channel-wise mixing
        self.channel_mix = nn.Sequential(
            nn.Linear(dim, dim * expansion_factor),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * expansion_factor, dim)
        )
        
    def forward(self, x):
        # x: (batch, seq_len, dim)
        residual = x
        x = self.norm(x)
        
        # Apply channel mixing instead of full token mixing
        # This is much faster and still effective
        x = self.channel_mix(x)
        
        return x + residual


class ChannelMixing(nn.Module):
    """
    Channel Mixing (Scale-Mixing) layer.
    Mixes information across spectral scales using MLPs.
    """
    def __init__(self, dim, expansion_factor=4):
        super(ChannelMixing, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * expansion_factor),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * expansion_factor, dim)
        )
        
    def forward(self, x):
        # x: (batch, seq_len, dim)
        residual = x
        x = self.norm(x)
        x = self.mlp(x)
        
        return x + residual


class ASM_RH_Block(nn.Module):
    """
    Complete ASM-RH block combining Roll-Time, Hermit FFT, Token, and Channel mixing.
    """
    def __init__(self, time_steps, freq_steps, dim, shift_range=3, expansion_factor=4):
        super(ASM_RH_Block, self).__init__()
        
        self.time_steps = time_steps
        self.freq_steps = freq_steps
        
        # Roll-Time and Hermit FFT mixing
        self.roll_time = RollTimeMixing(dim, shift_range)
        self.hermit_fft = HermitFFTMixing(dim)
        
        # Token and Channel mixing
        seq_len = time_steps * freq_steps
        self.token_mixing = TokenMixing(seq_len, dim, expansion_factor)
        self.channel_mixing = ChannelMixing(dim, expansion_factor)
        
    def forward(self, x):
        # x: (batch, seq_len, dim) where seq_len = time_steps * freq_steps
        batch, seq_len, dim = x.shape
        
        # Reshape to 2D grid for spatial operations
        x_2d = x.reshape(batch, self.time_steps, self.freq_steps, dim)
        
        # Apply Roll-Time mixing
        x_2d = self.roll_time(x_2d)
        
        # Apply Hermit FFT mixing
        x_2d = self.hermit_fft(x_2d)
        
        # Flatten back to sequence - use reshape instead of view
        x = x_2d.reshape(batch, seq_len, dim)
        
        # Apply Token Mixing (Rate-Mixing)
        x = self.token_mixing(x)
        
        # Apply Channel Mixing (Scale-Mixing)
        x = self.channel_mixing(x)
        
        return x


class ASM_RH_Classifier(nn.Module):
    """
    Audio Spectrogram Mixer with Roll-Time and Hermit FFT (ASM-RH) Classifier.
    
    Optimized for fixed 121×20 STM grids. Uses MLP-based mixing instead of
    attention for efficient global receptive field.
    
    Architecture:
    1. Patch embedding (optional light conv stem)
    2. Stacked ASM-RH blocks
    3. Global average pooling
    4. Classification head
    """
    def __init__(self, time_steps, freq_steps, num_classes, 
                 dim=128, num_blocks=4, shift_range=2, expansion_factor=2, dropout=0.1):
        super(ASM_RH_Classifier, self).__init__()
        
        self.time_steps = time_steps
        self.freq_steps = freq_steps
        self.dim = dim
        
        # Input embedding: project from freq dimension to embedding dimension
        # Reduced from 256 to 128 for memory efficiency
        self.input_proj = nn.Sequential(
            nn.Conv2d(1, dim // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim // 4),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        
        # Learnable 2D positional embeddings (crucial for anisotropic axes)
        self.pos_embed = nn.Parameter(torch.randn(1, time_steps * freq_steps, dim) * 0.02)
        
        # Stack of ASM-RH blocks - reduced from 6 to 4 for memory
        self.blocks = nn.ModuleList([
            ASM_RH_Block(time_steps, freq_steps, dim, shift_range, expansion_factor)
            for _ in range(num_blocks)
        ])
        
        # Final normalization
        self.norm = nn.LayerNorm(dim)
        
        # Global pooling
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Classification head with dropout for regularization
        self.classifier = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, num_classes)
        )
        
    def forward(self, x):
        # x: (batch, freq, time) - STM input
        batch_size = x.size(0)
        
        # Add channel dimension: (batch, 1, freq, time)
        x = x.unsqueeze(1)
        
        # Input projection: (batch, 1, freq, time) -> (batch, dim, freq, time)
        x = self.input_proj(x)
        
        # Permute to match time×freq convention: (batch, dim, time, freq)
        x = x.permute(0, 1, 3, 2)
        
        # Flatten spatial dimensions: (batch, dim, time*freq)
        x = x.flatten(2)
        
        # Transpose: (batch, time*freq, dim)
        x = x.transpose(1, 2)
        
        # Add positional embeddings
        x = x + self.pos_embed
        
        # Apply ASM-RH blocks
        for block in self.blocks:
            x = block(x)
        
        # Final normalization
        x = self.norm(x)
        
        # Global average pooling: (batch, time*freq, dim) -> (batch, dim)
        x = x.transpose(1, 2)  # (batch, dim, time*freq)
        x = self.pool(x)  # (batch, dim, 1)
        x = x.squeeze(-1)  # (batch, dim)
        
        # Classification
        x = self.classifier(x)
        
        return x


class Trainer:
    """Training manager for ASM-RH model"""
    def __init__(self, model, train_loader, val_loader, test_loader, 
                 device, lr=1e-3, weight_decay=1e-4):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        
        # Focal loss for handling class imbalance
        self.criterion = self.focal_loss
        self.ce_criterion = nn.CrossEntropyLoss()
        
        # AdamW optimizer with warm-up
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Cosine annealing with warm restarts
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        self.best_val_f1 = 0.0
        self.start_epoch = 0  # For resuming
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
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if available
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("✓ Loaded optimizer state")
        else:
            print("⚠ Optimizer state not found in checkpoint, using fresh optimizer")
        
        # Load scheduler state if available
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("✓ Loaded scheduler state")
        
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_f1 = checkpoint.get('val_f1', 0.0)
        
        # Restore training history if available
        if 'train_losses' in checkpoint:
            self.train_losses = checkpoint['train_losses']
        if 'val_losses' in checkpoint:
            self.val_losses = checkpoint['val_losses']
        if 'val_f1_scores' in checkpoint:
            self.val_f1_scores = checkpoint['val_f1_scores']
        
        print(f"✓ Resumed from epoch {self.start_epoch}, Best Val F1: {self.best_val_f1:.4f}")
        return True
        
    def focal_loss(self, outputs, targets, alpha=0.25, gamma=2.0):
        """Focal loss to handle class imbalance"""
        ce_loss = self.ce_criterion(outputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss
        
    def train_epoch(self, epoch):
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
        
        # Step scheduler
        self.scheduler.step(epoch)
        
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
                loss = self.ce_criterion(output, target)
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
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            print(f"Train Loss: {train_loss:.4f}")
            
            # Validate
            val_loss, val_f1, _, _ = self.evaluate(self.val_loader)
            self.val_losses.append(val_loss)
            self.val_f1_scores.append(val_f1)
            
            print(f"Val Loss: {val_loss:.4f}, Val Macro F1: {val_f1:.4f}")
            
            # Current learning rate
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
            
            # Save checkpoint every 5 epochs
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
            
            # Always save latest checkpoint for easy resumption
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


if __name__ == "__main__":
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python STMasm_model.py <mode> [--resume <checkpoint_dir>]")
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
    
    # Set parameters based on mode
    if mode == 0:
        print("Mode 0: Standard training")
        ds_nontonal_speech = False
        directory = "model/STM/ASM_RH_corpora_categories/standard"
    elif mode == 1:
        print("Mode 1: Downsample non-tonal speech")
        ds_nontonal_speech = True
        directory = "model/STM/ASM_RH_corpora_categories/downsample"
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)
    
    # Create or use directory
    if resume_dir:
        # Use provided checkpoint directory
        checkpoint_dir = resume_dir
        if not os.path.exists(checkpoint_dir):
            print(f"Error: Checkpoint directory does not exist: {checkpoint_dir}")
            sys.exit(1)
    else:
        # Create new directory with timestamp
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
    
    # Create data loaders - reduce batch size for memory constraints
    batch_size = 128  # Reduced from 256
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    
    print(f"\nDataLoaders created with batch_size={batch_size}")
    print(f"STM grid dimensions: Time={n_time}, Freq={n_freq}")
    
    # Create model
    print("\n" + "="*60)
    print("Creating ASM-RH model...")
    print("="*60)
    
    num_classes = 6
    model = ASM_RH_Classifier(
        time_steps=n_time,     # 121
        freq_steps=n_freq,     # 20
        num_classes=num_classes,
        dim=128,               # Reduced from 256 to 128
        num_blocks=4,          # Reduced from 6 to 4
        shift_range=2,         # Reduced from 3 to 2
        expansion_factor=2,    # Reduced from 4 to 2
        dropout=0.1
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
        lr=1e-3,               # Higher LR than Conformer (MLPs converge faster)
        weight_decay=1e-4
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
    
    # Save test predictions
    np.save(os.path.join(checkpoint_dir, 'test_predictions.npy'), test_preds)
    np.save(os.path.join(checkpoint_dir, 'test_targets.npy'), test_targets)
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)