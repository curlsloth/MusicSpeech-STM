#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Conformer model with Class Balancing for STM Classification

Improvements over base Conformer:
1. Class-weighted focal loss for severe imbalance (with sqrt-scaled weights)
2. Label smoothing for better generalization
3. SpecAugment-style augmentation during training
4. Warmup learning rate schedule
5. Better gradient clipping and regularization

Based on successful base Conformer (Test F1: 0.8636) with refined class balancing.

Version 2 changes:
- Square root scaling for class weights (gentler balancing)
- Weight capping to prevent over-emphasis
- Reduced focal loss gamma for stability
- Tuned SpecAugment parameters
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
from torchaudio.models import Conformer
import warnings

warnings.filterwarnings('ignore')

# Import data preparation from base model
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
    
    Version 2: Reduced masking parameters for less aggressive augmentation
    """
    def __init__(self, freq_mask_param=3, time_mask_param=15, n_freq_masks=1, n_time_masks=1):
        super(SpecAugment, self).__init__()
        self.freq_mask_param = freq_mask_param  # Reduced from 4
        self.time_mask_param = time_mask_param  # Reduced from 20
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks  # Reduced from 2
        
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
# Enhanced Loss Function
# ============================================================================

class WeightedFocalLoss(nn.Module):
    """
    Focal loss with sqrt-scaled class weights for gentler balancing.
    
    Version 2 improvements:
    - Square root scaling of weights
    - Weight capping at 3.0
    - Reduced gamma for less aggressive focusing
    
    Args:
        class_weights: Per-class weights (sqrt-scaled from data)
        gamma: Focusing parameter (reduced to 1.5)
        label_smoothing: Label smoothing factor (default: 0.1)
    """
    def __init__(self, class_weights=None, gamma=1.5, label_smoothing=0.1):
        super(WeightedFocalLoss, self).__init__()
        self.class_weights = class_weights
        self.gamma = gamma  # Reduced from 2.0
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
# Enhanced Conformer Classifier
# ============================================================================

class BalancedConformerClassifier(nn.Module):
    """
    Conformer-based classifier with SpecAugment for better generalization.
    
    Architecture (unchanged from base):
    1. Input projection to increase feature dimension
    2. Conformer blocks for feature extraction
    3. Global average pooling
    4. Classification head
    
    Version 2: Less aggressive SpecAugment
    """
    def __init__(self, input_dim, num_classes, d_model=128, num_heads=4, 
                 ffn_dim=512, num_layers=4, depthwise_conv_kernel_size=31, dropout=0.1):
        super(BalancedConformerClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # SpecAugment for training (less aggressive)
        self.spec_augment = SpecAugment(
            freq_mask_param=3, time_mask_param=15, 
            n_freq_masks=1, n_time_masks=1
        )
        
        # Input projection: (batch, freq, time) -> (batch, time, d_model)
        self.input_proj = nn.Sequential(
            nn.Conv1d(input_dim, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(d_model),
            nn.Dropout(dropout)
        )
        
        # Conformer blocks
        self.conformer = Conformer(
            input_dim=d_model,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            depthwise_conv_kernel_size=depthwise_conv_kernel_size,
            dropout=dropout,
            use_group_norm=True,
            convolution_first=False,
        )
        
        # Global pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
    def forward(self, x):
        # x: (batch, freq, time)
        batch_size = x.size(0)
        
        # Apply SpecAugment during training
        x = self.spec_augment(x)
        
        # Input projection: (batch, freq, time) -> (batch, d_model, time)
        x = self.input_proj(x)
        
        # Transpose for Conformer: (batch, d_model, time) -> (batch, time, d_model)
        x = x.transpose(1, 2)
        
        # Conformer expects (batch, time, d_model)
        lengths = torch.full((batch_size,), x.size(1), dtype=torch.long, device=x.device)
        x, _ = self.conformer(x, lengths)
        
        # Transpose back: (batch, time, d_model) -> (batch, d_model, time)
        x = x.transpose(1, 2)
        
        # Global pooling: (batch, d_model, time) -> (batch, d_model, 1)
        x = self.global_pool(x)
        
        # Flatten: (batch, d_model, 1) -> (batch, d_model)
        x = x.squeeze(-1)
        
        # Classifier
        x = self.classifier(x)
        
        return x


# ============================================================================
# Enhanced Trainer with Refined Class Balancing
# ============================================================================

class BalancedTrainer:
    """
    Enhanced training manager with refined class balancing strategy.
    
    Version 2 improvements:
    - Square root scaling for class weights
    - Weight capping at 3.0
    - Reduced focal loss gamma (1.5)
    - Better handling of minority classes
    """
    def __init__(self, model, train_loader, val_loader, test_loader, 
                 device, train_labels, lr=1e-4, weight_decay=1e-5, warmup_epochs=5):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.warmup_epochs = warmup_epochs
        
        # Compute refined class weights with sqrt scaling
        unique_classes = np.unique(train_labels)
        class_counts = np.bincount(train_labels)
        
        # Square root scaling: w_i = sqrt(N / (n_classes * n_i))
        total_samples = len(train_labels)
        n_classes = len(unique_classes)
        
        # Compute square root scaled weights
        sqrt_weights = np.sqrt(total_samples / (n_classes * class_counts))
        
        # Cap weights at 3.0 to prevent over-emphasis
        max_weight = 3.0
        capped_weights = np.minimum(sqrt_weights, max_weight)
        
        # Normalize weights to have mean of 1.0
        normalized_weights = capped_weights / capped_weights.mean()
        
        self.class_weights = torch.FloatTensor(normalized_weights).to(device)
        
        print(f"Class distribution: {class_counts}")
        print(f"Raw sqrt weights: {sqrt_weights}")
        print(f"Capped weights: {capped_weights}")
        print(f"Final normalized weights: {normalized_weights}")
        
        # Weighted focal loss with reduced gamma
        self.criterion = WeightedFocalLoss(
            class_weights=self.class_weights,
            gamma=1.5,  # Reduced from 2.0
            label_smoothing=0.1
        )
        
        # Standard cross-entropy for validation (no label smoothing)
        self.ce_criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Scheduler with warmup
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=3
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
                param_group['lr'] = lr_scale * 1e-4
        
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
        
        # Calculate metrics
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        macro_f1 = f1_score(all_targets, all_preds, average='macro')
        
        # Per-class F1 scores
        per_class_f1 = f1_score(all_targets, all_preds, average=None)
        
        return avg_loss, macro_f1, per_class_f1, all_preds, all_targets
    
    def train(self, num_epochs, checkpoint_dir):
        """Full training loop"""
        print(f"\nStarting balanced Conformer training for {num_epochs} epochs...")
        print(f"Starting from epoch {self.start_epoch + 1}")
        
        class_names = ['speech:non-tonal', 'speech:tonal', 'music:vocal', 
                      'music:non-vocal', 'env:urban', 'env:wildlife']
        
        for epoch in range(self.start_epoch, num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"{'='*60}")
            
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            print(f"Train Loss: {train_loss:.4f}")
            
            val_loss, val_f1, per_class_f1, _, _ = self.evaluate(self.val_loader)
            self.val_losses.append(val_loss)
            self.val_f1_scores.append(val_f1)
            
            print(f"Val Loss: {val_loss:.4f}, Val Macro F1: {val_f1:.4f}")
            print("Per-class F1 scores:")
            for i, name in enumerate(class_names):
                print(f"  {name}: {per_class_f1[i]:.4f}")
            
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Current learning rate: {current_lr:.6f}")
            
            # Learning rate scheduling (after warmup)
            if epoch >= self.warmup_epochs:
                self.scheduler.step(val_f1)
            
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
        print("Usage: python STMconformer_balanced.py <mode> [--resume <checkpoint_dir>]")
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
        directory = "model/STM/Conformer_Balanced_corpora_categories/standard"
    elif mode == 1:
        print("Mode 1: Downsample non-tonal speech")
        ds_nontonal_speech = True
        directory = "model/STM/Conformer_Balanced_corpora_categories/downsample"
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
    
    # Create balanced model (same architecture as base)
    print("\n" + "="*60)
    print("Creating Balanced Conformer model...")
    print("="*60)
    
    num_classes = 6
    model = BalancedConformerClassifier(
        input_dim=n_freq,      # 20
        num_classes=num_classes,
        d_model=128,
        num_heads=4,
        ffn_dim=512,
        num_layers=4,
        depthwise_conv_kernel_size=31,
        dropout=0.1
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create balanced trainer
    trainer = BalancedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        train_labels=train_labels,
        lr=1e-4,
        weight_decay=1e-5,
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
    
    test_loss, test_f1, per_class_f1, test_preds, test_targets = trainer.evaluate(test_loader)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Macro F1: {test_f1:.4f}")
    
    class_names = ['speech:non-tonal', 'speech:tonal', 'music:vocal', 
                  'music:non-vocal', 'env:urban', 'env:wildlife']
    print("\nPer-class Test F1 scores:")
    for i, name in enumerate(class_names):
        print(f"  {name}: {per_class_f1[i]:.4f}")
    
    # Save test predictions
    np.save(os.path.join(checkpoint_dir, 'test_predictions.npy'), test_preds)
    np.save(os.path.join(checkpoint_dir, 'test_targets.npy'), test_targets)
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)
