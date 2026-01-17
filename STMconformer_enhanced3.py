#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Conformer V3 - Minimal, Proven Improvements
Based on failure analysis of V2 and comparison with baseline.

Philosophy: Only add what demonstrably helps. Keep it simple.

Key changes from V2:
- Removed attention pooling (hurt performance)
- Removed SpecAugment (too aggressive)
- Removed mixup (buggy implementation)
- Kept label smoothing (reduced to 0.05)
- Fixed warmup implementation
- Kept simple cosine schedule (no restarts)
- Added gradient accumulation for speed
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import datetime
import os
import sys
import warnings

warnings.filterwarnings('ignore')


class LabelSmoothingCrossEntropy(nn.Module):
    """Reduced label smoothing (0.05 instead of 0.1)"""
    def __init__(self, smoothing=0.05):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        n_classes = pred.size(-1)
        log_preds = F.log_softmax(pred, dim=-1)
        loss = -log_preds.sum(dim=-1).mean()
        nll = F.nll_loss(log_preds, target, reduction='mean')
        return self.smoothing * loss / n_classes + (1 - self.smoothing) * nll


class MinimalEnhancedConformer(nn.Module):
    """Minimal enhancements over baseline"""
    def __init__(self, input_dim, num_classes, d_model=128, num_heads=4, 
                 ffn_dim=512, num_layers=4, dropout=0.1):
        super().__init__()
        
        # Simple input projection (like baseline)
        self.input_proj = nn.Sequential(
            nn.Conv1d(input_dim, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Conformer (same as baseline)
        from torchaudio.models import Conformer
        self.conformer = Conformer(
            input_dim=d_model,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            depthwise_conv_kernel_size=31,
            dropout=dropout,
            use_group_norm=True,
            convolution_first=False,
        )
        
        # Simple global average pooling (like baseline)
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Enhanced classifier with LayerNorm
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Project input
        x = self.input_proj(x)
        x = x.transpose(1, 2)
        
        # Conformer
        lengths = torch.full((batch_size,), x.size(1), dtype=torch.long, device=x.device)
        x, _ = self.conformer(x, lengths)
        
        # Pool and classify
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)
        x = self.classifier(x)
        
        return x


class FocusedTrainer:
    """Simplified trainer with proven techniques only"""
    def __init__(self, model, train_loader, val_loader, test_loader, 
                 device, lr=1e-4, weight_decay=1e-5,
                 label_smoothing=0.05, warmup_epochs=5, num_epochs=50,
                 accumulation_steps=1):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.warmup_epochs = warmup_epochs
        self.num_epochs = num_epochs
        self.accumulation_steps = accumulation_steps
        
        self.criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
        
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        self.base_lr = lr
        
        # Simple cosine decay
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs - warmup_epochs,
            eta_min=1e-6
        )
        
        self.best_val_f1 = 0.0
        self.best_epoch = 0
        self.train_losses = []
        self.val_losses = []
        self.val_f1_scores = []
        self.learning_rates = []
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        
        # Warmup
        if epoch < self.warmup_epochs:
            lr_scale = (epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr_scale * self.base_lr
        
        self.optimizer.zero_grad()
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            output = self.model(data)
            loss = self.criterion(output, target)
            loss = loss / self.accumulation_steps
            
            loss.backward()
            
            if (batch_idx + 1) % self.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * self.accumulation_steps
            
            if batch_idx % 200 == 0:
                print(f"  Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item() * self.accumulation_steps:.4f}")
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def evaluate(self, data_loader):
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
        
        from sklearn.metrics import f1_score
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        macro_f1 = f1_score(all_targets, all_preds, average='macro')
        
        return avg_loss, macro_f1, all_preds, all_targets
    
    def train(self, checkpoint_dir):
        print(f"\nStarting focused training for {self.num_epochs} epochs...")
        print(f"Warmup epochs: {self.warmup_epochs}")
        print(f"Base LR: {self.base_lr}")
        print(f"Label smoothing: {self.criterion.smoothing}")
        print(f"Gradient accumulation steps: {self.accumulation_steps}")
        
        for epoch in range(self.num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{self.num_epochs}")
            print(f"{'='*60}")
            
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            print(f"Train Loss: {train_loss:.4f}")
            
            val_loss, val_f1, _, _ = self.evaluate(self.val_loader)
            self.val_losses.append(val_loss)
            self.val_f1_scores.append(val_f1)
            
            print(f"Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")
            
            if epoch >= self.warmup_epochs:
                self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)
            print(f"Learning rate: {current_lr:.6f}")
            
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.best_epoch = epoch + 1
                checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_f1': val_f1,
                    'val_loss': val_loss,
                }, checkpoint_path)
                print(f"✓ NEW BEST! Saved model with Val F1: {val_f1:.4f}")
            else:
                print(f"  (Best: {self.best_val_f1:.4f} at epoch {self.best_epoch})")
            
            if (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_f1': val_f1,
                }, checkpoint_path)
        
        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Best Val F1: {self.best_val_f1:.4f} (Epoch {self.best_epoch})")
        print(f"{'='*60}")
        
        np.save(os.path.join(checkpoint_dir, 'train_losses.npy'), self.train_losses)
        np.save(os.path.join(checkpoint_dir, 'val_losses.npy'), self.val_losses)
        np.save(os.path.join(checkpoint_dir, 'val_f1_scores.npy'), self.val_f1_scores)
        np.save(os.path.join(checkpoint_dir, 'learning_rates.npy'), self.learning_rates)


from STMconformer_model import prepData_STM_Conformer


if __name__ == "__main__":
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    if len(sys.argv) < 2:
        print("Usage: python STMconformer_enhanced3.py <mode> [accumulation_steps]")
        print("Modes:")
        print("  0: Standard training")
        print("  1: Downsample non-tonal speech")
        print("Accumulation steps (optional, default=2):")
        print("  2: Effective batch size 256 (faster)")
        print("  4: Effective batch size 512 (even faster)")
        sys.exit(1)
    
    mode = int(sys.argv[1])
    accumulation_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    
    if mode == 0:
        print("Mode: Standard training")
        ds_nontonal_speech = False
        base_dir = "model/STM/Conformer_Enhanced3/standard"
    elif mode == 1:
        print("Mode: Downsample non-tonal speech")
        ds_nontonal_speech = True
        base_dir = "model/STM/Conformer_Enhanced3/downsample"
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)
    
    print(f"Gradient accumulation steps: {accumulation_steps}")
    print(f"Effective batch size: {128 * accumulation_steps}")
    
    time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    checkpoint_dir = os.path.join(base_dir, "ckpt", time_stamp)
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoint directory: {checkpoint_dir}")
    
    print("\n" + "="*60)
    print("Loading and preparing data...")
    print("="*60)
    
    data_prep = prepData_STM_Conformer(ds_nontonal_speech=ds_nontonal_speech)
    train_dataset, val_dataset, test_dataset, n_freq, n_time = data_prep.prepare_datasets()
    
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    
    print(f"\nDataLoaders created with batch_size={batch_size}")
    
    print("\n" + "="*60)
    print("Creating Minimal Enhanced Conformer...")
    print("="*60)
    
    num_classes = 6
    model = MinimalEnhancedConformer(
        input_dim=n_freq,
        num_classes=num_classes,
        d_model=128,
        num_heads=4,
        ffn_dim=512,
        num_layers=4,
        dropout=0.1
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    trainer = FocusedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        lr=1e-4,
        weight_decay=1e-5,
        label_smoothing=0.05,
        warmup_epochs=5,
        num_epochs=50,
        accumulation_steps=accumulation_steps
    )
    
    trainer.train(checkpoint_dir=checkpoint_dir)
    
    print("\n" + "="*60)
    print("Evaluating on test set...")
    print("="*60)
    
    checkpoint = torch.load(os.path.join(checkpoint_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_f1, test_preds, test_targets = trainer.evaluate(test_loader)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test F1: {test_f1:.4f}")
    
    np.save(os.path.join(checkpoint_dir, 'test_predictions.npy'), test_preds)
    np.save(os.path.join(checkpoint_dir, 'test_targets.npy'), test_targets)
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)
