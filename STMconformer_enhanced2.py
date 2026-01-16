#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Conformer v2 - Simplified and Stable
Based on lessons learned from v1 failure analysis.

Key changes from v1:
- Removed LR restarts (use simple cosine decay)
- Fixed warmup bug (store base LR)
- Reduced augmentation (pick ONE: SpecAugment OR Mixup)
- Removed stochastic depth (redundant with dropout)
- Simplified mixup (batch-level only)
- Better logging and monitoring
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
import warnings

warnings.filterwarnings('ignore')


class SpecAugment(nn.Module):
    """Conservative SpecAugment - reduced strength"""
    def __init__(self, freq_mask_param=2, time_mask_param=10, num_masks=1):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_masks = num_masks
    
    def forward(self, x):
        if not self.training:
            return x
        
        batch, freq, time = x.shape
        
        # Apply masks
        for _ in range(self.num_masks):
            # Frequency mask
            if self.freq_mask_param > 0:
                f = torch.randint(0, min(self.freq_mask_param, freq//4), (1,)).item()
                f0 = torch.randint(0, max(1, freq - f), (1,)).item()
                x[:, f0:f0+f, :] = 0
            
            # Time mask
            if self.time_mask_param > 0:
                t = torch.randint(0, min(self.time_mask_param, time//4), (1,)).item()
                t0 = torch.randint(0, max(1, time - t), (1,)).item()
                x[:, :, t0:t0+t] = 0
        
        return x


class MixupDataset(Dataset):
    """Simplified Mixup - batch-level lambda only"""
    def __init__(self, dataset, alpha=0.2, prob=0.5):
        self.dataset = dataset
        self.alpha = alpha
        self.prob = prob  # Probability of applying mixup
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        x1, y1 = self.dataset[idx]
        
        if np.random.rand() < self.prob:
            idx2 = np.random.randint(0, len(self.dataset))
            x2, y2 = self.dataset[idx2]
            lam = np.random.beta(self.alpha, self.alpha)
            x = lam * x1 + (1 - lam) * x2
            return x, y1, y2, lam
        else:
            return x1, y1, y1, 1.0


class AttentionPooling(nn.Module):
    """Attention-based pooling"""
    def __init__(self, d_model):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )
    
    def forward(self, x):
        # x: (batch, d_model, time)
        x = x.transpose(1, 2)  # (batch, time, d_model)
        attn_weights = self.attention(x)
        attn_weights = F.softmax(attn_weights, dim=1)
        x = torch.sum(x * attn_weights, dim=1)
        return x


class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing"""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        n_classes = pred.size(-1)
        log_preds = F.log_softmax(pred, dim=-1)
        loss = -log_preds.sum(dim=-1).mean()
        nll = F.nll_loss(log_preds, target, reduction='mean')
        return self.smoothing * loss / n_classes + (1 - self.smoothing) * nll


class EnhancedConformerV2(nn.Module):
    """Simplified Enhanced Conformer"""
    def __init__(self, input_dim, num_classes, d_model=128, num_heads=4, 
                 ffn_dim=512, num_layers=4, dropout=0.1, 
                 use_spec_augment=False):  # Default: no augmentation
        super().__init__()
        
        self.use_spec_augment = use_spec_augment
        
        if use_spec_augment:
            self.spec_augment = SpecAugment(freq_mask_param=2, time_mask_param=10, num_masks=1)
        
        # Input projection (simpler than v1)
        self.input_proj = nn.Sequential(
            nn.Conv1d(input_dim, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Conformer
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
        
        # Attention pooling
        self.attention_pool = AttentionPooling(d_model)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        if self.training and self.use_spec_augment:
            x = self.spec_augment(x)
        
        x = self.input_proj(x)
        x = x.transpose(1, 2)
        
        lengths = torch.full((batch_size,), x.size(1), dtype=torch.long, device=x.device)
        x, _ = self.conformer(x, lengths)
        
        x = x.transpose(1, 2)
        x = self.attention_pool(x)
        x = self.classifier(x)
        
        return x


class StableTrainer:
    """Simplified, stable trainer"""
    def __init__(self, model, train_loader, val_loader, test_loader, 
                 device, lr=1e-4, weight_decay=1e-5, use_mixup=False,
                 label_smoothing=0.1, warmup_epochs=5, num_epochs=50):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.use_mixup = use_mixup
        self.warmup_epochs = warmup_epochs
        self.num_epochs = num_epochs
        
        self.criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
        
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        # FIXED: Store base LR for warmup
        self.base_lr = lr
        
        # Simple cosine decay WITHOUT restarts
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
    
    def mixup_criterion(self, pred, y1, y2, lam):
        """Simple batch-level mixup loss"""
        return lam * self.criterion(pred, y1) + (1 - lam) * self.criterion(pred, y2)
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        
        # FIXED: Proper warmup implementation
        if epoch < self.warmup_epochs:
            lr_scale = (epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr_scale * self.base_lr
        
        for batch_idx, batch_data in enumerate(self.train_loader):
            if self.use_mixup:
                data, y1, y2, lam = batch_data
                data = data.to(self.device)
                y1 = y1.to(self.device)
                y2 = y2.to(self.device)
                # lam is always a scalar in this simplified version
                
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.mixup_criterion(output, y1, y2, lam)
            else:
                data, target = batch_data
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"  Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}")
        
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
        print(f"\nStarting stable training for {self.num_epochs} epochs...")
        print(f"Warmup epochs: {self.warmup_epochs}")
        print(f"Base LR: {self.base_lr}")
        print(f"Mixup: {'Enabled' if self.use_mixup else 'Disabled'}")
        print(f"SpecAugment: {'Enabled' if self.model.use_spec_augment else 'Disabled'}")
        
        for epoch in range(self.num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{self.num_epochs}")
            print(f"{'='*60}")
            
            # Train
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            print(f"Train Loss: {train_loss:.4f}")
            
            # Validate
            val_loss, val_f1, _, _ = self.evaluate(self.val_loader)
            self.val_losses.append(val_loss)
            self.val_f1_scores.append(val_f1)
            
            print(f"Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")
            
            # Learning rate scheduling (after warmup)
            if epoch >= self.warmup_epochs:
                self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)
            print(f"Learning rate: {current_lr:.6f}")
            
            # Save best model
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
            
            # Save checkpoint every 10 epochs
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
        
        # Save training history
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
    
    if len(sys.argv) < 3:
        print("Usage: python STMconformer_enhanced2.py <mode> <augmentation>")
        print("Modes:")
        print("  0: Standard training")
        print("  1: Downsample non-tonal speech")
        print("Augmentation:")
        print("  none: No augmentation (just label smoothing + attention pooling)")
        print("  spec: SpecAugment only")
        print("  mix: Mixup only")
        sys.exit(1)
    
    mode = int(sys.argv[1])
    aug_type = sys.argv[2].lower()
    
    if mode == 0:
        print("Mode: Standard training")
        ds_nontonal_speech = False
        base_dir = "model/STM/Conformer_Enhanced2/standard"
    elif mode == 1:
        print("Mode: Downsample non-tonal speech")
        ds_nontonal_speech = True
        base_dir = "model/STM/Conformer_Enhanced2/downsample"
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)
    
    # Configure augmentation
    use_spec_augment = (aug_type == 'spec')
    use_mixup = (aug_type == 'mix')
    
    if aug_type == 'none':
        print("Augmentation: None (baseline improvements only)")
        aug_dir = "none"
    elif use_spec_augment:
        print("Augmentation: SpecAugment")
        aug_dir = "specaugment"
    elif use_mixup:
        print("Augmentation: Mixup")
        aug_dir = "mixup"
    else:
        print(f"Unknown augmentation: {aug_type}")
        sys.exit(1)
    
    # Create directory
    time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    checkpoint_dir = os.path.join(base_dir, aug_dir, "ckpt", time_stamp)
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoint directory: {checkpoint_dir}")
    
    # Prepare data
    print("\n" + "="*60)
    print("Loading and preparing data...")
    print("="*60)
    
    data_prep = prepData_STM_Conformer(ds_nontonal_speech=ds_nontonal_speech)
    train_dataset, val_dataset, test_dataset, n_freq, n_time = data_prep.prepare_datasets()
    
    # Wrap with Mixup if needed
    if use_mixup:
        train_dataset = MixupDataset(train_dataset, alpha=0.2, prob=0.5)
    
    # Create data loaders
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    
    print(f"\nDataLoaders created with batch_size={batch_size}")
    
    # Create model
    print("\n" + "="*60)
    print("Creating Enhanced Conformer V2...")
    print("="*60)
    
    num_classes = 6
    model = EnhancedConformerV2(
        input_dim=n_freq,
        num_classes=num_classes,
        d_model=128,
        num_heads=4,
        ffn_dim=512,
        num_layers=4,
        dropout=0.1,
        use_spec_augment=use_spec_augment
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Create trainer
    trainer = StableTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        lr=1e-4,
        weight_decay=1e-5,
        use_mixup=use_mixup,
        label_smoothing=0.1,
        warmup_epochs=5,
        num_epochs=50
    )
    
    # Train
    trainer.train(checkpoint_dir=checkpoint_dir)
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("Evaluating on test set...")
    print("="*60)
    
    checkpoint = torch.load(os.path.join(checkpoint_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_f1, test_preds, test_targets = trainer.evaluate(test_loader)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test F1: {test_f1:.4f}")
    
    # Save results
    np.save(os.path.join(checkpoint_dir, 'test_predictions.npy'), test_preds)
    np.save(os.path.join(checkpoint_dir, 'test_targets.npy'), test_targets)
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)
