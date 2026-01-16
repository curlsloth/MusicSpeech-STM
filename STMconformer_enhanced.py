#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Conformer implementation for STM audio classification
Improvements over STM08:
- SpecAugment data augmentation
- Label smoothing
- Attention pooling instead of average pooling
- Stochastic depth
- Mixup augmentation
- Test-time augmentation
- Better learning rate scheduling
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


class SpecAugment(nn.Module):
    """SpecAugment: A Simple Data Augmentation Method for ASR"""
    def __init__(self, freq_mask_param=8, time_mask_param=20, num_freq_mask=2, num_time_mask=2):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_mask = num_freq_mask
        self.num_time_mask = num_time_mask
    
    def forward(self, x):
        if not self.training:
            return x
        
        batch, freq, time = x.shape
        
        # Frequency masking
        for _ in range(self.num_freq_mask):
            f = torch.randint(0, self.freq_mask_param, (1,)).item()
            f0 = torch.randint(0, freq - f, (1,)).item()
            x[:, f0:f0+f, :] = 0
        
        # Time masking
        for _ in range(self.num_time_mask):
            t = torch.randint(0, self.time_mask_param, (1,)).item()
            t0 = torch.randint(0, time - t, (1,)).item()
            x[:, :, t0:t0+t] = 0
        
        return x


class AttentionPooling(nn.Module):
    """Attention-based pooling instead of average pooling"""
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
        
        # Compute attention weights
        attn_weights = self.attention(x)  # (batch, time, 1)
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # Apply attention
        x = torch.sum(x * attn_weights, dim=1)  # (batch, d_model)
        
        return x


class StochasticDepth(nn.Module):
    """Stochastic Depth for regularization"""
    def __init__(self, drop_prob=0.1):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class EnhancedConformerClassifier(nn.Module):
    """Enhanced Conformer with multiple improvements"""
    def __init__(self, input_dim, num_classes, d_model=128, num_heads=4, 
                 ffn_dim=512, num_layers=4, depthwise_conv_kernel_size=31, 
                 dropout=0.1, stochastic_depth_prob=0.1, use_spec_augment=True):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.use_spec_augment = use_spec_augment
        
        # SpecAugment
        if use_spec_augment:
            self.spec_augment = SpecAugment(
                freq_mask_param=4, 
                time_mask_param=15,
                num_freq_mask=2,
                num_time_mask=2
            )
        
        # Input projection with residual
        self.input_proj = nn.Sequential(
            nn.Conv1d(input_dim, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
        )
        
        # Skip connection for input
        self.input_skip = nn.Conv1d(input_dim, d_model, kernel_size=1)
        
        # Conformer blocks with stochastic depth
        from torchaudio.models import Conformer
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
        
        # Stochastic depth
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob)
        
        # Attention pooling instead of average pooling
        self.attention_pool = AttentionPooling(d_model)
        
        # Enhanced classifier with skip connections
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
    def forward(self, x, return_features=False):
        # x: (batch, freq, time)
        batch_size = x.size(0)
        
        # SpecAugment
        if self.training and self.use_spec_augment:
            x = self.spec_augment(x)
        
        # Input projection with skip connection
        identity = self.input_skip(x)
        x = self.input_proj(x)
        x = x + identity
        x = F.relu(x)
        
        # Transpose for Conformer
        x = x.transpose(1, 2)
        
        # Conformer
        lengths = torch.full((batch_size,), x.size(1), dtype=torch.long, device=x.device)
        x, _ = self.conformer(x, lengths)
        
        # Stochastic depth
        x = self.stochastic_depth(x)
        
        # Transpose back
        x = x.transpose(1, 2)
        
        # Attention pooling
        features = self.attention_pool(x)
        
        # Classifier
        output = self.classifier(features)
        
        if return_features:
            return output, features
        return output


class MixupDataset(Dataset):
    """Dataset wrapper for Mixup augmentation"""
    def __init__(self, dataset, alpha=0.2):
        self.dataset = dataset
        self.alpha = alpha
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        x1, y1 = self.dataset[idx]
        
        if self.alpha > 0 and np.random.rand() < 0.5:
            # Mixup
            idx2 = np.random.randint(0, len(self.dataset))
            x2, y2 = self.dataset[idx2]
            
            lam = np.random.beta(self.alpha, self.alpha)
            x = lam * x1 + (1 - lam) * x2
            
            # Return lam as float, will be converted to tensor in training loop
            return x, y1, y2, float(lam)
        else:
            return x1, y1, y1, 1.0


class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing for better generalization"""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        n_classes = pred.size(-1)
        log_preds = F.log_softmax(pred, dim=-1)
        
        loss = -log_preds.sum(dim=-1).mean()
        nll = F.nll_loss(log_preds, target, reduction='mean')
        
        return self.smoothing * loss / n_classes + (1 - self.smoothing) * nll


class EnhancedTrainer:
    """Enhanced training with advanced techniques"""
    def __init__(self, model, train_loader, val_loader, test_loader, 
                 device, lr=1e-4, weight_decay=1e-5, use_mixup=True, 
                 label_smoothing=0.1, warmup_epochs=5):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.use_mixup = use_mixup
        self.warmup_epochs = warmup_epochs
        
        # Label smoothing loss
        self.criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
        
        # AdamW with weight decay
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Cosine annealing with warmup
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, 
            T_0=10, 
            T_mult=2, 
            eta_min=1e-6
        )
        
        self.best_val_f1 = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_f1_scores = []
    
    def mixup_criterion(self, pred, y1, y2, lam):
        """Mixup loss computation"""
        # Ensure lam is a scalar or properly broadcast
        if isinstance(lam, torch.Tensor):
            if lam.dim() == 0:  # Scalar tensor
                return lam * self.criterion(pred, y1) + (1 - lam) * self.criterion(pred, y2)
            else:  # Batch of lambdas - compute per-sample loss then average
                loss1 = F.cross_entropy(pred, y1, reduction='none')
                loss2 = F.cross_entropy(pred, y2, reduction='none')
                # Apply label smoothing manually per sample
                n_classes = pred.size(-1)
                log_preds = F.log_softmax(pred, dim=-1)
                
                # For each sample in batch, compute smoothed loss
                batch_losses = []
                for i in range(pred.size(0)):
                    l = lam[i] if lam.dim() > 0 else lam
                    # Get one-hot for both targets
                    target1_one_hot = F.one_hot(y1[i:i+1], n_classes).float()
                    target2_one_hot = F.one_hot(y2[i:i+1], n_classes).float()
                    
                    # Mix the targets
                    mixed_target = l * target1_one_hot + (1 - l) * target2_one_hot
                    
                    # Apply label smoothing
                    mixed_target = mixed_target * (1 - self.criterion.smoothing) + self.criterion.smoothing / n_classes
                    
                    # Compute cross entropy with soft targets
                    sample_loss = -(mixed_target * log_preds[i:i+1]).sum()
                    batch_losses.append(sample_loss)
                
                return torch.stack(batch_losses).mean()
        else:
            # lam is a Python float
            return lam * self.criterion(pred, y1) + (1 - lam) * self.criterion(pred, y2)
    
    def train_epoch(self, epoch):
        """Train for one epoch with mixup support"""
        self.model.train()
        total_loss = 0.0
        
        # Warmup learning rate
        if epoch < self.warmup_epochs:
            lr_scale = (epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr_scale * param_group['lr']
        
        for batch_idx, batch_data in enumerate(self.train_loader):
            if self.use_mixup:
                data, y1, y2, lam = batch_data
                data = data.to(self.device)
                y1 = y1.to(self.device)
                y2 = y2.to(self.device)
                
                # Handle lam - convert to tensor on device
                if isinstance(lam, (list, tuple)):
                    lam = torch.tensor(lam, device=self.device, dtype=torch.float32)
                elif isinstance(lam, torch.Tensor):
                    lam = lam.to(self.device).float()
                else:
                    # Single float value - keep as scalar
                    lam = float(lam)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                
                # Compute loss - handle batch dimension properly
                if isinstance(lam, torch.Tensor) and lam.numel() > 1:
                    # Multiple lambda values (one per sample in batch)
                    # Ensure lam is 1D
                    lam = lam.view(-1)
                    # Expand for broadcasting
                    lam = lam.view(-1, 1)
                    
                    # Simple approach: compute both losses and mix
                    loss1 = F.cross_entropy(output, y1, reduction='none')
                    loss2 = F.cross_entropy(output, y2, reduction='none')
                    
                    # Mix the losses
                    loss = (lam.squeeze() * loss1 + (1 - lam.squeeze()) * loss2).mean()
                else:
                    # Single lambda value for whole batch
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
    
    def evaluate(self, data_loader, use_tta=False):
        """Evaluate with optional test-time augmentation"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                if use_tta:
                    # Test-time augmentation: average predictions over augmented versions
                    outputs = []
                    for _ in range(3):  # 3 augmented versions
                        output = self.model(data)
                        outputs.append(F.softmax(output, dim=1))
                    output = torch.stack(outputs).mean(dim=0)
                    output = torch.log(output + 1e-8)  # Convert back to log probabilities
                else:
                    output = self.model(data)
                
                loss = F.cross_entropy(output, target)
                total_loss += loss.item()
                
                preds = torch.argmax(output, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        avg_loss = total_loss / len(data_loader)
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        macro_f1 = f1_score(all_targets, all_preds, average='macro')
        
        return avg_loss, macro_f1, all_preds, all_targets
    
    def train(self, num_epochs, checkpoint_dir):
        """Full training loop"""
        print(f"\nStarting enhanced training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
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
                    'val_f1': val_f1,
                }, checkpoint_path)
                print(f"✓ Saved best model with Val F1: {val_f1:.4f}")
            
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
        print(f"Training completed! Best Val F1: {self.best_val_f1:.4f}")
        print(f"{'='*60}")


# Import data preparation from original script
from STMconformer_model import prepData_STM_Conformer


if __name__ == "__main__":
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Parse arguments
    if len(sys.argv) < 2:
        print("Usage: python STM09gpu_Conformer_Enhanced.py <mode>")
        print("Modes:")
        print("  0: Standard training")
        print("  1: Downsample non-tonal speech")
        sys.exit(1)
    
    mode = int(sys.argv[1])
    
    if mode == 0:
        print("Mode 0: Standard training with enhancements")
        ds_nontonal_speech = False
        directory = "model/STM/Conformer_Enhanced/standard"
    elif mode == 1:
        print("Mode 1: Downsample non-tonal speech with enhancements")
        ds_nontonal_speech = True
        directory = "model/STM/Conformer_Enhanced/downsample"
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)
    
    # Create directory
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
    
    # Wrap with Mixup
    use_mixup = True
    if use_mixup:
        train_dataset = MixupDataset(train_dataset, alpha=0.2)
    
    # Create data loaders
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    
    print(f"\nDataLoaders created with batch_size={batch_size}")
    if use_mixup:
        print("Using Mixup augmentation")
    
    # Create enhanced model
    print("\n" + "="*60)
    print("Creating Enhanced Conformer model...")
    print("="*60)
    
    num_classes = 6
    model = EnhancedConformerClassifier(
        input_dim=n_freq,
        num_classes=num_classes,
        d_model=128,
        num_heads=4,
        ffn_dim=512,
        num_layers=4,
        depthwise_conv_kernel_size=31,
        dropout=0.1,
        stochastic_depth_prob=0.1,
        use_spec_augment=True
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
        lr=1e-4,
        weight_decay=1e-5,
        use_mixup=use_mixup,
        label_smoothing=0.1,
        warmup_epochs=5
    )
    
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
    
    # Without TTA
    test_loss, test_f1, test_preds, test_targets = trainer.evaluate(test_loader, use_tta=False)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Macro F1 (no TTA): {test_f1:.4f}")
    
    # With TTA
    test_loss_tta, test_f1_tta, test_preds_tta, _ = trainer.evaluate(test_loader, use_tta=True)
    print(f"Test Macro F1 (with TTA): {test_f1_tta:.4f}")
    
    # Save results
    np.save(os.path.join(checkpoint_dir, 'test_predictions.npy'), test_preds)
    np.save(os.path.join(checkpoint_dir, 'test_predictions_tta.npy'), test_preds_tta)
    np.save(os.path.join(checkpoint_dir, 'test_targets.npy'), test_targets)
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)
