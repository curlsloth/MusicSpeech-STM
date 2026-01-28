#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STM Classification with Vision Mamba (Vim)
Phase 2: The Modern Sequence Model

This implementation follows the Audio Classification Model Improvement document:
- Vision Mamba for linear-complexity global context
- Bidirectional scanning of the full 2420-token STM sequence
- Learnable positional embeddings (absolute position awareness)
- Balanced Softmax loss for imbalanced data

Installation Requirements:
    pip install mamba-ssm causal-conv1d>=1.2.0

Key Design Principles:
1. Process FULL STM resolution (2420 tokens = 20×121 bins)
2. Each bin becomes a token with learnable position embedding
3. SSM provides O(L) complexity vs O(L²) for Transformers
4. Bidirectional scanning captures non-causal dependencies
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

warnings.filterwarnings('ignore')

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
# Data Preparation (Same as CoordConv but flattened for sequence processing)
# ============================================================================

class prepData_STM_Mamba:
    """
    Data preparation for Vision Mamba model.
    Keeps STM features flattened as 2420-dimensional vectors (sequence of tokens).
    """
    def __init__(self, addAug=False, ds_nontonal_speech=False):
        self.addAug = addAug
        self.ds_nontonal_speech = ds_nontonal_speech
        
        # STM dimensions
        self.n_freq = 20
        self.n_time = 121
        self.seq_len = self.n_freq * self.n_time  # 2420
        
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
            # Forward scan
            self.mamba_forward = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            
            # Backward scan
            self.mamba_backward = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        else:
            # Placeholder: Use simple MLP
            hidden_dim = int(d_model * expand)
            self.mamba_forward = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, d_model)
            )
            self.mamba_backward = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, d_model)
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
# Vision Mamba Model
# ============================================================================

class VisionMamba(nn.Module):
    """
    Vision Mamba for STM Classification
    
    Architecture:
    1. Patch Embedding (1x1 patches = each bin is a token)
    2. Positional Embedding (learnable, absolute positions)
    3. Stack of Vim Blocks (bidirectional SSM)
    4. Global Average Pooling
    5. Classification Head
    """
    def __init__(self, seq_len=2420, num_classes=6, d_model=192, depth=12,
                 d_state=16, d_conv=4, expand=2, drop_path_rate=0.1, dropout=0.1):
        super(VisionMamba, self).__init__()
        
        self.seq_len = seq_len
        self.d_model = d_model
        
        # Patch embedding: each modulation bin becomes a token
        # Input: (batch, seq_len=2420) -> (batch, seq_len, d_model)
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
            x: (batch, seq_len=2420)
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
        
        # Global average pooling: (batch, seq_len, d_model) -> (batch, d_model)
        x = x.mean(dim=1)
        
        # Classification head
        x = self.head(x)
        
        return x


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
        
        # Convert frequencies to log space
        self.log_class_freq = torch.log(torch.FloatTensor(class_freq) + 1e-8)
        
    def forward(self, logits, target):
        """
        Args:
            logits: (batch, num_classes)
            target: (batch,)
        """
        # Move log_class_freq to same device
        log_freq = self.log_class_freq.to(logits.device)
        
        # Adjust logits
        adjusted_logits = logits + log_freq
        
        return F.cross_entropy(adjusted_logits, target)


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
        
        # Balanced Softmax loss
        self.criterion = BalancedSoftmaxLoss(class_freq)
        
        # Optimizer and scheduler
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50)
        
        # Tracking
        self.best_val_f1 = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_f1_scores = []
        
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
        
        for epoch in range(num_epochs):
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


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    
    # Check Mamba availability
    if not MAMBA_AVAILABLE:
        print("\n" + "="*60)
        print("WARNING: Running with placeholder Mamba implementation!")
        print("Install mamba-ssm for full functionality:")
        print("  pip install mamba-ssm causal-conv1d>=1.2.0")
        print("="*60 + "\n")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python STM_ViM.py <mode>")
        print("Modes:")
        print("  0: Standard training")
        print("  1: Downsample non-tonal speech")
        sys.exit(1)
    
    mode = int(sys.argv[1])
    
    # Set parameters based on mode
    if mode == 0:
        print("\nMode 0: Standard training (full dataset)")
        ds_nontonal_speech = False
        directory = "model/STM/ViM_corpora_categories/standard"
    elif mode == 1:
        print("\nMode 1: Downsample non-tonal speech to 100k")
        ds_nontonal_speech = True
        directory = "model/STM/ViM_corpora_categories/downsample"
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
    
    data_prep = prepData_STM_Mamba(ds_nontonal_speech=ds_nontonal_speech)
    train_dataset, val_dataset, test_dataset, class_freq = data_prep.prepare_datasets()
    
    # Create data loaders
    batch_size = 64  # Smaller due to long sequences (2420 tokens)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    
    print(f"\nDataLoaders created with batch_size={batch_size}")
    
    # Create model
    print("\n" + "="*60)
    print("Creating Vision Mamba model...")
    print("="*60)
    
    num_classes = 6
    model = VisionMamba(
        seq_len=2420,
        num_classes=num_classes,
        d_model=192,       # Model dimension (Vim-Small config)
        depth=12,          # Number of Vim blocks
        d_state=16,        # SSM state dimension
        d_conv=4,          # Convolution kernel size in SSM
        expand=2,          # Expansion factor in SSM
        drop_path_rate=0.1,
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
        class_freq=class_freq,
        lr=1e-4,
        weight_decay=1e-4
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
