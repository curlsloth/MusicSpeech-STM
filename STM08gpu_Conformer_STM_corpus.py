#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Conformer implementation for STM audio classification

This script uses PyTorch and Conformer architecture to classify audio STM features.
The Conformer model is suitable for 2D image-like STM representations.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import datetime
import os
import sys
import gc
from sklearn.metrics import roc_auc_score, f1_score
import subprocess
from torchaudio.models import Conformer
import warnings

warnings.filterwarnings('ignore')

def auto_git_push(path):
    subprocess.run(['git', 'add', path], check=True)
    subprocess.run(['git', 'commit', '-m', 'auto_sync: '+path], check=True)
    subprocess.run(['git', 'push'], check=True)


class prepData_STM_Conformer:
    """
    Data preparation for Conformer model.
    Loads flattened STM data and reshapes it back to 2D for Conformer input.
    """
    def __init__(self, addAug=False, ds_nontonal_speech=False, ablation_params=None):
        self.addAug = addAug
        self.ds_nontonal_speech = ds_nontonal_speech
        self.ablation_params = ablation_params
        
        # STM preprocessing parameters (from STM06)
        self.xmin = 190
        self.xmax = 310
        self.ymin = 75
        self.ymax = 114
        self.x_ds_factor = 1
        self.y_ds_factor = 2
        
        # Calculate 2D dimensions
        self.n_freq = (self.ymax - self.ymin + 1) // self.y_ds_factor  # 20
        self.n_time = (self.xmax - self.xmin + 1) // self.x_ds_factor  # 121
        
    def corpora_list(self):
        """Returns list of all corpora"""
        corpus_speech_list = [
            'BibleTTS/akuapem-twi',
            'BibleTTS/asante-twi',
            'BibleTTS/ewe',
            'BibleTTS/hausa',
            'BibleTTS/lingala',
            'BibleTTS/yoruba',
            'Buckeye',
            'EUROM',
            'HiltonMoser2022_speech',
            'LibriSpeech',
            'MediaSpeech/AR',
            'MediaSpeech/ES',
            'MediaSpeech/FR',
            'MediaSpeech/TR',
            'MozillaCommonVoice/ab',
            'MozillaCommonVoice/ar',
            'MozillaCommonVoice/ba',
            'MozillaCommonVoice/be',
            'MozillaCommonVoice/bg',
            'MozillaCommonVoice/bn',
            'MozillaCommonVoice/br',
            'MozillaCommonVoice/ca',
            'MozillaCommonVoice/ckb',
            'MozillaCommonVoice/cnh',
            'MozillaCommonVoice/cs',
            'MozillaCommonVoice/cv',
            'MozillaCommonVoice/cy',
            'MozillaCommonVoice/da',
            'MozillaCommonVoice/de',
            'MozillaCommonVoice/dv',
            'MozillaCommonVoice/el',
            'MozillaCommonVoice/en',
            'MozillaCommonVoice/eo',
            'MozillaCommonVoice/es',
            'MozillaCommonVoice/et',
            'MozillaCommonVoice/eu',
            'MozillaCommonVoice/fa',
            'MozillaCommonVoice/fi',
            'MozillaCommonVoice/fr',
            'MozillaCommonVoice/fy-NL',
            'MozillaCommonVoice/ga-IE',
            'MozillaCommonVoice/gl',
            'MozillaCommonVoice/gn',
            'MozillaCommonVoice/hi',
            'MozillaCommonVoice/hu',
            'MozillaCommonVoice/hy-AM',
            'MozillaCommonVoice/id',
            'MozillaCommonVoice/ig',
            'MozillaCommonVoice/it',
            'MozillaCommonVoice/ja',
            'MozillaCommonVoice/ka',
            'MozillaCommonVoice/kab',
            'MozillaCommonVoice/kk',
            'MozillaCommonVoice/kmr',
            'MozillaCommonVoice/ky',
            'MozillaCommonVoice/lg',
            'MozillaCommonVoice/lt',
            'MozillaCommonVoice/ltg',
            'MozillaCommonVoice/lv',
            'MozillaCommonVoice/mhr',
            'MozillaCommonVoice/ml',
            'MozillaCommonVoice/mn',
            'MozillaCommonVoice/mt',
            'MozillaCommonVoice/nan-tw',
            'MozillaCommonVoice/nl',
            'MozillaCommonVoice/oc',
            'MozillaCommonVoice/or',
            'MozillaCommonVoice/pl',
            'MozillaCommonVoice/pt',
            'MozillaCommonVoice/ro',
            'MozillaCommonVoice/ru',
            'MozillaCommonVoice/rw',
            'MozillaCommonVoice/sr',
            'MozillaCommonVoice/sv-SE',
            'MozillaCommonVoice/sw',
            'MozillaCommonVoice/ta',
            'MozillaCommonVoice/th',
            'MozillaCommonVoice/tr',
            'MozillaCommonVoice/tt',
            'MozillaCommonVoice/uk',
            'MozillaCommonVoice/vi',
            'MozillaCommonVoice/yo',
            'MozillaCommonVoice/zh-CN',
            'MozillaCommonVoice/zh-HK',
            'MozillaCommonVoice/zh-TW',
            'NIST2008_SRE',
            'TIMIT',
            'VoxCeleb',
        ]
        
        corpus_music_list = [
            'Albouy2020Science',
            'Bach10',
            'GTZAN',
            'HiltonMoser2022_music',
            'IRMAS',
            'MedleyDB',
            'MusicDelta/Beijing',
            'MusicDelta/Carnatic',
            'MusicDelta/Turkish',
            'MusicDelta/Western',
        ]
        
        corpus_env_list = [
            'SONYC',
        ]
        
        if self.addAug:
            corpus_env_list.append('SONYC_augmented')
            
        return corpus_speech_list + corpus_music_list + corpus_env_list
    
    def load_data(self):
        """Load and preprocess STM data"""
        corpus_list_all = self.corpora_list()
        
        STM_all = None
        for corp in corpus_list_all:
            filename = 'STM_output/corpSTMnpy/' + corp.replace('/', '-') + '_STMall.npy'
            if STM_all is None:
                STM_all = np.load(filename)
            else:
                STM_all = np.vstack((STM_all, np.load(filename)))
            print(f"Loaded: {filename}, shape: {np.load(filename).shape}")
        
        # Load metadata
        speech_corp_df1 = pd.read_csv('train_test_split/speech1_10folds_speakerGroupFold.csv', index_col=0)
        speech_corp_df2 = pd.read_csv('train_test_split/speech2_10folds_speakerGroupFold.csv', index_col=0)
        music_corp_df = pd.read_csv('train_test_split/music_10folds_speakerGroupFold.csv', index_col=0)
        df_SONYC = pd.read_csv('train_test_split/env_10folds_speakerGroupFold.csv', index_col=0)
        
        all_corp_df = pd.concat([speech_corp_df1, speech_corp_df2, music_corp_df, df_SONYC], ignore_index=True)
        
        # Handle augmented data
        if self.addAug:
            SONYC_aug_len = np.load('STM_output/corpSTMnpy/SONYC_augmented_STMall.npy').shape[0]
            target = pd.concat([all_corp_df['corpus_type'], pd.Series(['env'] * SONYC_aug_len)], 
                             ignore_index=True)
            data_split = pd.concat([all_corp_df['10fold_labels'], pd.Series([1] * SONYC_aug_len)],
                                 ignore_index=True)
        else:
            target = all_corp_df['corpus_type'].copy()
            data_split = all_corp_df['10fold_labels'].copy()
        
        # Map categories to integers
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
        
        print(f"Total samples: {len(STM_all)}")
        print(f"Train samples: {sum(train_ind)}")
        print(f"Val samples: {sum(val_ind)}")
        print(f"Test samples: {sum(test_ind)}")
        print(f"Feature dimension (flattened): {STM_all.shape[1]}")
        print(f"Expected 2D shape: ({self.n_freq}, {self.n_time})")
        
        return STM_all, target.values, train_ind, val_ind, test_ind
    
    def prepare_datasets(self):
        """Prepare PyTorch datasets and dataloaders"""
        STM_all, target, train_ind, val_ind, test_ind = self.load_data()
        
        # Reshape from flattened to 2D: (batch, freq, time)
        # Original flattened: (N, freq*time) -> Reshape to (N, freq, time)
        STM_all_2d = STM_all.reshape(-1, self.n_freq, self.n_time)
        
        # Normalize per sample
        means = STM_all_2d.mean(axis=(1, 2), keepdims=True)
        stds = STM_all_2d.std(axis=(1, 2), keepdims=True)
        STM_all_2d = (STM_all_2d - means) / (stds + 1e-8)
        
        # Convert to PyTorch tensors
        X_train = torch.FloatTensor(STM_all_2d[train_ind])
        y_train = torch.LongTensor(target[train_ind])
        
        X_val = torch.FloatTensor(STM_all_2d[val_ind])
        y_val = torch.LongTensor(target[val_ind])
        
        X_test = torch.FloatTensor(STM_all_2d[test_ind])
        y_test = torch.LongTensor(target[test_ind])
        
        # Create datasets
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)
        
        print(f"Train dataset shape: {X_train.shape}")
        print(f"Val dataset shape: {X_val.shape}")
        print(f"Test dataset shape: {X_test.shape}")
        
        return train_dataset, val_dataset, test_dataset, self.n_freq, self.n_time


class ConformerClassifier(nn.Module):
    """
    Conformer-based classifier for STM audio classification.
    
    Architecture:
    1. Input projection to increase feature dimension
    2. Conformer blocks for feature extraction
    3. Global average pooling
    4. Classification head
    """
    def __init__(self, input_dim, num_classes, d_model=128, num_heads=4, 
                 ffn_dim=512, num_layers=4, depthwise_conv_kernel_size=31, dropout=0.1):
        super(ConformerClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
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


class Trainer:
    """Training manager for Conformer model"""
    def __init__(self, model, train_loader, val_loader, test_loader, 
                 device, lr=1e-4, weight_decay=1e-5):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=3, verbose=True
        )
        
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
                loss = self.criterion(output, target)
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
            self.scheduler.step(val_f1)
            
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
            
            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
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


# %% Main execution
if __name__ == "__main__":
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python STM08gpu_Conformer_STM_corpus.py <mode>")
        print("Modes:")
        print("  0: Standard training")
        print("  1: Downsample non-tonal speech")
        sys.exit(1)
    
    mode = int(sys.argv[1])
    
    # Set parameters based on mode
    if mode == 0:
        print("Mode 0: Standard training")
        ds_nontonal_speech = False
        directory = "model/STM/Conformer_corpora_categories/standard"
    elif mode == 1:
        print("Mode 1: Downsample non-tonal speech")
        ds_nontonal_speech = True
        directory = "model/STM/Conformer_corpora_categories/downsample"
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
    
    # Create data loaders
    batch_size = 128  # Smaller batch size due to model complexity
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    
    print(f"\nDataLoaders created with batch_size={batch_size}")
    
    # Create model
    print("\n" + "="*60)
    print("Creating Conformer model...")
    print("="*60)
    
    num_classes = 6
    model = ConformerClassifier(
        input_dim=n_freq,      # 20
        num_classes=num_classes,
        d_model=128,           # Reduced from 256 for efficiency
        num_heads=4,
        ffn_dim=512,
        num_layers=4,          # Moderate depth
        depthwise_conv_kernel_size=31,
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
        lr=1e-4,
        weight_decay=1e-5
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
    
    # Save test predictions
    np.save(os.path.join(checkpoint_dir, 'test_predictions.npy'), test_preds)
    np.save(os.path.join(checkpoint_dir, 'test_targets.npy'), test_targets)
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)
