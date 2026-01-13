#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to check and visualize Conformer model training results
"""

import numpy as np
import torch
import os
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def load_checkpoint(checkpoint_path):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    return checkpoint

def plot_training_history(checkpoint_dir):
    """Plot training history if available"""
    # This would need to be saved during training
    # For now, just a placeholder
    print("Training history plotting not implemented yet")
    print("You can add history saving in the training loop")

def analyze_predictions(checkpoint_dir):
    """Analyze test set predictions"""
    pred_path = os.path.join(checkpoint_dir, 'test_predictions.npy')
    target_path = os.path.join(checkpoint_dir, 'test_targets.npy')
    
    if not os.path.exists(pred_path) or not os.path.exists(target_path):
        print("Prediction files not found!")
        return
    
    predictions = np.load(pred_path)
    targets = np.load(target_path)
    
    # Class names
    class_names = [
        'speech: non-tonal',
        'speech: tonal',
        'music: vocal',
        'music: non-vocal',
        'env: urban',
        'env: wildlife'
    ]
    
    # Classification report
    print("\n" + "="*60)
    print("Classification Report")
    print("="*60)
    print(classification_report(targets, predictions, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(targets, predictions)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    save_path = os.path.join(checkpoint_dir, 'confusion_matrix.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nConfusion matrix saved to: {save_path}")
    plt.close()
    
    # Per-class accuracy
    print("\n" + "="*60)
    print("Per-class Accuracy")
    print("="*60)
    for i, class_name in enumerate(class_names):
        class_mask = targets == i
        if class_mask.sum() > 0:
            accuracy = (predictions[class_mask] == i).sum() / class_mask.sum()
            print(f"{class_name}: {accuracy:.4f} ({class_mask.sum()} samples)")

def check_model_architecture(checkpoint_path):
    """Print model architecture information"""
    checkpoint = load_checkpoint(checkpoint_path)
    
    print("\n" + "="*60)
    print("Checkpoint Information")
    print("="*60)
    
    if 'epoch' in checkpoint:
        print(f"Epoch: {checkpoint['epoch']}")
    if 'val_f1' in checkpoint:
        print(f"Validation F1: {checkpoint['val_f1']:.4f}")
    
    # Count parameters
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        total_params = sum(p.numel() for p in state_dict.values())
        print(f"Total parameters: {total_params:,}")
        
        print("\nModel layers:")
        for key in state_dict.keys():
            print(f"  {key}: {state_dict[key].shape}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python STM08gpu-Conformer-1_checkmodel.py <checkpoint_dir>")
        print("Example: python STM08gpu-Conformer-1_checkmodel.py model/STM/Conformer_corpora_categories/standard/ckpt/2024-01-12_10-30")
        sys.exit(1)
    
    checkpoint_dir = sys.argv[1]
    
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory not found: {checkpoint_dir}")
        sys.exit(1)
    
    print(f"Checking checkpoint directory: {checkpoint_dir}")
    
    # Check for best model
    best_model_path = os.path.join(checkpoint_dir, 'best_model.pt')
    if os.path.exists(best_model_path):
        print("\nFound best_model.pt")
        check_model_architecture(best_model_path)
    else:
        print("\nNo best_model.pt found")
    
    # Analyze predictions
    analyze_predictions(checkpoint_dir)
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)

if __name__ == "__main__":
    main()
