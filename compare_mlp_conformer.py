#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare MLP vs Conformer model performance

This script helps compare the performance of MLP and Conformer models
on the same test set.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, f1_score, accuracy_score
import os
import sys

class_names = [
    'speech: non-tonal',
    'speech: tonal',
    'music: vocal',
    'music: non-vocal',
    'env: urban',
    'env: wildlife'
]

def load_predictions(checkpoint_dir):
    """Load predictions and targets from a checkpoint directory"""
    pred_path = os.path.join(checkpoint_dir, 'test_predictions.npy')
    target_path = os.path.join(checkpoint_dir, 'test_targets.npy')
    
    if not os.path.exists(pred_path) or not os.path.exists(target_path):
        print(f"Prediction files not found in {checkpoint_dir}")
        return None, None
    
    predictions = np.load(pred_path)
    targets = np.load(target_path)
    
    return predictions, targets

def compute_metrics(predictions, targets):
    """Compute classification metrics"""
    accuracy = accuracy_score(targets, predictions)
    macro_f1 = f1_score(targets, predictions, average='macro')
    weighted_f1 = f1_score(targets, predictions, average='weighted')
    
    per_class_f1 = f1_score(targets, predictions, average=None)
    
    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'per_class_f1': per_class_f1
    }

def plot_comparison(mlp_metrics, conformer_metrics, save_path=None):
    """Plot comparison between MLP and Conformer"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Overall metrics comparison
    ax1 = axes[0]
    metrics = ['accuracy', 'macro_f1', 'weighted_f1']
    mlp_values = [mlp_metrics[m] for m in metrics]
    conformer_values = [conformer_metrics[m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax1.bar(x - width/2, mlp_values, width, label='MLP', alpha=0.8)
    ax1.bar(x + width/2, conformer_values, width, label='Conformer', alpha=0.8)
    
    ax1.set_ylabel('Score')
    ax1.set_title('Overall Performance Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
    ax1.legend()
    ax1.set_ylim([0, 1])
    ax1.grid(axis='y', alpha=0.3)
    
    # Per-class F1 comparison
    ax2 = axes[1]
    mlp_per_class = mlp_metrics['per_class_f1']
    conformer_per_class = conformer_metrics['per_class_f1']
    
    x = np.arange(len(class_names))
    
    ax2.bar(x - width/2, mlp_per_class, width, label='MLP', alpha=0.8)
    ax2.bar(x + width/2, conformer_per_class, width, label='Conformer', alpha=0.8)
    
    ax2.set_ylabel('F1 Score')
    ax2.set_title('Per-Class F1 Score Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(class_names, rotation=45, ha='right')
    ax2.legend()
    ax2.set_ylim([0, 1])
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    
    plt.show()

def print_detailed_comparison(mlp_metrics, conformer_metrics):
    """Print detailed comparison"""
    print("\n" + "="*70)
    print("DETAILED COMPARISON: MLP vs CONFORMER")
    print("="*70)
    
    print("\n{:<20s} {:>15s} {:>15s} {:>15s}".format(
        "Metric", "MLP", "Conformer", "Difference"
    ))
    print("-"*70)
    
    for metric in ['accuracy', 'macro_f1', 'weighted_f1']:
        mlp_val = mlp_metrics[metric]
        conf_val = conformer_metrics[metric]
        diff = conf_val - mlp_val
        
        diff_str = f"+{diff:.4f}" if diff > 0 else f"{diff:.4f}"
        if abs(diff) < 0.001:
            diff_str += " (≈)"
        
        print("{:<20s} {:>15.4f} {:>15.4f} {:>15s}".format(
            metric.replace('_', ' ').title(), mlp_val, conf_val, diff_str
        ))
    
    print("\n" + "="*70)
    print("PER-CLASS F1 SCORES")
    print("="*70)
    print("\n{:<25s} {:>15s} {:>15s} {:>15s}".format(
        "Class", "MLP", "Conformer", "Difference"
    ))
    print("-"*70)
    
    for i, class_name in enumerate(class_names):
        mlp_val = mlp_metrics['per_class_f1'][i]
        conf_val = conformer_metrics['per_class_f1'][i]
        diff = conf_val - mlp_val
        
        diff_str = f"+{diff:.4f}" if diff > 0 else f"{diff:.4f}"
        
        print("{:<25s} {:>15.4f} {:>15.4f} {:>15s}".format(
            class_name, mlp_val, conf_val, diff_str
        ))
    
    print("\n" + "="*70)
    
    # Summary
    improvements = sum(1 for i in range(6) 
                      if conformer_metrics['per_class_f1'][i] > mlp_metrics['per_class_f1'][i])
    
    print(f"\nSummary:")
    print(f"  - Conformer improved on {improvements}/6 classes")
    
    if conformer_metrics['macro_f1'] > mlp_metrics['macro_f1']:
        print(f"  - Conformer has BETTER overall macro F1 score")
    elif conformer_metrics['macro_f1'] < mlp_metrics['macro_f1']:
        print(f"  - MLP has BETTER overall macro F1 score")
    else:
        print(f"  - Models have SIMILAR overall macro F1 score")
    
    print("="*70)

def main():
    if len(sys.argv) < 3:
        print("Usage: python compare_mlp_conformer.py <mlp_checkpoint_dir> <conformer_checkpoint_dir>")
        print("\nExample:")
        print("  python compare_mlp_conformer.py \\")
        print("    model/STM/MLP_corpora_categories/PCA/Dropout/macroF1/MLP_2024-04-26_21-50 \\")
        print("    model/STM/Conformer_corpora_categories/standard/ckpt/2024-01-12_10-30")
        sys.exit(1)
    
    mlp_dir = sys.argv[1]
    conformer_dir = sys.argv[2]
    
    print("Loading MLP predictions...")
    mlp_preds, mlp_targets = load_predictions(mlp_dir)
    
    if mlp_preds is None:
        print(f"ERROR: Could not load MLP predictions from {mlp_dir}")
        sys.exit(1)
    
    print("Loading Conformer predictions...")
    conformer_preds, conformer_targets = load_predictions(conformer_dir)
    
    if conformer_preds is None:
        print(f"ERROR: Could not load Conformer predictions from {conformer_dir}")
        sys.exit(1)
    
    # Verify same test set
    if not np.array_equal(mlp_targets, conformer_targets):
        print("WARNING: MLP and Conformer appear to use different test sets!")
        print("Results may not be directly comparable.")
    
    print("\nComputing metrics...")
    mlp_metrics = compute_metrics(mlp_preds, mlp_targets)
    conformer_metrics = compute_metrics(conformer_preds, conformer_targets)
    
    # Print detailed comparison
    print_detailed_comparison(mlp_metrics, conformer_metrics)
    
    # Plot comparison
    save_path = "model_comparison_MLP_vs_Conformer.png"
    plot_comparison(mlp_metrics, conformer_metrics, save_path)
    
    print(f"\n✓ Comparison complete!")

if __name__ == "__main__":
    main()
