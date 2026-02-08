# STM_CoordConvLDAM_preIN7: Class-Balanced Sampling Only (No DRW)

## Overview

V7 is a refined version that addresses V5's performance regression by **eliminating DRW** while keeping class-balanced sampling as the sole rebalancing mechanism. V5's double reweighting (class-balanced sampling + DRW) overcorrected for minority classes.

## Version Comparison

| Metric | V3 | V5 | V7 (Expected) |
|--------|----|----|---------------|
| Test Macro F1 | 0.8646 | 0.8577 | 0.87-0.89 |
| DRW | Epoch 50+ | Epoch 30+ | **DISABLED** |
| Class-Balanced Sampling | ❌ | ✅ | ✅ |
| SpecAugment Prob | - | 50% | **30%** |
| SpecAugment Time Mask | - | 10 bins | **6 bins** |
| SpecAugment Freq Mask | - | 3 bins | **2 bins** |

## Key Changes from V5

### 1. DRW Disabled (`drw_start_epoch = 999`)

**Problem**: V5 used both class-balanced sampling AND DRW, causing double reweighting:
- Class-balanced sampling: Minority classes sampled ~52x more often
- DRW: Loss term multiplied by inverse class frequency

This overcorrected, degrading majority class (speech:non-tonal) performance.

**Solution**: Disable DRW entirely. Class-balanced sampling already ensures equal class exposure per epoch.

### 2. Reduced SpecAugment

| Parameter | V5 | V7 |
|-----------|----|----|
| Probability | 50% | **30%** |
| Time mask param | 10 | **6** |
| Freq mask param | 3 | **2** |

**Rationale**: With class-balanced sampling making training harder (more minority class exposure), aggressive augmentation further destabilizes learning. Reduced SpecAugment provides regularization without excessive difficulty.

### 3. Confusion Matrix Analysis

V7 now generates:
- `confusion_matrix.png`: Normalized heatmap visualization
- `confusion_matrix.npy`: Raw counts for programmatic analysis
- Detailed per-class confusion analysis in stdout

## Architecture (Unchanged from V3/V5)

```
Input: (B, 2, 20, 121) - 2-channel STM difference maps
  ↓
CoordConv Stem: 4-ch input (2 STM + 2 coordinate channels)
  ↓
Layer1 (64 ch): 2 BasicBlocks + Coordinate Attention
  ↓
Layer2 (128 ch): 2 BasicBlocks + Coordinate Attention  
  ↓
Layer3 (256 ch): 2 BasicBlocks + Squeeze-Excitation
  ↓
Layer4 (512 ch): 2 BasicBlocks + Squeeze-Excitation
  ↓
Global Average Pool → Dropout (0.3) → FC (6 classes)
```

## Training Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Backbone | ResNet-18 | ImageNet pretrained |
| Optimizer | AdamW | - |
| Base LR | 1e-4 | - |
| Discriminative LR | Stem/L1: 0.1x, L2-3: 0.5x, L4/Head: 1.0x | Transfer learning |
| Weight Decay | 2e-4 | Moderate regularization |
| Batch Size | 256 | - |
| Scheduler | ReduceLROnPlateau | factor=0.5, patience=7 |
| Early Stopping | 20 epochs | Based on Val F1 |
| Loss | LDAM | max_m=0.5, s=30, label_smooth=0.05 |
| DRW | **DISABLED** | drw_start_epoch=999 |
| Class-Balanced Sampling | Enabled | WeightedRandomSampler |
| Mixup | 30% prob, α=0.3 | - |
| SpecAugment | 30% prob | time=6, freq=2, num_masks=2 each |
| Head Dropout | 0.3 | - |
| Block Dropout | 0.05 | - |

## Class Distribution

| Class | Train Samples | Weight | Effect |
|-------|---------------|--------|--------|
| speech:non-tonal | 497,475 (64.6%) | 0.000002 | 1x |
| speech:tonal | 80,258 (10.4%) | 0.000012 | 6.2x |
| music:vocal | 101,893 (13.2%) | 0.000010 | 4.9x |
| music:non-vocal | 53,538 (6.9%) | 0.000019 | 9.3x |
| env:urban | 9,580 (1.2%) | 0.000104 | **51.9x** |
| env:wildlife | 27,649 (3.6%) | 0.000036 | 18.0x |

## Expected Training Dynamics

### Without DRW

Unlike V3/V5, the training loss will be more stable throughout as no DRW activation occurs:
- Epochs 1-20: Rapid improvement, learning general features
- Epochs 20-50: Gradual refinement with stable loss
- Epochs 50+: Convergence plateau

### Validation F1 Trajectory

Expected improvement over V3's 0.8646 baseline:
- Class-balanced sampling increases minority class exposure
- No DRW interference maintains majority class performance
- Reduced SpecAugment allows faster convergence

## Usage

```bash
# Standard mode (full dataset)
python STM_CoordConvLDAM_preIN7.py 0

# Downsampled mode (speech:non-tonal capped at 100k)
python STM_CoordConvLDAM_preIN7.py 1
```

## Output Files

```
model/STM/CoordConvLDAM_preIN7_corpora_categories/standard/ckpt/<timestamp>/
├── best_model.pt              # Best model checkpoint
├── checkpoint_epoch_*.pt      # Periodic checkpoints
├── test_predictions.npy       # Final predictions
├── test_targets.npy           # Ground truth labels
├── confusion_matrix.npy       # Raw confusion matrix
└── confusion_matrix.png       # Visualization
```

## Confusion Matrix Interpretation

The confusion matrix helps identify:

1. **What each class is confused AS** (row-wise errors):
   - E.g., music:non-vocal often predicted as music:vocal
   
2. **What is misclassified INTO each class** (column-wise errors):
   - E.g., speech:tonal samples incorrectly predicted as speech:non-tonal

### Key Patterns to Watch

| True Class | Common Confusion | Why |
|------------|------------------|-----|
| music:non-vocal | music:vocal | Both music, vocal detection difficult |
| speech:tonal | speech:non-tonal | Tonal languages can have similar STM patterns |
| env:urban | speech:non-tonal | Urban noise can overlap with speech spectrum |

## Theoretical Basis

### Why Class-Balanced Sampling Alone Works Better

1. **Single Rebalancing Mechanism**: No competing signals during training
2. **Natural Gradient Flow**: Loss gradients reflect true class difficulty, not artificial weighting
3. **Consistent Training Signal**: Each batch has balanced class representation

### Why DRW + Balanced Sampling Fails

1. **Double Counting**: Both mechanisms boost minority classes
2. **Majority Suppression**: Excessive penalty on majority class predictions
3. **Unstable Gradients**: Conflicting reweighting signals

## Success Criteria

- [ ] Test Macro F1 > 0.86 (beat V3)
- [ ] Test Macro F1 > 0.8577 (beat V5)
- [ ] music:non-vocal F1 ≥ 0.67 (at least maintain)
- [ ] speech:non-tonal F1 > 0.95 (maintain majority class)
- [ ] Confusion matrix shows clear diagonal dominance

## Future Improvements (if V7 underperforms)

1. **Focal Loss**: Replace LDAM entirely, dynamic hard example mining
2. **Square-root Sampling**: Gentler `1/sqrt(n)` instead of `1/n` weighting
3. **Curriculum Learning**: Start with uniform sampling, gradually shift to balanced
4. **Multi-scale Fusion**: Combine layer3 + layer4 features for better discrimination
