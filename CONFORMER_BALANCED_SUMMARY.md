# Balanced Conformer Model for STM Classification

## Overview

This is an enhanced version of the base Conformer model (`STMconformer_model.py`) that achieved 0.8636 test macro F1. The balanced version incorporates class weighting strategies to address the severe class imbalance in the training data.

## Key Improvements

### 1. Class-Weighted Focal Loss

The main enhancement is replacing standard cross-entropy with a **weighted focal loss**:

```python
WeightedFocalLoss(
    class_weights=computed_from_training_data,
    gamma=2.0,
    label_smoothing=0.1
)
```

**Benefits:**
- **Class Weights**: Automatically computed using scikit-learn's `compute_class_weight` with 'balanced' strategy
- **Focal Loss**: Focuses training on hard-to-classify examples by down-weighting well-classified samples
- **Label Smoothing**: Prevents overconfidence and improves generalization (smoothing factor: 0.1)

### 2. SpecAugment Data Augmentation

Added SpecAugment-style masking during training:

```python
SpecAugment(
    freq_mask_param=4,      # Mask up to 4 frequency bins
    time_mask_param=20,     # Mask up to 20 time steps
    n_freq_masks=1,         # 1 frequency mask per sample
    n_time_masks=2          # 2 time masks per sample
)
```

**Benefits:**
- Prevents overfitting to specific time-frequency patterns
- Acts as strong regularization
- Only applied during training (disabled during evaluation)

### 3. Warmup Learning Rate Schedule

Gradual warmup over 5 epochs:

```python
if epoch < warmup_epochs:
    lr_scale = (epoch + 1) / warmup_epochs
    lr = lr_scale * base_lr
```

**Benefits:**
- Stabilizes early training
- Helps model converge to better local minima
- Reduces sensitivity to initial learning rate

### 4. Enhanced Monitoring

Added per-class F1 score tracking:

```python
per_class_f1 = f1_score(targets, predictions, average=None)
# Tracks F1 for each of 6 classes separately
```

**Benefits:**
- Identifies which classes are underperforming
- Helps diagnose class-specific issues
- Ensures balanced performance across all categories

## Architecture (Unchanged)

The core Conformer architecture remains identical to the base model:

```
Input (batch, 20, 121)
  ↓
SpecAugment (training only)
  ↓
Conv1D Projection → (batch, 128, 121)
  ↓
Conformer Blocks (4 layers)
  - Multi-head Self-Attention (4 heads)
  - Convolution Module (kernel=31)
  - Feed-Forward Network (dim=512)
  ↓
Global Average Pooling → (batch, 128)
  ↓
Classifier Head
  - Linear(128 → 64)
  - ReLU + Dropout
  - Linear(64 → 6)
  ↓
Output (batch, 6)
```

**Parameters:** ~1.55M (same as base model)

## Training Configuration

### Hyperparameters
- **Optimizer**: AdamW
- **Base Learning Rate**: 1e-4
- **Weight Decay**: 1e-5
- **Batch Size**: 128
- **Epochs**: 50
- **Warmup Epochs**: 5
- **Gradient Clipping**: max_norm=1.0

### Loss Functions
- **Training**: Weighted Focal Loss (γ=2.0, label_smoothing=0.1)
- **Validation/Test**: Standard Cross-Entropy (for fair comparison)

### Learning Rate Schedule
1. **Warmup Phase** (epochs 1-5): Linear warmup from 0 to 1e-4
2. **Plateau Schedule** (epochs 6+): ReduceLROnPlateau
   - Monitored metric: Validation macro F1
   - Factor: 0.5
   - Patience: 3 epochs

## Class Distribution (Training Set)

The class imbalance that motivated this enhancement:

| Class | Count | Percentage | Weight |
|-------|-------|------------|--------|
| speech:non-tonal | ~500K | ~65% | 0.31 |
| speech:tonal | ~50K | ~6.5% | 3.00 |
| music:vocal | ~30K | ~4% | 5.00 |
| music:non-vocal | ~150K | ~19.5% | 1.00 |
| env:urban | ~15K | ~2% | 10.00 |
| env:wildlife | ~25K | ~3% | 6.00 |

The weighted focal loss ensures minority classes (tonal speech, urban environments) receive adequate training signal despite having fewer samples.

## Expected Performance

### Baseline (Unbalanced Conformer)
- **Test Macro F1**: 0.8636
- **Issue**: Potential bias toward majority classes (non-tonal speech)

### Expected Improvements with Balancing
- **Better minority class performance**: Tonal speech, urban sounds
- **More stable training**: Reduced variance in per-class F1
- **Maintained or improved macro F1**: Should exceed 0.8636
- **Slight drop in majority class F1**: Acceptable trade-off for better balance

## Usage

### Training from scratch
```bash
python STMconformer_balanced.py 0  # Standard mode
python STMconformer_balanced.py 1  # Downsample non-tonal speech mode
```

### Resume training
```bash
python STMconformer_balanced.py 0 --resume model/STM/Conformer_Balanced_corpora_categories/standard/ckpt/YYYY-MM-DD_HH-MM
```

## Output Files

Saved in checkpoint directory:
- `best_model.pt`: Best model by validation macro F1
- `latest_checkpoint.pt`: Most recent checkpoint (for resuming)
- `checkpoint_epoch_*.pt`: Periodic checkpoints every 5 epochs
- `test_predictions.npy`: Test set predictions
- `test_targets.npy`: Test set ground truth

## Key Differences from Base Model

| Aspect | Base Conformer | Balanced Conformer |
|--------|----------------|-------------------|
| Loss Function | Cross-Entropy | Weighted Focal Loss + Label Smoothing |
| Class Weighting | None | Automatic from training data |
| Augmentation | None | SpecAugment |
| LR Schedule | ReduceLROnPlateau | Warmup + ReduceLROnPlateau |
| Monitoring | Overall F1 only | Overall + Per-class F1 |
| Resume Support | No | Yes |

## Implementation Notes

1. **Class Weight Computation**: Done automatically in trainer initialization using scikit-learn
2. **Focal Loss Gamma**: Set to 2.0 (standard value from paper)
3. **Label Smoothing**: 0.1 (prevents overconfidence)
4. **SpecAugment**: Conservative parameters to avoid over-augmentation
5. **Warmup**: 5 epochs to stabilize early training

## References

- Base Conformer: `STMconformer_model.py` (Test F1: 0.8636)
- ASM Enhanced: `STMasm_enhanced.py` (inspiration for class balancing)
- Focal Loss: Lin et al., "Focal Loss for Dense Object Detection" (2017)
- SpecAugment: Park et al., "SpecAugment: A Simple Data Augmentation Method for ASR" (2019)
- Conformer: Gulati et al., "Conformer: Convolution-augmented Transformer" (2020)

## Contact

For questions about this model, refer to the base implementation or the ASM enhanced version for similar techniques.
