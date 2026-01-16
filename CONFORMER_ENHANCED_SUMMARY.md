# Summary: Enhanced Conformer Implementation for STM Audio Classification

## Overview

This document describes the **enhanced Conformer implementation** (`STMconformer_enhanced.py`) that builds upon the baseline Conformer model with multiple state-of-the-art improvements to push performance beyond the current **Test F1: 0.8636**.

---

## What's New: Baseline vs Enhanced

| Feature | Baseline (STM08) | Enhanced (STM09) |
|---------|------------------|------------------|
| **Data Augmentation** | None | ✓ SpecAugment (freq/time masking) |
| **Mixup** | ❌ | ✓ Sample mixing during training |
| **Loss Function** | Cross Entropy | ✓ Label Smoothing (0.1) |
| **Pooling** | Global Average | ✓ Attention Pooling |
| **Regularization** | Dropout only | ✓ Dropout + Stochastic Depth |
| **Architecture** | Single conv projection | ✓ Residual projection (2 layers) |
| **Classifier** | 2-layer MLP | ✓ 3-layer with LayerNorm |
| **LR Schedule** | ReduceLROnPlateau | ✓ CosineAnnealing + Warmup |
| **Test-Time Augmentation** | ❌ | ✓ Ensemble 3 predictions |
| **Parameters** | 1.56M | ~1.65M (+6%) |

**Expected Improvement**: +0.5-2.0% F1 score (target: 0.87-0.88)

---

## Key Enhancements Explained

### 1. SpecAugment (Data Augmentation)
```python
# Randomly masks frequency and time regions during training
SpecAugment(
    freq_mask_param=4,      # Mask up to 4 frequency bins
    time_mask_param=15,     # Mask up to 15 time steps
    num_freq_mask=2,        # Apply 2 frequency masks
    num_time_mask=2         # Apply 2 time masks
)
```

**Why it helps**: Forces model to be robust to missing spectral/temporal information, reducing overfitting.

**Note**: SpecAugment is applied directly to the input before the model projection, masking regions by setting them to zero.

---

### 2. Mixup Augmentation
```python
# Mixes two training samples with per-sample or batch-level lambda
x_mixed = λ * x1 + (1-λ) * x2

# Loss computed with proper handling for batch/sample-level mixing
if isinstance(lam, torch.Tensor) and lam.numel() > 1:
    # Per-sample lambda: compute individual losses and mix
    loss1 = F.cross_entropy(output, y1, reduction='none')
    loss2 = F.cross_entropy(output, y2, reduction='none')
    loss = (lam.squeeze() * loss1 + (1 - lam.squeeze()) * loss2).mean()
else:
    # Batch-level lambda: use mixed criterion
    loss = mixup_criterion(output, y1, y2, lam)
```

**Why it helps**: Creates smoother decision boundaries, improves generalization to unseen data.

**Alpha = 0.2**: Moderate mixing strength (good for 6-class problem)

**Implementation note**: The code supports both batch-level and per-sample lambda values with proper broadcasting.

---

### 3. Label Smoothing
```python
# Softens hard labels: [0,0,1,0,0,0] → [0.02, 0.02, 0.9, 0.02, 0.02, 0.02]
LabelSmoothingCrossEntropy(smoothing=0.1)
```

**Why it helps**: Prevents overconfidence, better calibrated predictions, improves generalization.

---

### 4. Attention Pooling
```python
# Instead of simple average pooling
# Learns which time steps are important
attn_weights = softmax(MLP(features))  # (batch, time, 1)
output = sum(features * attn_weights)  # (batch, d_model)
```

**Why it helps**: Focuses on discriminative parts of the sequence, better than uniform averaging.

---

### 5. Stochastic Depth
```python
# Randomly drops entire Conformer layers during training
# Similar to Dropout but for layers
StochasticDepth(drop_prob=0.1)
```

**Why it helps**: Acts as implicit ensemble, reduces co-adaptation between layers.

---

### 6. Residual Input Projection
```python
# Baseline: single conv layer
# Enhanced: two conv layers + skip connection
x = Conv(input) + Conv(Conv(input))
```

**Why it helps**: Easier gradient flow, richer feature extraction from input.

---

### 7. Enhanced Classifier Head
```python
# Baseline: Linear → ReLU → Linear
# Enhanced: Linear → LayerNorm → ReLU → Dropout → (repeat) → Linear
```

**Why it helps**: Better feature transformation, more stable training with LayerNorm.

---

### 8. Cosine Annealing with Warmup
```python
# Learning rate schedule:
# Epochs 1-5: Linear warmup (0 → lr)
# Epochs 6+: Cosine annealing with restarts
CosineAnnealingWarmRestarts(T_0=10, T_mult=2)
```

**Why it helps**: 
- Warmup: Stabilizes early training
- Cosine: Smooth decay encourages better convergence
- Restarts: Helps escape local minima

---

### 9. Test-Time Augmentation (TTA)
```python
# At inference, average predictions over multiple forward passes
# Note: Current implementation averages multiple forward passes 
# without explicit augmentation
outputs = []
for _ in range(3):
    output = model(data)
    outputs.append(F.softmax(output, dim=1))
final_pred = torch.stack(outputs).mean(dim=0)
```

**Why it helps**: Reduces variance in predictions, acts as implicit ensemble (typically +0.1-0.3% improvement).

**Note**: Current implementation performs 3 forward passes and averages the softmax probabilities. For true TTA with augmentation, consider adding random crops or minor transformations during the loop.

---

## Architecture Comparison

### Baseline Conformer
```
Input (20, 121)
  ↓
Conv1D Projection (→128)
  ↓
Conformer Blocks (×4)
  ↓
Global Average Pooling
  ↓
Linear → ReLU → Linear
  ↓
Output (6 classes)
```

### Enhanced Conformer
```
Input (20, 121)
  ↓
[SpecAugment during training]
  ↓
Residual Conv Projection (→128)
  ↓
Conformer Blocks (×4)
  ↓
[Stochastic Depth during training]
  ↓
Attention Pooling
  ↓
Linear → LayerNorm → ReLU → Dropout → (×2) → Linear
  ↓
Output (6 classes)
```

---

## Training Configuration

### Hyperparameters
```python
# Model
d_model = 128
num_heads = 4
ffn_dim = 512
num_layers = 4
dropout = 0.1
stochastic_depth_prob = 0.1

# Training
batch_size = 128
learning_rate = 1e-4
weight_decay = 1e-5
num_epochs = 50
warmup_epochs = 5

# Augmentation
mixup_alpha = 0.2
mixup_probability = 0.5  # 50% chance to apply mixup per sample
label_smoothing = 0.1
spec_augment = True  # freq_mask=4, time_mask=15, num_masks=2 each
```

### Why these values?
- **d_model=128**: Balanced capacity for 1M samples
- **num_layers=4**: Deep enough for complex patterns, not too deep to overfit
- **dropout=0.1**: Light regularization (we have many other regularization techniques)
- **stochastic_depth=0.1**: 10% chance to drop layers (conservative)
- **label_smoothing=0.1**: Standard value for multi-class classification
- **mixup_alpha=0.2**: Moderate mixing (0.2-0.4 typical for audio)

---

## Usage

### Basic Training
```bash
# Standard training
python STMconformer_enhanced.py 0

# With downsampling
python STMconformer_enhanced.py 1
```

### HPC Batch Job
```bash
#!/bin/bash
#SBATCH --job-name=conformer_enhanced
#SBATCH --time=48:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

python STMconformer_enhanced.py 0
```

### Evaluating Results
```python
import torch
import numpy as np

# Load checkpoint
checkpoint = torch.load('model/STM/Conformer_Enhanced/standard/ckpt/best_model.pt')

# Check validation F1
print(f"Best Val F1: {checkpoint['val_f1']:.4f}")

# Load test predictions
test_preds = np.load('test_predictions.npy')
test_preds_tta = np.load('test_predictions_tta.npy')
test_targets = np.load('test_targets.npy')

# Compare
from sklearn.metrics import f1_score
f1_standard = f1_score(test_targets, test_preds, average='macro')
f1_tta = f1_score(test_targets, test_preds_tta, average='macro')

print(f"Test F1 (standard): {f1_standard:.4f}")
print(f"Test F1 (with TTA): {f1_tta:.4f}")
print(f"TTA improvement: {(f1_tta - f1_standard)*100:.2f}%")
```

---

## Expected Performance

### Baseline (STM08)
- **Best Val F1**: 0.8521
- **Test F1**: 0.8636
- **Training time**: ~2-3 hours/epoch

### Enhanced (STM09) - Projected
- **Best Val F1**: 0.86-0.87 (target)
- **Test F1**: 0.87-0.88 (target)
- **Test F1 with TTA**: 0.875-0.885 (target)
- **Training time**: ~2.5-3.5 hours/epoch (slightly slower due to augmentations)

### Breakdown of Expected Improvements
| Technique | Expected Gain | Implementation Notes |
|-----------|---------------|----------------------|
| SpecAugment | +0.2-0.5% | Masking at input level |
| Mixup | +0.2-0.4% | 50% probability, alpha=0.2 |
| Label Smoothing | +0.1-0.3% | Smoothing factor 0.1 |
| Attention Pooling | +0.1-0.2% | 2-layer attention network |
| Stochastic Depth | +0.1-0.2% | 10% drop probability |
| Enhanced Architecture | +0.1-0.2% | Residual projection + 3-layer classifier |
| Better LR Schedule | +0.1-0.2% | Cosine with warmup |
| **Total (without TTA)** | **+0.9-2.0%** | |
| Test-Time Augmentation | +0.1-0.3% | 3x forward passes averaged |
| **Grand Total** | **+1.0-2.3%** | |

**Conservative estimate**: 0.87-0.88 F1  
**Optimistic estimate**: 0.88-0.89 F1

**Note**: TTA improvement is more modest than initially documented due to averaging forward passes without explicit augmentation.

---

## Output Structure

```
model/STM/Conformer_Enhanced/
├── standard/
│   └── ckpt/
│       └── 2024-01-15_14-30/
│           ├── best_model.pt              # Best model by val F1
│           ├── checkpoint_epoch_10.pt     # Periodic checkpoints
│           ├── checkpoint_epoch_20.pt
│           ├── ...
│           ├── test_predictions.npy       # Standard predictions
│           ├── test_predictions_tta.npy   # With TTA
│           └── test_targets.npy           # Ground truth
└── downsample/
    └── ckpt/
        └── ...
```

---

## Monitoring Training

### What to Watch
```python
# Key metrics during training
1. Train Loss: Should decrease smoothly (but slower than baseline due to augmentation)
2. Val Loss: Should decrease and stabilize
3. Val F1: Should increase, best model saved automatically
4. Learning Rate: Check warmup and cosine schedule working

# Good signs:
- Train loss slightly higher than baseline (due to mixup/label smoothing)
- Val loss similar or lower than baseline
- Val F1 improving consistently
- No sudden spikes (gradient clipping working)

# Warning signs:
- Val loss increasing while train loss decreasing → overfitting
  (though less likely with all the regularization)
- Very erratic training loss → check augmentation parameters
- Extremely slow convergence → check learning rate schedule
```

---

## Ablation Study (Optional)

To understand which improvements matter most:

### 1. Train with individual improvements
```bash
# Disable specific features in code:
use_spec_augment = False
use_mixup = False
label_smoothing = 0.0
# etc.
```

### 2. Compare results
```python
results = {
    'baseline': 0.8636,
    'enhanced_full': 0.87xx,
    'no_specaugment': 0.86xx,
    'no_mixup': 0.87xx,
    'no_label_smooth': 0.87xx,
    # etc.
}
```

---

## Comparison with Baseline

### When to Use Enhanced vs Baseline

**Use Enhanced (STM09) when**:
- ✓ You want maximum performance
- ✓ You have sufficient GPU memory (8-12 GB)
- ✓ Training time is not critical
- ✓ You need robust predictions

**Use Baseline (STM08) when**:
- ✓ You want faster training
- ✓ Limited GPU memory (<8 GB)
- ✓ You need simpler model for deployment
- ✓ 0.86 F1 is sufficient for your application

---

## Troubleshooting

### Issue: Training slower than expected
**Solutions**:
- Disable TTA during validation (only use at final test)
- Reduce mixup probability from 0.5 to 0.3
- Use fewer SpecAugment masks (1 instead of 2 per dimension)
- Reduce batch_size if memory-bound

### Issue: Validation F1 not improving
**Solutions**:
- Check if warmup is working (LR should increase first 5 epochs)
- Verify augmentations not too strong (try disabling SpecAugment temporarily)
- Ensure data loading is correct (check MixupDataset returns 4 values)
- Train longer (some techniques need 20-30 epochs to show benefits)

### Issue: CUDA out of memory
**Solutions**:
- Reduce batch_size to 64 or 32
- Disable TTA during training/validation
- Use gradient accumulation (accumulate 2-4 steps before optimizer.step())
- Reduce d_model from 128 to 96

### Issue: Results worse than baseline
**Solutions**:
- Train longer (enhanced model needs more epochs due to regularization)
- Verify implementation (compare layer outputs with baseline)
- Check if augmentations too aggressive:
  - Reduce SpecAugment: freq_mask_param=2, time_mask_param=10
  - Reduce Mixup alpha to 0.1
  - Reduce label_smoothing to 0.05
- Ensure proper learning rate schedule (check warmup is working)

### Issue: Mixup causing NaN losses
**Solutions**:
- Verify lambda values are in [0, 1]
- Check that MixupDataset returns proper tensor shapes
- Ensure y1, y2 are valid class indices [0, 5]
- Add gradient clipping (already included: max_norm=1.0)

---

## Technical Details

### Mixup Implementation Details

The enhanced implementation supports two modes of Mixup:

1. **Batch-level mixing**: Single lambda for entire batch
2. **Sample-level mixing**: Different lambda per sample (if provided by DataLoader)

```python
# The MixupDataset returns lambda as float for batch-level
# or can be extended to return tensor for sample-level mixing

# Loss computation handles both cases:
if isinstance(lam, torch.Tensor) and lam.numel() > 1:
    # Per-sample: compute losses separately and mix
    loss = (lam * loss1 + (1-lam) * loss2).mean()
else:
    # Batch-level: use criterion with mixed targets
    loss = mixup_criterion(pred, y1, y2, lam)
```

### Label Smoothing Details

Label smoothing is implemented in the `LabelSmoothingCrossEntropy` class:

```python
# Converts hard labels to soft distributions
# target: [0, 0, 1, 0, 0, 0] (one-hot)
# smoothed: [ε/K, ε/K, 1-ε+ε/K, ε/K, ε/K, ε/K]
# where ε=0.1, K=6 classes

loss = smoothing * (-log_preds.sum() / n_classes) + (1 - smoothing) * nll_loss
```

### Learning Rate Schedule

The training uses a two-phase learning rate schedule:

1. **Warmup phase** (epochs 1-5): Linear increase from 0 to `lr`
2. **Cosine annealing** (epochs 6+): `CosineAnnealingWarmRestarts` with:
   - T_0 = 10 (restart every 10 epochs)
   - T_mult = 2 (double the period after each restart)
   - eta_min = 1e-6 (minimum learning rate)

```python
# Warmup implementation
if epoch < warmup_epochs:
    lr_scale = (epoch + 1) / warmup_epochs
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_scale * base_lr

# Cosine annealing kicks in after warmup
scheduler.step()  # CosineAnnealingWarmRestarts
```

---

## Next Steps

### Immediate
1. **Run test**: `python test_conformer_implementation.py` (works for both versions)
2. **Small trial**: Train for 5 epochs to verify setup
3. **Full training**: Run on HPC with 50 epochs

### After Training
1. **Evaluate**: Compare with baseline using test predictions
2. **Analyze**: Check which classes improved most
3. **Visualize**: Plot training curves, confusion matrices
4. **Document**: Record results for future reference

### Optional Improvements
- [ ] Increase model size (d_model=256)
- [ ] Add more Conformer layers (6-8)
- [ ] Experiment with different augmentation strengths
- [ ] Try ensemble of multiple models
- [ ] Use self-distillation for further gains

---

## Files Overview

| File | Purpose | Lines |
|------|---------|-------|
| `STMconformer_enhanced.py` | Main training script | ~600 |
| `STM08gpu_Conformer_STM_corpus.py` | Baseline (for comparison) | ~400 |
| `test_conformer_implementation.py` | Testing both versions | ~200 |
| `CONFORMER_ENHANCED_SUMMARY.md` | This document | ~700 |

**Total enhancement**: ~800 new lines of advanced techniques!

---

## References & Inspiration

1. **SpecAugment**: Park et al., "SpecAugment: A Simple Data Augmentation Method for ASR" (2019)
2. **Mixup**: Zhang et al., "mixup: Beyond Empirical Risk Minimization" (2018)
3. **Label Smoothing**: Szegedy et al., "Rethinking the Inception Architecture" (2016)
4. **Stochastic Depth**: Huang et al., "Deep Networks with Stochastic Depth" (2016)
5. **Conformer**: Gulati et al., "Conformer: Convolution-augmented Transformer" (2020)

---

## Questions?

**For baseline Conformer**: See `CONFORMER_SUMMARY.md`  
**For implementation details**: See code comments in `STMconformer_enhanced.py`  
**For original MLP**: See existing documentation

---

## Changelog

**2024-01-15**: Initial enhanced implementation
- Added all 9 improvements over baseline
- Documented expected performance gains
- Created comprehensive summary

---

## Actual Training Results (2026-01-16)

**Best Val F1**: 0.8181 (Epoch 27)
**Test F1**: 0.8309 (no TTA)
**Test F1 with TTA**: 0.8309 (identical - TTA not working)

### Performance vs Baseline
- Baseline (STM08): Test F1 = 0.8636
- Enhanced (STM09): Test F1 = 0.8309
- **Difference**: -0.0327 (-3.3% worse!)

### Why Did It Fail?

**Primary Issues** (Occam's Razor):

1. **Training Instability**: Val F1 oscillated wildly (0.32 → 0.82 → 0.57)
   - Cause: Too many augmentations applied simultaneously
   - Solution: Reduce augmentation strength or apply fewer at once

2. **Learning Rate Restarts**: CosineAnnealingWarmRestarts disrupted convergence
   - Cause: LR jumped back to max every 10 epochs, destroying good solutions
   - Solution: Use simple CosineAnnealingLR without restarts

3. **Warmup Bug**: LR warmup multiplied decayed LR instead of base LR
   - Cause: `param_group['lr'] = lr_scale * param_group['lr']` (wrong!)
   - Solution: Store base LR and scale from that

4. **Over-Regularization**: SpecAugment + Mixup + Label Smoothing + Dropout + Stochastic Depth
   - Cause: Model never saw clean data clearly
   - Solution: Use 1-2 techniques, not all 5

5. **TTA Doesn't Actually Augment**: Just runs same input 3 times
   - Cause: No augmentation applied in TTA loop
   - Solution: Not a priority (won't help much anyway)

6. **Mixup Loss Complexity**: Complex branching for per-sample vs batch-level lambda
   - Cause: Overly complex implementation, potential bugs
   - Solution: Simplify to batch-level only

### Training Curve Analysis

```
Epoch   Val F1   Notes
1       0.49     Initial
6       0.64     Good progress
7       0.32     COLLAPSE (too aggressive augmentation?)
10      0.59     LR restart #1
20      0.52     LR restart #2
27      0.82     BEST (just before restart #3)
28      0.57     Restart destroyed progress
30      0.39     LR restart #3
50      0.59     Never recovered
```

**Key Observation**: Model found good solution at epoch 27 (F1=0.82) but LR restart at epoch 30 destroyed it.

---

## Lessons Learned

1. **Occam's Razor**: Simpler is better. Don't combine too many techniques.
2. **One change at a time**: Should have ablated each improvement individually.
3. **LR schedules matter**: Restarts can hurt more than help.
4. **Training stability > Fancy techniques**: A stable baseline beats an unstable "enhancement".
5. **Always validate incrementally**: Should have tested each feature separately.

## Next Steps

### Immediate Fixes (Priority Order)

1. **Remove LR restarts** - Use simple `CosineAnnealingLR` instead
2. **Fix warmup bug** - Store and scale base LR correctly
3. **Reduce augmentation** - Pick **one**: either SpecAugment OR Mixup, not both
4. **Simplify Mixup** - Remove per-sample lambda complexity
5. **Reduce regularization** - Remove stochastic depth (redundant with dropout)

### Recommended Minimal Enhancement

Start with **just 2-3 improvements**:
- Label smoothing (safe, always helps)
- Attention pooling (modest improvement, stable)
- Better LR schedule (without restarts)

**Test each individually** before combining!
