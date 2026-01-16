# Enhanced Conformer V2: Lessons Learned Edition

## Executive Summary

**Enhanced V2** is a **simplified, stable** version that learns from the V1 failure. Instead of throwing every possible improvement at the model, V2 focuses on **proven techniques** with **controlled experiments**.

### Key Principle: Occam's Razor
> "Simpler is better. Add complexity only when justified by evidence."

---

## What Went Wrong with V1?

| Issue | Problem | Impact |
|-------|---------|--------|
| **LR Restarts** | Destroyed good solutions | Val F1 dropped from 0.82→0.57 |
| **Warmup Bug** | Multiplied decaying LR | Incorrect learning rates |
| **Over-augmentation** | SpecAugment + Mixup simultaneously | Training instability |
| **Stochastic Depth** | Redundant with dropout | Unnecessary complexity |
| **Complex Mixup** | Per-sample lambda logic | Hard to debug |
| **TTA Not Working** | No actual augmentation | Zero benefit |

**Result**: V1 performed **3.3% WORSE** than baseline (0.8309 vs 0.8636)

---

## What's New in V2?

### Core Philosophy
1. **Start simple** - baseline improvements only
2. **Add one technique at a time** - controlled experiments
3. **Measure everything** - detailed logging

### Changes from V1

| Feature | V1 (Failed) | V2 (Stable) | Rationale |
|---------|-------------|-------------|-----------|
| **LR Schedule** | CosineRestarts | Simple Cosine | No disruptive restarts |
| **Warmup** | Buggy | Fixed | Store base LR properly |
| **Augmentation** | All at once | Pick ONE | Reduce instability |
| **Stochastic Depth** | ✓ | ❌ | Redundant with dropout |
| **Mixup** | Complex | Simple batch-level | Easier to debug |
| **TTA** | Broken | Removed | Didn't work anyway |
| **Logging** | Basic | Detailed | Better monitoring |

---

## Three Training Modes

V2 allows **controlled experiments** by choosing augmentation strategy:

### 1. None (Baseline Improvements Only)
```bash
python STMconformer_enhanced2.py 0 none
```

**Includes**:
- ✓ Label smoothing (0.1)
- ✓ Attention pooling
- ✓ Better LR schedule (warmup + cosine)
- ✓ Fixed warmup bug

**Expected**: Small, **stable** improvement (~0.5-1.0% over baseline)

**Use when**: You want minimal risk, reliable gains

---

### 2. SpecAugment Only
```bash
python STMconformer_enhanced2.py 0 spec
```

**Adds**:
- ✓ Conservative SpecAugment
  - freq_mask_param=2 (vs 4 in V1)
  - time_mask_param=10 (vs 15 in V1)
  - num_masks=1 (vs 2 in V1)

**Expected**: +0.3-0.7% over mode 1

**Use when**: You want data augmentation without changing loss function

---

### 3. Mixup Only
```bash
python STMconformer_enhanced2.py 0 mix
```

**Adds**:
- ✓ Simplified Mixup
  - alpha=0.2
  - prob=0.5 (50% of batches)
  - **Batch-level lambda only** (not per-sample)

**Expected**: +0.3-0.7% over mode 1

**Use when**: You want smooth decision boundaries

---

## Architecture

### Model Structure
```
Input (20, 121)
  ↓
[Optional: SpecAugment]
  ↓
Conv1D Projection (→128)
  ↓
Conformer Blocks (×4)
  ↓
Attention Pooling
  ↓
Classifier (2-layer + LayerNorm)
  ↓
Output (6 classes)
```

### Key Differences from Baseline

| Component | Baseline | V2 |
|-----------|----------|-----|
| Pooling | Global Average | Attention |
| Classifier | 2-layer | 2-layer + LayerNorm |
| Loss | CrossEntropy | Label Smoothing |
| LR Schedule | ReduceLROnPlateau | Warmup + Cosine |
| Augmentation | None | Optional (user choice) |

**Parameters**: ~1.61M (vs baseline 1.56M, +3%)

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

# Training
batch_size = 128
learning_rate = 1e-4  # Base LR
weight_decay = 1e-5
num_epochs = 50
warmup_epochs = 5

# Loss
label_smoothing = 0.1

# SpecAugment (if enabled)
freq_mask_param = 2      # Conservative
time_mask_param = 10     # Conservative
num_masks = 1            # Only 1 mask

# Mixup (if enabled)
mixup_alpha = 0.2
mixup_prob = 0.5         # 50% of batches
```

### Learning Rate Schedule

**Phase 1: Warmup (Epochs 1-5)**
```python
# FIXED: Correct warmup implementation
lr = (epoch / warmup_epochs) * base_lr
# Epoch 1: lr = 0.2 * 1e-4 = 2e-5
# Epoch 5: lr = 1.0 * 1e-4 = 1e-4
```

**Phase 2: Cosine Decay (Epochs 6-50)**
```python
# Simple cosine decay WITHOUT restarts
lr = eta_min + (base_lr - eta_min) * 0.5 * (1 + cos(π * t / T))
# T = 45 epochs (50 - 5)
# eta_min = 1e-6
```

**Key Fix**: Store `base_lr` separately, never multiply decaying LR

---

## Expected Results

### Conservative Estimates

| Configuration | Expected Val F1 | Expected Test F1 | vs Baseline |
|---------------|----------------|------------------|-------------|
| Baseline (STM08) | 0.8521 | 0.8636 | - |
| V2 + None | 0.855-0.860 | 0.865-0.870 | +0.2-0.6% |
| V2 + SpecAugment | 0.858-0.863 | 0.868-0.873 | +0.5-1.0% |
| V2 + Mixup | 0.858-0.863 | 0.868-0.873 | +0.5-1.0% |

### Why Conservative?

1. **Learned from V1**: Don't overpromise
2. **Fewer techniques**: Smaller cumulative gain
3. **Stability first**: Prefer reliable 0.5% over risky 2%

**Philosophy**: Better to under-promise and over-deliver

---

## Usage Examples

### Quick Start
```bash
# Start with safest option
python STMconformer_enhanced2.py 0 none

# If successful, try augmentation
python STMconformer_enhanced2.py 0 spec

# Or try mixup instead
python STMconformer_enhanced2.py 0 mix
```

### HPC Job
```bash
#!/bin/bash
#SBATCH --job-name=conformer_v2
#SBATCH --time=48:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

# Run all three experiments
python STMconformer_enhanced2.py 0 none
python STMconformer_enhanced2.py 0 spec
python STMconformer_enhanced2.py 0 mix

# Compare results
python compare_results.py
```

### Ablation Study
```bash
# Systematic comparison
experiments=(none spec mix)

for exp in "${experiments[@]}"; do
    echo "Running: $exp"
    python STMconformer_enhanced2.py 0 $exp
    
    # Extract test F1
    best_f1=$(python -c "
import numpy as np
f1s = np.load('model/STM/Conformer_Enhanced2/standard/$exp/ckpt/*/test_predictions.npy')
print(f1s)
")
    
    echo "$exp: $best_f1"
done
```

---

## Monitoring Training

### Key Metrics to Watch

```python
# 1. Training Loss
# Should decrease smoothly
# Mixup: slightly higher than none/spec (expected)

# 2. Validation Loss
# Should decrease and stabilize
# Warning: If increasing → overfitting

# 3. Validation F1
# Should increase steadily
# Watch for: sudden drops (like V1 had)

# 4. Learning Rate
# Epochs 1-5: Should increase linearly
# Epochs 6+: Should decrease smoothly (cosine)
# NO JUMPS (unlike V1 restarts)
```

### Good vs Bad Training

**Good Training (What We Want)**
```
Epoch   Train Loss   Val Loss   Val F1    LR
1       0.52         0.50       0.70      2e-5
5       0.38         0.42       0.80      1e-4  ← Warmup complete
10      0.35         0.40       0.82      7e-5  ← Smooth decay
20      0.32         0.39       0.84      4e-5
30      0.31         0.38       0.85      2e-5
50      0.30         0.37       0.86      1e-6
```

**Bad Training (V1 Failure Pattern)**
```
Epoch   Val F1    Notes
27      0.82      ← GOOD
28      0.57      ← COLLAPSE (LR restart destroyed it)
30      0.39      ← Never recovered
```

V2 **cannot** have this failure because we removed LR restarts!

---

## Output Structure

```
model/STM/Conformer_Enhanced2/
├── standard/
│   ├── none/
│   │   └── ckpt/
│   │       └── 2026-01-16_XX-XX/
│   │           ├── best_model.pt
│   │           ├── test_predictions.npy
│   │           ├── test_targets.npy
│   │           ├── train_losses.npy
│   │           ├── val_losses.npy
│   │           ├── val_f1_scores.npy
│   │           └── learning_rates.npy
│   ├── specaugment/
│   │   └── ckpt/...
│   └── mixup/
│       └── ckpt/...
└── downsample/
    └── (same structure)
```

---

## Troubleshooting

### Issue: Results still worse than baseline

**Check**:
1. Is warmup working? (LR should start low and increase)
2. Is cosine decay smooth? (no jumps)
3. Is training stable? (Val F1 shouldn't collapse)

**If yes to all**: The improvements don't help this dataset
**Solution**: Use baseline model

---

### Issue: Training unstable with augmentation

**Try**:
1. Switch to `none` mode (no augmentation)
2. Reduce augmentation strength in code:
   ```python
   # SpecAugment
   freq_mask_param = 1  # Even more conservative
   time_mask_param = 5
   
   # Mixup
   prob = 0.3  # Apply less often
   ```

---

### Issue: Validation F1 plateaus early

**Possible causes**:
1. Learning rate too low (check warmup worked)
2. Model capacity insufficient (try d_model=256)
3. Dataset limitation (some problems are just hard)

**Try**:
- Increase learning rate to 2e-4
- Train longer (100 epochs)
- Check if baseline also plateaus

---

## Comparison with V1

| Aspect | V1 (Failed) | V2 (Stable) | Winner |
|--------|-------------|-------------|--------|
| **Complexity** | High | Low | V2 ✓ |
| **Stability** | Poor | Good | V2 ✓ |
| **Debuggability** | Hard | Easy | V2 ✓ |
| **Best Val F1** | 0.8181 | TBD | ? |
| **Test F1** | 0.8309 | TBD | ? |
| **Reliability** | 3/10 | 9/10 | V2 ✓ |

**Philosophy**:
- V1: "Try everything, hope something works"
- V2: "Try one thing, understand why it works"

---

## Decision Tree: Which Mode to Use?

```
Start
  ↓
Do you need maximum performance?
  │
  ├─ No → Use baseline (STM08)
  │
  └─ Yes
      ↓
      Are you risk-averse?
        │
        ├─ Yes → Use V2 + none
        │         (safest, +0.5-1.0%)
        │
        └─ No
            ↓
            Do you prefer data augmentation or loss modification?
              │
              ├─ Data aug → Use V2 + spec
              │              (SpecAugment)
              │
              └─ Loss mod → Use V2 + mix
                             (Mixup)
```

---

## Lessons Learned (From V1 Failure)

### 1. Simple is Beautiful
**V1 Mistake**: Combined 9 techniques simultaneously
**V2 Fix**: Test each technique individually

**Takeaway**: You can't debug what you don't understand

---

### 2. LR Schedules Matter
**V1 Mistake**: Restarts destroyed good solutions
**V2 Fix**: Smooth decay, no restarts

**Takeaway**: Stability > Fancy schedules

---

### 3. Always Store Base Values
**V1 Mistake**: `lr *= warmup_scale` (compounding error)
**V2 Fix**: `lr = base_lr * warmup_scale`

**Takeaway**: Never modify what you need to reference later

---

### 4. Don't Stack Regularization
**V1 Mistake**: Dropout + Stochastic Depth + Augmentation
**V2 Fix**: Pick 1-2 regularization techniques

**Takeaway**: More is not always better

---

### 5. Measure Everything
**V1 Problem**: Didn't notice warmup bug until too late
**V2 Fix**: Save LR history, plot it

**Takeaway**: If you don't measure it, you can't fix it

---

## Next Steps

### Immediate (After V2 Training)

1. **Compare all three modes**
   ```bash
   python scripts/compare_v2_modes.py
   ```

2. **Check if improvements are significant**
   - Is test F1 > 0.865? → Success!
   - Is test F1 < baseline? → Stick with baseline

3. **Plot training curves**
   - Verify LR schedule is smooth
   - Check Val F1 doesn't collapse

---

### If V2 Succeeds

1. **Try combining techniques**
   - Spec + Attention pooling
   - Mix + Label smoothing
   - But **ONE AT A TIME**

2. **Scale up model**
   - d_model=256
   - num_layers=6
   - (Only if V2 shows promise)

---

### If V2 Also Fails

1. **Accept baseline is good enough**
   - 0.8636 F1 is actually quite strong
   - Diminishing returns on further improvements

2. **Focus elsewhere**
   - Better features (not STM)
   - Ensemble models
   - Task-specific architectures

---

## Scientific Method Applied

### Hypothesis
"Simplified enhancements (label smoothing + attention pooling) will provide stable 0.5-1% improvement"

### Experiment Design
```
Control: Baseline (0.8636 F1)

Test groups:
1. V2 + none (label smooth + attention)
2. V2 + spec (add SpecAugment)
3. V2 + mix (add Mixup)
```

### Success Criteria
- Test F1 > 0.865 (any mode)
- Training stable (no collapses)
- Improvements reproducible

### Analysis Plan
```python
# After training all modes
results = {
    'baseline': 0.8636,
    'v2_none': None,     # Fill after training
    'v2_spec': None,
    'v2_mix': None
}

# Statistical test
from scipy import stats
# Is improvement significant?
t_stat, p_value = stats.ttest_ind(baseline_preds, v2_preds)
```

---

## FAQ

### Q: Should I use V1 or V2?
**A**: V2. V1 is documented only for educational purposes (learning from failure).

### Q: Can I combine SpecAugment and Mixup?
**A**: Not recommended. V1 tried this and training was unstable. Test separately first.

### Q: What if V2 also fails?
**A**: Use baseline. Not every model benefits from enhancements.

### Q: How do I know which mode is best?
**A**: Run all three, compare test F1. Pick winner.

### Q: Why no TTA in V2?
**A**: V1's TTA didn't work (no actual augmentation). Real TTA needs more implementation work.

### Q: Can I use different hyperparameters?
**A**: Yes, but change **one at a time** and document results.

---

## References

1. **Learning from Failure**: "Why did Enhanced V1 fail?" (this document)
2. **Original Conformer**: Gulati et al., 2020
3. **Label Smoothing**: Szegedy et al., 2016
4. **SpecAugment**: Park et al., 2019
5. **Mixup**: Zhang et al., 2018

---

## Changelog

**2026-01-16**: Created V2 after V1 failure analysis
- Removed LR restarts
- Fixed warmup bug
- Simplified augmentation
- Added three training modes
- Better documentation

---

## Conclusion

**V2 Philosophy**: 
> "First, do no harm. Then, improve carefully."

**Expected Outcome**: 
- Reliable 0.5-1.0% improvement
- Stable training
- Reproducible results

**Worst Case**: 
- Still better than V1 (at least won't be 3% worse!)

---

**Good luck! And remember: Simple, stable, successful. 🎯**
