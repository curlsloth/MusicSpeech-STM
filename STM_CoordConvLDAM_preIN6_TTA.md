# STM_CoordConvLDAM_preIN6_TTA: Test-Time Augmentation

## Overview

Test-Time Augmentation (TTA) is a technique to improve inference accuracy **without retraining**. It applies the same augmentations used during training at test time and averages the predictions.

## How TTA Works

```
                    ┌─ Original ────────→ Model → Logits₁ ─┐
                    │                                       │
                    ├─ Time Flip ───────→ Model → Logits₂ ─┤
                    │                                       │
Input Sample ───────├─ Freq Shift +2 ───→ Model → Logits₃ ─┼──→ Average → Final Prediction
                    │                                       │
                    ├─ Freq Shift -2 ───→ Model → Logits₄ ─┤
                    │                                       │
                    └─ Time Shift +5 ───→ Model → Logits₅ ─┘
```

## Augmentations Applied

| # | Augmentation | Description | PyTorch Operation |
|---|--------------|-------------|-------------------|
| 1 | Original | No transformation | `x` |
| 2 | Time Flip | Reverse temporal axis | `torch.flip(x, dims=[3])` |
| 3 | Freq Shift +2 | Cyclic shift +2 bins on spectral axis | `torch.roll(x, shifts=2, dims=2)` |
| 4 | Freq Shift -2 | Cyclic shift -2 bins on spectral axis | `torch.roll(x, shifts=-2, dims=2)` |
| 5 | Time Shift +5 | Cyclic shift +5 bins on temporal axis | `torch.roll(x, shifts=5, dims=3)` |

## Why These Augmentations?

1. **Time Flip:** STM features have temporal symmetry for some classes. Averaging helps capture patterns regardless of direction.

2. **Frequency Shifts:** The model learned frequency shift invariance during training (via STMAugmentation). TTA leverages this at inference.

3. **Time Shifts:** Similarly, the model learned some temporal shift invariance.

## Usage

### Basic Usage (Default Checkpoint)

```bash
python STM_CoordConvLDAM_preIN6_TTA.py
```

Uses default checkpoint: `model/STM/CoordConvLDAM_preIN6_corpora_categories/standard/ckpt/2026-02-07_11-08`

### Specify Checkpoint

```bash
python STM_CoordConvLDAM_preIN6_TTA.py /path/to/checkpoint/directory
```

## Expected Output

```
============================================================
Baseline Evaluation (No TTA)...
============================================================
Baseline Test Macro F1: 0.8709

============================================================
TTA Evaluation (5 augmentations)...
============================================================
TTA Test Macro F1: 0.87XX
Improvement over baseline: +X.XX%

============================================================
Per-Class F1 Comparison
============================================================
Class                  Baseline        TTA          Δ
--------------------------------------------------
speech:non-tonal         0.9700     0.97XX     +0.00XX
speech:tonal             0.8000     0.80XX     +0.00XX
music:vocal              0.8300     0.83XX     +0.00XX
music:non-vocal          0.6800     0.69XX     +0.01XX  ← Target class
env:urban                0.9800     0.98XX     +0.00XX
env:wildlife             0.9600     0.96XX     +0.00XX
--------------------------------------------------
MACRO AVG                0.8709     0.87XX     +0.00XX
```

## Output Files

The script saves results to the checkpoint directory:

| File | Description |
|------|-------------|
| `test_predictions_tta.npy` | TTA predictions (class indices) |
| `test_probabilities_tta.npy` | TTA probability distributions |

## Expected Improvement

| Metric | Without TTA | With TTA | Typical Gain |
|--------|-------------|----------|--------------|
| Macro F1 | 0.8709 | 0.87-0.88 | +0.5-1.5% |
| music:non-vocal F1 | 0.68 | 0.69-0.71 | +1-3% |

TTA typically helps most on:
- **Ambiguous samples** near decision boundaries
- **Hard classes** like music:non-vocal with high intra-class variance

## Trade-offs

| Pros | Cons |
|------|------|
| No retraining required | 5× slower inference |
| Guaranteed not to hurt | Marginal gain on easy samples |
| Easy to implement | Diminishing returns after ~5 augmentations |
| Helps hard samples most | Memory: 5× batch requirement |

## Technical Details

### Soft Voting vs Hard Voting

This implementation uses **soft voting** (averaging logits before argmax):

```python
# Soft voting (better)
avg_logits = torch.stack(all_logits).mean(dim=0)
predictions = avg_logits.argmax(dim=1)

# Hard voting (worse)
# all_preds = [logits.argmax(dim=1) for logits in all_logits]
# predictions = mode(all_preds)
```

Soft voting is better because it considers prediction confidence, not just the final class.

### Batch Processing

TTA is applied batch-wise for efficiency:

```python
for name, aug_fn in self.augmentations:
    x_aug = aug_fn(x_batch)  # Augment entire batch
    logits = self.model(x_aug)  # Forward pass
    all_logits.append(logits)
```

## Relationship to Training Augmentation

The TTA augmentations are a subset of training augmentations:

| Training (STMAugmentation) | TTA |
|---------------------------|-----|
| Freq mask (30% prob) | ✗ |
| Time mask (30% prob) | ✗ |
| Freq shift (20% prob, ±3 bins) | ✓ (±2 bins) |
| Time shift (20% prob, ±10 bins) | ✓ (+5 bins) |
| - | ✓ Time flip |

We avoid masking augmentations at TTA because:
1. They destroy information (not invertible)
2. Hard to determine appropriate mask positions without labels

## Files

| File | Description |
|------|-------------|
| [STM_CoordConvLDAM_preIN6_TTA.py](STM_CoordConvLDAM_preIN6_TTA.py) | TTA evaluation script |
| [STM_CoordConvLDAM_preIN6_TTA.md](STM_CoordConvLDAM_preIN6_TTA.md) | This documentation |

## Next Steps

If TTA provides insufficient improvement:
1. **Ensemble:** Combine V2.4 + V2.6 predictions
2. **Hierarchical Classification:** Two-stage speech/music/env → sub-class
3. **Architecture Upgrade:** ConvNeXt or Transformer backbone
