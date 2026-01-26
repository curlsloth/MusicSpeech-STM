# ASM Enhanced v5 Summary

## Overview

Enhanced Audio Spectrogram Mixer v5 (ASM-RH v5) builds on v4 by **removing class weights** to evaluate the model's performance when relying solely on the Focal Loss mechanism for handling class imbalance. This simplifies the training process and tests whether the combination of symmetric STM processing and unweighted Focal Loss can achieve competitive results.

**Key Innovation from v4**: NO class weighting — allows the model to learn natural class distributions with Focal Loss's inherent imbalance handling.

---

## Differences from v4

### What's Changed

| Component | v4 | v5 |
|-----------|----|----|
| **Class Weights** | ✓ Confusion-aware weights | ✗ **NO weights (alpha=None)** |
| **Loss Function** | Weighted Focal Loss | **Unweighted Focal Loss** |
| **Weight Computation** | `compute_confusion_aware_weights()` | **Removed** |
| **Trainer Signature** | `class_weights` parameter required | **NO `class_weights` parameter** |
| **Training Philosophy** | Manual class balancing | **Natural learning with Focal Loss** |

### What's Unchanged

All other components remain identical to v4:
- ✓ Symmetric STM processing (121 → 61 modulation rates)
- ✓ Model architecture (6 blocks, 160 dims)
- ✓ Contrastive regularization for similar classes
- ✓ Focal Loss with gamma=2.0
- ✓ Label smoothing (0.01)
- ✓ Same hyperparameters (lr, scheduler, batch size)

---

## Signal Processing Pipeline (from v4)

The symmetric STM processing is **unchanged** from v4:

### Symmetric STM Processing

**Input Data Structure**: STM data has shape `(freq_bands=20, mod_rates=121)`
- 20 frequency bands (spectral channels)
- 121 modulation rate bins (−15 Hz to +15 Hz in 0.25 Hz steps)

#### Processing Steps
1. **Separate**: Negative rates [0:60] and positive rates [61:121] along modulation axis
2. **Flip**: Reverse negative chunk to align with positive chunk
3. **Average**: Element-wise average of flipped negative and positive chunks
4. **Concatenate**: Prepend DC component (0 Hz) at index 0

**Result**: `(freq_bands=20, mod_rates=61)` — 0 Hz + 60 positive rates (50% reduction)

---

## Model Architecture (same as v4)

```
Input: (batch, freq_bands=20, mod_rates=61)
  ↓
SpecAugment (freq_mask=4, time_mask=20)
  ↓
Input Projection (Conv2d layers)
  (20, 61) → dim/4=40 → dim=160
  ↓
6x ASM-RH Blocks (each containing):
  ├─ Enhanced2DPositionalEncoding (mod_rate × freq_band)
  ├─ RollTimeMixing (shift_range=2, along mod_rate axis)
  ├─ HermitFFTMixing (frequency domain processing)
  ├─ TokenMixing (seq_len = 20 × 61 = 1220)
  └─ ChannelMixing (dim=160)
  ↓
LayerNorm + AdaptiveAvgPool
  ↓
Feature Extractor (dim=160 → dim/2=80)
  ↓
Classifier (80 → 6 classes)
```

**Total Parameters**: ~2.0M (same as v4)

---

## Loss Function Changes

### v4: Weighted Focal Loss

```python
# v4 approach
class_weights = compute_confusion_aware_weights(train_labels)
loss = ContrastiveFocalLoss(
    alpha=class_weights.to(device),  # Manual weighting
    gamma=2.0,
    label_smoothing=0.01,
    contrastive_weight=0.1
)
```

**v4 Class Weights Example**:
- Class 0: 1.0
- Class 1: 1.3 (boosted for tonal speech)
- Class 2: 0.9
- Class 3: 1.3 (boosted for env/music confusion)
- Class 4: 0.7 (reduced for env:urban)
- Class 5: 0.8 (reduced for env:wildlife)

### v5: Unweighted Focal Loss

```python
# v5 approach - NO class weights
loss = ContrastiveFocalLoss(
    # NO alpha parameter
    gamma=2.0,
    label_smoothing=0.01,
    contrastive_weight=0.1
)
```

**v5 Philosophy**:
- Let Focal Loss handle imbalance naturally through $(1-p_t)^\gamma$ term
- Hard examples (low $p_t$) get higher loss regardless of class
- Minority classes often have lower $p_t$ → automatically upweighted
- Simpler training without manual weight tuning

---

## Training Strategy

### Hyperparameters (unchanged from v4)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Optimizer | AdamW | Weight decay = 1e-4 |
| Learning rate | 1e-3 | With warmup over 5 epochs |
| Scheduler | CosineAnnealingWarmRestarts | T_0=10, T_mult=2 |
| Batch size | 128 | |
| Epochs | 50 | |
| Gradient clipping | 1.0 | Prevents instability |

### Data Augmentation (unchanged)

- **SpecAugment**: 
  - Frequency masking: 1 mask, max 4 bins
  - Time masking: 2 masks, max 20 frames
- **Implicit**: Symmetric averaging acts as denoising

### Contrastive Regularization (unchanged)

Maintains contrastive loss for similar class pairs:
- Class 0 (non-tonal speech) ↔ Class 1 (tonal speech)
- Class 2 (music) ↔ Class 3 (env/music boundary)

**Goal**: Maximize inter-class distance in feature space

---

## Expected Behavior

### Potential Advantages (vs v4)

1. **Simpler Training**: No need to tune class weights or confusion-aware adjustments
2. **Natural Learning**: Model discovers optimal class emphasis from data
3. **Better Generalization**: May avoid overfitting to manually-tuned weight scheme
4. **Reduced Hyperparameter Space**: One less set of parameters to optimize

### Potential Disadvantages (vs v4)

1. **Lower Minority Class Performance**: Without boosting, minority classes may suffer
2. **Increased Confusion**: Confusable pairs (0↔1, 2↔3) may not get special attention
3. **Slower Convergence**: May take longer to learn proper class balance

### Comparison Metrics

Key metrics to monitor vs v4:
- **Macro F1**: Overall balanced performance (expect: **similar or slightly lower** than v4)
- **Per-Class F1**: Especially for minority classes 4 and 5 (expect: **potential drop**)
- **Confusion Pairs**: 0↔1 and 2↔3 mistakes (expect: **may increase** without boosting)
- **Training Stability**: Loss curves (expect: **similar or smoother**)

---

## Usage

### Training from Scratch

```bash
# Mode 0: Standard class distribution
python STMasm_enhanced5.py 0

# Mode 1: Downsample non-tonal speech
python STMasm_enhanced5.py 1
```

### Resume Training

```bash
python STMasm_enhanced5.py 0 --resume model/STM/ASM_Enhanced5_corpora_categories/standard/ckpt/2026-01-25_10-30
```

### Checkpoints Saved

- `best_model.pt`: Best validation F1 score
- `latest_checkpoint.pt`: Latest epoch (for resuming)
- `checkpoint_epoch_N.pt`: Every 5 epochs

---

## Comparison with Previous Versions

### Evolution: v1 → v2 → v3 → v4 → v5

| Metric | v1 | v2 | v3 | v4 | v5 (Expected) |
|--------|----|----|----|----|---------------|
| **Input Shape** | (20, 121) | (20, 121) | (20, 121) | (20, 61) | **(20, 61)** |
| **Model Dim** | 128 | 128 | 128 | 160 | **160** |
| **Num Blocks** | 4 | 4 | 4 | 6 | **6** |
| **Class Weights** | Inverse freq | Inverse freq | sqrt(inv freq) | sqrt(inv freq) + boost | **NONE** |
| **Contrastive Loss** | ✗ | ✗ | ✓ | ✓ | ✓ |
| **Symmetric STM** | ✗ | ✗ | ✗ | ✓ | ✓ |
| **Val Macro F1** | ~0.64 | ~0.66 | ~0.68 | ~0.70 (target) | **0.68–0.72** |

### Key Innovations by Version

- **v1**: Base ASM-RH architecture with focal loss
- **v2**: Enhanced positional encoding, adjusted class weights
- **v3**: Confusion-aware loss, contrastive regularization
- **v4**: Symmetric STM processing (121→61), increased capacity
- **v5**: **Removed class weights for natural learning**

---

## Theoretical Rationale

### Why Remove Class Weights?

#### Focal Loss Theory

The Focal Loss is designed to handle class imbalance:

$$
FL(p_t) = -(1 - p_t)^\gamma \log(p_t)
$$

Where $p_t$ is the predicted probability for the true class.

**Key Properties**:
1. **Easy examples** ($p_t$ high): Loss down-weighted by $(1-p_t)^\gamma$
2. **Hard examples** ($p_t$ low): Full loss contribution
3. **Natural balancing**: Minority classes often harder → higher loss → more focus

#### Manual Weighting May Be Redundant

With Focal Loss, explicit class weighting may be:
1. **Redundant**: FL already focuses on hard examples (often minorities)
2. **Suboptimal**: Manual weights may conflict with FL's adaptive mechanism
3. **Overfit-prone**: Hand-tuned weights may not generalize across datasets

### Contrastive Regularization Still Needed

While class weights are removed, contrastive loss remains because:
- Focal Loss handles **quantity imbalance** (sample counts)
- Contrastive loss handles **quality confusion** (similar class pairs)
- Different mechanisms, complementary effects

---

## Implementation Details

### Code Changes from v4

#### 1. Removed Function

```python
# v4: Compute class weights
def compute_confusion_aware_weights(y_train, num_classes=6):
    # ... computes weights with confusion-aware boosting ...
    
# v5: Function completely removed
```

#### 2. Updated Loss Class

```python
# v4: ContrastiveFocalLoss with alpha
class ContrastiveFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, ...):
        self.alpha = alpha  # Class weights
        
    def forward(self, inputs, targets, features=None):
        focal_loss = (1 - p_t) ** self.gamma * ce_loss
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss  # Apply weights

# v5: NO alpha in constructor or forward
class ContrastiveFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, ...):  # NO alpha parameter
        # self.alpha removed
        
    def forward(self, inputs, targets, features=None):
        focal_loss = (1 - p_t) ** self.gamma * ce_loss
        # NO alpha weighting applied
        focal_loss = focal_loss.mean()
```

#### 3. Updated Trainer

```python
# v4: Trainer requires class_weights
class EnhancedTrainerV4:
    def __init__(self, model, train_loader, val_loader, test_loader, 
                 device, class_weights, ...):  # class_weights required
        self.criterion = ContrastiveFocalLoss(
            alpha=class_weights.to(device),  # Pass weights
            ...
        )

# v5: NO class_weights parameter
class EnhancedTrainerV5:
    def __init__(self, model, train_loader, val_loader, test_loader, 
                 device, ...):  # NO class_weights
        self.criterion = ContrastiveFocalLoss(
            # NO alpha
            ...
        )
```

#### 4. Updated Main Execution

```python
# v4: Compute and use weights
train_labels = extract_labels(train_dataset)
class_weights = compute_confusion_aware_weights(train_labels)
trainer = EnhancedTrainerV4(..., class_weights=class_weights)

# v5: Display distribution but don't compute weights
train_labels = extract_labels(train_dataset)
print("Class distribution:")  # Informational only
for i in range(6):
    print(f"  Class {i}: {count} samples ({pct}%)")
print("NOTE: v5 does NOT use class weights")
trainer = EnhancedTrainerV5(...)  # NO class_weights parameter
```

---

## Monitoring and Evaluation

### Key Metrics to Track

1. **Macro F1 Score**: Overall balanced performance
   - Target: 0.68–0.72 (similar to v4's 0.70 target)
   
2. **Per-Class F1**: Identify class-specific impacts
   - Special attention to minority classes 4 (env:urban) and 5 (env:wildlife)
   - Without boosting, these may see lower F1
   
3. **Confusion Pairs**: Monitor 0↔1 and 2↔3 mistakes
   - v4 boosted classes 1 and 3 specifically for these pairs
   - v5 relies on Focal Loss alone
   
4. **Training Dynamics**: Loss curves and convergence
   - May be smoother without weight conflicts
   - Could be slower without explicit minority boosting

### Expected Training Dynamics

- **Warmup (Epochs 1–5)**: Gradual LR increase, stable loss decrease
- **Main Training (Epochs 6–40)**: Cosine annealing, steady F1 improvement
- **Convergence (Epochs 41–50)**: Plateau around best F1

### Checkpoint Analysis

```python
# Compare v4 vs v5 confusion matrices
ckpt_v4 = torch.load('v4/best_model.pt')
ckpt_v5 = torch.load('v5/best_model.pt')

cm_v4 = ckpt_v4['confusion_history'][-1]
cm_v5 = ckpt_v5['confusion_history'][-1]

# Analyze confusion differences
print("Class 0↔1 confusion:")
print(f"  v4: {cm_v4[0,1] + cm_v4[1,0]}")
print(f"  v5: {cm_v5[0,1] + cm_v5[1,0]}")
```

---

## Experimental Goals

### Primary Research Questions

1. **Is class weighting necessary?**
   - Does Focal Loss alone handle imbalance effectively?
   - Are manual weights adding value or noise?

2. **How do minority classes perform?**
   - Classes 4 and 5 (env:urban, env:wildlife) are smallest
   - v4 reduced their weights (0.7×, 0.8×) — was this helpful?

3. **Does simplicity improve generalization?**
   - Fewer hyperparameters → less overfitting risk
   - May generalize better to new datasets

4. **Training stability comparison**
   - Are weight conflicts causing instability in v4?
   - Is v5 training smoother?

### Success Criteria

**Minimum Viable Performance**:
- Val Macro F1 ≥ 0.68 (match v3 baseline)
- Test Macro F1 ≥ 0.66

**Competitive with v4**:
- Val Macro F1 ≥ 0.69 (within 1% of v4's target)
- Per-class F1 not significantly worse for any class

**Exceeds v4**:
- Val Macro F1 > 0.70
- Better generalization (smaller train-test gap)
- Simpler to deploy (no weight computation needed)

---

## Troubleshooting

### Common Issues

**1. Poor Minority Class Performance**
- **Symptom**: Classes 4 and 5 have very low F1
- **Cause**: Focal Loss may not focus enough on minorities
- **Fix**: Consider increasing `gamma` (e.g., 2.0 → 2.5) or reintroduce minimal weights

**2. High Confusion on Target Pairs**
- **Symptom**: 0↔1 and 2↔3 confusion higher than v4
- **Cause**: No explicit boosting for classes 1 and 3
- **Fix**: Increase `contrastive_weight` (e.g., 0.1 → 0.15) or reintroduce selective weights

**3. Slower Convergence**
- **Symptom**: Takes more epochs to reach plateau
- **Cause**: Natural learning without weight guidance
- **Fix**: Acceptable trade-off; extend training to 60–70 epochs if needed

**4. Training Instability**
- **Symptom**: Loss oscillations, NaN losses
- **Cause**: Gradient issues (unrelated to weight removal)
- **Fix**: Same as v4 — check gradient clipping, reduce LR, add epsilon

### Debugging Commands

```bash
# Check model architecture consistency
python -c "
from STMasm_enhanced5 import EnhancedASM_RH_Classifier
model = EnhancedASM_RH_Classifier(time_steps=61, freq_steps=20, num_classes=6)
print(f'Total params: {sum(p.numel() for p in model.parameters()):,}')
# Should be ~2.0M (same as v4)
"

# Verify loss function has no alpha
python -c "
from STMasm_enhanced5 import ContrastiveFocalLoss
loss_fn = ContrastiveFocalLoss()
print(f'Has alpha: {hasattr(loss_fn, \"alpha\")}')  # Should be False
"

# Test forward pass
python -c "
import torch
from STMasm_enhanced5 import EnhancedASM_RH_Classifier
model = EnhancedASM_RH_Classifier(time_steps=61, freq_steps=20, num_classes=6)
x = torch.randn(2, 20, 61)
out = model(x)
print(f'Output shape: {out.shape}')  # Should be (2, 6)
"
```

---

## Future Directions

### Potential Enhancements

1. **Adaptive Gamma Scheduling**:
   - Start with high gamma (e.g., 3.0) for strong minority focus
   - Gradually reduce to 2.0 as training progresses
   - May combine benefits of weighting without manual tuning

2. **Per-Class Gamma**:
   - Different gamma values for different classes
   - Learnable gamma parameters
   - More flexible than fixed weights

3. **Hybrid Approach (v6?)**:
   - Minimal class weights (e.g., just 2× for minorities)
   - Less aggressive than v4's confusion-aware scheme
   - Middle ground between v4 and v5

4. **Uncertainty Quantification**:
   - Estimate model confidence per class
   - Use uncertainty to guide sample weighting
   - Dynamic adjustment during training

### Ablation Studies

To validate v5 design:
1. **Vary gamma**: Test 1.5, 2.0, 2.5, 3.0 to find optimal
2. **Remove contrastive loss**: Test if it's still beneficial without class weights
3. **Minimal weights**: Try simple 2× minority boost (not confusion-aware)
4. **Label smoothing variations**: Test 0.0, 0.01, 0.05, 0.1

---

## Comparison Table: v4 vs v5

| Aspect | v4 | v5 |
|--------|----|----|
| **Class Weights** | ✓ sqrt(inverse freq) + confusion boost | ✗ **NONE** |
| **Loss Function** | Weighted Focal Loss | **Unweighted Focal Loss** |
| **Training Complexity** | High (weight tuning) | **Low (no tuning)** |
| **Hyperparameters** | More (weight adjustments) | **Fewer** |
| **Generalization** | May overfit to weight scheme | **May generalize better** |
| **Minority Classes** | Explicitly boosted (or reduced) | **Relies on Focal Loss** |
| **Confusion Pairs** | Classes 1&3 boosted 1.3× | **Relies on contrastive loss** |
| **Code Complexity** | Higher (weight computation) | **Lower** |
| **Deployment** | Requires weight computation | **Simpler (plug-and-play)** |
| **Expected Val F1** | 0.70–0.75 | **0.68–0.72** |
| **Use Case** | Known confusion patterns | **General-purpose / new datasets** |

---

## Citations and References

### Related Work

1. **Focal Loss**: Lin et al., "Focal Loss for Dense Object Detection" (2017)
   - Demonstrates FL's effectiveness without explicit weights
2. **Class Imbalance**: Cui et al., "Class-Balanced Loss Based on Effective Number of Samples" (2019)
   - Compares weighting strategies
3. **Contrastive Learning**: Chen et al., "A Simple Framework for Contrastive Learning" (2020)
4. **ASM Architecture**: Bai et al., "Audio Spectrogram Mixer" (2022)

### Acknowledgments

- **v4 Foundation**: Symmetric STM processing and increased model capacity
- **Focal Loss Theory**: Naturally handles imbalance, inspired removal of weights
- **Implementation**: Built on PyTorch and scikit-learn

---

## Conclusion

ASM Enhanced v5 represents an experiment in simplification: **Can we achieve competitive results without manual class weighting?**

By relying on Focal Loss's inherent imbalance handling and maintaining contrastive regularization for confusion pairs, v5 offers:
- **Simpler training** (no weight tuning)
- **Easier deployment** (no weight computation)
- **Better generalization potential** (fewer hyperparameters)

The trade-off is potential performance loss on minority classes and confusion pairs that previously benefited from explicit boosting.

**Next Steps**: 
1. Train v5 on full dataset
2. Compare against v4 baseline (especially minority classes and confusion pairs)
3. Analyze training dynamics (convergence, stability)
4. Consider hybrid approaches if v5 underperforms

**Success Metric**: If v5 achieves **≥0.69 Val Macro F1** (within 1% of v4), the simplification is validated.

---

*Document Version: 1.0*  
*Last Updated: 2026-01-25*  
*Author: ASM v5 Development Team*
