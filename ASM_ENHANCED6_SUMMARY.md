# ASM Enhanced v6 Summary

## Overview

Enhanced Audio Spectrogram Mixer v6 (ASM-RH v6) builds on v5 by introducing **balanced batch sampling** to ensure equal class representation during training. This provides a data-level solution to class imbalance, complementing v5's unweighted loss function approach.

**Key Innovation from v5**: Balanced batch sampling — each training batch contains equal samples from each class, ensuring balanced learning without loss function manipulation.

---

## Differences from v5

### What's Changed

| Component | v5 | v6 |
|-----------|----|----|
| **Batch Sampling** | Standard random shuffle | **Balanced batch sampler** |
| **Training DataLoader** | `shuffle=True` | **`batch_sampler=BalancedBatchSampler`** |
| **Batch Composition** | Random (imbalanced) | **Equal samples per class** |
| **Balancing Strategy** | Loss-level only (Focal Loss) | **Data-level (sampling) + loss-level** |

### What's Unchanged

All other components remain identical to v5:
- ✓ NO class weights in loss function
- ✓ Symmetric STM processing (121 → 61 modulation rates)
- ✓ Model architecture (6 blocks, 160 dims)
- ✓ Contrastive regularization for similar classes
- ✓ Focal Loss with gamma=2.0
- ✓ Label smoothing (0.01)
- ✓ Same hyperparameters (lr, scheduler, etc.)

---

## Balanced Batch Sampler

### Concept

Instead of randomly sampling from the dataset (which produces imbalanced batches), the balanced batch sampler ensures:
- Each batch contains **n_samples** from **n_classes**
- All classes get equal representation in every batch
- Automatic reshuffling when a class runs out of samples

### Implementation

```python
class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, n_classes, n_samples):
        # n_classes: number of classes to sample per batch (e.g., 6)
        # n_samples: number of samples per class (e.g., 21)
        # batch_size = n_classes * n_samples (e.g., 6 * 21 = 126)
```

### Configuration (Default)

```python
n_classes_per_batch = 6   # Sample from all 6 classes
n_samples_per_class = 21  # 21 samples per class
# → Effective batch size = 6 × 21 = 126
```

### Sampling Strategy

**Per Batch**:
1. Randomly select `n_classes` classes (in default config: all 6)
2. For each selected class:
   - Take next `n_samples` samples from that class's pool
   - If pool exhausted, reshuffle and restart from beginning
3. Yield batch with equal class representation

**Example Batch Composition** (with default settings):
- Class 0: 21 samples
- Class 1: 21 samples
- Class 2: 21 samples
- Class 3: 21 samples
- Class 4: 21 samples
- Class 5: 21 samples
- **Total: 126 samples (perfectly balanced)**

---

## Signal Processing Pipeline (from v4/v5)

The symmetric STM processing is **unchanged**:

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

## Model Architecture (same as v4/v5)

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

**Total Parameters**: ~2.0M (same as v4/v5)

---

## Balancing Strategy Comparison

### v4: Loss-Level Weighting

```python
# Explicit class weights
class_weights = compute_confusion_aware_weights(train_labels)
loss = ContrastiveFocalLoss(alpha=class_weights, ...)

# Batches: Imbalanced (natural distribution)
# Loss: Weighted (manual adjustment)
```

**Pros**: Direct control over class importance  
**Cons**: Hyperparameter tuning, may conflict with Focal Loss

### v5: Natural Focal Loss

```python
# No class weights
loss = ContrastiveFocalLoss(gamma=2.0, ...)

# Batches: Imbalanced (natural distribution)
# Loss: Unweighted (Focal Loss handles imbalance)
```

**Pros**: Simple, no tuning, Focal Loss natural behavior  
**Cons**: May not focus enough on minorities

### v6: Balanced Sampling + Focal Loss (NEW)

```python
# No class weights + balanced sampling
sampler = BalancedBatchSampler(dataset, n_classes=6, n_samples=21)
train_loader = DataLoader(dataset, batch_sampler=sampler)
loss = ContrastiveFocalLoss(gamma=2.0, ...)

# Batches: Balanced (forced equal representation)
# Loss: Unweighted (all classes seen equally)
```

**Pros**: 
- Ensures all classes trained equally per epoch
- No loss function complexity
- Better gradient signal for minorities
- Prevents majority class dominance

**Cons**: 
- Minority classes sampled more frequently (with replacement effect)
- May overfit on minority classes
- Slightly smaller effective dataset per epoch

---

## Training Strategy

### Hyperparameters (unchanged from v5)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Optimizer | AdamW | Weight decay = 1e-4 |
| Learning rate | 1e-3 | With warmup over 5 epochs |
| Scheduler | CosineAnnealingWarmRestarts | T_0=10, T_mult=2 |
| Effective batch size | 126 | 6 classes × 21 samples |
| Epochs | 50 | |
| Gradient clipping | 1.0 | Prevents instability |

### Batch Size Calculation

```python
n_classes_per_batch = 6
n_samples_per_class = 21
effective_batch_size = 6 × 21 = 126
```

**Why 126?**
- Close to original batch size of 128
- Divisible by 6 (number of classes)
- Allows 21 samples per class (reasonable representation)

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

### Potential Advantages (vs v5)

1. **Guaranteed Class Balance**: Every batch sees all classes equally
2. **Better Minority Class Learning**: Minorities trained as much as majorities
3. **Stable Gradients**: No batch-to-batch class distribution variance
4. **Reduced Majority Dominance**: Prevents majorities from overwhelming training
5. **Complementary to Focal Loss**: Data-level + loss-level balance

### Potential Advantages (vs v4)

1. **Simpler Than Weighting**: No manual weight tuning
2. **More Natural**: Data augmentation vs loss manipulation
3. **Better Generalization**: Less risk of loss function overfitting

### Potential Disadvantages

1. **Minority Oversampling**: Small classes sampled with replacement more often
   - May lead to overfitting on minority class examples
   - Effective training set smaller (limited by minority class size)

2. **Altered Data Distribution**: Training distribution ≠ test distribution
   - Model sees balanced distribution
   - Evaluation sees natural (imbalanced) distribution
   - May affect calibration

3. **Computational Overhead**: Slightly slower due to per-class indexing

### Comparison Metrics

Key metrics to monitor vs v4 and v5:
- **Macro F1**: Overall balanced performance (expect: **≥ v5**, potentially > v4)
- **Per-Class F1**: Especially minorities (expect: **improvement**)
- **Confusion Pairs**: 0↔1 and 2↔3 (expect: **similar or better** than v5)
- **Training Stability**: Loss curves (expect: **smoother** than v5)
- **Overfitting**: Train-test gap (expect: **monitor carefully**)

---

## Usage

### Training from Scratch

```bash
# Mode 0: Standard class distribution
python STMasm_enhanced6.py 0

# Mode 1: Downsample non-tonal speech
python STMasm_enhanced6.py 1
```

### Resume Training

```bash
python STMasm_enhanced6.py 0 --resume model/STM/ASM_Enhanced6_corpora_categories/standard/ckpt/2026-01-26_10-30
```

### Checkpoints Saved

- `best_model.pt`: Best validation F1 score
- `latest_checkpoint.pt`: Latest epoch (for resuming)
- `checkpoint_epoch_N.pt`: Every 5 epochs

---

## Comparison with Previous Versions

### Evolution: v1 → v2 → v3 → v4 → v5 → v6

| Metric | v1 | v2 | v3 | v4 | v5 | v6 (Expected) |
|--------|----|----|----|----|----|----|
| **Input Shape** | (20, 121) | (20, 121) | (20, 121) | (20, 61) | (20, 61) | **(20, 61)** |
| **Model Dim** | 128 | 128 | 128 | 160 | 160 | **160** |
| **Num Blocks** | 4 | 4 | 4 | 6 | 6 | **6** |
| **Class Weights** | Inverse | Inverse | sqrt(inv) | sqrt+boost | NONE | **NONE** |
| **Batch Sampling** | Random | Random | Random | Random | Random | **Balanced** |
| **Contrastive Loss** | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ |
| **Symmetric STM** | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ |
| **Val Macro F1** | ~0.64 | ~0.66 | ~0.68 | ~0.70 | ~0.68-0.72 | **0.70–0.74** |

### Key Innovations by Version

- **v1**: Base ASM-RH architecture with focal loss
- **v2**: Enhanced positional encoding, adjusted class weights
- **v3**: Confusion-aware loss, contrastive regularization
- **v4**: Symmetric STM processing (121→61), increased capacity
- **v5**: Removed class weights for natural learning
- **v6**: **Balanced batch sampling for data-level balancing**

---

## Theoretical Rationale

### Why Balanced Batch Sampling?

#### Problem with Imbalanced Batches

In standard random sampling with imbalanced datasets:
```
Batch 1: [Class 0: 60, Class 1: 40, Class 2: 20, Class 3: 5, Class 4: 2, Class 5: 1]
Batch 2: [Class 0: 55, Class 1: 45, Class 2: 18, Class 3: 7, Class 4: 3, Class 5: 0]
Batch 3: [Class 0: 58, Class 1: 42, Class 2: 22, Class 3: 4, Class 4: 1, Class 5: 1]
```

**Issues**:
1. Minority classes rarely appear (sometimes 0 samples)
2. Majority classes dominate gradient updates
3. Model biased toward majorities even with Focal Loss
4. High variance in batch statistics

#### Solution: Balanced Batches

```
Every Batch: [Class 0: 21, Class 1: 21, Class 2: 21, Class 3: 21, Class 4: 21, Class 5: 21]
```

**Benefits**:
1. Every class contributes equally to every gradient update
2. Minority classes get sufficient training signal
3. Consistent batch statistics across epochs
4. Natural way to handle imbalance without loss manipulation

### Complementarity with Focal Loss

**Balanced Sampling**: Ensures equal **exposure** (data-level)  
**Focal Loss**: Focuses on **hard examples** (loss-level)

These mechanisms are complementary:
- Balanced sampling → equal training opportunities
- Focal Loss → efficient learning from each opportunity
- Together → robust learning for all classes

### Relationship to Oversampling

Balanced batch sampling is similar to **class-balanced oversampling**, but more efficient:

**Traditional Oversampling**:
- Duplicate minority samples to match majority
- Creates artificially large dataset
- Higher memory and computation

**Balanced Batch Sampling**:
- Sample minorities more frequently within batches
- No dataset duplication
- Efficient memory usage
- Implicit oversampling with reshuffling

---

## Implementation Details

### Code Changes from v5

#### 1. Added BalancedBatchSampler Class

```python
# NEW in v6
class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, n_classes, n_samples):
        # Extract labels and create class-to-indices mapping
        self.labels = [label.item() for _, label in dataset]
        self.label_to_indices = {
            label: np.where(np.array(self.labels) == label)[0]
            for label in set(self.labels)
        }
        # Shuffle initially
        for label in self.label_to_indices:
            np.random.shuffle(self.label_to_indices[label])
    
    def __iter__(self):
        # Yield batches with n_samples from n_classes
        while self.count + self.batch_size <= self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                # Get n_samples for this class
                # Reshuffle if exhausted
            yield indices
```

#### 2. Updated DataLoader Creation

```python
# v5: Standard random sampling
train_loader = DataLoader(
    train_dataset, 
    batch_size=128,          # Fixed batch size
    shuffle=True,            # Random shuffle
    num_workers=4, 
    pin_memory=True
)

# v6: Balanced batch sampling
balanced_sampler = BalancedBatchSampler(
    train_dataset, 
    n_classes=6,             # Sample from all classes
    n_samples=21             # 21 per class
)
train_loader = DataLoader(
    train_dataset, 
    batch_sampler=balanced_sampler,  # Use balanced sampler
    num_workers=4, 
    pin_memory=True
)
```

#### 3. Validation/Test Loaders Unchanged

```python
# Val and test use standard sampling (no balancing needed)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, ...)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, ...)
```

#### 4. Updated Main Execution Messages

```python
print("NOTE: v6 uses balanced batch sampling - each batch has equal representation")
print(f"Balanced Batch Sampler Configuration:")
print(f"  Classes per batch: {n_classes_per_batch}")
print(f"  Samples per class: {n_samples_per_class}")
print(f"  Effective batch size: {n_classes_per_batch * n_samples_per_class}")
```

---

## Monitoring and Evaluation

### Key Metrics to Track

1. **Macro F1 Score**: Overall balanced performance
   - Target: 0.70–0.74 (improvement over v5's 0.68–0.72)
   
2. **Per-Class F1**: Identify class-specific impacts
   - **Special attention to minorities** (classes 4 and 5)
   - Expect significant improvement due to balanced training
   
3. **Confusion Pairs**: Monitor 0↔1 and 2↔3 mistakes
   - Should be similar or better than v5
   - Balanced exposure may help discrimination
   
4. **Training Dynamics**: Loss curves and convergence
   - Expect smoother curves (consistent batch composition)
   - May converge faster (better gradient signal)
   
5. **Overfitting**: Train-test F1 gap
   - **Critical to monitor**: balanced training ≠ balanced test
   - Watch for overfitting on minority classes

### Expected Training Dynamics

- **Warmup (Epochs 1–5)**: Gradual LR increase, stable loss decrease
  - Should be smoother than v5 (consistent batches)
- **Main Training (Epochs 6–40)**: Cosine annealing, steady F1 improvement
  - Minority class F1 should improve faster than v5
- **Convergence (Epochs 41–50)**: Plateau around best F1
  - Watch for train-test gap widening

### Checkpoint Analysis

```python
# Compare v5 vs v6 per-class performance
ckpt_v5 = torch.load('v5/best_model.pt')
ckpt_v6 = torch.load('v6/best_model.pt')

cm_v5 = ckpt_v5['confusion_history'][-1]
cm_v6 = ckpt_v6['confusion_history'][-1]

# Analyze minority class improvement
for class_id in [4, 5]:  # env:urban, env:wildlife
    recall_v5 = cm_v5[class_id, class_id] / cm_v5[class_id, :].sum()
    recall_v6 = cm_v6[class_id, class_id] / cm_v6[class_id, :].sum()
    print(f"Class {class_id} recall improvement: {recall_v6 - recall_v5:.3f}")
```

---

## Experimental Goals

### Primary Research Questions

1. **Does balanced sampling improve minority class performance?**
   - Expected: Yes, significantly
   - Classes 4 and 5 should see largest gains

2. **How does it compare to loss-based weighting (v4)?**
   - Which is more effective: data-level or loss-level balancing?
   - Or is combination best?

3. **What is the impact on overfitting?**
   - Does minority oversampling cause overfitting?
   - How large is the train-test gap?

4. **Does it improve overall performance vs v5?**
   - Is data-level balancing better than pure Focal Loss?

### Success Criteria

**Minimum Viable Performance**:
- Val Macro F1 ≥ 0.69 (match v5 upper bound)
- Test Macro F1 ≥ 0.67

**Competitive with v4**:
- Val Macro F1 ≥ 0.70 (match v4's target)
- Better minority class F1 than v4

**Exceeds v5**:
- Val Macro F1 > 0.70
- Minority class F1 > v5 by at least 5%
- Similar or better confusion pair performance

**Ideal Outcome**:
- Val Macro F1 ≥ 0.72
- All class F1 > 0.60
- Best overall performance among all versions

---

## Troubleshooting

### Common Issues

**1. Overfitting on Minority Classes**
- **Symptom**: High train F1, low test F1 for classes 4 and 5
- **Cause**: Repeated sampling of same minority examples
- **Fix**: 
  - Reduce `n_samples_per_class` (e.g., 21 → 15)
  - Increase dropout (0.1 → 0.15)
  - Stronger data augmentation

**2. Training Instability**
- **Symptom**: Loss oscillations
- **Cause**: Small effective batch size per class
- **Fix**: 
  - Increase `n_samples_per_class` (21 → 25)
  - Adjust learning rate
  - Check for NaN losses

**3. Slower Convergence**
- **Symptom**: Takes more epochs than v5
- **Cause**: Smaller effective dataset per epoch
- **Fix**: 
  - Acceptable trade-off for better balance
  - Consider extending training to 60 epochs

**4. Poor Generalization**
- **Symptom**: High train F1, low test F1 overall
- **Cause**: Train distribution ≠ test distribution
- **Fix**: 
  - Consider hybrid approach (balanced sampling + slight test-time adjustment)
  - Stronger regularization

### Debugging Commands

```bash
# Check sampler is working
python -c "
from STMasm_enhanced6 import BalancedBatchSampler, SymmetricSTMDataset
import torch
from torch.utils.data import TensorDataset

# Create dummy dataset
data = torch.randn(1000, 20, 121)
labels = torch.tensor([0]*300 + [1]*250 + [2]*200 + [3]*150 + [4]*60 + [5]*40)
dataset = TensorDataset(data, labels)
dataset = SymmetricSTMDataset(dataset)

# Create sampler
sampler = BalancedBatchSampler(dataset, n_classes=6, n_samples=10)

# Check first batch
batch_indices = next(iter(sampler))
batch_labels = [labels[i].item() for i in batch_indices]
from collections import Counter
print('Batch composition:', Counter(batch_labels))
# Should show: {0: 10, 1: 10, 2: 10, 3: 10, 4: 10, 5: 10}
"

# Verify batch size
python -c "
from STMasm_enhanced6 import *
# ... setup code ...
print(f'Batches per epoch: {len(train_loader)}')
print(f'Expected: ~{len(train_dataset) // 126}')
"
```

---

## Hyperparameter Tuning

### Batch Sampler Parameters

#### n_samples_per_class

**Trade-offs**:
- **Higher** (e.g., 25–30):
  - Pros: More stable gradients, better generalization
  - Cons: Fewer batches per epoch, slower convergence
- **Lower** (e.g., 10–15):
  - Pros: More batches per epoch, faster iterations
  - Cons: Noisier gradients, may need more epochs

**Recommendation**: Start with 21 (default), adjust based on:
- Memory constraints
- Training stability
- Convergence speed

#### n_classes_per_batch

**Options**:
- **All classes (6)**: Current default
  - Pros: Perfect balance every batch
  - Cons: Largest batch size
- **Subset (e.g., 3–4)**: Alternative
  - Pros: Smaller batch size, more batches
  - Cons: Less consistent balance

**Recommendation**: Keep at 6 for maximum balance

### Alternative Configurations

```python
# Configuration 1: Smaller batches, more iterations
n_classes_per_batch = 6
n_samples_per_class = 15
# → batch_size = 90, more batches per epoch

# Configuration 2: Larger batches, fewer iterations
n_classes_per_batch = 6
n_samples_per_class = 25
# → batch_size = 150, fewer batches per epoch

# Configuration 3: Subset sampling
n_classes_per_batch = 4
n_samples_per_class = 30
# → batch_size = 120, variable class selection
```

---

## Future Directions

### Potential Enhancements

1. **Adaptive Sampling Rates**:
   - Sample easier classes less frequently as training progresses
   - Focus on difficult classes in later epochs
   - Dynamic `n_samples_per_class` based on per-class loss

2. **Hybrid Sampling Strategy**:
   - Start with balanced sampling (epochs 1–30)
   - Gradually transition to natural sampling (epochs 31–50)
   - Best of both worlds: balanced learning + natural distribution

3. **Difficulty-Aware Sampling**:
   - Sample hard examples more frequently
   - Combine with balanced batch sampling
   - Per-sample difficulty estimated from loss history

4. **Class-Specific Sample Counts**:
   - Different `n_samples` for different classes
   - More samples for confusable pairs (0↔1, 2↔3)
   - Fewer samples for easy classes

### Ablation Studies

To validate v6 design:
1. **Vary n_samples_per_class**: Test 10, 15, 21, 25, 30
2. **Vary n_classes_per_batch**: Test 3, 4, 5, 6
3. **With vs without contrastive loss**: Is it still beneficial with balanced sampling?
4. **Hybrid sampling schedule**: Compare always-balanced vs progressive shift

---

## Comparison Table: v4 vs v5 vs v6

| Aspect | v4 | v5 | v6 |
|--------|----|----|-----|
| **Class Weights** | ✓ sqrt + boost | ✗ NONE | ✗ **NONE** |
| **Batch Sampling** | Random | Random | **Balanced** |
| **Loss Function** | Weighted Focal | Unweighted Focal | **Unweighted Focal** |
| **Balancing Level** | Loss-level | None (Focal only) | **Data-level** |
| **Training Complexity** | High (weight tuning) | Low | **Medium (sampler config)** |
| **Batch Consistency** | Variable | Variable | **Fixed (balanced)** |
| **Minority Exposure** | Natural | Natural | **Increased (equal)** |
| **Overfitting Risk** | Low-Medium | Low | **Medium (minority oversampling)** |
| **Generalization** | May overfit to weights | Good | **Monitor train-test gap** |
| **Code Complexity** | Higher | Lower | **Medium** |
| **Expected Val F1** | 0.70–0.75 | 0.68–0.72 | **0.70–0.74** |
| **Best For** | Known confusion patterns | General-purpose | **Imbalanced datasets** |

---

## Implementation Checklist

When using v6, ensure:

- [ ] Balanced sampler properly configured (n_classes, n_samples)
- [ ] DataLoader uses `batch_sampler` (not `batch_size` + `shuffle`)
- [ ] Validation/test loaders use standard sampling (no balancing)
- [ ] Monitor per-class F1 (especially minorities)
- [ ] Check train-test gap for overfitting
- [ ] Verify batch composition in first epoch (all classes present)
- [ ] Adjust sampler parameters if needed based on early results

---

## Citations and References

### Related Work

1. **Focal Loss**: Lin et al., "Focal Loss for Dense Object Detection" (2017)
2. **Balanced Batch Sampling**: Similar to episodic sampling in meta-learning
3. **Class Imbalance**: Cui et al., "Class-Balanced Loss Based on Effective Number of Samples" (2019)
4. **Oversampling**: Chawla et al., "SMOTE: Synthetic Minority Over-sampling Technique" (2002)
5. **ASM Architecture**: Bai et al., "Audio Spectrogram Mixer" (2022)

### Acknowledgments

- **v5 Foundation**: Unweighted Focal Loss approach
- **Balanced Sampling**: Inspired by episodic training in few-shot learning
- **Implementation**: Built on PyTorch and scikit-learn

---

## Conclusion

ASM Enhanced v6 explores **data-level balancing** through balanced batch sampling, providing an alternative to loss-based weighting (v4) and pure Focal Loss (v5).

**Key Advantages**:
- **Guaranteed balance**: Every class trained equally
- **Simple**: No loss function complexity
- **Natural**: Data augmentation rather than loss manipulation
- **Effective**: Direct control over class exposure

**Key Considerations**:
- **Overfitting risk**: Monitor minority classes carefully
- **Distribution mismatch**: Train ≠ test distribution
- **Computational**: Slightly more complex sampling logic

**Expected Outcome**: v6 should achieve the **best minority class performance** among all versions, with competitive or superior overall F1 scores. The combination of balanced sampling and Focal Loss provides robust handling of class imbalance at both data and loss levels.

**Next Steps**: 
1. Train v6 on full dataset
2. Compare against v4 (loss-level) and v5 (natural)
3. Analyze per-class performance (especially minorities)
4. Monitor train-test gap for overfitting
5. Consider hybrid approach if pure balanced sampling shows issues

**Success Metric**: If v6 achieves **≥0.72 Val Macro F1** with strong minority class performance, it validates data-level balancing as a superior approach for imbalanced audio classification.

---

*Document Version: 1.0*  
*Last Updated: 2026-01-26*  
*Author: ASM v6 Development Team*
