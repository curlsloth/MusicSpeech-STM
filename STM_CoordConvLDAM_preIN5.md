# STM_CoordConvLDAM_preIN5: Enhanced Training with Class-Balanced Sampling & SpecAugment

**Version:** 5.0 (Targeted Minority Class Improvements)  
**Parent:** STM_CoordConvLDAM_preIN3 (V3)  
**Target:** 0.88-0.90 Macro F1  
**Date:** February 2026

---

## Executive Summary

STM_CoordConvLDAM_preIN5 addresses the key bottlenecks identified in V3's performance by implementing three targeted improvements:

1. **Earlier DRW Activation** - Epoch 30 instead of 50
2. **Class-Balanced Sampling** - WeightedRandomSampler for equal class exposure
3. **SpecAugment Masking** - Time/frequency masking for better generalization

These changes specifically target the underperforming classes (`music:non-vocal` at 0.67 F1 and `speech:tonal` at 0.72 recall) while preserving V3's excellent performance on majority classes.

---

## V3 Performance Analysis (Baseline)

### Test Results (V3)
```
Test Macro F1: 0.8646

                  precision    recall  f1-score   support
speech:non-tonal       0.95      0.98      0.96     70117  ✅ Excellent
    speech:tonal       0.88      0.72      0.79     13495  ⚠️ Low recall
     music:vocal       0.83      0.85      0.84     12165  ✓ Good
 music:non-vocal       0.70      0.64      0.67      6018  ❌ Worst
       env:urban       0.95      0.98      0.97       769  ✅ Excellent
    env:wildlife       0.95      0.96      0.96      3411  ✅ Excellent
```

### Identified Issues

| Problem | Root Cause | V5 Solution |
|---------|------------|-------------|
| music:non-vocal: 0.67 F1 | Only 6k samples (0.8% of data) | Class-balanced sampling |
| speech:tonal: 0.72 recall | DRW too late (epoch 51) | Earlier DRW (epoch 31) |
| Limited augmentation variety | Only mixup | Add SpecAugment masking |

---

## V5 Improvements

### 1. Earlier DRW Activation (Epoch 30)

**Problem:** In V3, DRW activated at epoch 51 but early stopping triggered immediately after (epoch 51), giving only 1 epoch of reweighted training.

**Solution:** Activate DRW at epoch 30 instead of 50.

```python
# V3 (epoch 50)
use_drw = epoch > (num_epochs // 2)  # epoch > 50

# V5 (epoch 30)
self.drw_start_epoch = 30
use_drw = epoch > self.drw_start_epoch  # epoch > 30
```

**Rationale:**
- V3 validation shows model reaches near-peak performance by epoch 25-30
- 20+ epochs of DRW training (vs 1 epoch in V3)
- More time to refine decision boundaries for minority classes

**Expected Impact:**
- +1-2% F1 on speech:tonal and music:non-vocal
- Minority classes get more gradient signal

---

### 2. Class-Balanced Sampling via WeightedRandomSampler

**Problem:** Training data is highly imbalanced:
```
speech:non-tonal: 556,741 (72.3%)  ← Dominates training
speech:tonal:      97,091 (12.6%)
music:vocal:       87,478 (11.4%)
music:non-vocal:   25,052 (3.3%)   ← Rarely sampled
env:urban:          1,544 (0.2%)
env:wildlife:       2,487 (0.3%)
```

**Solution:** Weight each sample inversely proportional to class frequency.

```python
# Compute class weights (inverse frequency)
class_weights = 1.0 / class_sample_counts

# Assign weight to each sample
sample_weights = class_weights[train_labels]

# Create sampler
train_sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

# Use sampler instead of shuffle
train_loader = DataLoader(train_dataset, sampler=train_sampler, ...)
```

**Effect:** Each class sampled with equal probability per batch:
- music:non-vocal: Sampled ~22x more often (vs uniform)
- speech:tonal: Sampled ~5.7x more often
- env:urban/wildlife: Sampled ~360x more often

**Mathematical Basis:**
```
P(sample from class k) = 1/K  (where K=6 classes)

Before (uniform):
  P(music:non-vocal) = 25,052 / 770,393 = 3.3%

After (balanced):
  P(music:non-vocal) = 1/6 = 16.7%
```

**Trade-off:** Majority class (speech:non-tonal) sees fewer unique samples per epoch, but model already performs excellently on it (0.98 recall).

**Expected Impact:**
- +2-4% F1 on music:non-vocal
- +1-2% recall on speech:tonal
- May slightly reduce speech:non-tonal recall (acceptable)

---

### 3. SpecAugment Masking

**From:** "SpecAugment: A Simple Data Augmentation Method for ASR" (Park et al., Interspeech 2019)

**Adapted for STM:** Masks regions in spectral modulation (frequency) and temporal modulation (time) dimensions.

```python
def spec_augment(x, time_mask_param=10, freq_mask_param=3, 
                 num_time_masks=2, num_freq_masks=2):
    """
    Input: x of shape (batch, 2, 20, 121)
    - Freq dimension: 20 bins (spectral modulation)
    - Time dimension: 121 bins (temporal modulation)
    
    Masking:
    - Time: Mask up to 10 consecutive bins, apply 2 times
    - Freq: Mask up to 3 consecutive bins, apply 2 times
    """
    for i in range(batch_size):
        # Frequency masks (spectral modulation)
        for _ in range(num_freq_masks):
            f = random(0, freq_mask_param)  # 0-3 bins
            f0 = random start position
            x[i, :, f0:f0+f, :] = 0
        
        # Time masks (temporal modulation)
        for _ in range(num_time_masks):
            t = random(0, time_mask_param)  # 0-10 bins
            t0 = random start position
            x[i, :, :, t0:t0+t] = 0
    return x
```

**Application:** 50% probability per batch during training.

**Why These Parameters:**
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| time_mask_param | 10 | ~8% of 121 bins, similar to speech ASR |
| freq_mask_param | 3 | ~15% of 20 bins, preserves major patterns |
| num_time_masks | 2 | Moderate augmentation |
| num_freq_masks | 2 | Moderate augmentation |
| apply_prob | 50% | Combined with mixup (30%), sufficient variety |

**Force Robustness To:**
- Partial temporal patterns (incomplete phrases)
- Partial spectral patterns (missing frequency bands)
- Model learns to use contextual information

**Expected Impact:**
- +0.5-1.5% overall F1 through regularization
- Better generalization to unseen recordings
- Reduced overfitting on specific patterns

---

## Training Configuration

### Complete Hyperparameters

```python
# Model Architecture (from V3)
num_classes = 6
dropout = 0.3
block_dropout = 0.05
use_pretrained = True  # ImageNet ResNet-18

# V5 Enhancements
drw_start_epoch = 30          # Earlier than V3's 50
use_class_balanced_sampling = True
use_specaugment = True
specaugment_prob = 0.5
time_mask_param = 10
freq_mask_param = 3
num_time_masks = 2
num_freq_masks = 2

# Optimizer (from V3)
base_lr = 1e-4
weight_decay = 2e-4

# Discriminative LR (from V3)
lr_stem_layer1 = 0.1 * base_lr   # 1e-5
lr_layer2_layer3 = 0.5 * base_lr # 5e-5
lr_layer4_head = 1.0 * base_lr   # 1e-4
lr_attention = 1.0 * base_lr     # 1e-4
lr_batchnorm = 1.0 * base_lr     # 1e-4

# Scheduler
scheduler = ReduceLROnPlateau(mode='max', factor=0.5, patience=7, min_lr=1e-6)

# Loss (from V3)
criterion = LDAMLoss(max_m=0.5, s=30, label_smooth=0.05)

# Mixup (from V3)
mixup_alpha = 0.3
mixup_prob = 0.3

# Training
batch_size = 256
max_epochs = 100
early_stop_patience = 20
grad_clip = 1.0
```

---

## Expected Training Dynamics

### Phase Breakdown

```
Phase 1: Warm-up (Epochs 1-10)
├─ Model learns basic features
├─ Class-balanced sampling ensures all classes seen
├─ SpecAugment creates varied training examples
├─ Val F1: 0.78 → 0.84
└─ Key: Minority classes get equal exposure from start

Phase 2: Feature Learning (Epochs 10-30)
├─ Model refines decision boundaries
├─ Attention stabilizes
├─ Minority classes continue getting balanced exposure
├─ Val F1: 0.84 → 0.86
└─ Key: Pre-DRW learning with balanced data

Phase 3: DRW Activated (Epochs 31-50)
├─ Loss reweighted for minority classes
├─ Combined with balanced sampling = double emphasis
├─ speech:tonal and music:non-vocal get extra gradient
├─ Val F1: 0.86 → 0.88
└─ Key: 20 epochs of minority-focused refinement

Phase 4: Fine-tuning (Epochs 50-70)
├─ Learning rate reduced via plateau scheduler
├─ Final boundary refinement
├─ Val F1: 0.88 → 0.885
└─ Early stopping likely triggers here

Expected Convergence: 60-80 epochs (vs V3's 51)
Expected Best Val F1: 0.875-0.89
```

---

## Comparison Table: V3 vs V5

| Component | V3 | V5 | Change |
|-----------|-----|-----|--------|
| **DRW Start** | Epoch 51 | **Epoch 31** | 20 epochs earlier |
| **Sampling** | Uniform random | **Class-balanced** | Equal class probability |
| **SpecAugment** | None | **Yes (50% prob)** | New augmentation |
| **Time mask** | N/A | **10 bins, 2 masks** | New |
| **Freq mask** | N/A | **3 bins, 2 masks** | New |
| Architecture | ImageNet ResNet-18 + CA/SE | Same | Unchanged |
| Discriminative LR | Yes | Yes | Unchanged |
| Mixup | α=0.3, 30% prob | α=0.3, 30% prob | Unchanged |
| LDAM Loss | V2 implementation | V2 implementation | Unchanged |
| Dropout | 0.3 | 0.3 | Unchanged |
| Weight decay | 2e-4 | 2e-4 | Unchanged |

---

## Expected Performance

### Overall Metrics

| Metric | V3 Baseline | V5 Target | Gain |
|--------|-------------|-----------|------|
| **Test Macro F1** | 0.8646 | **0.88-0.90** | +1.5-3.5% |
| **Val Macro F1** | 0.8513 | **0.87-0.89** | +2-4% |
| **Epochs to best** | 31 | **50-70** | More refinement |

### Per-Class Targets

| Class | V3 F1 | V3 Recall | V5 Target F1 | V5 Target Recall | Improvement Source |
|-------|------|-----------|--------------|------------------|-------------------|
| speech:non-tonal | 0.96 | 0.98 | 0.95-0.96 | 0.96-0.98 | May decrease slightly |
| **speech:tonal** | **0.79** | **0.72** | **0.82-0.85** | **0.77-0.82** | DRW + balanced sampling |
| music:vocal | 0.84 | 0.85 | 0.85-0.87 | 0.86-0.88 | SpecAugment + sampling |
| **music:non-vocal** | **0.67** | **0.64** | **0.72-0.78** | **0.70-0.76** | Sampling (22x more) |
| env:urban | 0.97 | 0.98 | 0.96-0.98 | 0.97-0.99 | Maintained |
| env:wildlife | 0.96 | 0.96 | 0.95-0.97 | 0.96-0.98 | Maintained |

---

## Risk Assessment

| Change | Risk Level | Potential Issue | Mitigation |
|--------|-----------|-----------------|------------|
| Earlier DRW | Low | None expected | Proven technique |
| Class-balanced sampling | Medium | Majority class degradation | Monitor speech:non-tonal |
| SpecAugment | Low | Over-masking breaks patterns | Conservative params |

### Monitoring Points

1. **speech:non-tonal recall** - Should stay >0.95
2. **Early convergence check** - Val F1 should reach 0.84 by epoch 15
3. **DRW activation effect** - Should see minority class improvement after epoch 31

---

## Usage

### Training

```bash
# Standard training (full dataset)
python STM_CoordConvLDAM_preIN5.py 0

# With downsampled non-tonal speech
python STM_CoordConvLDAM_preIN5.py 1
```

### Output Directory

```
model/STM/CoordConvLDAM_preIN5_corpora_categories/
├── standard/
│   └── ckpt/
│       └── YYYY-MM-DD_HH-MM/
│           ├── best_model.pt
│           ├── checkpoint_epoch_10.pt
│           ├── checkpoint_epoch_20.pt
│           ├── ...
│           ├── test_predictions.npy
│           └── test_targets.npy
└── downsample/
    └── ...
```

---

## Expected Console Output

```
============================================================
Loading and preparing data...
============================================================
[... data loading ...]

Class-Balanced Sampling (V5 Enhancement):
  Original class distribution:
    Class 0: 556,741 samples, weight=0.000002
    Class 1: 97,091 samples, weight=0.000010
    Class 2: 87,478 samples, weight=0.000011
    Class 3: 25,052 samples, weight=0.000040
    Class 4: 1,544 samples, weight=0.000648
    Class 5: 2,487 samples, weight=0.000402
  Effect: Minority classes sampled ~360x more often

DataLoaders created with batch_size=256
  Train: WeightedRandomSampler (class-balanced)
  Val/Test: Sequential (no sampling)

============================================================
Creating ImageNet-Pretrained ResNet-18 for STM (V5)...
============================================================

V5 Training Configuration:
  • Earlier DRW: Activated at epoch 31 (vs V3's epoch 51)
  • Class-Balanced Sampling: WeightedRandomSampler
  • SpecAugment: Time mask (10 bins), Freq mask (3 bins)
  • Discriminative LR: Stem/L1 (0.1x), L2-3 (0.5x), L4/Head (1.0x)
  • Weight decay: 2e-4 (V2 proven)
  • Mixup alpha: 0.3 (30% probability)
  • Head dropout: 0.3
  • Early stopping: 20 epochs patience

============================================================
Starting training...
============================================================

Epoch 1/100
============================================================
  Batch 0/3010, Loss: 2.1234, DRW: False, SpecAug: True
  Batch 500/3010, Loss: 1.8765, DRW: False, SpecAug: True
  ...
Train Loss: 1.9234
Val Loss: 1.6543, Val Macro F1: 0.7856
✓ Saved best model with Val F1: 0.7856

[... training continues ...]

Epoch 31/100
============================================================

*** Activating Deferred Reweighting (DRW) at epoch 31 ***

  Batch 0/3010, Loss: 1.2345, DRW: True, SpecAug: True
  ...
Train Loss: 1.2876
Val Loss: 1.4321, Val Macro F1: 0.8678
✓ Saved best model with Val F1: 0.8678

[... training continues ...]

Epoch 65/100
============================================================
No improvement for 20 epoch(s)
Early stopping triggered after 65 epochs

============================================================
Training completed! Best Val F1: 0.8812
============================================================

============================================================
Evaluating on test set...
============================================================
Test Loss: 1.4123
Test Macro F1: 0.8856

Classification Report:
                  precision    recall  f1-score   support

speech:non-tonal       0.94      0.97      0.95     70117
    speech:tonal       0.85      0.79      0.82     13495  ← Improved!
     music:vocal       0.85      0.87      0.86     12165  ← Improved!
 music:non-vocal       0.75      0.72      0.73      6018  ← Improved!
       env:urban       0.95      0.98      0.96       769
    env:wildlife       0.95      0.96      0.96      3411
```

---

## Success Criteria

### Primary Goals (Must Achieve)
- ✅ **Test Macro F1 > 0.87** (beats V3's 0.8646)
- ✅ **music:non-vocal F1 > 0.70** (V3: 0.67)
- ✅ **speech:tonal recall > 0.75** (V3: 0.72)

### Secondary Goals (Should Achieve)
- ✅ **Test Macro F1 > 0.88** (significant improvement)
- ✅ **music:non-vocal F1 > 0.73** (closing gap to majority classes)
- ✅ **No class below 0.70 F1** (balanced performance)

### Stretch Goals (May Achieve)
- 🎯 **Test Macro F1 > 0.89** (approaching SOTA)
- 🎯 **All minority classes > 0.80 F1**

---

## References

1. **SpecAugment:**
   - Park et al., "SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition", Interspeech 2019

2. **Class Imbalance:**
   - Cao et al., "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss", NeurIPS 2019
   - Kang et al., "Decoupling Representation and Classifier for Long-Tailed Recognition", ICLR 2020

3. **Sampling Strategies:**
   - Buda et al., "A Systematic Study of the Class Imbalance Problem in CNNs", Neural Networks 2018

4. **Architecture (from V3):**
   - Hou et al., "Coordinate Attention for Efficient Mobile Network Design", CVPR 2021
   - Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018
   - Liu et al., "An Intriguing Failing of CNNs and the CoordConv Solution", NeurIPS 2018

---

## Conclusion

V5 implements three targeted improvements to address V3's minority class underperformance:

1. **Earlier DRW** → More time for minority class boundary refinement
2. **Class-balanced sampling** → Equal exposure for all classes during training
3. **SpecAugment** → Better generalization through masking augmentation

These changes work synergistically:
- Balanced sampling ensures minority classes are seen equally often
- Earlier DRW amplifies their loss contribution
- SpecAugment prevents overfitting to their limited unique patterns

**Expected Outcome:** 0.88-0.90 Macro F1 with significant improvements on music:non-vocal (+5-10% F1) and speech:tonal (+3-5% recall).
