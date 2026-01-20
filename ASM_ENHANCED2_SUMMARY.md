# ASM Enhanced v2 Summary

## Problem Analysis from v1 Results

### v1 Performance (Macro F1: 0.8566)
```
Class 0 (Speech: Non-tonal):  F1=0.9592  ✓ Excellent
Class 1 (Speech: Tonal):      F1=0.7893  ✗ Poor (confused with Class 0)
Class 2 (Music):              F1=0.8349  ~ Good
Class 3 (Environment):        F1=0.6709  ✗ Poor (confused with Class 2)
Class 4 (Small minority):     F1=0.9547  ✓ Excellent (over-weighted?)
Class 5 (Small minority):     F1=0.9306  ✓ Excellent (over-weighted?)
```

### Root Cause Identified

**Issue**: Inverse-frequency class weighting treats all minority classes equally, but:
- **Classes 4 & 5** are truly distinct → High weights work well
- **Classes 1 & 3** are similar to dominant classes → High weights cause confusion

**Confusable Pairs**:
1. **Class 1 (Speech: Tonal) ↔ Class 0 (Speech: Non-tonal)**
   - Both are speech, differ only in tonality
   - Model over-predicts Class 0 due to its dominance
   - High weight on Class 1 helps recall but hurts precision

2. **Class 3 (Environment) ↔ Class 2 (Music)**
   - Acoustic similarity (background music, nature sounds, etc.)
   - Model struggles to separate decision boundary
   - High weight on Class 3 doesn't address core similarity

## v2 Solution: Confusion-Aware Learning

### 1. Confusion-Aware Class Weighting

**Strategy**: Reduce weights for confusable pairs by 30%

```python
# Before (v1)
class_weights = [0.5, 1.8, 1.2, 1.9, 2.5, 2.3]  # Inverse frequency

# After (v2)
class_weights = [0.35, 1.26, 0.84, 1.33, 2.5, 2.3]  # Adjusted for pairs (0,1) and (2,3)
```

**Why it works**: 
- Prevents over-prediction of minority classes in confusable pairs
- Maintains high weights for truly distinct minorities (4, 5)
- Balances precision/recall trade-off

### 2. Asymmetric Focal Loss Gamma

**Strategy**: Higher gamma (2.5) for confusable pairs vs base (2.0)

```python
gamma = {
    Class 0: 2.5,  # Confusable with 1
    Class 1: 2.5,  # Confusable with 0
    Class 2: 2.5,  # Confusable with 3
    Class 3: 2.5,  # Confusable with 2
    Class 4: 2.0,  # Distinct
    Class 5: 2.0   # Distinct
}
```

**Why it works**:
- Higher gamma → Model focuses more on hard negatives
- Forces model to learn subtle differences between similar classes
- Less aggressive for already-distinct classes

### 3. Adaptive Label Smoothing

**Strategy**: Extra smoothing between confusable pairs

```python
# Standard smoothing (all classes)
smooth = 0.1

# Additional smoothing for confusable pairs
smooth_pairs = {
    (0, 1): 0.2,  # Class 1 samples get 20% probability to Class 0
    (2, 3): 0.2   # Class 3 samples get 20% probability to Class 2
}
```

**Why it works**:
- Acknowledges inherent ambiguity between similar classes
- Prevents over-confidence on boundary cases
- Improves calibration for confusable pairs

### 4. Inter-Class Margin Regularization (NEW)

**Strategy**: Explicit loss term to push confusable classes apart in feature space

```python
margin_loss = max(0, margin - distance(features_class_i, features_class_j))
```

**Why it works**:
- Forces model to learn more discriminative features
- Creates larger decision margin between confusable classes
- Metric learning approach proven effective for fine-grained classification

**Implementation**:
```python
class MarginRankingLoss:
    def __init__(self, margin=0.5):
        # Penalizes if distance < 0.5 in feature space
    
    def forward(self, features, targets, pairs):
        # Computes pairwise distances for confusable samples
        # Returns penalty if too close
```

### 5. Auxiliary Binary Classifiers (NEW)

**Strategy**: Train dedicated binary classifiers for confusable pairs

```python
# Additional heads in model
binary_classifier_01 = nn.Linear(dim, 2)  # Class 0 vs 1
binary_classifier_23 = nn.Linear(dim, 2)  # Class 2 vs 3

# Multi-task loss
total_loss = main_loss + 0.1 * margin_loss + 0.2 * binary_loss
```

**Why it works**:
- Provides focused signal for hard distinctions
- Binary classification is easier than 6-class
- Multi-task learning improves feature quality

## Expected Improvements

### Target Performance (Test Macro F1: ~0.88)

**Conservative estimate**:
```
Class 0: 0.96 → 0.96  (no change, already excellent)
Class 1: 0.79 → 0.84  (+5% from confusion reduction)
Class 2: 0.83 → 0.85  (+2% from better features)
Class 3: 0.67 → 0.75  (+8% from margin + binary classifier)
Class 4: 0.95 → 0.94  (-1% acceptable trade-off)
Class 5: 0.93 → 0.92  (-1% acceptable trade-off)

Macro F1: 0.8566 → 0.8767 (+2.0%)
```

**Key improvements**:
- **Class 1**: Better separation from Class 0 via margin loss
- **Class 3**: Stronger discrimination from Class 2 via binary classifier
- **Classes 4, 5**: Slight decrease acceptable (still >0.90)

## Implementation Details

### Model Architecture (Unchanged)
- Same ASM-RH blocks as v1
- Same 2D positional encoding
- Same SpecAugment
- **Added**: 2 auxiliary binary classifiers

### Loss Function (New)
```python
# Main: Confusion-aware focal loss
main_loss = ConfusionAwareFocalLoss(
    adjusted_weights, 
    confusion_pairs=[(0,1), (2,3)],
    gamma_confusable=2.5
)

# Auxiliary: Margin ranking
margin_loss = MarginRankingLoss(margin=0.5)

# Auxiliary: Binary classification
binary_loss = CrossEntropyLoss(binary_01) + CrossEntropyLoss(binary_23)

# Combined
total_loss = main_loss + 0.1 * margin_loss + 0.2 * binary_loss
```

### Training (Enhanced)
- Warmup: 5 epochs
- Base LR: 1e-3
- Scheduler: Cosine annealing
- **New**: Multi-task gradient balancing
- **New**: Per-batch confusion monitoring

## Key Differences from v1

| Aspect | v1 | v2 |
|--------|----|----|
| Class weights | Uniform inverse-frequency | Confusion-aware adjustment |
| Focal gamma | Fixed 2.0 | Asymmetric (2.0 vs 2.5) |
| Label smoothing | Uniform 0.1 | Adaptive (0.1 vs 0.2) |
| Feature learning | Implicit | Explicit margin loss |
| Classifier | Single head | Multi-task (main + 2 binary) |
| Loss components | 1 (focal) | 3 (focal + margin + binary) |

## Usage

```bash
# Train v2 (standard mode)
python STMasm_enhanced2.py 0

# Train with downsampling
python STMasm_enhanced2.py 1

# Resume training
python STMasm_enhanced2.py 0 --resume model/STM/ASM_Enhanced2_corpora_categories/standard/ckpt/TIMESTAMP
```

## Monitoring Training

**Key metrics to watch**:
1. **Main loss**: Should converge slower than v1 (more regularization)
2. **Margin loss**: Should decrease steadily (classes separating)
3. **Binary losses**: Should be low (<0.1 each)
4. **Confusion matrix**: Monitor (0,1) and (2,3) confusion rates

**Expected training curve**:
```
Epoch 1:  Main=0.05, Margin=0.3, Binary=0.4  → Total=0.12
Epoch 10: Main=0.02, Margin=0.1, Binary=0.1  → Total=0.04
Epoch 30: Main=0.005, Margin=0.02, Binary=0.02 → Total=0.01
```

## Troubleshooting

### If Class 1/3 still perform poorly:
1. Increase margin: 0.5 → 0.7
2. Increase binary weight: 0.2 → 0.3
3. Reduce confusion pair weights further: 0.7 → 0.5

### If Classes 4/5 drop too much:
1. Restore partial weighting: 0.7 → 0.85
2. Reduce overall regularization

### If training is unstable:
1. Reduce margin weight: 0.1 → 0.05
2. Increase warmup: 5 → 10 epochs

## Theoretical Justification

1. **Confusion-aware weighting**: Addresses class imbalance while respecting similarity
2. **Asymmetric focal loss**: Proven in hard negative mining literature
3. **Margin loss**: Standard in metric learning (triplet loss, contrastive loss)
4. **Multi-task learning**: Improves feature quality via auxiliary tasks
5. **Adaptive smoothing**: Calibration technique for uncertain boundaries

## References

- Base ASM: `/vast/ac8888/MusicSpeech-STM/STMasm_model.py`
- Enhanced v1: `/vast/ac8888/MusicSpeech-STM/STMasm_enhanced.py`
- Enhanced v2: `/vast/ac8888/MusicSpeech-STM/STMasm_enhanced2.py`
- v1 Results: `model/STM/ASM_Enhanced_corpora_categories/standard/ckpt/2026-01-18_07-29/`

## Summary

v2 addresses v1's core issue: **treating all minority classes equally despite different confusion patterns**. By explicitly modeling class similarity and adding targeted regularization, we expect:

✓ **Better discrimination** between confusable pairs (1 vs 0, 3 vs 2)  
✓ **Maintained performance** on distinct minorities (4, 5)  
✓ **Overall improvement** of ~2% macro F1 (0.8566 → 0.8767)  
✓ **More balanced** per-class performance
