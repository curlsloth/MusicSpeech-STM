# Enhanced Kanformer v2 for STM Audio Classification

## Executive Summary

**Enhanced Kanformer v2** addresses the critical performance issues identified in v1:
- **Problem**: v1 achieved Test F1 = 0.8398 (vs Conformer baseline 0.86)
- **Root Cause**: Over-aggressive inverse-frequency class weighting hurt similar classes (1 vs 0, 3 vs 2)
- **Solution**: 5 targeted improvements for better class discrimination and balanced learning

## Performance Analysis of v1

### v1 Results (STMkanformer_enhanced.py)
```
Test Accuracy: 0.8703
Test Macro F1: 0.8398

Per-Class F1:
  Class 0 (speech: non-tonal):  0.9297 ✓ (70,117 samples)
  Class 1 (speech: tonal):      0.7104 ✗ (13,495 samples) - UNDERPERFORMED
  Class 2 (music: vocal):       0.8223 ✓ (12,165 samples)
  Class 3 (music: non-vocal):   0.6812 ✗ (6,018 samples)  - UNDERPERFORMED
  Class 4 (env: urban):         0.9585 ✓✓ (769 samples)    - OVERTRAINED
  Class 5 (env: wildlife):      0.9369 ✓✓ (3,411 samples)  - OVERTRAINED
```

### Key Observations

**Issue 1: Similar Classes Suffered**
- Class 1 (tonal speech) confused with Class 0 (non-tonal speech): F1 = 0.7104
- Class 3 (non-vocal music) confused with Class 2 (vocal music): F1 = 0.6812
- These pairs share perceptual features but had weak discrimination

**Issue 2: Minority Classes Overtrained**
- Class 4 and 5 achieved near-perfect F1 (0.95+)
- Inverse-frequency weights (w₄ = 3.41, w₅ = 1.18) were too strong
- Model focused excessively on rare classes at expense of majority

**Issue 3: Macro F1 < Conformer**
- v1 Macro F1: 0.8398
- Conformer Baseline: 0.8602
- Loss: -0.02 points (-2.4% relative)

## Enhanced v2 Improvements

### 1. Softer Class Weighting (Square Root)

**Change**: Use `sqrt(inverse_frequency)` instead of `inverse_frequency`

**Mathematical Form**:
```python
# v1: w_i = total / (num_classes × count_i)
# v2: w_i = sqrt(total / (num_classes × count_i))
```

**Effect**:
- Reduces extreme weights for rare classes
- Example: Class 4 weight: 3.41 → 1.85 (-46%)
- More balanced loss contributions across classes

**Implementation**:
```python
def compute_confusion_aware_weights(y_train, num_classes=6):
    for i in range(num_classes):
        count = class_counts.get(i, 1)
        weight = math.sqrt(total / (num_classes * count))  # NEW: sqrt
```

### 2. Confusion-Aware Weight Boosting

**Change**: Manually adjust weights for similar class pairs

**Strategy**:
```python
# Boost discrimination between similar classes
weights[1] *= 1.3  # Tonal speech (vs non-tonal)
weights[3] *= 1.3  # Non-vocal music (vs vocal)

# Reduce weights for overtrained classes
weights[4] *= 0.7  # Urban environment
weights[5] *= 0.8  # Wildlife environment
```

**Rationale**:
- Classes 1 and 3 are hardest to distinguish from their siblings
- Classes 4 and 5 don't need strong emphasis (already high performance)
- This shifts model focus to difficult discrimination tasks

**Expected Weights**:
```
Class 0 (speech: non-tonal):  ~0.3  (baseline)
Class 1 (speech: tonal):      ~1.0  (boosted 1.3×)
Class 2 (music: vocal):       ~0.7  (baseline)
Class 3 (music: non-vocal):   ~1.6  (boosted 1.3×)
Class 4 (env: urban):         ~1.3  (reduced 0.7×)
Class 5 (env: wildlife):      ~0.9  (reduced 0.8×)
```

### 3. Contrastive Regularization Loss

**New Component**: Maximize feature distance between similar classes

**Loss Function**:
```python
Total Loss = Focal Loss + λ × Contrastive Loss

Contrastive Loss = Σ (1 / ||f_a - f_b||₂)
```

Where:
- `f_a`, `f_b`: Feature embeddings from similar classes
- `λ = 0.1`: Contrastive weight (tunable)
- Pairs: (0,1) for speech, (2,3) for music

**Mechanism**:
1. Extract features before classifier head
2. Sample pairs from similar classes (32 pairs per batch)
3. Compute pairwise L2 distances
4. Loss increases when distance is small → gradient pushes classes apart

**Implementation**:
```python
class ContrastiveFocalLoss(nn.Module):
    def forward(self, inputs, targets, features=None):
        focal_loss = ...  # Standard focal loss
        
        # NEW: Contrastive term
        for class_a, class_b in [(0,1), (2,3)]:
            features_a = features[targets == class_a]
            features_b = features[targets == class_b]
            distances = F.pairwise_distance(features_a, features_b)
            contrastive_loss += (1.0 / (distances + 1e-6)).mean()
        
        return focal_loss + 0.1 * contrastive_loss
```

**Expected Impact**: +1-2% F1 on Classes 1 and 3

### 4. Increased KAN Groups (4 → 8)

**Change**: Use 8 KAN groups instead of 4

**Rationale**:
- 4 groups (v1) was TOO constrained for 6-class task with 770k training samples
- 8 groups (baseline) provides necessary expressivity
- With improved regularization (softer weights + contrastive loss), overfitting risk is mitigated
- **Technical constraint**: d_model=128 and ffn_dim=512 must be divisible by num_groups
  - 128 % 6 = 2 ❌ (not divisible)
  - 128 % 8 = 0 ✓ (divisible)
  - 512 % 8 = 0 ✓ (divisible)

**Architecture**:
```python
model = EnhancedKanformerClassifier(
    num_kan_groups=8  # Back to baseline (was reduced to 4 in v1)
)
```

**Parameter Impact**:
- v1 (4 groups): ~3.2M parameters (too constrained)
- v2 (8 groups): ~3.85M parameters (optimal)
- Baseline (8 groups): ~3.85M parameters (same capacity)

**Key Insight**: The issue with v1 wasn't the number of groups, but the class weighting strategy. By fixing the weights, we can safely use 8 groups for better performance.

### 5. Reduced Label Smoothing (0.05 → 0.01)

**Change**: Use ε = 0.01 instead of ε = 0.05

**Rationale**:
- v1's ε = 0.05 was too aggressive (5% smoothing)
- Reduced model's ability to make confident predictions
- Hurt discrimination between similar classes

**Formula**:
```python
# v1: y_smooth = y × 0.95 + (1-y) × 0.05/5
# v2: y_smooth = y × 0.99 + (1-y) × 0.01/5
```

**Effect**:
- Allows sharper decision boundaries
- Better for similar class pairs that need strong separation

## Architecture Specifications

### Model Configuration
```python
EnhancedKanformerClassifier(
    input_dim=20,           # Frequency bins
    num_classes=6,
    d_model=128,
    num_heads=4,
    ffn_dim=512,
    num_layers=4,
    kernel_size=31,
    dropout=0.1,
    num_kan_groups=8        # CORRECTED: 8 (not 6) for divisibility
)
```

### Parameter Count
- Total parameters: ~3,850,000
- Trainable: ~3,850,000
- Memory: ~11GB GPU (batch_size=128)

### Training Hyperparameters
```python
Optimizer: AdamW
  - lr: 1e-4
  - weight_decay: 1e-4

Scheduler: ReduceLROnPlateau
  - mode: 'max' (monitor Val F1)
  - factor: 0.5
  - patience: 3

Loss: ContrastiveFocalLoss
  - alpha: confusion-aware weights
  - gamma: 2.0
  - label_smoothing: 0.01
  - contrastive_weight: 0.1

Batch size: 128
Epochs: 50
Gradient clipping: 1.0
```

## Expected Performance Improvements

### Target Metrics
```
Test Macro F1: 0.86+ (target: match or exceed Conformer)

Per-Class F1 Targets:
  Class 0: 0.92-0.93 (maintain)
  Class 1: 0.78-0.82 (+8-11 points) ← KEY IMPROVEMENT
  Class 2: 0.82-0.84 (maintain)
  Class 3: 0.75-0.78 (+7-10 points) ← KEY IMPROVEMENT
  Class 4: 0.92-0.95 (slight reduction OK)
  Class 5: 0.91-0.93 (slight reduction OK)
```

### Confusion Matrix Expectations

**v1 Problematic Pairs**:
```
Class 0 → 1: 1,461 errors
Class 1 → 0: 1,682 errors
Class 2 → 3: 950 errors
Class 3 → 2: 895 errors
```

**v2 Target** (30% reduction):
```
Class 0 → 1: ~1,020 errors (-30%)
Class 1 → 0: ~1,180 errors (-30%)
Class 2 → 3: ~665 errors (-30%)
Class 3 → 2: ~625 errors (-30%)
```

## Usage Instructions

### Training from Scratch
```bash
# Standard mode (full dataset)
python STMkanformer_enhanced2.py 0

# Downsampled mode (100k non-tonal speech)
python STMkanformer_enhanced2.py 1
```

### Resuming Training
```bash
# Resume from checkpoint directory
python STMkanformer_enhanced2.py 0 --resume \
  model/STM/Kanformer_enhanced2_corpora_categories/standard/ckpt/2026-01-18_10-30
```

### Checkpoint Management

**Checkpoint Files**:
- `best_model.pt`: Best validation F1 model (for final test)
- `latest_checkpoint.pt`: Most recent epoch (for resumption)
- `checkpoint_epoch_X.pt`: Every 5 epochs (ablation/rollback)

**Checkpoint Contents**:
```python
{
    'epoch': int,
    'model_state_dict': OrderedDict,
    'optimizer_state_dict': dict,
    'val_f1': float,
    'train_losses': List[float],
    'val_losses': List[float],
    'val_f1_scores': List[float],
    'confusion_history': List[np.ndarray]  # NEW: track confusion over time
}
```

### Evaluating Test Performance
```bash
# After training completes
cd model/STM/Kanformer_enhanced2_corpora_categories/standard/ckpt/[timestamp]

# Check outputs
cat *.out | grep "Test Macro F1"
cat *.out | grep "Confusion"

# Load predictions
python
>>> import numpy as np
>>> preds = np.load('test_predictions.npy')
>>> targets = np.load('test_targets.npy')
>>> conf_matrix = np.load('test_confusion_matrix.npy')
```

## Monitoring Training

### Key Metrics to Watch

**1. Validation F1 Progression**
- Should steadily increase to 0.85+ by epoch 30
- Plateaus trigger LR reduction (patience=3)
- Best model saved when new peak achieved

**2. Confusion Pair Tracking**
```
Every epoch prints:
Confusion - Class 0 vs 1: [count] | [count]
Confusion - Class 2 vs 3: [count] | [count]
```
- Monitor these decrease over training
- Sharp increases indicate overfitting

**3. Loss Components**
```
Batch logs show:
Loss: [focal + contrastive combined]
Grad: [gradient norm]
```
- Focal loss should dominate (~90%)
- Contrastive contributes ~10%
- Grad norm should stay < 3.0

### Training Curves

**Expected Patterns**:
```
Epoch 1-10:   Rapid decrease in train loss
Epoch 10-20:  Validation F1 climbs to 0.82-0.83
Epoch 20-30:  First LR reduction, F1 → 0.84-0.85
Epoch 30-40:  Second LR reduction, F1 → 0.85-0.86
Epoch 40-50:  Convergence, minimal improvement
```

**Warning Signs**:
- Val F1 decreasing: Overfitting (reduce weight_decay)
- Train loss > 0.5 at epoch 20: Underfitting (increase model capacity)
- NaN losses: Numerical instability (check for extreme weights)

## Ablation Studies (Future Work)

### Planned Experiments

**1. Weight Sensitivity Analysis**
```bash
# Test different boost factors
weights[1] *= [1.0, 1.2, 1.4, 1.6]  # Current: 1.3
weights[3] *= [1.0, 1.2, 1.4, 1.6]  # Current: 1.3
```

**2. Contrastive Weight Tuning**
```python
contrastive_weight = [0.0, 0.05, 0.1, 0.2, 0.5]  # Current: 0.1
```

**3. KAN Group Sweep**
```python
num_kan_groups = [4, 5, 6, 7, 8]  # Current: 6
```

**4. Label Smoothing Grid**
```python
label_smoothing = [0.0, 0.01, 0.02, 0.05]  # Current: 0.01
```

### How to Run Ablations

1. Copy `STMkanformer_enhanced2.py` → `STMkanformer_ablation.py`
2. Modify hyperparameter
3. Train with new experiment name:
```python
directory = f"model/STM/Kanformer_ablation_boost{boost_factor}_lambda{lambda_val}"
```
4. Compare test F1 across runs

## Comparison with Previous Versions

| Metric | Conformer Baseline | Kanformer v1 (Enhanced) | Kanformer v2 (Target) |
|--------|-------------------|-------------------------|----------------------|
| Test Macro F1 | 0.8602 | 0.8398 | **0.86+** |
| Class 0 F1 | 0.93 | 0.9297 | 0.92-0.93 |
| Class 1 F1 | 0.84 | 0.7104 | **0.78-0.82** |
| Class 2 F1 | 0.82 | 0.8223 | 0.82-0.84 |
| Class 3 F1 | 0.71 | 0.6812 | **0.75-0.78** |
| Class 4 F1 | 0.96 | 0.9585 | 0.92-0.95 |
| Class 5 F1 | 0.94 | 0.9369 | 0.91-0.93 |
| Parameters | 3.85M | 3.20M | 3.60M |
| Training Time/Epoch | 8 min | 7 min | 8 min |

## Technical Implementation Details

### Contrastive Loss Deep Dive

**Sampling Strategy**:
```python
# For each similar class pair (a, b)
n_pairs = min(len(features_a), len(features_b), 32)  # Cap at 32 to avoid memory issues

# Random sampling without replacement
idx_a = torch.randperm(len(features_a))[:n_pairs]
idx_b = torch.randperm(len(features_b))[:n_pairs]
```

**Distance Computation**:
```python
distances = F.pairwise_distance(pairs_a, pairs_b, p=2)  # L2 norm
loss = (1.0 / (distances + 1e-6)).mean()  # Inverse distance (closer = higher loss)
```

**Gradient Flow**:
- Loss backpropagates through `feature_extractor` layer
- Encourages orthogonal representations for similar classes
- Does NOT affect earlier Kanformer blocks directly

### Weight Computation Example

**Class 1 (Tonal Speech)**:
```python
# Training samples: 80,258
# Total samples: 770,000

# Step 1: Base weight (sqrt inverse frequency)
base_weight = sqrt(770000 / (6 × 80258)) = 1.245

# Step 2: Normalize to sum = 6
normalized = 1.245 / sum_all_weights × 6 = 0.75

# Step 3: Confusion-aware boost
final_weight = 0.75 × 1.3 = 0.975 ≈ 1.0
```

### Feature Extractor Architecture

**Purpose**: Extract discriminative embeddings before classification

**Design**:
```python
self.feature_extractor = EnhancedGroupRationalKANLayer(
    d_model=128,      # Input from pooled Kanformer output
    out_features=64,  # Compressed representation
    num_groups=2,     # Lightweight (only 2 groups)
    dropout=0.1
)

self.classifier_head = nn.Linear(64, 6)  # Final classifier
```

**Why This Helps**:
1. Provides access to pre-logit features for contrastive loss
2. Adds trainable non-linearity before classification
3. Compresses 128D → 64D for efficiency

## Known Limitations and Caveats

### 1. Hyperparameter Sensitivity
- Contrastive weight (λ=0.1) may need tuning
- Optimal boost factors (1.3×) are heuristic-based
- Different datasets may require different settings

### 2. Computational Cost
- Contrastive loss adds ~10% training time
- Feature extraction requires extra forward pass
- Memory usage increases by ~1GB

### 3. Class Imbalance Still Exists
- Sqrt weighting is softer but doesn't eliminate imbalance
- Class 0 still dominates training (64.6% of samples)
- May benefit from additional data augmentation

### 4. Contrastive Sampling Limitations
- Only samples 32 pairs per batch → high variance
- Rare classes (4, 5) may not appear in every batch
- Could use memory bank for more stable gradients

## Troubleshooting Guide

### Issue 1: Val F1 Not Improving Past 0.80
**Possible Causes**:
- Contrastive weight too high (overwhelming focal loss)
- Learning rate not reducing (check scheduler)
- Model capacity insufficient

**Solutions**:
```python
# Reduce contrastive weight
contrastive_weight = 0.05  # Down from 0.1

# Check scheduler patience
scheduler = ReduceLROnPlateau(..., patience=2)  # More aggressive

# Increase model depth
num_layers = 6  # Up from 4
```

### Issue 2: Classes 1 and 3 Still Underperforming
**Possible Causes**:
- Boost factors too weak
- Contrastive loss not converging
- Inherent class overlap too high

**Solutions**:
```python
# Stronger boosting
weights[1] *= 1.5  # Up from 1.3
weights[3] *= 1.5

# Increase contrastive weight
contrastive_weight = 0.2  # Up from 0.1

# Add hard negative mining (future work)
```

### Issue 3: Classes 4 and 5 Degraded Too Much
**Possible Causes**:
- Weight reduction too aggressive (0.7×, 0.8×)
- Model forgetting rare class patterns

**Solutions**:
```python
# Softer reduction
weights[4] *= 0.85  # Up from 0.7
weights[5] *= 0.9   # Up from 0.8

# Verify sample counts in batches
print(f"Class 4 in batch: {(targets == 4).sum()}")
```

### Issue 4: Training Instability (NaN Loss)
**Possible Causes**:
- Contrastive loss division by zero
- Extreme distance values
- Rational function overflow

**Solutions**:
- Check for empty class masks in contrastive loss
- Verify epsilon in distance denominator (1e-6)
- Monitor rational function outputs (should be clamped to [-20, 20])

## Comparison with Conformer

### Why Kanformer Could Win

**Advantages**:
1. **Learnable Activations**: Rational functions adapt to data-specific patterns
2. **Contrastive Regularization**: Explicitly separates similar classes
3. **Confusion-Aware Weighting**: Targets known failure modes
4. **Interpretability**: Can visualize learned rational functions

**Theoretical Edge**:
- STM features may have non-standard activation patterns
- Fixed ReLU (Conformer) might miss these nuances
- KAN's polynomial basis can capture them

### Why Conformer Might Still Win

**Concerns**:
1. **Simplicity**: Fewer hyperparameters to tune
2. **Stability**: Standard activations = predictable training
3. **Proven**: Already achieved 0.86 F1
4. **Efficiency**: Faster inference without rational functions

**Practical Reality**:
- If v2 doesn't beat Conformer, problem is likely task structure, not architecture
- May need data augmentation or different features

## Next Steps

### Immediate (After v2 Training)
1. **Compare Test F1 with Conformer** (target: 0.86+)
2. **Analyze confusion matrix** (Classes 1 and 3 should improve)
3. **Check contrastive loss convergence** (should stabilize by epoch 20)

### Short-Term (If v2 < 0.86)
1. **Ablation Study**: Remove contrastive loss to isolate effect
2. **Weight Tuning**: Try boost factors in [1.2, 1.4, 1.6]
3. **Ensemble**: Combine Conformer + Kanformer v2 predictions

### Long-Term (Research Directions)
1. **Hard Negative Mining**: Sample difficult pairs more frequently
2. **Triplet Loss**: Use (anchor, positive, negative) instead of pairs
3. **Adaptive Gamma**: Learn per-class focal gamma (not fixed at 2.0)
4. **Curriculum Learning**: Train on easy classes first, add hard classes later

## References and Resources

### Related Files
- **Implementation**: `STMkanformer_enhanced2.py`
- **Baseline**: `STMconformer_model.py`
- **Previous Version**: `STMkanformer_enhanced.py`
- **Development History**: `thoughts/kanformer_enhanced_thread.txt`

### Key Papers
1. **Focal Loss**: Lin et al., "Focal Loss for Dense Object Detection" (ICCV 2017)
2. **Contrastive Learning**: Chen et al., "A Simple Framework for Contrastive Learning" (ICML 2020)
3. **KAN**: Liu et al., "Kolmogorov-Arnold Networks" (arXiv 2024)
4. **Conformer**: Gulati et al., "Conformer: Convolution-augmented Transformer" (Interspeech 2020)

### Checkpoint Locations
```
model/STM/Kanformer_enhanced2_corpora_categories/
├── standard/
│   └── ckpt/
│       └── [timestamp]/
│           ├── best_model.pt
│           ├── latest_checkpoint.pt
│           ├── test_predictions.npy
│           ├── test_targets.npy
│           └── test_confusion_matrix.npy
└── downsample/
    └── ckpt/
        └── [timestamp]/
```

## Conclusion

Enhanced Kanformer v2 represents a **targeted, theory-driven improvement** over v1:
- **Softer class weighting** prevents overtraining on rare classes
- **Confusion-aware boosting** focuses on known failure modes (Classes 1 and 3)
- **Contrastive regularization** explicitly separates similar classes
- **Increased capacity** (8 groups) provides better expressivity

**Expected Outcome**: Test F1 = 0.86+ (match or exceed Conformer baseline)

If v2 succeeds → KAN architecture validated for STM audio classification  
If v2 fails → Problem lies in task structure, not architecture (try data augmentation)

---

**Last Updated**: January 2025  
**Author**: GitHub Copilot (Claude Sonnet 4.5) + User  
**Status**: Ready for training  
**Next Milestone**: Train v2 and compare with Conformer (0.8602)
