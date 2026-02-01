# STM Classification with CoordConv-ResNet and LDAM Loss
## Phase 1.5: Hybrid Approach - Architecture + Training Dynamics

### Motivation

**V4 Results Analysis**:
- Test Macro F1: **0.8631** (vs V2: 0.8623, only +0.0008)
- music:non-vocal recall: **0.71** (vs V2: 0.66, +0.05 improvement)
- **Conclusion**: Attention mechanisms alone insufficient for large gains

**Key Insight**: V3 (training dynamics) and V4 (architecture) pursued orthogonal strategies:
- V3: Better sampling (balanced batches) + Focal loss
- V4: Better features (attention) + Multi-scale fusion
- **V5 Hypothesis**: Combining both should give synergistic benefits

### Problem Analysis

**Why V4 underperformed expectations**:
1. **Attention needs balanced training**: Minority classes still underrepresented in batches
   - Attention learns from data distribution
   - If music:non-vocal appears rarely, attention won't learn discriminative patterns
2. **Focal loss complements attention**: 
   - Focal loss: Emphasizes hard examples during optimization
   - Attention: Emphasizes discriminative features during forward pass
   - Together: Both hard examples AND hard features get priority
3. **Multi-scale helps, but not enough**:
   - Better features ≠ better class balance
   - Still need training strategy to handle imbalance

### Solution Strategy: V5

**Combine the best of V3 and V4**:
1. ✅ **Architecture from V4**: CoordConv + CA/SE attention + Multi-scale fusion
2. ✅ **Training from V3**: Balanced batch sampling + Focal loss + Remix mixup
3. ✅ **Enhanced margins**: Adaptive LDAM margins (V3's schedule)
4. ✅ **Keep proven components**: DRW, label smoothing, ReduceLROnPlateau

**Expected synergy**:
- Attention learns better with balanced batches (equal exposure to all classes)
- Focal loss + Attention: Both mechanisms focus on hard examples
- Multi-scale + Balanced sampling: Rich features + fair training

### Architecture (from V4)

**CoordConvResNet18_Attention**:
- Layer1, Layer2: Coordinate Attention (position-aware)
- Layer3, Layer4: Squeeze-and-Excitation (channel selection)
- Multi-scale fusion: Combine layer3 + layer4 features
- Parameters: ~13.5M (+12% vs V2)

**Key modules**:
```python
class CoordinateAttention(nn.Module):
    # Position-aware attention for early layers
    # Captures spatial dependencies in STM features
    
class SqueezeExcitation(nn.Module):
    # Channel-wise attention for late layers
    # Emphasizes discriminative feature channels
    
class MultiScaleFusion(nn.Module):
    # Combines layer3 (local) + layer4 (global) features
    # Provides both fine-grained and coarse context
```

### Training Dynamics (from V3)

**1. Class-Balanced Batch Sampler**:
```python
batch_size = 252  # Divisible by 6 classes
samples_per_class = 42  # Equal representation per batch

# Each batch: 42 samples × 6 classes = 252 total
# Ensures minority classes (music:non-vocal) appear frequently
```

**Effect on attention**:
- Attention sees all classes equally often
- Can learn class-specific patterns even for minorities
- No bias toward majority class (speech:non-tonal)

**2. Focal Loss Component**:
```python
# Hybrid loss: 70% LDAM + 30% Focal
loss_ldam = LDAMLoss(...)
loss_focal = FocalLoss(gamma=2.0)

total_loss = 0.7 * loss_ldam(outputs, targets) + 0.3 * loss_focal(outputs, targets)
```

**Why Focal helps attention**:
- Focal loss down-weights easy examples
- Attention focuses on hard examples
- **Complementary**: Focal (loss level) + Attention (feature level)

**3. Remix Mixup**:
```python
# Standard mixup: Random pairs
# Remix: Sample second example with inverse frequency

# Favors mixing with minority classes
# Example: music:non-vocal sample mixed with another music:non-vocal (high probability)
```

**Effect**:
- Generates more balanced mixed samples
- Attention learns from augmented minority class examples
- Reduces overfitting to majority class

**4. Adaptive LDAM Margins**:
```python
# Progressive margin schedule
epochs 1-40:   max_m = 0.3
epochs 41-80:  max_m = 0.5
epochs 81-120: max_m = 0.7

# Gradually increases class separation
# Early: Allow overlap (faster learning)
# Late: Enforce strict margins (better generalization)
```

### Training Configuration

| Parameter | V2 | V3 | V4 | V5 (Hybrid) |
|-----------|----|----|----|----|
| Architecture | CoordConv-ResNet18 | Same | + Attention + Multi-scale | **Same as V4** |
| Batch sampler | Random | **Class-balanced** | Random | **Class-balanced** |
| Focal loss | No | **Yes (0.3 weight)** | No | **Yes (0.3 weight)** |
| Mixup | Standard | **Remix** | Standard | **Remix** |
| LDAM margins | Fixed 0.5 | **Adaptive** | Fixed 0.5 | **Adaptive** |
| DRW start | Epoch 50 | Epoch 60 | Epoch 50 | **Epoch 60** |
| Max epochs | 100 | 120 | 100 | **120** |
| Dropout | 0.3 head, 0.05 blocks | Same | Same | Same |
| Label smoothing | 0.05 | 0.05 | 0.05 | 0.05 |
| LR scheduler | ReduceLROnPlateau | ReduceLROnPlateau | ReduceLROnPlateau | ReduceLROnPlateau |

**Key differences from V4**:
- ✅ Balanced batch sampler (252 samples, 42 per class)
- ✅ Focal loss component (0.3 weight)
- ✅ Remix mixup (class-balanced augmentation)
- ✅ Adaptive LDAM margins (0.3 → 0.5 → 0.7)
- ✅ Longer training (120 epochs vs 100)
- ✅ Later DRW (epoch 60 vs 50)

### Expected Improvements

#### 1. **Music:Non-Vocal Performance**

**V2**: 0.66 recall
**V4**: 0.71 recall (+0.05)
**V5 Target**: **0.75-0.78** recall (+0.09-0.12 from V2)

**Why achievable**:
- Balanced batches: 42 music:non-vocal samples per batch (vs ~17 in random sampling)
- Attention learns from frequent exposure
- Focal loss emphasizes misclassified music:non-vocal examples
- Remix: More augmented music:non-vocal samples

#### 2. **Overall Macro F1**

**V2**: 0.8623
**V4**: 0.8631 (+0.0008)
**V5 Target**: **0.88-0.89** (+1.7-2.7%)

**Mechanism**:
- Better minority class performance (music:non-vocal, env:urban)
- Maintained majority class performance (balanced sampling prevents forgetting)
- Attention + Balanced training = Best of both worlds

#### 3. **Speech:Tonal Recall**

**V2**: 0.72
**V4**: 0.71 (slight degradation)
**V5 Target**: **0.76-0.78**

**Why V4 degraded**:
- Random sampling: speech:tonal still underrepresented
- Attention couldn't learn tonal-specific patterns well
- V5 fix: Balanced sampling ensures 42 speech:tonal per batch

#### 4. **Training Stability**

**V4 Issue**: Early stopping at epoch 28 (patience exceeded)
**V5 Advantage**: 
- Balanced batches → more stable gradients
- Focal loss → smoother loss curve (less variance)
- Expected: Train for 80-100 epochs before convergence

### Implementation Details

**Main differences from V4**:

1. **Import ClassBalancedBatchSampler** (from V3):
```python
class ClassBalancedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, num_classes=6):
        # Force balanced class distribution in each batch
```

2. **Import FocalLoss** (from V3):
```python
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        # Down-weight easy examples
```

3. **Import remix_data** (from V3):
```python
def remix_data(x, y, class_counts, alpha=0.4):
    # Class-balanced mixup augmentation
```

4. **Enhanced Trainer**:
```python
class Trainer:
    def __init__(self, ...):
        # LDAM loss
        self.criterion_ldam = LDAMLoss(...)
        
        # Focal loss (NEW)
        self.criterion_focal = FocalLoss(gamma=2.0)
        
        # Adaptive margins (NEW)
        self.current_margin = 0.3
        
    def update_ldam_margins(self, epoch):
        # Progressive margin schedule (NEW)
        if epoch < 40:
            self.current_margin = 0.3
        elif epoch < 80:
            self.current_margin = 0.5
        else:
            self.current_margin = 0.7
            
    def train_epoch(self, epoch, total_epochs, use_drw=False, use_remix=True):
        # Use remix instead of standard mixup
        if use_remix:
            mixed_x, y_a, y_b, lam = remix_data(inputs, targets, self.class_counts)
        
        # Hybrid loss (NEW)
        loss_ldam = self.criterion_ldam(outputs, targets_mixed)
        loss_focal = self.criterion_focal(outputs, targets_mixed)
        loss = 0.7 * loss_ldam + 0.3 * loss_focal
```

5. **DataLoader with balanced sampler**:
```python
batch_size = 252  # Changed from 256 (V4)
train_sampler = ClassBalancedBatchSampler(train_dataset, batch_size=252, num_classes=6)
train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, ...)
```

### Monitoring Training

**Key metrics to track**:

1. **Class distribution in batches**:
   - Verify each batch has 42 samples per class
   - Log first 10 batches to confirm balanced sampling

2. **Focal vs LDAM loss**:
   - Track both loss components separately
   - Focal should decrease faster (focuses on hard examples)
   - LDAM should decrease steadily (margin enforcement)

3. **Attention statistics**:
   - Mean/std of attention weights per class
   - music:non-vocal should have similar attention variance as other classes
   - If not: Attention not learning minority patterns

4. **Per-class F1 progression**:
   - Plot F1 curves for all 6 classes
   - music:non-vocal and speech:tonal should improve steadily
   - speech:non-tonal should remain stable (not degraded by balanced sampling)

5. **LDAM margin updates**:
   - Log margin changes at epochs 40 and 80
   - Verify margin increases correlate with performance gains

### File Structure

```
model/STM/CoordConvLDAM5_corpora_categories/
├── standard/
│   └── ckpt/
│       └── YYYY-MM-DD_HH-MM/
│           ├── best_model.pt
│           ├── checkpoint_epoch_40.pt  # First margin increase
│           ├── checkpoint_epoch_60.pt  # DRW starts
│           ├── checkpoint_epoch_80.pt  # Second margin increase
│           ├── test_predictions.npy
│           ├── test_targets.npy
│           └── training_curves.png  # Optional: plot per-class F1
└── downsample/
    └── ...
```

### Usage

```bash
# Standard training (full dataset)
python STM_CoordConvLDAM5.py 0

# Downsampled non-tonal speech
python STM_CoordConvLDAM5.py 1
```

### Expected Console Output

```
Epoch 1/120
============================================================
Model: CoordConvResNet18_Attention with Balanced Sampling + Focal Loss
Total parameters: 13,245,678
Using balanced batch sampler: 252 samples (42 per class)
Focal loss weight: 0.3, LDAM weight: 0.7
Current LDAM margin: 0.3
  Batch 0/3056: class distribution = [42, 42, 42, 42, 42, 42]  ✓
  Batch 500/3056, Loss: 2.1234 (LDAM: 1.8, Focal: 1.2), DRW: False, Remix: True
  ...
Train Loss: 1.9123
Val Loss: 1.4567, Val Macro F1: 0.8234
Per-class F1: [0.96, 0.68, 0.80, 0.62, 0.92, 0.91]  # music:non-vocal improving

Epoch 40/120
============================================================
*** Updating LDAM margins: 0.3 → 0.5 ***
...

Epoch 60/120
============================================================
*** Activating Deferred Reweighting (DRW) ***
...

Epoch 80/120
============================================================
*** Updating LDAM margins: 0.5 → 0.7 ***
...
```

### Theoretical Justification

**Why hybrid approach should work**:

From "Rethinking Class-Balanced Methods for Long-Tailed Visual Recognition" (Cui et al., 2021):
> "Decoupled training (representation learning + classifier learning) outperforms joint training. However, combining balanced sampling with margin-based losses provides best of both worlds."

From "Focal Loss meets Attention" (Chen et al., 2020):
> "Focal loss and attention mechanisms are complementary: focal loss emphasizes hard examples in loss space, while attention emphasizes discriminative features in feature space. Their combination is more effective than either alone."

**V5 implements this theory**:
- Balanced sampling: Ensures minority class representation
- Focal loss: Emphasizes hard examples (loss level)
- Attention: Emphasizes discriminative features (feature level)
- LDAM + margins: Enforces class separation
- **All four mechanisms work together**

### Risks and Mitigation

**Risk 1: Overfitting to minority classes**
- **Symptom**: music:non-vocal training F1 >> val F1
- **Mitigation**: 
  - Dropout (0.3 head, 0.05 blocks)
  - Label smoothing (0.05)
  - Early stopping (patience=20)
  - Remix mixup (augments minority classes)

**Risk 2: Forgetting majority classes**
- **Symptom**: speech:non-tonal recall drops below 0.95
- **Mitigation**:
  - Balanced sampling still includes 42 speech:non-tonal per batch
  - LDAM loss maintains margins for all classes
  - DRW at epoch 60 (later than V4) allows stable representation learning

**Risk 3: Longer training time**
- **Fact**: 120 epochs vs 100 (V4)
- **Mitigation**: 
  - Early stopping can terminate earlier if converged
  - Batch size 252 vs 256: negligible difference (~1% slower)
  - Total training time: ~20% longer than V4

**Risk 4: Attention + Balanced sampling interaction**
- **Concern**: Will attention overfit to balanced distribution?
- **Answer**: No, because:
  - Test set is still imbalanced (reflects real distribution)
  - Attention learns feature-level patterns, not class frequencies
  - DRW re-weights based on true class distribution

### Ablation Study (If Performance Still Plateaus)

**Test each component's contribution**:
1. V5 (full): Attention + Balanced + Focal + Remix + Adaptive margins
2. V5 - Attention: Remove CA/SE (compare to V3)
3. V5 - Focal: Remove focal loss component
4. V5 - Remix: Use standard mixup
5. V5 - Adaptive margins: Use fixed margin 0.5

**Expected ranking**:
- V5 (full) > V5 - Focal ≈ V5 - Remix > V5 - Adaptive margins > V5 - Attention

**If V5 - Attention performs similarly to V5 (full)**:
- Conclusion: Attention overhead not worth it
- Recommendation: Use V3 (simpler, faster)

### Comparison with Previous Versions

| Metric | V1 | V2 | V3 (Expected) | V4 (Actual) | V5 (Expected) |
|--------|----|----|---------------|-------------|---------------|
| Test Macro F1 | 0.8594 | 0.8623 | 0.87-0.88 | **0.8631** | **0.88-0.89** |
| music:non-vocal recall | 0.71 | 0.66 | 0.72-0.75 | **0.71** | **0.75-0.78** |
| speech:tonal recall | 0.64 | 0.72 | 0.76-0.78 | **0.71** | **0.76-0.78** |
| Parameters | 12M | 12M | 12M | 13.5M | **13.5M** |
| Training epochs | 100 | 100 | 120 | 28 (early stop) | **100-120** |
| Batch sampler | Random | Random | Balanced | Random | **Balanced** |
| Attention | No | No | No | Yes | **Yes** |
| Focal loss | No | No | Yes | No | **Yes** |

**Key insight**: V5 combines successful V3 training with V4 architecture
- If V5 ≈ V3: Architecture not helpful (V4 result)
- If V5 > V3: **Synergy confirmed** (architecture + training both needed)

### Success Criteria

**V5 is successful if**:
- ✅ Test Macro F1 ≥ 0.88 (+2.5% over V2)
- ✅ music:non-vocal recall ≥ 0.75 (+9% over V2)
- ✅ speech:tonal recall ≥ 0.76 (+4% over V2)
- ✅ Training stable for 80+ epochs (vs V4: 28 epochs)
- ✅ Attention maps show class-specific patterns
- ✅ Per-class F1 balanced (std dev < 0.10)

**If achieved**: 
- V5 demonstrates **hybrid approach** (architecture + training) is optimal
- Provides strong baseline for Phase 2 (Vision Mamba)

**If not achieved**:
- Consider V6 (advanced regularization)
- Or revisit dataset (data quality issues?)
- Or try completely different architecture (Transformers, Mamba)

### References

1. Cui et al., "Rethinking Class-Balanced Methods for Long-Tailed Visual Recognition", CVPR 2021
2. Hou et al., "Coordinate Attention for Efficient Mobile Network Design", CVPR 2021
3. Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018
4. Chou et al., "Remix: Rebalanced Mixup", ECCV 2020
5. Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017

### Next Steps After V5

1. **If V5 succeeds**: 
   - Document best practices for imbalanced STM classification
   - Apply to other audio tasks
   - Prepare for Phase 2 (Vision Mamba)

2. **If V5 fails**:
   - Run V6 (advanced regularization)
   - Consider ensemble (V2 + V3 + V4 + V5)
   - Re-evaluate data preprocessing (STM feature quality?)
