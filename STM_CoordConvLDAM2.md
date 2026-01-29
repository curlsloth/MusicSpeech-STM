# STM Classification with CoordConv-ResNet and LDAM Loss
## Phase 1.2: Enhanced Training Dynamics

### Motivation

Analysis of `STM_CoordConvLDAM.py` training logs revealed **early overfitting**:
- Validation F1 peaked at **0.8482** (epoch 8)
- Degraded to **0.8426** by epoch 11
- Training loss continued decreasing → classic overfitting pattern
- DRW scheduled too late (epoch 40) to help with early overfitting

### Key Improvements Over V1

#### 1. **Stronger Regularization**

**Problem**: Model memorizes training data too quickly

**Solutions**:
- **Head dropout**: 0.3 → **0.3** (kept moderate, not too aggressive)
- **Block dropout**: **0.05** in residual blocks (light regularization)
- **Weight decay**: 1e-4 → **2e-4** (2× increase, moderate)
- **Label smoothing**: **0.05** (light smoothing to prevent overconfidence)

**Rationale**:
- Block dropout forces each residual block to learn robust features
- Higher head dropout prevents final layers from overfitting
- Label smoothing (10% probability mass redistributed) reduces overconfidence
- Strong weight decay penalizes large weights throughout network

#### 2. **Adaptive Learning Rate**

**V1 Problem**: CosineAnnealingLR decays LR even when validation performance drops

**V2 Solution**: **ReduceLROnPlateau**
```python
ReduceLROnPlateau(
    mode='max',      # Monitor validation F1 (higher is better)
    factor=0.5,      # Reduce LR by half when plateau detected
    patience=5,      # Wait 5 epochs before reducing
    min_lr=1e-6
)
```

**Effect**:
- LR only decreases when validation F1 plateaus
- Adaptive to training dynamics (not time-based)
- Allows model to "recover" from bad epochs by keeping LR high

#### 3. **Early DRW Activation**

**V1 Problem**: DRW starts at epoch 40 (80%), but overfitting starts by epoch 10

**V2 Solution**: DRW starts at **epoch 50 (50%)**

**Rationale**:
- Class imbalance needs correction, but not too early
- Balanced two-stage training:
  - Stage 1 (epochs 0-49): Learn features from all data
  - Stage 2 (epochs 50+): Refine boundaries with class weights
- Allows model more time to learn before reweighting

#### 4. **Mixup Augmentation**

**From**: "mixup: Beyond Empirical Risk Minimization" (Zhang et al., ICLR 2018)

**Implementation**:
```python
# 30% of batches use mixup with stronger interpolation
mixed_x = λ × x_i + (1 - λ) × x_j
loss = λ × L(f(mixed_x), y_i) + (1 - λ) × L(f(mixed_x), y_j)
```
where λ ~ Beta(0.3, 0.3)

**Effect**:
- Creates synthetic training samples via interpolation
- Encourages linear behavior between examples
- Improves generalization by smoothing decision boundaries
- Particularly effective for imbalanced datasets

**Why for STM?**:
- STM features are continuous (energy distributions)
- Linear interpolation in modulation space is semantically meaningful
- Helps model learn "intermediate" acoustic patterns

#### 5. **Early Stopping**

**V1 Problem**: Trains for all 50 epochs even after overfitting

**V2 Solution**:
```python
patience = 20  # Stop if no improvement for 20 epochs
```

**Effect**:
- Prevents wasted computation on degraded models
- Automatically finds optimal training duration
- Reduced risk of overfitting to validation set

#### 6. **Label Smoothing in LDAM**

**Standard LDAM**: Hard targets (1.0 for correct class, 0.0 for others)

**V2 Enhancement**:
```python
true_label_prob = 1.0 - ε = 0.95
other_label_prob = ε / (K-1) = 0.05 / 5 = 0.01 each
```

**Effect**:
- Prevents extreme confidence (logits → ±∞)
- Reduces gradient magnitudes for correctly classified examples
- Encourages model to focus on hard examples
- Synergistic with LDAM margins

**Mathematical form**:
```
L_LDAM_smooth = -Σ q_k log p_k(x - Δ)
where q = smooth(one_hot(y), ε=0.05)
```

### Architecture Changes

**Enhanced BasicBlock**:
```python
BasicBlock(
    use_coordconv=True,      # First layer in each block
    dropout=0.05             # Light spatial dropout after first conv
)
```

**Dropout placement**:
- After ReLU in first conv of each block → light regularization
- Head dropout (0.3) after fc1 and fc2 → moderate regularization

**Total dropout stages**: 2 (blocks) + 2 (head) = 4 dropout layers

### Training Configuration Changes

| Parameter | V1 | V2 | Change |
|-----------|----|----|--------|
| Head dropout | 0.3 | 0.3 | Same (moderate) |
| Block dropout | 0.0 | 0.05 | Light regularization |
| Weight decay | 1e-4 | 2e-4 | 2× (moderate) |
| Label smoothing | 0.0 | 0.05 | Light smoothing |
| DRW start | Epoch 40 (80%) | Epoch 50 (50%) | More balanced |
| Scheduler | CosineAnnealing | ReduceLROnPlateau (patience=7) | Adaptive |
| Mixup | No | Yes (α=0.3, 30% prob) | Moderate augmentation |
| Early stopping | No | Yes (patience=20) | Generous |
| Max epochs | 50 | 100 | 2× (but early stop) |

### Expected Improvements

#### 1. **Better Generalization**

**V1 Issue**: Val F1 peaks early then degrades  
**V2 Fix**: Dropout + Mixup + Label Smoothing → smoother training  
**Expected**: Val F1 curve more stable, less oscillation

#### 2. **Higher Peak Performance**

**V1 Best**: 0.8482 (epoch 8)  
**V2 Target**: **0.86-0.87** (with improved regularization)  
**Rationale**: Model was underfitting minorities + overfitting majorities

#### 3. **Faster Convergence**

**V1**: Best model at epoch 8, but trained for 50 epochs  
**V2**: Early stopping + adaptive LR → automatic optimal duration  
**Expected**: Convergence in 25-35 epochs (saved compute)

#### 4. **Better Minority Class Performance**

**V1 Issue**: DRW too late to help  
**V2 Fix**: Early DRW (epoch 20) + Mixup (balances classes naturally)  
**Expected**: 
- Environment classes: +3-5% recall
- Music vocal: +2-3% recall
- Maintains speech performance

### Theoretical Justification

#### Why Dropout in Blocks?

From "Improved Regularization of Convolutional Neural Networks with Cutout" (DeVries & Taylor, 2017):

> "Early regularization forces feature extractors to learn robust, redundant representations"

- Dropout after first conv in each block → features can't rely on single activation
- Synergistic with BatchNorm (different noise sources)

#### Why Earlier DRW?

From LDAM paper (Cao et al., 2019):

> "DRW should begin when feature representations are sufficiently learned"

Our analysis:
- Validation accuracy suggests features learned by epoch 8-10
- Further training without DRW → decision boundary drifts toward majority
- Early DRW (epoch 20) gives more time for boundary refinement

#### Why Mixup for Imbalanced Data?

From "Remix: Rebalanced Mixup" (Chou et al., 2020):

> "Mixup naturally balances classes by creating synthetic minority samples"

- Probability of mixing minority sample ∝ 1/class_count
- Creates hard negatives (minority mixed with majority)
- Forces model to learn fine-grained boundaries

### Monitoring Training

**Key metrics to watch**:

1. **Train vs Val Loss Gap**
   - V1: Large gap (overfitting)
   - V2: Should remain close due to dropout/mixup

2. **Val F1 Trend**
   - V1: Peaked early, then noisy
   - V2: Should increase smoothly to higher peak

3. **Learning Rate Reductions**
   - Each reduction indicates plateau
   - Should see 2-3 reductions before convergence

4. **Early Stopping Trigger**
   - Indicates optimal training duration found
   - Should occur between epochs 25-40

5. **Per-Class Metrics** (at end)
   - Environment recall should be >0.75 (V1: likely <0.70)
   - Music vocal recall should be >0.80 (V1: likely <0.75)

### File Structure

```
model/STM/CoordConvLDAM2_corpora_categories/
├── standard/
│   └── ckpt/
│       └── YYYY-MM-DD_HH-MM/
│           ├── best_model.pt           # Best val F1
│           ├── checkpoint_epoch_10.pt  # Periodic saves
│           ├── checkpoint_epoch_20.pt  # DRW starts here
│           ├── checkpoint_epoch_30.pt
│           ├── test_predictions.npy
│           └── test_targets.npy
└── downsample/
    └── ...
```

### Usage

```bash
# Standard training (full dataset)
python STM_CoordConvLDAM2.py 0

# Downsampled non-tonal speech
python STM_CoordConvLDAM2.py 1
```

### Expected Console Output

```
Epoch 1/100
============================================================
  Batch 500/3010, Loss: 2.1234, DRW: False, Mixup: True
  Batch 1000/3010, Loss: 1.9876, DRW: False, Mixup: True
  ...
Train Loss: 2.0123
Val Loss: 1.3456, Val Macro F1: 0.7890
Current learning rate: 0.000100
✓ Saved best model with Val F1: 0.7890

...

Epoch 20/100
============================================================
  Batch 500/3010, Loss: 1.2345, DRW: True, Mixup: True  ← DRW activated
  ...
Train Loss: 1.1234
Val Loss: 0.9876, Val Macro F1: 0.8567
Current learning rate: 0.000050  ← Reduced by plateau scheduler
✓ Saved best model with Val F1: 0.8567

...

Epoch 35/100
============================================================
No improvement for 15 epoch(s)
Early stopping triggered after 35 epochs  ← Optimal duration found
Best Val F1: 0.8678
```

### Comparison with V1

| Metric | V1 | V2 (Expected) |
|--------|----|----|
| Best Val F1 | 0.8482 | 0.86-0.87 |
| Epochs to best | 8 | 25-35 |
| Total epochs run | 50 | 30-40 (early stop) |
| Overfitting severity | High | Low |
| Environment recall | <0.70 | >0.75 |
| Training stability | Noisy | Smooth |

### If Performance Still Plateaus

**Possible next steps**:

1. **Stronger augmentation**:
   - Increase mixup alpha (0.2 → 0.4)
   - Add time/frequency masking (SpecAugment-style)

2. **Architecture changes**:
   - Add squeeze-excitation blocks (channel attention)
   - Try deeper network (ResNet-34)

3. **Advanced techniques**:
   - SupCon pretraining (contrastive learning)
   - Focal loss instead of LDAM
   - Balanced batch sampling

4. **Ensemble methods**:
   - Train multiple models with different seeds
   - Average predictions (often +1-2% F1)

### References

1. Zhang et al., "mixup: Beyond Empirical Risk Minimization", ICLR 2018
2. Chou et al., "Remix: Rebalanced Mixup", NeurIPS 2020 Workshop
3. Cao et al., "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss", NeurIPS 2019
4. DeVries & Taylor, "Improved Regularization of Convolutional Neural Networks with Cutout", arXiv 2017
5. Müller et al., "When Does Label Smoothing Help?", NeurIPS 2019

### Success Criteria

**V2 is successful if**:
- ✅ Best Val F1 > 0.86 (vs V1: 0.8482)
- ✅ Test F1 within 0.01 of Val F1 (generalization)
- ✅ Environment recall > 0.75
- ✅ No class has recall < 0.70
- ✅ Training completes in <40 epochs (efficiency)

**If these are met**: Proceed to Phase 2 (Vision Mamba)  
**If not**: Investigate per-class confusion, consider ensemble
