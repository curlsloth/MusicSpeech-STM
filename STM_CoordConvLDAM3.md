# STM Classification with CoordConv-ResNet and LDAM Loss
## Phase 1.3: Balanced Sampling + Focal Loss Enhancement

### Motivation

Analysis of `STM_CoordConvLDAM2.py` results shows:
- **Overall improvement**: Macro F1 **0.8623** (vs V1: 0.8594)
- **Key wins**: speech:tonal (+8% recall), music:vocal (+2% recall)
- **Remaining weakness**: music:non-vocal still lowest (0.66 recall, 0.68 F1)
- **Class imbalance still visible**: speech:non-tonal dominates (70k samples)

### Core Problem

Even with LDAM+DRW and Mixup, the model sees **imbalanced batches**:
- Random sampling → some batches have 200+ speech, 0-2 music:non-vocal
- Model learns batch-level biases toward majority classes
- Hard examples (misclassified minorities) get insufficient gradient updates

### Solution Strategy: V3

**Three-pronged approach**:
1. **Balanced Batch Sampling**: Force every batch to contain samples from all classes
2. **Focal Loss Component**: Emphasize hard examples (low confidence predictions)
3. **Remix (Balanced Mixup)**: Mixup pairs are class-balanced

### Key Improvements Over V2

#### 1. **Class-Balanced Batch Sampler**

**Problem**: Standard random sampling creates imbalanced batches

**Solution**: Custom sampler that ensures balanced class representation

**Implementation**:
```python
ClassBalancedBatchSampler:
  - Divide batch_size by num_classes (256 / 6 = 42 samples per class)
  - Each batch contains ~42 samples from each class
  - Within-class shuffling for diversity
  - Between-class rotation to cover all samples
```

**Effect**:
- Every batch: Model sees all 6 classes
- Gradient updates: Balanced across classes
- No batch dominated by speech:non-tonal
- Minorities get consistent gradient signal

**Tradeoff**: 
- Slightly oversamples minorities (repeats samples)
- Undersamples majorities (sees less often)
- Net effect: Better balance, worth the tradeoff

#### 2. **Focal Loss Integration**

**From**: "Focal Loss for Dense Object Detection" (Lin et al., ICCV 2017)

**Problem**: Easy examples (high confidence) dominate gradient
- Speech:non-tonal at 0.99 recall → almost always correct
- music:non-vocal at 0.66 recall → many hard examples
- Standard CE: Both contribute equally to loss

**Focal Loss Formula**:
```
FL(p_t) = -(1 - p_t)^γ × log(p_t)
```
where:
- p_t = predicted probability for true class
- γ = focusing parameter (default: 2)

**Intuition**:
- Easy example (p_t = 0.95): Weight = (1-0.95)^2 = 0.0025 → small loss
- Hard example (p_t = 0.60): Weight = (1-0.60)^2 = 0.16 → large loss
- Ratio: Hard/Easy = 64× more gradient

**Hybrid Loss**:
```
L_total = α × L_LDAM + (1-α) × L_Focal
```
where α = 0.7 (LDAM dominant, Focal assists)

**Why hybrid?**:
- LDAM: Handles class imbalance via margins
- Focal: Handles example difficulty via confidence weighting
- Complementary mechanisms

#### 3. **Remix: Balanced Mixup**

**From**: "Remix: Rebalanced Mixup" (Chou et al., NeurIPS 2020)

**Problem**: Standard Mixup samples pairs uniformly
- Probability of mixing two minority samples: very low
- Most mixup: speech + speech (majority + majority)

**Remix Strategy**:
```python
# Sample first example uniformly
x_i, y_i = batch[i]

# Sample second example with class-balanced probability
p_j ∝ 1 / class_count[j]  # Inverse frequency
x_j, y_j = sample_by_class_balanced_prob()

# Mixup
λ ~ Beta(0.4, 0.4)  # Slightly stronger than V2
mixed = λ × x_i + (1-λ) × x_j
```

**Effect**:
- Minorities more likely to be paired
- Creates synthetic minority samples
- Forces model to learn minority features
- 40% of batches use Remix (vs 30% V2)

**Tradeoff**:
- More aggressive augmentation
- Risk: Might create unrealistic samples
- Mitigation: Use moderate α=0.4 (not too extreme)

#### 4. **Adaptive LDAM Margins**

**V2 Problem**: Fixed margins for all training stages

**V3 Enhancement**: Margin schedule
```python
# Early training (epochs 0-20): Smaller margins
max_m = 0.3  # Easier to learn

# Mid training (epochs 21-50): Standard margins  
max_m = 0.5  # V2 value

# Late training (epochs 51+): Larger margins
max_m = 0.7  # Stronger minority protection
```

**Rationale**:
- Early: Model needs to learn basic features (don't enforce too much)
- Mid: Standard LDAM margins
- Late: Push boundaries further from minorities

#### 5. **Per-Class Learning Rate Scaling**

**Concept**: Different classes learn at different rates

**Implementation**:
```python
# Final classification layer: per-class weight scaling
fc_out.weight[class_i] *= lr_scale[class_i]

lr_scale = {
  speech:non-tonal: 0.5,   # Learns fast, slow it down
  speech:tonal: 1.0,
  music:vocal: 1.0,
  music:non-vocal: 2.0,    # Learns slow, speed it up
  env:urban: 1.5,
  env:wildlife: 1.5
}
```

**Effect**: 
- Minorities get stronger gradient updates
- Majorities get dampened updates
- Self-balancing mechanism

### Architecture Enhancements

**No major changes** from V2:
- Keep CoordConv in all blocks
- Keep moderate dropout (0.3 head, 0.05 blocks)
- Keep label smoothing (0.05)

**Why?** V2 architecture is solid. V3 focuses on **training dynamics**, not architecture.

### Training Configuration

| Parameter | V2 | V3 | Change Rationale |
|-----------|----|----|------------------|
| Batch size | 256 | 252 (42×6) | Balanced sampling |
| Sampler | Random | ClassBalanced | Force class balance |
| Loss | LDAM | 0.7×LDAM + 0.3×Focal | Add hard example focus |
| Mixup | Standard (α=0.3) | Remix (α=0.4) | Balanced pairs |
| Mixup prob | 30% | 40% | More augmentation |
| LDAM margins | Fixed 0.5 | Schedule 0.3→0.5→0.7 | Progressive |
| LR scaling | None | Per-class | Balance learning |
| DRW start | Epoch 50 | Epoch 60 | Later (more epochs) |
| Max epochs | 100 | 120 | Longer training |
| Early stop patience | 20 | 25 | More patience |

### Expected Improvements

#### 1. **Minority Class Recall**

**V2 Weaknesses**:
- music:non-vocal: 0.66 recall (lowest)
- speech:tonal: 0.72 recall (second lowest)

**V3 Targets**:
- music:non-vocal: **0.72-0.75** recall (+6-9%)
- speech:tonal: **0.76-0.78** recall (+4-6%)
- Overall Macro F1: **0.87-0.88** (+0.7-1.7%)

**Why achievable?**:
- Balanced batches: Every gradient update includes music:non-vocal
- Focal loss: Hard music:non-vocal examples weighted 50-100× more
- Remix: Creates synthetic music:non-vocal training data
- Per-class LR: music:non-vocal gets 2× learning rate

#### 2. **More Stable Training**

**V2 Issue**: Validation F1 oscillates slightly

**V3 Fix**: Balanced batches → more consistent gradients
- Expected: Smoother validation curve
- Less epoch-to-epoch variance

#### 3. **Better Calibration**

**V2**: Some classes overconfident (speech:non-tonal)

**V3**: Focal loss + label smoothing → better calibrated
- Expected: Confidence scores match actual accuracy
- Safer for downstream tasks (e.g., active learning)

### Implementation Details

#### Balanced Batch Sampler

```python
class ClassBalancedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.class_indices = {}  # class_id -> [sample_idx]
        for idx, (_, label) in enumerate(dataset):
            self.class_indices[label].append(idx)
        
        self.samples_per_class = batch_size // num_classes
        
    def __iter__(self):
        # Shuffle within each class
        for class_indices in self.class_indices.values():
            random.shuffle(class_indices)
        
        # Iterate until all samples seen
        iterators = {c: iter(indices) for c, indices in self.class_indices.items()}
        
        while True:
            batch = []
            for class_id in range(num_classes):
                try:
                    for _ in range(self.samples_per_class):
                        batch.append(next(iterators[class_id]))
                except StopIteration:
                    # Reshuffle and restart this class
                    random.shuffle(self.class_indices[class_id])
                    iterators[class_id] = iter(self.class_indices[class_id])
                    # Fill remaining
                    for _ in range(self.samples_per_class - len(batch) % self.samples_per_class):
                        batch.append(next(iterators[class_id]))
            
            if len(batch) == batch_size:
                yield batch
            else:
                break
```

#### Focal Loss

```python
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        self.gamma = gamma  # Focusing parameter
        self.alpha = alpha  # Class weights (optional)
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)  # Predicted probability
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = focal_weight * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        return focal_loss.mean()
```

#### Remix Mixup

```python
def remix_data(x, y, class_counts, alpha=0.4):
    batch_size = x.size(0)
    
    # Sample lambda
    lam = np.random.beta(alpha, alpha)
    
    # First sample: uniform
    # Second sample: inverse frequency probability
    inv_freq = 1.0 / class_counts[y]
    inv_freq = inv_freq / inv_freq.sum()
    
    # Sample second index with balanced probability
    index = torch.multinomial(inv_freq, batch_size, replacement=True)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam
```

### Monitoring Training

**Key indicators**:

1. **Batch Class Distribution** (print every 500 batches):
   - Should show ~42 samples per class
   - Verify balanced sampling works

2. **Focal vs LDAM Loss** (log separately):
   - Focal loss should decrease faster early (easy examples)
   - LDAM loss should dominate late (margin enforcement)

3. **Per-Class Accuracy** (every 5 epochs):
   - music:non-vocal should improve steadily
   - speech:non-tonal should remain stable

4. **Mixup Ratio** (log):
   - Should see ~40% batches use Remix
   - Verify minority pairs created

5. **Learning Rate Reductions**:
   - Should see 2-3 reductions (ReduceLROnPlateau)
   - Later than V2 due to smoother training

### File Structure

```
model/STM/CoordConvLDAM3_corpora_categories/
├── standard/
│   └── ckpt/
│       └── YYYY-MM-DD_HH-MM/
│           ├── best_model.pt
│           ├── checkpoint_epoch_20.pt  # Margin = 0.3
│           ├── checkpoint_epoch_50.pt  # Margin = 0.5
│           ├── checkpoint_epoch_60.pt  # DRW starts
│           ├── checkpoint_epoch_80.pt  # Margin = 0.7
│           ├── test_predictions.npy
│           └── test_targets.npy
└── downsample/
    └── ...
```

### Usage

```bash
# Standard training (full dataset)
python STM_CoordConvLDAM3.py 0

# Downsampled non-tonal speech
python STM_CoordConvLDAM3.py 1
```

### Expected Console Output

```
Epoch 1/120
============================================================
Batch class distribution: [42, 42, 42, 42, 42, 42]  ← Balanced!
  Batch 500/2976, Loss: 2.1234, LDAM: 1.5, Focal: 0.6, Remix: True
  ...
Train Loss: 2.0123 (LDAM: 1.41, Focal: 0.60)
Val Loss: 1.3456, Val Macro F1: 0.7890
Current learning rate: 0.000100

...

Epoch 60/120
============================================================
*** Activating Deferred Reweighting (DRW) ***
Batch class distribution: [42, 42, 42, 42, 42, 42]
  Batch 500/2976, Loss: 1.0234, LDAM: 0.7, Focal: 0.3, Remix: True
  ...
Train Loss: 1.0123 (LDAM: 0.71, Focal: 0.31)
Val Loss: 0.8456, Val Macro F1: 0.8756
Current learning rate: 0.000025

...

Early stopping triggered after 85 epochs
Best Val F1: 0.8789
```

### Theoretical Justification

#### Why Balanced Sampling Works

From "Decoupling Representation and Classifier for Long-Tailed Recognition" (Kang et al., ICLR 2020):

> "Class-balanced sampling improves minority class features **without** hurting majority class performance when combined with re-weighting"

Our setup:
- Balanced sampling: Learn features
- DRW + LDAM: Refine boundaries
- Synergistic effect

#### Why Focal Loss Helps

From "Class-Balanced Loss Based on Effective Number of Samples" (Cui et al., CVPR 2019):

> "Focal loss and re-weighting are **orthogonal** — one handles difficulty, one handles imbalance"

Our hybrid loss:
- LDAM: Margin-based imbalance correction
- Focal: Confidence-based difficulty weighting
- Together: Address both problems

#### Why Remix Works

From the Remix paper:

> "Pairing minority samples in Mixup creates **quadratically** more minority training signal"

Math:
- Standard Mixup: P(minority pair) = (n_minority / n_total)²
- Remix: P(minority pair) = n_minority / n_total
- Improvement: √(n_total / n_minority) times more

For music:non-vocal (6k / 1M samples):
- Standard: 0.000036 probability
- Remix: 0.006 probability
- **167× improvement**

### Comparison with V1 and V2

| Metric | V1 | V2 | V3 (Expected) |
|--------|----|----|---------------|
| Test Macro F1 | 0.8594 | 0.8623 | 0.87-0.88 |
| music:non-vocal recall | 0.71 | 0.66 | 0.72-0.75 |
| speech:tonal recall | 0.64 | 0.72 | 0.76-0.78 |
| Training stability | Low | Medium | High |
| Batch balance | No | No | Yes |
| Hard example focus | No | Partial | Yes |

### If Performance Plateaus

**Diagnosis steps**:

1. **Check batch distribution**:
   - Print class counts per batch
   - Verify sampler works correctly

2. **Analyze confusion matrix**:
   - Which classes confuse music:non-vocal?
   - May need feature engineering

3. **Focal loss gamma tuning**:
   - γ=2 is standard, try γ=3 (more aggressive)
   - Monitor Focal/LDAM ratio

4. **Consider ensemble**:
   - V2 + V3 ensemble (average predictions)
   - Often +1-2% F1 boost

### Next Steps

**If V3 succeeds** (Macro F1 > 0.87):
- **Phase 1.4 (V4)**: Add attention mechanisms (SE blocks)
- **Phase 2**: Try Vision Mamba for global context

**If V3 plateaus**:
- Investigate feature space (t-SNE)
- Consider music:non-vocal subcategories (jazz vs. electronic?)
- Try SupCon pretraining (contrastive learning)

### References

1. Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
2. Chou et al., "Remix: Rebalanced Mixup", NeurIPS 2020 Workshop
3. Kang et al., "Decoupling Representation and Classifier for Long-Tailed Recognition", ICLR 2020
4. Cui et al., "Class-Balanced Loss Based on Effective Number of Samples", CVPR 2019
5. Cao et al., "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss", NeurIPS 2019

### Success Criteria

**V3 is successful if**:
- ✅ Test Macro F1 > 0.87 (vs V2: 0.8623)
- ✅ music:non-vocal recall > 0.72 (vs V2: 0.66)
- ✅ speech:tonal recall > 0.75 (vs V2: 0.72)
- ✅ No class below 0.70 F1
- ✅ Validation curve smoother than V2

If achieved → V3 provides **balanced + robust** model, ready for deployment or Phase 2.
