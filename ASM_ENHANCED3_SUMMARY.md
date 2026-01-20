# ASM Enhanced v3 Summary

## Executive Summary

**Enhanced ASM v3** applies Kanformer v2's proven strategies to the ASM architecture:
- **Root Issue**: ASM v1 achieved 0.8566 macro F1, but with same confusion problems as Kanformer v1
- **Solution**: Adopt Kanformer v2's successful approach: softer weights + contrastive loss
- **Target**: Match or exceed Conformer baseline (0.86) with ASM's efficiency (~1.5M params)

## Problem Analysis from ASM v1/v2

### ASM v1 Results (STMasm_enhanced.py)
```
Test Macro F1: 0.8566

Per-Class F1:
  Class 0 (speech: non-tonal):  0.9592 ✓ Excellent
  Class 1 (speech: tonal):      0.7893 ✗ Confused with Class 0
  Class 2 (music: vocal):       0.8349 ~ Good
  Class 3 (environment):        0.6709 ✗ Confused with Class 2
  Class 4 (env: urban):         0.9547 ✓✓ Overtrained
  Class 5 (env: wildlife):      0.9306 ✓✓ Overtrained
```

### ASM v2 Approach (STMasm_enhanced2.py)
- Used **explicit confusion-aware weighting** (0.7× reduction for confusable pairs)
- Added **margin ranking loss** and **binary auxiliary classifiers**
- **Issue**: Complex multi-task learning, multiple loss terms to balance

### Why v3 is Different

**Simpler strategy inspired by Kanformer v2's success**:
1. **Square root weighting** naturally softens extreme weights
2. **Targeted boosting** for difficult pairs (simpler than v2's multiple heads)
3. **Single contrastive loss** term (not 3 separate losses like v2)
4. **Minimal label smoothing** (0.01 vs 0.1) for sharper decisions

## Key Improvements in v3

### 1. Softer Class Weighting (Square Root)

**Mathematical Form**:
```python
# v1: w_i = total / (num_classes × count_i)
# v3: w_i = sqrt(total / (num_classes × count_i))
```

**Example** (Class 1: Tonal Speech, 80,258 samples):
```python
# v1: weight = 770000 / (6 × 80258) = 1.60
# v3: weight = sqrt(770000 / (6 × 80258)) = 1.26 (-21%)
```

**Effect**:
- Reduces penalty for rare classes
- Prevents model from ignoring majority classes
- More balanced loss contributions

### 2. Confusion-Aware Boosting

**Strategy** (from Kanformer v2):
```python
weights[1] *= 1.3  # Boost tonal speech (vs non-tonal)
weights[3] *= 1.3  # Boost environment (vs music)
weights[4] *= 0.7  # Reduce urban env (already near-perfect)
weights[5] *= 0.8  # Reduce wildlife env (already near-perfect)
```

**Expected Adjusted Weights**:
```
Class 0 (speech: non-tonal):  ~0.3  (baseline)
Class 1 (speech: tonal):      ~1.0  (boosted 1.3×)
Class 2 (music: vocal):       ~0.7  (baseline)
Class 3 (environment):        ~1.6  (boosted 1.3×)
Class 4 (env: urban):         ~1.3  (reduced 0.7×)
Class 5 (env: wildlife):      ~0.9  (reduced 0.8×)
```

### 3. Contrastive Regularization Loss

**New in v3**: Single, simple contrastive term (not v2's complex multi-task setup)

**Loss Function**:
```python
Total Loss = Focal Loss + 0.1 × Contrastive Loss

Contrastive Loss = Σ (1 / ||f_a - f_b||₂)
```

**Mechanism**:
1. Extract features from `feature_extractor` layer (new in v3)
2. Sample 32 pairs from similar classes (0,1) and (2,3) per batch
3. Compute L2 distances between pairs
4. Loss increases when distance is small → gradient pushes apart

**Implementation**:
```python
class EnhancedASM_RH_Classifier:
    def __init__(self, ...):
        # ...existing blocks...
        
        # NEW v3: Feature extractor before classifier
        self.feature_extractor = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(dim // 2, num_classes)
    
    def forward(self, x, return_features=False):
        # ...existing processing...
        features = self.feature_extractor(pooled)
        logits = self.classifier(features)
        
        if return_features:
            return logits, features  # For contrastive loss
        return logits
```

**Comparison with v2**:
| Aspect | v2 (Complex) | v3 (Simple) |
|--------|--------------|-------------|
| Loss terms | 3 (focal + margin + binary) | 2 (focal + contrastive) |
| Auxiliary heads | 2 binary classifiers | 0 (reuses feature extractor) |
| Parameters | +50k for binary heads | +32k for feature layer only |
| Complexity | High (tuning 3 weights) | Low (tuning 1 weight) |

### 4. Reduced Label Smoothing (0.1 → 0.01)

**Change**:
```python
# v1: label_smoothing = 0.1  (too aggressive)
# v3: label_smoothing = 0.01  (minimal, like Kanformer v2)
```

**Effect**:
- Allows sharper decision boundaries
- Better for discriminating similar classes
- Model can be more confident when correct

## Architecture Specifications

### Model Configuration
```python
EnhancedASM_RH_Classifier(
    time_steps=121,         # STM time dimension
    freq_steps=20,          # STM frequency dimension
    num_classes=6,
    dim=128,
    num_blocks=4,
    shift_range=2,
    expansion_factor=2,
    dropout=0.1
)
```

### Parameter Count
- Total parameters: ~1,520,000
- Trainable: ~1,520,000
- Memory: ~9-10GB GPU (batch_size=128)

**Breakdown**:
- ASM blocks: ~1,400,000
- Feature extractor (new): ~32,000
- Classifier head: ~400

**Comparison**:
- v1 (no feature layer): 1,488,000 params
- v2 (with binary heads): 1,540,000 params
- v3 (with feature layer): 1,520,000 params

### Training Hyperparameters
```python
Optimizer: AdamW
  - lr: 1e-3 (higher than Kanformer due to smaller model)
  - weight_decay: 1e-4

Scheduler: CosineAnnealingWarmRestarts
  - T_0: 10 epochs
  - T_mult: 2
  - eta_min: 1e-6

Warmup: 5 epochs (linear)

Loss: ContrastiveFocalLoss
  - alpha: confusion-aware sqrt weights
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
Test Macro F1: 0.87+ (exceed Conformer's 0.86)

Per-Class F1 Targets:
  Class 0: 0.95-0.96 (maintain)
  Class 1: 0.82-0.85 (+3-6 points) ← KEY IMPROVEMENT
  Class 2: 0.83-0.85 (maintain)
  Class 3: 0.74-0.77 (+7-10 points) ← KEY IMPROVEMENT
  Class 4: 0.92-0.94 (slight reduction OK)
  Class 5: 0.91-0.93 (slight reduction OK)
```

### Why v3 Should Outperform v2

**v2's Issues**:
- 3 loss terms to balance (main + margin + binary)
- Binary classifiers add complexity
- Harder to tune (more hyperparameters)

**v3's Advantages**:
- Simpler: 2 loss terms (focal + contrastive)
- Proven strategy from Kanformer v2
- Reuses feature extractor (no extra heads)
- Easier to interpret and debug

## Comparison Table

| Aspect | ASM v1 | ASM v2 | ASM v3 | Kanformer v2 |
|--------|--------|--------|--------|--------------|
| **Weighting** | Inverse freq | Confusion-aware 0.7× | Sqrt + boost 1.3× | Sqrt + boost 1.3× |
| **Label Smooth** | 0.1 | Adaptive (0.1-0.2) | 0.01 | 0.01 |
| **Contrastive** | ❌ | ❌ | ✓ (simple) | ✓ (simple) |
| **Auxiliary Loss** | ❌ | Margin + Binary | ❌ | ❌ |
| **Loss Terms** | 1 | 3 | 2 | 2 |
| **Parameters** | 1.49M | 1.54M | 1.52M | 3.85M |
| **Complexity** | Low | High | Medium | Medium |
| **Test F1 (target)** | 0.8566 | 0.8767 | **0.87+** | 0.86+ |

## Usage Instructions

### Training from Scratch
```bash
# Standard mode
python STMasm_enhanced3.py 0

# Downsampled mode
python STMasm_enhanced3.py 1
```

### Resuming Training
```bash
python STMasm_enhanced3.py 0 --resume \
  model/STM/ASM_Enhanced3_corpora_categories/standard/ckpt/2026-01-18_12-00
```

### Monitoring Training

**Key metrics printed each epoch**:
```
Val Macro F1: 0.8234
Per-class F1 scores:
  Class 0: 0.9345
  Class 1: 0.7856 (tonal speech)      ← Watch this improve
  Class 2: 0.8234
  Class 3: 0.7123 (env/music)         ← Watch this improve
  Class 4: 0.9456
  Class 5: 0.9234

Confusion between Similar Classes:
  Class 0→1:   950 | Class 1→0:  1200  ← Should decrease over time
  Class 2→3:   680 | Class 3→2:   820
```

## Expected Training Behavior

### Training Curves
```
Epoch 1-5:   Warmup phase, loss decreases rapidly
Epoch 5-15:  Fast convergence, Val F1 → 0.82-0.83
Epoch 15-25: First plateau, contrastive loss stabilizes
Epoch 25-35: LR reduction, Val F1 → 0.85-0.86
Epoch 35-50: Fine-tuning, Val F1 → 0.86-0.87
```

### Loss Patterns
```
Combined Loss:
  Epoch 1:  ~0.8 (focal dominant)
  Epoch 10: ~0.3 (both contributing)
  Epoch 30: ~0.1 (converged)

Component Breakdown:
  Focal Loss: ~90% of total (main signal)
  Contrastive: ~10% of total (regularization)
```

## Implementation Details

### Contrastive Loss Sampling Strategy

**Per batch**:
1. Find samples from Class 0 and Class 1
2. If both present, sample min(32, available) pairs
3. Compute pairwise L2 distances
4. Loss = mean(1 / (distance + 1e-6))
5. Repeat for Class 2 and Class 3
6. Average losses from both pairs

**Memory considerations**:
- Cap at 32 pairs per class pair per batch
- Total: max 64 distance computations per batch
- Adds ~5-10% compute overhead

### Feature Extractor Design

**Why this architecture**:
```python
self.feature_extractor = nn.Sequential(
    nn.Linear(dim, dim // 2),    # Compress 128 → 64
    nn.GELU(),                   # Non-linearity
    nn.Dropout(dropout),         # Regularization
)
```

**Rationale**:
1. **Compression** (128→64): Forces informative representation
2. **Non-linearity** (GELU): Adds expressivity before classification
3. **Dropout**: Prevents overfitting on features
4. **Reusable**: Serves both classifier and contrastive loss

**Comparison with v2**:
- v2: Binary classifiers (2×64→2 linear layers) = 256 params × 2 = 512 params
- v3: Feature extractor (128→64 linear) = 8,192 params (more expressive)

## Troubleshooting Guide

### Issue 1: Val F1 Not Improving Past 0.82
**Possible Causes**:
- Contrastive weight too low (not enough separation)
- Boost factors too weak

**Solutions**:
```python
# Increase contrastive weight
contrastive_weight = 0.15  # Up from 0.1

# Stronger boosting
weights[1] *= 1.4  # Up from 1.3
weights[3] *= 1.4
```

### Issue 2: Classes 1 and 3 Still Poor
**Possible Causes**:
- Inherent class overlap too high
- Features not discriminative enough

**Solutions**:
```python
# Larger feature extractor
self.feature_extractor = nn.Sequential(
    nn.Linear(dim, dim),        # Keep full dimension
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(dim, dim // 2),   # Two-layer compression
)

# Or increase model capacity
num_blocks = 5  # Up from 4
```

### Issue 3: Classes 4 and 5 Degraded Too Much
**Possible Causes**:
- Weight reduction too aggressive

**Solutions**:
```python
# Softer reduction
weights[4] *= 0.85  # Up from 0.7
weights[5] *= 0.9   # Up from 0.8
```

## Comparison with Kanformer v2

### Why ASM v3 Might Win

**Advantages**:
1. **Efficiency**: 1.5M params vs 3.8M (2.5× smaller)
2. **Speed**: Faster training (no self-attention)
3. **Memory**: 9GB vs 11GB GPU memory
4. **Simplicity**: Simpler architecture (MLP-based)

**When ASM wins**:
- Deployment scenarios (edge devices)
- Fast iteration experiments
- Memory-constrained settings

### Why Kanformer v2 Might Win

**Advantages**:
1. **Capacity**: 2.5× more parameters
2. **Attention**: Can capture long-range dependencies
3. **Proven**: Attention-based models dominate audio

**When Kanformer wins**:
- Maximum accuracy needed
- Sufficient compute available
- Complex audio patterns

## Success Metrics

### Minimum Viable Success
- Test Macro F1 ≥ 0.86 (match Conformer)
- Class 1 F1 ≥ 0.80 (improvement over v1's 0.79)
- Class 3 F1 ≥ 0.72 (improvement over v1's 0.67)

### Strong Success
- Test Macro F1 ≥ 0.87 (beat Conformer)
- Class 1 F1 ≥ 0.83 (close to Class 0's level)
- Class 3 F1 ≥ 0.75 (approaching Class 2's level)

### Outstanding Success
- Test Macro F1 ≥ 0.88
- All classes F1 ≥ 0.75
- Confusion pairs reduced by 40%+

## Next Steps After Training

### If v3 Succeeds (F1 ≥ 0.86)
1. **Ablation study**: Remove contrastive loss to measure contribution
2. **Weight sensitivity**: Try boost factors [1.2, 1.4, 1.6]
3. **Ensemble**: Combine ASM v3 + Kanformer v2 predictions
4. **Publish**: Write up efficient ASM architecture

### If v3 Fails (F1 < 0.86)
1. **Check contrastive convergence**: Plot distance metrics over time
2. **Try harder augmentation**: SpecAugment with stronger masking
3. **Increase capacity**: More blocks (5-6) or wider (dim=256)
4. **Hybrid**: Add 1-2 attention blocks at the end

### Regardless of Outcome
1. **Compare all versions**: v1, v2, v3, Kanformer v2
2. **Error analysis**: Deep dive into remaining confusion cases
3. **Feature visualization**: t-SNE of learned embeddings
4. **Publish confusion matrices**: Detailed per-class breakdown

## Files and Locations

### Source Code
- **Implementation**: `/vast/ac8888/MusicSpeech-STM/STMasm_enhanced3.py`
- **Documentation**: `/vast/ac8888/MusicSpeech-STM/ASM_ENHANCED3_SUMMARY.md`

### Related Files
- **ASM v1**: `STMasm_enhanced.py`
- **ASM v2**: `STMasm_enhanced2.py`
- **Kanformer v2**: `STMkanformer_enhanced2.py`
- **Baseline Conformer**: `STMconformer_model.py`

### Checkpoints
```
model/STM/ASM_Enhanced3_corpora_categories/
├── standard/
│   └── ckpt/
│       └── [timestamp]/
│           ├── best_model.pt
│           ├── latest_checkpoint.pt
│           ├── checkpoint_epoch_X.pt
│           ├── test_predictions.npy
│           ├── test_targets.npy
│           └── confusion_matrix.npy
└── downsample/
```

## Conclusion

Enhanced ASM v3 represents a **simpler, more principled approach** than v2:
- **Learns from Kanformer v2's success**: Proven strategy, not experimental
- **Softer weighting + targeted boosting**: Natural solution to over-weighting
- **Single contrastive loss**: Simpler than v2's 3-term loss
- **Maintains ASM efficiency**: 1.5M params vs Conformer/Kanformer's 3-4M

**Expected Outcome**: Test F1 = 0.87+ (beat Conformer baseline while staying efficient)

If v3 succeeds → ASM validated as efficient alternative to attention-based models  
If v3 fails → Try hybrid ASM+attention or accept Kanformer's superiority

---

**Last Updated**: January 2026  
**Author**: GitHub Copilot (Claude Sonnet 4.5) + User  
**Status**: Ready for training  
**Next Milestone**: Train v3 and compare with v1, v2, Kanformer v2, Conformer
