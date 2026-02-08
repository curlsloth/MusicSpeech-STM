# STM_CoordConvLDAM_preIN6: V2.6 Enhancement Summary

## Overview
V2.6 builds upon V2.4 (0.8682 Test F1) with three targeted improvements to address the **music:non-vocal bottleneck** (F1=0.68), which is dragging overall performance.

## Key Improvements in V2.6

### 1. Three-Scale Feature Fusion (Deeper Multi-Scale)
**Previous (V2.4):** 2-scale fusion (layer3 + layer4)
**New (V2.6):** 3-scale fusion (layer2 + layer3 + layer4)

| Layer | Channels | Spatial Size | Captures |
|-------|----------|--------------|----------|
| layer2 | 128 | 10×61 | **Fine-grained temporal patterns** |
| layer3 | 256 | 5×31 | Mid-level features |
| layer4 | 512 | 3×16 | High-level semantics |

**Implementation:**
- All features interpolated to match layer3 spatial dims (5×31)
- Concatenation: 128 + 256 + 512 = 896 channels
- 1×1 conv fusion: 896 → 512 channels

**Why it helps music:non-vocal:**
- Music:non-vocal has high intra-class variance (jazz, classical, electronic)
- layer2's higher temporal resolution (61 vs 31 vs 16) captures subtle rhythmic/timbral differences
- More discriminative features for classes with similar high-level semantics

### 2. Hybrid LDAM + Focal Loss

**Loss Formula:**
```
L_hybrid = 0.7 × L_LDAM + 0.3 × L_Focal
```

**Components:**
- **LDAM Loss (70%):** Margin-based loss for class-imbalanced learning
  - Class-dependent margins: larger margins for minority classes
  - Maintains proven long-tail handling from V2.4
  
- **Focal Loss (30%):** Hard example mining
  - FL(p_t) = -(1 - p_t)^γ × log(p_t), γ=2
  - Down-weights easy examples (p_t > 0.5)
  - Focuses learning on hard-to-classify samples

**Why it helps music:non-vocal:**
- Music:non-vocal samples are often misclassified as music:vocal
- Focal loss forces the model to focus on these ambiguous samples
- γ=2 provides moderate focusing without being too aggressive

### 3. Earlier DRW Activation

**Previous (V2.4):** DRW at epoch 50 (num_epochs // 2)
**New (V2.6):** DRW at epoch 30

**Deferred Re-Weighting (DRW) Schedule:**
| Epochs | Strategy | Phase |
|--------|----------|-------|
| 1-10 | Layer1-2 frozen, uniform weights | Warmup |
| 11-30 | All layers trainable, uniform weights | Feature learning |
| 31-100 | All layers trainable, **class weights** | Rebalanced learning |

**Why it helps minority classes:**
- 70 epochs of class-weighted training (vs 50 in V2.4)
- More time for minority classes (music:non-vocal, speech categories) to be emphasized
- Early epochs still use uniform weights to learn good representations

## Architecture Summary

```
Input: 2×20×121 (Symmetric + Asymmetric Difference Maps)
   ↓
CoordConv Stem (2+2 → 64)
   ↓
Layer1 [64ch, 20×121] + Coordinate Attention
   ↓
Layer2 [128ch, 10×61] + Coordinate Attention ─────┐
   ↓                                               │
Layer3 [256ch, 5×31] + SE Attention ──────────────┼──→ 3-Scale Fusion
   ↓                                               │      (896→512ch)
Layer4 [512ch, 3×16] + SE Attention ──────────────┘
                                                    ↓
                                            Global Avg Pool
                                                    ↓
                                            FC Head (512→256→128→6)
                                                    ↓
                                              6-class output
```

## Training Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Loss** | HybridLDAMFocalLoss | 70% LDAM + 30% Focal |
| **Focal γ** | 2.0 | Moderate focusing |
| **LDAM max_m** | 0.5 | Standard margin |
| **Label smoothing** | 0.05 | Prevents overconfidence |
| **DRW activation** | Epoch 31 | Earlier than V2.4 (epoch 51) |
| **Layer freezing** | Epochs 1-10 | layer1-2 frozen |
| **Learning rates** | Discriminative | layer1: 0.1×, layer2-3: 0.5×, layer4+new: 1.0× |
| **Optimizer** | AdamW | weight_decay=0.01 |
| **Scheduler** | ReduceLROnPlateau | patience=7, factor=0.5 |
| **Mixup** | α=0.4 | Applied when model.training |
| **STM Augmentation** | Freq/time masking + shifts | SpecAugment-style |
| **Batch size** | 64 | |
| **Epochs** | 100 | |

## Expected Improvements

| Metric | V2.4 | V2.6 Target | Notes |
|--------|------|-------------|-------|
| **Test Macro F1** | 0.8682 | 0.88-0.90 | Approaching SOTA |
| **music:non-vocal F1** | 0.68 | 0.75-0.80 | Primary target |
| **Balanced classes** | 0.88-0.98 | Maintain | Should not regress |

## Files

| File | Description |
|------|-------------|
| `STM_CoordConvLDAM_preIN6.py` | Main training script |
| `STM_CoordConvLDAM_preIN6.md` | This documentation |

## Usage

```bash
# Standard mode (full training data)
python STM_CoordConvLDAM_preIN6.py standard

# Downsampled mode (for debugging/quick tests)
python STM_CoordConvLDAM_preIN6.py downsample
```

## Output Directories

- Standard: `model/STM/CoordConvLDAM_preIN6_corpora_categories/standard/`
- Downsample: `model/STM/CoordConvLDAM_preIN6_corpora_categories/downsample/`

## Comparison to Previous Versions

| Version | Key Changes | Test F1 |
|---------|-------------|---------|
| V2.0 (preIN) | ImageNet pretrained, CoordConv, LDAM | 0.8594 |
| V2.1 (preIN2) | +CA/SE attention, +discriminative LR | 0.8618 |
| V2.4 (preIN4) | +2-scale fusion (L3+L4), +STM aug, +layer freezing | 0.8682 |
| **V2.6 (preIN6)** | +3-scale fusion (L2+L3+L4), +Focal loss, +earlier DRW | TBD |

## Rationale for Changes

### Why Deeper Multi-Scale Fusion?
The confusion matrix shows music:non-vocal has high variance—the class contains diverse genres from jazz to electronic. Higher-resolution features from layer2 capture finer temporal patterns that distinguish these subgenres, reducing intra-class confusion.

### Why Add Focal Loss?
Hard examples (samples near decision boundaries) are under-represented in standard loss functions. Focal loss mathematically up-weights these samples. Combined with LDAM, we get both margin-based separation AND hard example emphasis.

### Why Earlier DRW?
At epoch 50, the model may already have converged to suboptimal patterns on minority classes. Starting DRW at epoch 30 gives 40% more weighted training time for minority classes while still allowing 30 epochs of unweighted feature learning.

## Next Steps After Running V2.6

If results are unsatisfactory:
1. **If music:non-vocal improves but overall F1 drops:** Reduce Focal weight (try 0.2)
2. **If no improvement:** Consider architecture change (ConvNeXt or Transformer)
3. **If close to 0.89:** Try ensemble with V2.4
