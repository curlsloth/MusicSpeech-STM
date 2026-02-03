# C3NeXt2: Focal-LDAM + Deeper Network

## Overview

C3NeXt2 addresses the weak performance on **speech:tonal (recall 0.63)** and **music:non-vocal (recall 0.60)** through:

1. **Focal-LDAM Loss**: Combines focal loss hard example mining with LDAM class balancing
2. **Deeper Network**: 24 blocks [4, 12, 8] instead of 18 blocks [3, 9, 6]
3. **Earlier DRW**: Starts at epoch 40 instead of 50
4. **Stronger Regularization**: Higher drop path (0.2 vs 0.1), stronger mixup (alpha=0.4 vs 0.3)

## Problem Analysis

### Why These Classes Fail

**Speech:Tonal (Recall 0.63)**:
- Problem: Confused with speech:non-tonal
- Root cause: Pitch contours are subtle in STM representation
- Need: Better discrimination of tonal patterns across spectral scales

**Music:Non-Vocal (Recall 0.60)**:
- Problem: Confused with speech:non-tonal
- Root cause: Both have irregular, non-harmonic patterns
- Need: Better separation based on rhythmic structure

### Current C3NeXt Limitations

1. **Insufficient Capacity**: 18 blocks may not capture subtle distinctions
2. **Imbalanced Learning**: Majority classes (speech:non-tonal) dominate gradients
3. **Easy Example Bias**: Network focuses on easy samples, ignores hard cases
4. **Late Reweighting**: DRW at epoch 50 may be too late for minority classes

## Solution 1: Focal-LDAM Loss

### Motivation

**Standard LDAM**:
- Adds larger margins to minority classes
- But treats all examples equally within a class
- Easy examples (high confidence) still contribute significant gradients

**Problem**: Network "satisfied" with easy examples, doesn't focus on hard cases
- Easy speech:tonal sample (strong pitch contours): Learned well
- Hard speech:tonal sample (weak pitch): Still misclassified

**Focal Loss Insight**:
- Down-weight easy examples: $(1 - p_t)^\gamma \cdot \mathcal{L}$
- Force network to focus on hard examples

### Focal-LDAM Formula

$$
\mathcal{L}_{\text{Focal-LDAM}} = (1 - p_t)^\gamma \cdot \left( -\log \frac{e^{s(\mathbf{z}_y - \Delta_y)}}{\sum_{j=1}^C e^{s\mathbf{z}_j}} \right)
$$

Where:
- $p_t$: Predicted probability for true class
- $(1 - p_t)^\gamma$: Focal weight (down-weights easy examples)
- $\gamma = 2.0$: Focusing parameter
- $\Delta_y$: LDAM margin (larger for minority classes)

**Effect on Training**:

| Sample Type | Confidence $p_t$ | Focal Weight $(1-p_t)^2$ | Effective Loss |
|-------------|------------------|--------------------------|----------------|
| Easy speech:non-tonal | 0.95 | 0.0025 | Very low |
| Hard speech:non-tonal | 0.60 | 0.16 | Medium |
| Easy speech:tonal | 0.90 | 0.01 | Low |
| **Hard speech:tonal** | **0.40** | **0.36** | **High** |

**Result**: Hard minority class examples get 30-100× more gradient contribution than easy majority class examples

### Expected Improvement

**Mechanism**:
1. Early epochs: Learn general features from all classes
2. Mid epochs: Focal loss shifts focus to hard examples
3. Late epochs: Fine-tune decision boundaries for minority classes

**Target**:
- Speech:tonal: 0.63 → 0.70-0.72 (+10-14%)
- Music:non-vocal: 0.60 → 0.67-0.69 (+11-15%)

## Solution 2: Deeper Network

### Architecture Changes

**C3NeXt (18 blocks)**:
```
Stage 1: 3 blocks (96 channels)
Stage 2: 9 blocks (192 channels)
Stage 3: 6 blocks (384 channels)
Total: 18 blocks
```

**C3NeXt2 (24 blocks)**:
```
Stage 1: 4 blocks (96 channels)     [+1 block]
Stage 2: 12 blocks (192 channels)   [+3 blocks]
Stage 3: 8 blocks (384 channels)    [+2 blocks]
Total: 24 blocks (+33%)
```

### Why More Depth Helps

**Problem**: Complex inter-class confusion patterns
- Speech:tonal vs speech:non-tonal: Requires multi-scale pitch analysis
- Music:non-vocal vs speech:non-tonal: Requires rhythmic structure understanding

**Deeper network benefits**:

1. **More hierarchical features**:
   - Shallow layers: Basic patterns (energy peaks, edges)
   - Mid layers: Composite patterns (rhythm units, harmonic stacks)
   - Deep layers: Abstract concepts (tonality, musical structure)

2. **Better feature composition**:
   - 18 blocks: Limited composition depth
   - 24 blocks: Can build more complex feature hierarchies

3. **Improved discrimination**:
   - Each additional block adds non-linear discrimination capacity
   - 33% more blocks → significantly better separation

### Parameter Analysis

**C3NeXt**: ~8M parameters
**C3NeXt2**: ~10.5M parameters (+31%)

**Still smaller than**:
- ResNet-50: ~25M parameters
- ConvNeXt-Small: ~50M parameters

**Justification**: More parameters allocated to feature learning vs classification

## Solution 3: Earlier DRW

### Problem with Late Reweighting

**Current C3NeXt (DRW at epoch 50)**:
- Epochs 1-49: Learn without class weights
- Problem: Majority classes dominate for 50% of training
- Minority class features may be under-learned

**Example gradient contributions (without DRW)**:
- Speech:non-tonal (497K samples): 64.6% of gradients
- Speech:tonal (80K samples): 10.4% of gradients
- Music:non-vocal (54K samples): 6.9% of gradients

**Issue**: By epoch 50, feature extractor is already "biased" toward majority classes

### C3NeXt2 Solution: DRW at Epoch 40

**Earlier reweighting**:
- Epochs 1-39: Learn general features (60% minority class data seen 2× vs 1×)
- Epochs 40-100: Balance training (60% of training with weights)

**DRW Weight Formula**:
$$
w_i = \frac{1 - \beta}{1 - \beta^{n_i}}, \quad \beta = 0.9999
$$

**Effective weights**:
| Class | Count | DRW Weight | Effective Samples |
|-------|-------|------------|-------------------|
| speech:non-tonal | 497K | 0.17 | 84K |
| speech:tonal | 80K | 1.06 | 85K |
| music:vocal | 102K | 0.83 | 85K |
| music:non-vocal | 54K | 1.57 | 85K |
| env:urban | 10K | 8.47 | 85K |
| env:wildlife | 28K | 3.03 | 85K |

**Result**: All classes contribute ~equally to gradients

**Why epoch 40?**
- Too early (e.g., epoch 20): Network hasn't learned general features yet
- Too late (e.g., epoch 60): Only 40% of training benefits from reweighting
- Epoch 40: Balance between general and class-specific learning

## Solution 4: Stronger Regularization

### Increased Drop Path

**C3NeXt**: drop_path_rate = 0.1
**C3NeXt2**: drop_path_rate = 0.2

**Effect**: 
- Randomly drop 20% of blocks during training (vs 10%)
- Forces network to learn redundant features
- Prevents overfitting to majority class patterns

**Benefit for minority classes**:
- Network can't rely on single pathway
- Must learn multiple ways to recognize each class
- Better generalization to hard examples

### Stronger Mixup

**C3NeXt**: alpha = 0.3, 30% of batches
**C3NeXt2**: alpha = 0.4, 40% of batches

**Mixup effect**:
$$
\tilde{x} = \lambda x_i + (1-\lambda) x_j, \quad \lambda \sim \text{Beta}(0.4, 0.4)
$$

**Alpha=0.4 distribution**:
- More balanced mixing (closer to 50-50 blends)
- Smoother decision boundaries
- Better interpolation between classes

**Example**:
```
speech:tonal sample + music:vocal sample (λ=0.6)
→ Network learns: "Shared tonal characteristics"
→ Better separation from non-tonal classes
```

### Higher Head Dropout

**C3NeXt**: dropout = 0.3
**C3NeXt2**: dropout = 0.4

**Effect**: More aggressive regularization in classifier
- Prevents overfitting to training distribution
- Especially important for minority classes (limited data)

## Expected Performance

### Baseline (C3NeXt)

| Metric | C3NeXt |
|--------|--------|
| Test Macro F1 | 0.8393 |
| speech:tonal recall | 0.63 |
| music:non-vocal recall | 0.60 |

### Target (C3NeXt2)

| Metric | Target | Improvement |
|--------|--------|-------------|
| Test Macro F1 | **0.855-0.865** | +1.5-2.5% |
| speech:tonal recall | **0.70-0.72** | +11-14% |
| music:non-vocal recall | **0.67-0.69** | +11-15% |

### Mechanism Summary

1. **Focal-LDAM**: Forces focus on hard minority class examples
   - Effect: +5-7% recall on hard classes
   
2. **Deeper Network**: Better feature hierarchies for subtle distinctions
   - Effect: +3-4% recall across all classes
   
3. **Earlier DRW**: More balanced learning throughout training
   - Effect: +2-3% recall on minority classes
   
4. **Stronger Regularization**: Better generalization
   - Effect: +1-2% overall improvement

**Combined effect**: +11-14% on weakest classes

## Implementation Details

### Training Schedule

- Total epochs: 100
- DRW start: Epoch 40 (vs 50 in C3NeXt)
- Learning rate: 1e-4 initial
- ReduceLROnPlateau: factor=0.5, patience=10
- Early stopping: patience=20

### Focal-LDAM Parameters

- gamma: 2.0 (standard focal loss)
- max_m: 0.5 (LDAM margin)
- s: 30 (scaling)
- label_smooth: 0.05

### Regularization

- drop_path_rate: 0.2 (vs 0.1)
- head_dropout: 0.4 (vs 0.3)
- mixup alpha: 0.4 (vs 0.3)
- mixup frequency: 40% (vs 30%)

## Computational Cost

**Training time per epoch**: ~15% slower than C3NeXt
- Reason: 33% more blocks, focal loss computation

**Total parameters**: ~10.5M (vs ~8M)
**GPU memory**: ~8GB (vs ~7GB)

**Justification**: Modest increase for significant performance gain on critical minority classes

## Summary

C3NeXt2 is specifically designed to address the weak performance on **speech:tonal** and **music:non-vocal** by:

1. **Focusing on hard examples** through Focal-LDAM
2. **Increasing capacity** with 24 blocks
3. **Earlier balancing** with DRW at epoch 40
4. **Stronger regularization** to prevent overfitting

Expected improvement: +11-14% recall on weakest classes, +1.5-2.5% macro F1.
