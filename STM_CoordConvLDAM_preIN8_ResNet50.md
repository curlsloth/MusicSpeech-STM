# STM_CoordConvLDAM_preIN8_ResNet50: Medium-Depth Backbone for STM Classification

## Version Summary

**Version: 2.8-R50**  
**Base: V2.8 (preIN8)**  
**Key Innovation: ResNet-50 Backbone (replacing ResNet-18)**

## What's New in V2.8-R50

### 1. ResNet-50 Backbone (vs ResNet-18)

| Feature | ResNet-18 (V2.8) | ResNet-50 (V2.8-R50) |
|---------|------------------|----------------------|
| Layers | 18 | 50 |
| Block Type | BasicBlock (2 convs) | Bottleneck (3 convs) |
| Blocks per Stage | [2, 2, 2, 2] | [3, 4, 6, 3] |
| Parameters | ~11.7M | ~25.6M |
| Feature Dimension | 512 | 2048 |

**Channel Dimensions:**

| Layer | ResNet-18 | ResNet-50 |
|-------|-----------|-----------|
| layer1 | 64 | 256 (64×4) |
| layer2 | 128 | 512 (128×4) |
| layer3 | 256 | 1024 (256×4) |
| layer4 | 512 | 2048 (512×4) |

#### Why ResNet-50?
- **Balanced trade-off**: More capacity than ResNet-18, more efficient than ResNet-101
- **Deeper feature hierarchy**: 50 layers vs 18 layers for richer intermediate representations
- **Good compromise**: ~2× parameters of ResNet-18, ~0.5× of ResNet-101
- **Well-studied**: Often the sweet spot for transfer learning tasks

### 2. Updated Three-Scale Feature Fusion

**V2.8 (ResNet-18):**
```
layer2 (128ch) + layer3 (256ch) + layer4 (512ch) = 896ch → 512ch
```

**V2.8-R50 (ResNet-50):**
```
layer2 (512ch) + layer3 (1024ch) + layer4 (2048ch) = 3584ch → 2048ch
```

The larger fusion dimensionality provides:
- Richer multi-scale feature representation
- Better preservation of fine-grained details from earlier layers
- More capacity for discriminating subtle class differences

### 3. Higher-Dimensional Center Loss

- Feature dimension increased from 512 to 2048
- More expressive feature space for intra-class compactness
- Center parameters: 6 classes × 2048 dims = 12,288 parameters

### 4. Adjusted Training Configuration

- **Batch size**: Reduced from 256 to 128 (due to larger model memory)
- **All other hyperparameters**: Same as V2.8 for fair comparison

## Architecture Overview

```
Input: (B, 2, 20, 121) - 2-channel Difference Map (Symmetric + Asymmetric)
    ↓
CoordConv Stem (4-ch: 2 STM + 2 coords) → 64 ch
    ↓
Layer1 (3 Bottlenecks, 256 ch) + Coordinate Attention
Layer2 (4 Bottlenecks, 512 ch, stride=2) + Coordinate Attention → feat_layer2
Layer3 (6 Bottlenecks, 1024 ch, stride=2) + Squeeze-Excitation → feat_layer3
Layer4 (3 Bottlenecks, 2048 ch, stride=2) + Squeeze-Excitation → feat_layer4
    ↓
Three-Scale Fusion: cat(feat_layer2↓, feat_layer3, feat_layer4↑) → 3584 ch → 2048 ch
    ↓
Global Average Pool → (B, 2048)
    ↓
Features (2048-dim) → Center Loss computation
    ↓
Dropout(0.4) → FC(2048 → 6) → Logits
```

### Bottleneck Block Structure

Each Bottleneck block in ResNet-50:
```
Input (C_in)
    ↓
1×1 Conv → BN → ReLU (reduce to C_in/4)
    ↓
3×3 Conv → BN → ReLU (maintain C_in/4)
    ↓
1×1 Conv → BN (expand to C_out = C_in × expansion)
    ↓
+ Skip Connection → ReLU
    ↓
Attention (CA or SE)
    ↓
Dropout(0.05)
```

Expansion factor = 4, which is why output channels are 4× the bottleneck width.

## Training Configuration

| Parameter | V2.8 (ResNet-18) | V2.8-R50 (ResNet-50) |
|-----------|------------------|----------------------|
| Backbone | ResNet-18 | **ResNet-50** |
| Base LR | 1e-4 | 1e-4 |
| Discriminative LR | Stem/L1: 0.1x, L2-3: 0.5x, L4/Head: 1.0x | Same |
| Weight Decay | 5e-4 | 5e-4 |
| **Batch Size** | 256 | **128** |
| Epochs | 100 (max) | 100 (max) |
| Early Stopping | 20 patience | 20 patience |
| Layer Freezing | L1-L2 frozen first 10 epochs | Same |
| Mixup | α=0.4, 30% probability | Same |
| Loss | Hybrid LDAM (70%) + Focal (30%) | Same |
| Center Loss λ | 0.1 | 0.1 |
| **Center Feature Dim** | 512 | **2048** |
| Center LR | 0.5 (SGD) | 0.5 (SGD) |
| DRW | Enabled after epoch 30 | Same |

## Loss Components

### 1. LDAM Loss (70%)
Label-Distribution-Aware Margin loss with class-dependent margins:
$$m_j = \frac{C}{n_j^{1/4}}$$

### 2. Focal Loss (30%)
Hard example mining with γ=2:
$$FL = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

### 3. Center Loss
$$L_{center} = \frac{1}{2B} \sum_i ||f_i - c_{y_i}||^2$$

Where $f_i$ is now a 2048-dim feature vector.

### Combined Loss
$$L_{total} = 0.7 \cdot L_{LDAM} + 0.3 \cdot L_{Focal} + 0.1 \cdot L_{center}$$

## STM Augmentation (SpecAugment-style)

Same as V2.8:
```python
STMAugmentation(
    freq_mask_prob=0.3, freq_mask_width=3,
    time_mask_prob=0.3, time_mask_width=15,
    freq_shift_prob=0.2, max_freq_shift=3,
    time_shift_prob=0.2, max_time_shift=10
)
```

## Test-Time Augmentation (TTA)

Same 5 augmentations as V2.8:

| # | Augmentation | Description |
|---|--------------|-------------|
| 1 | Original | No augmentation |
| 2 | Time Flip | Reverse temporal axis |
| 3 | Freq Shift +2 | Roll frequency by +2 bins |
| 4 | Freq Shift -2 | Roll frequency by -2 bins |
| 5 | Time Shift +5 | Roll time by +5 frames |

## Usage

```bash
# Standard mode (full dataset)
python STM_CoordConvLDAM_preIN8_ResNet50.py 0

# Downsample mode (non-tonal speech downsampled to 100k)
python STM_CoordConvLDAM_preIN8_ResNet50.py 1
```

Checkpoints saved to:
- `model/STM/CoordConvLDAM_preIN8_ResNet50_corpora_categories/standard/ckpt/<timestamp>/`
- `model/STM/CoordConvLDAM_preIN8_ResNet50_corpora_categories/downsample/ckpt/<timestamp>/`

## Output Files

| File | Description |
|------|-------------|
| `best_model.pt` | Best model checkpoint |
| `training_history.json` | Training metrics per epoch |
| `test_predictions.npy` | Test predictions (no TTA) |
| `test_targets.npy` | Test ground truth |
| `tta_predictions.npy` | TTA predictions |
| `tta_probs.npy` | TTA probability distributions |
| `confusion_matrix.png` | Visual confusion matrix |
| `confusion_matrix.npy` | Raw confusion matrix |

## Expected Results & Memory Requirements

### Performance Expectations
Based on V2.8 baseline (Test F1: ~0.87):

| Metric | V2.8 (ResNet-18) | V2.8-R50 Target |
|--------|------------------|-----------------|
| Test F1 (No TTA) | ~0.87 | 0.87-0.88 |
| Test F1 (TTA) | ~0.87-0.88 | 0.88-0.89 |
| music:non-vocal F1 | ~0.68-0.70 | 0.70+ |

### GPU Memory Requirements

| Metric | V2.8 (ResNet-18) | V2.8-R50 |
|--------|------------------|----------|
| Model Parameters | ~12M | ~26M |
| Batch Size | 256 | 128 |
| Est. GPU Memory | ~4-6 GB | ~8-12 GB |

**Note:** ResNet-50 is a good balance between:
- ResNet-18: Smaller, faster, but potentially limited capacity
- ResNet-101: Larger capacity but 2× the memory

## Comparison: ResNet-18 vs ResNet-50 vs ResNet-101

| Aspect | ResNet-18 | ResNet-50 | ResNet-101 |
|--------|-----------|-----------|------------|
| Parameters | ~12M | ~26M | ~45M |
| Blocks | [2,2,2,2] | [3,4,6,3] | [3,4,23,3] |
| GPU Memory | ~4-6 GB | ~8-12 GB | ~12-16 GB |
| Training Speed | Fast | Medium | Slow |
| Feature Dim | 512 | 2048 | 2048 |

### When to Use ResNet-50:
- **Good balance**: When ResNet-18 feels too limited but ResNet-101 is overkill
- **Memory constraints**: When you have 8-12GB GPU memory
- **Sweet spot**: Often the best trade-off for transfer learning tasks
- **Production**: Good balance between accuracy and inference speed

## Version History

| Version | Key Changes | Test F1 |
|---------|-------------|---------|
| V2.0 | ImageNet pretrained backbone (ResNet-18) | 0.8489 |
| V2.1 | Coord Attention, Layer Freezing | 0.8591 |
| V2.4 | Multi-Scale Fusion, SpecAugment | 0.8682 |
| V2.6 | Three-Scale Fusion, Hybrid Loss | 0.8709 |
| V2.6+TTA | Test-Time Augmentation | 0.8726 |
| V2.8 | Center Loss + Integrated TTA | ~0.87+ |
| **V2.8-R50** | **ResNet-50 Backbone** | **TBD** |

## Technical Notes

### Weight Cloning Strategy
Same as V2.8: Red channel weights from ImageNet pretrained conv1 are cloned to all 4 input channels (2 STM + 2 coords), scaled by √(3/4) to maintain variance.

### ResNet-50 vs ResNet-101 for STM
ResNet-50 may be sufficient for STM classification because:
- STM input is relatively small (20×121)
- 6 classes with clear distinctions
- Most discriminative features may not require the depth of ResNet-101
- Risk of overfitting with very deep models

### Reference

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. CVPR.
