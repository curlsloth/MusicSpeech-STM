# STM_CoordConvLDAM_preIN8_ConvNeXt: Modern CNN Backbone for STM Classification

## Version Summary

**Version: 2.8-ConvNeXt**  
**Base: V2.8 (preIN8)**  
**Key Innovation: ConvNeXt-Tiny Backbone (replacing ResNet-18)**

## What's New in V2.8-ConvNeXt

### 1. ConvNeXt-Tiny Backbone

ConvNeXt (CVPR 2022) is a modern pure ConvNet architecture that rivals Vision Transformers while maintaining the simplicity and efficiency of CNNs.

| Feature | ResNet-18 (V2.8) | ConvNeXt-Tiny (V2.8-ConvNeXt) |
|---------|------------------|-------------------------------|
| Architecture | Traditional ResNet | Modern ConvNet |
| Parameters | ~12M | ~28.6M |
| Feature Dimension | 512 | 768 |
| Normalization | BatchNorm | LayerNorm (GroupNorm num_groups=1) |
| Activation | ReLU | GELU |
| Block Design | BasicBlock (2 convs) | Inverted Bottleneck (expand→depthwise→project) |
| Kernel Size | 3×3 | 7×7 (large kernels) |

**Channel Dimensions:**

| Stage | ConvNeXt-Tiny Channels |
|-------|------------------------|
| stem | → 96 |
| stage0 | 96 |
| stage1 | 192 |
| stage2 | 384 |
| stage3 | 768 |

#### Why ConvNeXt?
- **Modern design**: Incorporates insights from Vision Transformers (ViT)
- **Large receptive field**: 7×7 convolutions capture wider context
- **Better feature representation**: GELU activation + LayerNorm
- **Strong ImageNet performance**: Competitive with Swin Transformer
- **Training efficiency**: Pure ConvNet, no attention mechanism overhead

### 2. ConvNeXt Block Structure

Each ConvNeXt block:
```
Input (C)
    ↓
7×7 Depthwise Conv → LayerNorm (C)
    ↓
1×1 Conv → GELU (expand to 4×C)
    ↓
1×1 Conv (project back to C)
    ↓
+ Skip Connection (with Stochastic Depth)
```

Key features:
- **Depthwise separable convolution**: Efficient computation
- **Inverted bottleneck**: Expand→process→project (like MobileNetV2)
- **Stochastic depth (DropPath)**: Built-in regularization

### 3. Updated Three-Scale Feature Fusion

**V2.8 (ResNet-18):**
```
layer2 (128ch) + layer3 (256ch) + layer4 (512ch) = 896ch → 512ch
```

**V2.8-ConvNeXt (ConvNeXt-Tiny):**
```
stage1 (192ch) + stage2 (384ch) + stage3 (768ch) = 1344ch → 768ch
```

The larger fusion dimensionality provides:
- 50% more channels for multi-scale representation
- Better preservation of ConvNeXt's rich features
- Higher capacity for discriminating subtle class differences

### 4. Higher-Dimensional Center Loss

- Feature dimension increased from 512 to 768
- More expressive feature space for intra-class compactness
- Center parameters: 6 classes × 768 dims = 4,608 parameters

### 5. Adjusted Training Configuration

- **Batch size**: Reduced from 256 to 128 (due to larger model memory)
- **No separate attention modules**: ConvNeXt has inherent attention-like properties through large kernels
- **All other hyperparameters**: Same as V2.8 for fair comparison

## Architecture Overview

```
Input: (B, 2, 20, 121) - 2-channel Difference Map (Symmetric + Asymmetric)
    ↓
Custom Stem (4-ch CoordConv: 2 STM + 2 coords) → 96 ch
    ↓
Stage0 (96ch ConvNeXt Blocks)
    ↓
Downsample → Stage1 (192ch) → feat_stage1
    ↓
Downsample → Stage2 (384ch) → feat_stage2
    ↓
Downsample → Stage3 (768ch) → feat_stage3
    ↓
Three-Scale Fusion: cat(feat_stage1↑, feat_stage2, feat_stage3↓) → 1344 ch → 768 ch
    ↓
Global Average Pool → (B, 768)
    ↓
Features (768-dim) → Center Loss computation
    ↓
Dropout(0.4) → FC(768 → 6) → Logits
```

## Training Configuration

| Parameter | V2.8 (ResNet-18) | V2.8-ConvNeXt |
|-----------|------------------|----------------|
| Backbone | ResNet-18 | **ConvNeXt-Tiny** |
| Base LR | 1e-4 | 1e-4 |
| Discriminative LR | Stem/L1: 0.1x, L2-3: 0.5x, L4/Head: 1.0x | **Stem/S0: 0.1x, S1: 0.3x, S2: 0.5x, S3/Head: 1.0x** |
| Weight Decay | 5e-4 | 5e-4 |
| **Batch Size** | 256 | **128** |
| Epochs | 100 (max) | 100 (max) |
| Early Stopping | 20 patience | 20 patience |
| Layer Freezing | L1-L2 frozen first 10 epochs | **Stem + S0-S1 frozen first 10 epochs** |
| Mixup | α=0.4, 30% probability | Same |
| Loss | Hybrid LDAM (70%) + Focal (30%) | Same |
| Center Loss λ | 0.1 | 0.1 |
| **Center Feature Dim** | 512 | **768** |
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

Where $f_i$ is now a 768-dim feature vector.

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
python STM_CoordConvLDAM_preIN8_ConvNeXt.py 0

# Downsample mode (non-tonal speech downsampled to 100k)
python STM_CoordConvLDAM_preIN8_ConvNeXt.py 1
```

Checkpoints saved to:
- `model/STM/CoordConvLDAM_preIN8_ConvNeXt_corpora_categories/standard/ckpt/<timestamp>/`
- `model/STM/CoordConvLDAM_preIN8_ConvNeXt_corpora_categories/downsample/ckpt/<timestamp>/`

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

| Metric | V2.8 (ResNet-18) | V2.8-ConvNeXt Target |
|--------|------------------|----------------------|
| Test F1 (No TTA) | ~0.87 | 0.87-0.89 |
| Test F1 (TTA) | ~0.87-0.88 | 0.88-0.90 |
| music:non-vocal F1 | ~0.68-0.70 | 0.70+ |

### GPU Memory Requirements

| Metric | V2.8 (ResNet-18) | V2.8-ConvNeXt |
|--------|------------------|---------------|
| Model Parameters | ~12M | ~29M |
| Batch Size | 256 | 128 |
| Est. GPU Memory | ~4-6 GB | ~10-14 GB |

**Note:** ConvNeXt-Tiny is larger than ResNet-18 but more efficient than attention-based models.

## Comparison: ResNet-18 vs ConvNeXt-Tiny

| Aspect | ResNet-18 | ConvNeXt-Tiny |
|--------|-----------|---------------|
| Parameters | ~12M | ~29M |
| Feature Dim | 512 | 768 |
| ImageNet Top-1 | 69.8% | 82.1% |
| Architecture | Traditional | Modern (ViT-inspired) |
| Kernel Size | 3×3 | 7×7 |
| Normalization | BatchNorm | LayerNorm |
| Activation | ReLU | GELU |
| GPU Memory | ~4-6 GB | ~10-14 GB |
| Training Speed | Fast | Moderate |

### When to Use ConvNeXt:
- **Higher capacity**: When ResNet-18 seems to plateau
- **Better features**: ConvNeXt learns more expressive representations
- **Modern architecture**: Benefits from recent architectural advances
- **Strong baseline**: Well-validated on ImageNet and downstream tasks

### Potential Concerns:
- **Memory**: ~2.5× more memory than ResNet-18
- **Training time**: Slightly slower per epoch
- **Overfitting risk**: Larger model may overfit on small datasets

## Version History

| Version | Key Changes | Test F1 |
|---------|-------------|---------|
| V2.0 | ImageNet pretrained backbone (ResNet-18) | 0.8489 |
| V2.1 | Coord Attention, Layer Freezing | 0.8591 |
| V2.4 | Multi-Scale Fusion, SpecAugment | 0.8682 |
| V2.6 | Three-Scale Fusion, Hybrid Loss | 0.8709 |
| V2.6+TTA | Test-Time Augmentation | 0.8726 |
| V2.8 | Center Loss + Integrated TTA | ~0.87+ |
| **V2.8-ConvNeXt** | **ConvNeXt-Tiny Backbone** | **TBD** |

## Technical Notes

### Weight Cloning Strategy
For the custom stem, we clone ImageNet-pretrained weights from the 3-channel input:
- Take mean of RGB channel weights
- Replicate to 4 channels (2 STM + 2 coords)
- Scale by √(3/4) to maintain variance

### ConvNeXt vs ResNet for STM
ConvNeXt may be particularly suitable for STM because:
- **Large kernels (7×7)**: Better capture spectro-temporal patterns that span multiple bins
- **GELU activation**: Smoother gradients may help with subtle feature differences
- **LayerNorm**: More stable for varying input distributions
- **Stochastic depth**: Built-in regularization helps with potential overfitting

### References

1. Liu, Z., et al. (2022). "A ConvNet for the 2020s." CVPR.
2. He, K., et al. (2016). "Deep residual learning for image recognition." CVPR.
3. Wen, Y., et al. (2016). "A discriminative feature learning approach for deep face recognition." ECCV.
