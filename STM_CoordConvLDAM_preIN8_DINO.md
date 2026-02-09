# STM_CoordConvLDAM_preIN8_DINO: Self-Supervised Vision Transformer for STM Classification

## Version Summary

**Version: 2.8-DINO**  
**Base: V2.8 (preIN8)**  
**Key Innovation: DINO ViT-Small Backbone (Self-Supervised Vision Transformer)**

## What's New in V2.8-DINO

### 1. DINO ViT-Small Backbone

DINO (Self-DIstillation with NO labels) is a self-supervised Vision Transformer from Facebook AI Research (ICCV 2021) that learns semantic features without labeled data.

| Feature | ResNet-18 (V2.8) | DINO ViT-Small (V2.8-DINO) |
|---------|------------------|----------------------------|
| Architecture | CNN (ResNet) | Vision Transformer |
| Pretraining | Supervised (ImageNet labels) | Self-Supervised (no labels) |
| Parameters | ~12M | ~22M |
| Feature Dimension | 512 | 384 |
| Blocks/Layers | 4 ResNet layers | 12 Transformer blocks |
| Attention | SE + CA (added) | Multi-Head Self-Attention (built-in) |

**DINO ViT-Small Architecture:**

| Component | Specification |
|-----------|---------------|
| Hidden Dimension | 384 |
| Attention Heads | 6 |
| Transformer Blocks | 12 |
| Patch Size | 4×4 (custom for STM) |
| Input Patches | 155 (5×31 grid) |
| MLP Ratio | 4× |

#### Why DINO?
- **Self-supervised pretraining**: Learns richer semantic features without label bias
- **Attention visualization**: DINO features are highly interpretable (attends to semantic regions)
- **Strong transfer learning**: Outperforms supervised pretraining on many downstream tasks
- **Global receptive field**: Each token can attend to all other tokens

### 2. Custom Patch Embedding for STM

Since STM input (20×121) is much smaller than typical ViT input (224×224), we use:

**Input Processing:**
```
STM Input: (B, 2, 20, 121)
    ↓
Add Coordinate Channels: (B, 4, 20, 121)
    ↓
Pad Width: (B, 4, 20, 124) [divisible by patch_size=4]
    ↓
Patch Embedding (4×4 patches): (B, 155, 384)
    ↓
Add [CLS] Token: (B, 156, 384)
    ↓
Add Position Embedding: (B, 156, 384)
```

- **Patch size**: 4×4 (vs 16×16 in original DINO)
- **Patch grid**: 5 × 31 = 155 patches
- **Position embedding**: Interpolated from original 14×14 grid

### 3. Multi-Scale Feature Extraction

Extract features from multiple transformer blocks:

```
Block 4:  (B, 156, 384) → [CLS] token → 384-dim
Block 8:  (B, 156, 384) → [CLS] token → 384-dim
Block 12: (B, 156, 384) → [CLS] token → 384-dim
    ↓
Concatenate: 384 × 3 = 1152-dim
    ↓
Fusion: 1152 → 384-dim
```

This captures:
- **Early features (block 4)**: Low-level patterns
- **Middle features (block 8)**: Mid-level structures
- **Late features (block 12)**: High-level semantics

### 4. Center Loss for 384-dim Features

- Feature dimension: 384 (vs 512 for ResNet-18)
- Center parameters: 6 classes × 384 dims = 2,304 parameters

### 5. Adjusted Training Configuration

- **Batch size**: Reduced from 256 to 64 (due to transformer memory)
- **Layer freezing**: First 6 transformer blocks frozen for 10 epochs

## Architecture Overview

```
Input: (B, 2, 20, 121) - 2-channel Difference Map (Symmetric + Asymmetric)
    ↓
Add Coordinate Channels → (B, 4, 20, 121)
    ↓
Pad Width → (B, 4, 20, 124)
    ↓
Patch Embedding (4×4) → (B, 155, 384)
    ↓
[CLS] Token + Position Embedding → (B, 156, 384)
    ↓
Transformer Block 1-4 → Extract [CLS] at block 4
Transformer Block 5-8 → Extract [CLS] at block 8
Transformer Block 9-12 → Extract [CLS] at block 12
    ↓
Final LayerNorm
    ↓
Multi-Scale Fusion: (384 + 384 + 384) → 1152 → 384
    ↓
Features (384-dim) → Center Loss computation
    ↓
Dropout(0.4) → FC(384 → 6) → Logits
```

## Training Configuration

| Parameter | V2.8 (ResNet-18) | V2.8-DINO (ViT-Small) |
|-----------|------------------|------------------------|
| Backbone | ResNet-18 | **DINO ViT-Small** |
| Pretraining | Supervised | **Self-Supervised** |
| Base LR | 1e-4 | 1e-4 |
| Discriminative LR | Stem/L1: 0.1x, L2-3: 0.5x | **Embed/Early: 0.1x, Late: 0.5x** |
| Weight Decay | 5e-4 | 5e-4 |
| **Batch Size** | 256 | **64** |
| Epochs | 100 (max) | 100 (max) |
| Early Stopping | 20 patience | 20 patience |
| **Layer Freezing** | L1-L2 frozen first 10 epochs | **Blocks 0-5 frozen first 10 epochs** |
| Mixup | α=0.4, 30% probability | Same |
| Loss | Hybrid LDAM (70%) + Focal (30%) | Same |
| Center Loss λ | 0.1 | 0.1 |
| **Center Feature Dim** | 512 | **384** |
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

Where $f_i$ is now a 384-dim feature vector.

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
python STM_CoordConvLDAM_preIN8_DINO.py 0

# Downsample mode (non-tonal speech downsampled to 100k)
python STM_CoordConvLDAM_preIN8_DINO.py 1
```

Checkpoints saved to:
- `model/STM/CoordConvLDAM_preIN8_DINO_corpora_categories/standard/ckpt/<timestamp>/`
- `model/STM/CoordConvLDAM_preIN8_DINO_corpora_categories/downsample/ckpt/<timestamp>/`

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

| Metric | V2.8 (ResNet-18) | V2.8-DINO Target |
|--------|------------------|------------------|
| Test F1 (No TTA) | ~0.87 | 0.87-0.89 |
| Test F1 (TTA) | ~0.87-0.88 | 0.88-0.90 |
| music:non-vocal F1 | ~0.68-0.70 | 0.70+ |

### GPU Memory Requirements

| Metric | V2.8 (ResNet-18) | V2.8-DINO |
|--------|------------------|-----------|
| Model Parameters | ~12M | ~22M |
| Batch Size | 256 | 64 |
| Est. GPU Memory | ~4-6 GB | ~12-16 GB |

**Note:** Vision Transformers have quadratic memory complexity with sequence length due to self-attention.

## Comparison: ResNet-18 vs DINO ViT-Small

| Aspect | ResNet-18 | DINO ViT-Small |
|--------|-----------|----------------|
| Architecture | CNN | Vision Transformer |
| Pretraining | Supervised | Self-Supervised |
| Parameters | ~12M | ~22M |
| Feature Dim | 512 | 384 |
| Receptive Field | Local (grows with depth) | Global (each layer) |
| Computation | O(HW) | O(N²) where N = patches |
| GPU Memory | ~4-6 GB | ~12-16 GB |
| Training Speed | Fast | Moderate |

### When to Use DINO:
- **Better semantic features**: Self-supervised pretraining captures richer semantics
- **Interpretability**: Attention maps show what the model focuses on
- **Global context**: When long-range dependencies matter
- **Transfer learning**: Strong performance across diverse downstream tasks

### Potential Concerns:
- **Memory**: ~3× more memory than ResNet-18
- **Training time**: Slower due to self-attention
- **Small input size**: May not fully utilize ViT's strengths (designed for 224×224)

## Version History

| Version | Key Changes | Test F1 |
|---------|-------------|---------|
| V2.0 | ImageNet pretrained backbone (ResNet-18) | 0.8489 |
| V2.1 | Coord Attention, Layer Freezing | 0.8591 |
| V2.4 | Multi-Scale Fusion, SpecAugment | 0.8682 |
| V2.6 | Three-Scale Fusion, Hybrid Loss | 0.8709 |
| V2.6+TTA | Test-Time Augmentation | 0.8726 |
| V2.8 | Center Loss + Integrated TTA | ~0.87+ |
| **V2.8-DINO** | **DINO ViT-Small Backbone** | **TBD** |

## Technical Notes

### DINO Loading
DINO is loaded from Facebook's torch.hub:
```python
dino_model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=True)
```

### Position Embedding Interpolation
Original DINO uses 14×14 patch grid (196 patches). We interpolate to our 5×31 grid (155 patches):
```python
# Reshape to 2D
patch_pos_embed = orig.reshape(1, 14, 14, 384)
# Interpolate
patch_pos_embed = F.interpolate(patch_pos_embed, size=(5, 31))
# Flatten back
patch_pos_embed = patch_pos_embed.reshape(1, 155, 384)
```

### DINO vs ViT vs DeiT
- **ViT**: Original Vision Transformer (supervised on ImageNet-21K)
- **DeiT**: Data-efficient ViT (supervised on ImageNet-1K with distillation)
- **DINO**: Self-supervised ViT (no labels, self-distillation)

DINO often produces better features for transfer learning because:
1. No label bias from supervised training
2. Self-distillation encourages consistent representations
3. Learns to attend to semantically meaningful regions

### References

1. Caron, M., et al. (2021). "Emerging Properties in Self-Supervised Vision Transformers." ICCV.
2. Dosovitskiy, A., et al. (2021). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." ICLR.
3. Wen, Y., et al. (2016). "A discriminative feature learning approach for deep face recognition." ECCV.
