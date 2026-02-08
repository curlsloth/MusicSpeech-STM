# STM_CoordConvLDAM_preIN8: Center Loss for Intra-Class Compactness

## Version Summary

**Version: 2.8 (preIN8)**  
**Base: V2.6 (preIN6)**  
**Key Innovation: Center Loss + Integrated TTA Evaluation**

## What's New in V2.8

### 1. Center Loss for Intra-Class Compactness

Center Loss penalizes the distance between sample features and their learned class centers:

$$L_{center} = \frac{1}{2B} \sum_{i=1}^{B} ||f_i - c_{y_i}||^2$$

Where:
- $f_i$ is the 512-dim feature vector (before dropout/fc)
- $c_{y_i}$ is the learnable center for class $y_i$
- $B$ is the batch size

**Total Loss:**
$$L_{total} = L_{hybrid} + \lambda \times L_{center}$$

Where $L_{hybrid} = 0.7 \times L_{LDAM} + 0.3 \times L_{Focal}$ and $\lambda = 0.1$ (default).

#### Why Center Loss?
- **Intra-class compactness**: Pulls samples of the same class closer in feature space
- **Addresses high intra-class variance**: Particularly beneficial for `music:non-vocal` which has diverse instrumental timbres
- **Complementary to LDAM**: LDAM focuses on inter-class margins; Center Loss focuses on intra-class clustering

#### Implementation Details
```python
class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim):
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
    
    def forward(self, features, targets):
        centers_batch = self.centers[targets]
        diff = features - centers_batch
        loss = torch.sum(diff ** 2) / (2.0 * batch_size)
        return loss
```

- Centers are learnable parameters
- Separate SGD optimizer with LR=0.5 (higher than main optimizer)
- Feature dimension: 512 (output of fc1 layer before dropout)

### 2. Integrated TTA Evaluation

Test-Time Augmentation is now integrated into the training script:

| # | Augmentation | Description |
|---|--------------|-------------|
| 1 | Original | No augmentation |
| 2 | Time Flip | Reverse temporal axis |
| 3 | Freq Shift +2 | Roll frequency by +2 bins |
| 4 | Freq Shift -2 | Roll frequency by -2 bins |
| 5 | Time Shift +5 | Roll time by +5 frames |

**Soft Voting:** Average logits across all 5 augmentations, then argmax.

### 3. Confusion Matrix Analysis

The script now generates:
- **Text-based confusion matrix** in terminal (rows=true, cols=predicted)
- **Per-class confusion analysis** (most confused AS, most misclassified FROM)
- **Visual confusion matrix** saved as PNG with:
  - Normalized values (recall per row)
  - Raw counts overlay
- **Saved artifacts:**
  - `confusion_matrix.png` - Visual heatmap
  - `confusion_matrix.npy` - Raw matrix
  - `tta_predictions.npy` - TTA predictions
  - `tta_probs.npy` - TTA probability distributions

## Architecture Overview

```
Input: (B, 2, 20, 121) - 2-channel Difference Map (Symmetric + Asymmetric)
    ↓
CoordConv Stem (4-ch: 2 STM + 2 coords) → 64 ch
    ↓
Layer1 (64 ch) + Coordinate Attention
Layer2 (128 ch, stride=2) + Coordinate Attention → feat_layer2
Layer3 (256 ch, stride=2) + Squeeze-Excitation → feat_layer3
Layer4 (512 ch, stride=2) + Squeeze-Excitation → feat_layer4
    ↓
Three-Scale Fusion: cat(feat_layer2↓, feat_layer3, feat_layer4↑) → 896 ch → 512 ch
    ↓
Global Average Pool → (B, 512)
    ↓
Features (512-dim) → Center Loss computation
    ↓
Dropout(0.4) → FC(512 → 6) → Logits
```

## Training Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Base LR | 1e-4 | AdamW optimizer |
| Discriminative LR | Stem/L1: 0.1x, L2-3: 0.5x, L4/Head: 1.0x | |
| Weight Decay | 5e-4 | |
| Batch Size | 256 | |
| Epochs | 100 (max) | Early stopping: 20 patience |
| Layer Freezing | L1-L2 frozen first 10 epochs | |
| Mixup | α=0.4, 30% probability | |
| Loss | Hybrid LDAM (70%) + Focal (30%) | |
| **Center Loss λ** | **0.1** | **NEW in V2.8** |
| Center LR | 0.5 (SGD) | Separate optimizer |
| DRW | Enabled after epoch 30 | |

## STM Augmentation (SpecAugment-style)

```python
STMAugmentation(
    freq_mask_prob=0.3, freq_mask_width=3,
    time_mask_prob=0.3, time_mask_width=15,
    freq_shift_prob=0.2, max_freq_shift=3,
    time_shift_prob=0.2, max_time_shift=10
)
```

## Loss Components

### 1. LDAM Loss (70%)
Label-Distribution-Aware Margin loss with class-dependent margins:
$$m_j = \frac{C}{n_j^{1/4}}$$

### 2. Focal Loss (30%)
Hard example mining with γ=2:
$$FL = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

### 3. Center Loss (NEW)
$$L_{center} = \frac{1}{2B} \sum_i ||f_i - c_{y_i}||^2$$

### Combined Loss
$$L_{total} = 0.7 \cdot L_{LDAM} + 0.3 \cdot L_{Focal} + 0.1 \cdot L_{center}$$

## Usage

```bash
# Standard mode (full dataset)
python STM_CoordConvLDAM_preIN8.py 0

# Downsample mode (non-tonal speech downsampled to 100k)
python STM_CoordConvLDAM_preIN8.py 1
```

Checkpoints saved to:
- `model/STM/CoordConvLDAM_preIN8_corpora_categories/standard/ckpt/<timestamp>/`
- `model/STM/CoordConvLDAM_preIN8_corpora_categories/downsample/ckpt/<timestamp>/`

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

## Expected Results

Based on V2.6 baseline (Test F1: 0.8709, TTA: 0.8726):

| Metric | V2.6 | V2.8 Target |
|--------|------|-------------|
| Test F1 (No TTA) | 0.8709 | 0.87+ |
| Test F1 (TTA) | 0.8726 | 0.875+ |
| music:non-vocal F1 | 0.68 | 0.70+ |

Center Loss should particularly help:
- **music:non-vocal**: High intra-class variance (diverse instruments)
- **speech classes**: Similar acoustic characteristics with subtle differences

## Version History

| Version | Key Changes | Test F1 |
|---------|-------------|---------|
| V2.0 | ImageNet pretrained backbone | 0.8489 |
| V2.1 | Coord Attention, Layer Freezing | 0.8591 |
| V2.4 | Multi-Scale Fusion, SpecAugment | 0.8682 |
| V2.6 | Three-Scale Fusion, Hybrid Loss | 0.8709 |
| V2.6+TTA | Test-Time Augmentation | 0.8726 |
| **V2.8** | **Center Loss + Integrated TTA** | **TBD** |

## Theoretical Foundation

### Why Center Loss Works for STM

1. **High Intra-Class Variance**: Music:non-vocal includes piano, guitar, drums, orchestral - very different timbres but same class. Center Loss encourages clustering despite variance.

2. **Similar Inter-Class Features**: Speech:tonal and Music:vocal share melodic characteristics. LDAM pushes them apart; Center Loss ensures within-class samples don't drift toward the margin.

3. **Feature Space Geometry**: 
   - LDAM → Large-margin decision boundaries
   - Center Loss → Compact class clusters
   - Combined → Well-separated, compact clusters

### Reference

Wen et al., "A Discriminative Feature Learning Approach for Deep Face Recognition" (ECCV 2016)

## Notes

- Center Loss adds ~6 trainable parameters per class (512-dim centers × 6 classes = 3072 params)
- Negligible compute overhead (one L2 distance per sample per batch)
- λ=0.1 is conservative; can experiment with 0.05-0.2 if needed
