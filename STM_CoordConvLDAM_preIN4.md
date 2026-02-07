# STM_CoordConvLDAM_preIN4: Multi-Scale Fusion + Advanced Augmentation

**Version:** 2.4 (Multi-Scale + STM Augmentation + Layer Freezing)  
**Parent:** STM_CoordConvLDAM_preIN2 (V2.1)  
**Target:** 0.88-0.90 Macro F1 (Surpass V2.1's 0.8618, Approach SOTA 0.89+)  
**Date:** February 2026

---

## Executive Summary

STM_CoordConvLDAM_preIN4 (V2.4) builds on V2.1's success (0.8618 Test F1) by incorporating three key innovations: **Multi-Scale Feature Fusion**, **STM-specific augmentation**, and **layer freezing during warmup**. These enhancements target the remaining performance gap to SOTA (0.89+) by improving feature representation, preventing overfitting, and preserving ImageNet pretrained knowledge.

### V2.1 Performance Summary

**V2.1 (STM_CoordConvLDAM_preIN2):**
- Test Macro F1: **0.8618** (+0.24% over V2.0's 0.8594)
- Best Val F1: 0.8500
- Early stopped at epoch 50
- **Issues identified:**
  - speech:tonal recall only 0.68 (needs improvement)
  - music:non-vocal F1 only 0.67 (high variance class)
  - Still gap to SOTA 0.89

### V2.4 Enhancements Strategy

| Enhancement | Target Issue | Expected Gain |
|------------|--------------|---------------|
| Multi-Scale Fusion | Missing mid-level features | +1-2% F1 |
| STM Augmentation | Limited data diversity | +0.5-1% F1 |
| Layer Freezing | Pretrained knowledge loss | +0.5-1% F1 |
| **Combined** | | **+2-4% F1** |

---

## V2.4 Innovation #1: Multi-Scale Feature Fusion

### Problem Analysis

V2.1's architecture uses only layer4 (512ch, 3×16) features for classification. This loses mid-level information from layer3 (256ch, 5×31) that may be crucial for:

1. **speech:tonal** - Pitch modulation patterns appear at intermediate scales
2. **music:non-vocal** - Rhythmic/timbral features span multiple resolutions
3. **music:vocal** - Vocal/instrumental distinction needs both fine and coarse features

### Solution: Combining Layer3 + Layer4

```
Layer3: (B, 256, 5, 31) → Mid-level patterns
                           ↘
                            Upsample + Concat + 1×1 Conv → (B, 512, 5, 31)
                           ↗
Layer4: (B, 512, 3, 16) → High-level semantics
                           (upsampled to 5×31)
```

**Implementation:**
```python
# In PretrainedSTMResNet18.__init__:
self.multi_scale_fusion = nn.Sequential(
    nn.Conv2d(256 + 512, 512, kernel_size=1, bias=False),  # Channel reduction
    nn.BatchNorm2d(512),
    nn.ReLU(inplace=True)
)

# In forward():
feat_layer4_up = F.interpolate(feat_layer4, size=feat_layer3.shape[-2:], 
                               mode='bilinear', align_corners=False)
feat_concat = torch.cat([feat_layer3, feat_layer4_up], dim=1)  # (B, 768, 5, 31)
feat_fused = self.multi_scale_fusion(feat_concat)  # (B, 512, 5, 31)
```

### Why This Helps

| Feature Type | Layer3 (256ch) | Layer4 (512ch) | Fused (512ch) |
|--------------|----------------|----------------|---------------|
| **Spatial Resolution** | 5×31 (high) | 3×16 (low) | 5×31 (high) |
| **Semantic Level** | Mid-level patterns | Abstract concepts | Both |
| **Pitch modulation** | ✓ Strong | ✗ Lost | ✓ Preserved |
| **Rhythmic patterns** | ✓ Partial | ✓ Strong | ✓ Strong |
| **Class discrimination** | Good for tonal | Good for non-vocal | Best combined |

**Expected Impact:**
- speech:tonal: +3-5% recall (pitch patterns from layer3)
- music:non-vocal: +2-3% F1 (rhythm from layer4 + timbre from layer3)
- Overall: +1-2% Macro F1

---

## V2.4 Innovation #2: STM-Specific Augmentation

### Problem Analysis

V2.1 uses only Mixup augmentation (α=0.4, 30% probability). This limits data diversity and doesn't exploit the specific structure of STM features.

### Solution: SpecAugment-Style Masking + Axis Shifts

Inspired by SpecAugment (Park et al., 2019) for speech recognition, we adapt the technique for STM modulation features:

#### 2a. Frequency Masking (Spectral Modulation Axis)

Masks random contiguous bands along the spectral modulation axis (20 bins):

```
STM Input (2, 20, 121):
┌─────────────────────────────────────────┐
│ * * * * * * * * * * * * * * * ... │ ω=0  (DC)
│ * * * * * * * * * * * * * * * ... │ ω=1
│ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ... │ ω=2  ← MASKED
│ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ... │ ω=3  ← MASKED
│ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ... │ ω=4  ← MASKED
│ * * * * * * * * * * * * * * * ... │ ω=5
│ ...                                 │
└─────────────────────────────────────────┘
```

**Parameters:**
- `freq_mask_prob=0.3` - Apply masking with 30% probability
- `freq_mask_width=3` - Maximum 3 bins masked at once (15% of spectrum)

**Rationale:**
- Forces model to classify without relying on specific frequency bands
- Improves robustness to speaker variation (different formant frequencies)
- Simulates natural variation in spectral tilt/emphasis

#### 2b. Time Masking (Temporal Modulation Axis)

Masks random contiguous bands along the temporal modulation axis (121 bins):

```
STM Input (2, 20, 121):
         0   10  20  30  40  50  60  70  80  90  100 110 120
ω=0  [ * * * * * 0 0 0 0 0 0 0 0 0 0 * * * * * * * * * ]
ω=1  [ * * * * * 0 0 0 0 0 0 0 0 0 0 * * * * * * * * * ]
...                  ↑ MASKED (width=15) ↑
```

**Parameters:**
- `time_mask_prob=0.3` - Apply masking with 30% probability
- `time_mask_width=15` - Maximum 15 bins masked (12% of time axis)

**Rationale:**
- Forces model to classify without relying on specific temporal modulation rates
- Improves robustness to speaking rate variation
- Simulates natural variation in rhythm/tempo

#### 2c. Frequency Shift (Cyclic)

Cyclically shifts features along the spectral modulation axis:

```
Before:  [ω0, ω1, ω2, ..., ω18, ω19]
After:   [ω17, ω18, ω19, ω0, ω1, ..., ω16]  (shift=+3)
```

**Parameters:**
- `freq_shift_prob=0.2` - Apply shift with 20% probability
- `max_freq_shift=3` - Maximum ±3 bins shift

**Rationale:**
- Simulates variation in vocal tract size (formant scaling)
- Helps with pitch range variation (tonal languages)
- Improves generalization across speakers

#### 2d. Time Shift (Cyclic)

Cyclically shifts features along the temporal modulation axis:

**Parameters:**
- `time_shift_prob=0.2` - Apply shift with 20% probability
- `max_time_shift=10` - Maximum ±10 bins shift

**Rationale:**
- Simulates tempo/rhythm variation
- Helps with speaking rate differences
- Improves robustness to onset alignment

### Combined Augmentation Pipeline

```python
class STMAugmentation:
    def __call__(self, x):
        # x: (batch, 2, 20, 121)
        x = self.freq_mask(x)   # 30% prob, mask up to 3 bins
        x = self.time_mask(x)   # 30% prob, mask up to 15 bins
        x = self.freq_shift(x)  # 20% prob, shift ±3 bins
        x = self.time_shift(x)  # 20% prob, shift ±10 bins
        return x

# Applied during training before mixup
if use_stm_aug:
    data = self.stm_augment(data)
```

**Expected Impact:**
- Reduced overfitting (more diverse training data)
- Better minority class performance (speech:tonal, music:non-vocal)
- Overall: +0.5-1% Macro F1

---

## V2.4 Innovation #3: Layer Freezing Strategy

### Problem Analysis

V2.1 trains all layers from epoch 1, which can:
1. Corrupt ImageNet pretrained features before attention modules stabilize
2. Waste gradient updates on already-good low-level features
3. Cause oscillation in early training

### Solution: Freeze Layer1-2 for First 10 Epochs

```
Epochs 1-10 (Warmup):
┌─────────────────────────────────────────┐
│ conv1, bn1  │ FROZEN - preserve ImageNet │
│ layer1 (CA) │ FROZEN - preserve features │
│ layer2 (CA) │ FROZEN - preserve features │
├─────────────────────────────────────────┤
│ layer3 (SE) │ TRAINING - adapt semantic  │
│ layer4 (SE) │ TRAINING - adapt semantic  │
│ fusion      │ TRAINING - learn fusion    │
│ fc          │ TRAINING - learn classifier│
│ attention   │ TRAINING - learn selection │
└─────────────────────────────────────────┘

Epochs 11+ (Full Training):
┌─────────────────────────────────────────┐
│ ALL LAYERS  │ TRAINING - fine-tune all  │
└─────────────────────────────────────────┘
```

### Implementation

```python
# In PretrainedSTMResNet18:
def freeze_early_layers(self):
    """Freeze stem, layer1, and layer2 for warmup phase."""
    for param in self.conv1.parameters():
        param.requires_grad = False
    for param in self.bn1.parameters():
        param.requires_grad = False
    for param in self.layer1.parameters():
        param.requires_grad = False
    for param in self.layer2.parameters():
        param.requires_grad = False

def unfreeze_all_layers(self):
    """Unfreeze all layers after warmup phase."""
    for param in self.parameters():
        param.requires_grad = True

# In Trainer:
def _unfreeze_if_needed(self, epoch):
    if self.layers_frozen and epoch > self.freeze_epochs:
        self.model.unfreeze_all_layers()
        self.layers_frozen = False
```

### Why This Helps

**Benefits of freezing:**
1. **Attention stabilization:** CA/SE modules in layer1-2 train without affecting the features they attend to
2. **Gradient efficiency:** Focus gradients on layers that need adaptation
3. **Feature preservation:** Maintain ImageNet texture bias during early training

**Why only 10 epochs:**
- Long enough for attention to stabilize
- Short enough to allow full adaptation
- Balances preservation vs. adaptation

**Expected Impact:**
- Faster early convergence (fewer parameters to train)
- Better final performance (preserved pretrained knowledge)
- Overall: +0.5-1% Macro F1

---

## Architecture Comparison

| Component | V2.1 (preIN2) | V2.4 (preIN4) |
|-----------|---------------|---------------|
| **Backbone** | ImageNet ResNet-18 | ImageNet ResNet-18 |
| **Stem** | 4-ch CoordConv | 4-ch CoordConv |
| **Attention** | CA (L1-2), SE (L3-4) | CA (L1-2), SE (L3-4) |
| **Multi-Scale Fusion** | No | **Yes (L3+L4)** ✓ |
| **Block Dropout** | 0.05 | 0.05 |
| **Head Dropout** | 0.4 | 0.4 |
| **STM Augmentation** | No | **Yes (SpecAugment-style)** ✓ |
| **Layer Freezing** | No | **Yes (10 epochs)** ✓ |
| **Mixup α** | 0.4 | 0.4 |
| **Weight Decay** | 5e-4 | 5e-4 |
| **Parameters** | ~11.5M | ~11.9M (+3.5%) |
| **Val F1** | 0.8500 | Target: **0.870+** |
| **Test F1** | 0.8618 | Target: **0.880-0.900** |

---

## Expected Performance Gains

### Quantitative Targets

**Overall:**
- Val F1: 0.8500 → **0.870-0.880** (+2-3%)
- Test F1: 0.8618 → **0.880-0.900** (+1.8-3.8%)
- Val-Test Gap: Maintained ~0.01

### Per-Class Improvements (Test Set)

| Class | V2.1 Recall | V2.4 Target | Δ | Mechanism |
|-------|-------------|-------------|---|-----------|
| **speech:tonal** | 0.68 | **0.74-0.78** | +6-10% | Multi-scale captures pitch patterns; freq mask improves robustness |
| **music:non-vocal** | 0.65 | **0.72-0.76** | +7-11% | Fusion combines rhythm (L4) + timbre (L3); time mask helps diversity |
| **music:vocal** | 0.85 | **0.87-0.89** | +2-4% | Better fusion of vocal harmonics |
| **speech:non-tonal** | 0.99 | **0.99** | 0% | Already saturated |
| **env:urban** | 0.99 | **0.99** | 0% | Already excellent |
| **env:wildlife** | 0.95 | **0.96-0.97** | +1-2% | Augmentation helps sparse class |

### Training Dynamics Prediction

```
Epochs 1-10: Warmup (frozen layer1-2)
  - Attention modules learn without disturbing features
  - Multi-scale fusion converges
  - STM augmentation provides diverse samples
  - Val F1: 0.78 → 0.84

Epochs 11-20: Unfreezing + Refinement
  - Layer1-2 fine-tune with discriminative LR
  - All components adapt together
  - Val F1: 0.84 → 0.86

Epochs 20-40: Main Training
  - Steady improvement with augmentation
  - LR decay kicks in
  - Val F1: 0.86 → 0.87

Epochs 40-50+: DRW Phase
  - Deferred Reweighting activates (epoch 50)
  - Minority class improvement
  - Val F1: 0.87 → 0.875

Expected convergence: ~60-80 epochs
```

---

## Training Configuration

### Hyperparameters

```python
# Model
num_classes = 6
dropout = 0.4          # Head dropout
block_dropout = 0.05   # Block dropout

# Optimizer
base_lr = 1e-4
weight_decay = 5e-4

# Discriminative LR (same as V2.1)
lr_stem_layer1 = 0.1 * base_lr  # 1e-5
lr_layer2_layer3 = 0.5 * base_lr  # 5e-5
lr_layer4_fusion_head = 1.0 * base_lr  # 1e-4
lr_attention = 1.0 * base_lr    # 1e-4
lr_batchnorm = 1.0 * base_lr    # 1e-4

# Scheduler
scheduler = ReduceLROnPlateau(mode='max', factor=0.5, patience=7, min_lr=1e-6)

# Loss
criterion = LDAMLoss(max_m=0.5, s=30, label_smooth=0.05)
drw_start_epoch = 50  # Deferred Reweighting

# Augmentation
mixup_alpha = 0.4
mixup_prob = 0.3
stm_augmentation = True
freeze_epochs = 10

# Training
batch_size = 256
max_epochs = 100
early_stop_patience = 20
grad_clip = 1.0
```

### STM Augmentation Parameters

```python
stm_augment = STMAugmentation(
    # Frequency masking (spectral modulation axis)
    freq_mask_prob=0.3,      # 30% probability
    freq_mask_width=3,       # Max 3 bins (15% of ω axis)
    
    # Time masking (temporal modulation axis)
    time_mask_prob=0.3,      # 30% probability
    time_mask_width=15,      # Max 15 bins (12% of Ω axis)
    
    # Frequency shift (cyclic)
    freq_shift_prob=0.2,     # 20% probability
    max_freq_shift=3,        # Max ±3 bins
    
    # Time shift (cyclic)
    time_shift_prob=0.2,     # 20% probability
    max_time_shift=10        # Max ±10 bins
)
```

---

## Running the Model

### Command

```bash
# Standard training (full dataset)
python STM_CoordConvLDAM_preIN4.py 0

# Downsampled training (100k non-tonal speech)
python STM_CoordConvLDAM_preIN4.py 1
```

### Expected Output

```
Using device: cuda
GPU: NVIDIA H200
Memory: 80.00 GB
Checkpoint directory: model/STM/CoordConvLDAM_preIN4_corpora_categories/standard/ckpt/2026-02-06_XX-XX

Loading ImageNet-pretrained ResNet-18...
✓ Successfully loaded ImageNet pretrained weights
✓ Cloned ImageNet weights to 4-channel CoordConv stem
✓ Created STM-adapted ResNet-18 V2.4 with ImageNet pretrained weights (6 classes)
Total parameters: 11,900,XXX
Trainable parameters: 11,900,XXX

V2.4 Enhancements:
  • Multi-Scale Fusion: layer3 (256ch) + layer4 (512ch)
  • Attention: CA (layer1-2), SE (layer3-4)
  • Block dropout: 0.05
  • Head dropout: 0.4
  • Layer freezing support (first 10 epochs)
  • STM augmentation: SpecAugment-style masking + axis shifts

Discriminative Learning Rates:
  stem_layer1         : LR = 0.000010 (XX params)
  layer2_layer3       : LR = 0.000050 (XX params)
  layer4              : LR = 0.000100 (XX params)
  multi_scale_fusion  : LR = 0.000100 (XX params)
  head                : LR = 0.000100 (XX params)
  attention           : LR = 0.000100 (XX params)
  batchnorm           : LR = 0.000100 (XX params)

✓ Frozen: conv1, bn1, layer1, layer2
→ Layer1-2 frozen for first 10 epochs

V2.4 Training Configuration:
  • Multi-Scale Fusion: layer3 (256ch) + layer4 (512ch)
  • Discriminative LR: Stem/L1 (0.1x), L2-3 (0.5x), L4/Head (1.0x)
  • Layer Freezing: layer1-2 frozen for first 10 epochs
  • STM Augmentation: SpecAugment-style masking + axis shifts
  • Weight decay: 5e-4
  • Mixup alpha: 0.4
  • LDAM + DRW: Enabled (epoch 50+)
  • Early stopping: 20 epochs patience

============================================================
Starting training...
============================================================
```

---

## Comparison with Previous Versions

| Version | Key Changes | Test F1 | Notes |
|---------|-------------|---------|-------|
| V2.0 (preIN) | ImageNet pretrained + CoordConv + LDAM | 0.8594 | Baseline pretrained |
| V2.1 (preIN2) | + Attention (CA/SE) + Discriminative LR | 0.8618 | +0.24% |
| **V2.4 (preIN4)** | + Multi-Scale Fusion + STM Aug + Freezing | **Target: 0.88-0.90** | **+2-4% expected** |
| SOTA | Unknown (research benchmark) | 0.89+ | Target to match |

---

## Potential Further Improvements

If V2.4 still underperforms SOTA:

1. **More aggressive augmentation:**
   - Increase mask probabilities/widths
   - Add noise injection
   - Add pitch/tempo perturbation before STM extraction

2. **Architecture changes:**
   - Try EfficientNet or ConvNeXt backbone (different inductive bias)
   - Add transformer blocks for global context
   - Experiment with larger models (ResNet-34/50)

3. **Training strategies:**
   - Longer training with cosine annealing
   - Self-training / pseudo-labeling
   - Contrastive learning pretraining on STM

4. **Ensemble:**
   - Combine V2.1 + V2.4 predictions
   - Multi-seed ensemble averaging

---

## References

1. **Multi-Scale Fusion:** Feature Pyramid Networks (Lin et al., 2017)
2. **SpecAugment:** Park et al., "SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition", Interspeech 2019
3. **Layer Freezing:** "How transferable are features in deep neural networks?" (Yosinski et al., 2014)
4. **Coordinate Attention:** Hou et al., CVPR 2021
5. **Squeeze-and-Excitation:** Hu et al., CVPR 2018
6. **LDAM Loss:** Cao et al., NeurIPS 2019
