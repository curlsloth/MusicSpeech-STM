# STM_CoordConvLDAM_preIN2: Attention-Enhanced Transfer Learning

**Version:** 2.1 (Attention + Discriminative LR)  
**Parent:** STM_CoordConvLDAM_preIN (V2.0)  
**Target:** 0.87-0.885 Macro F1 (Beat V4's 0.865, Approach SOTA 0.89)  
**Date:** February 2026

---

## Executive Summary

STM_CoordConvLDAM_preIN2 represents a strategic enhancement of V2.0 by addressing its primary weakness: **overfitting due to insufficient regularization and uniform learning rates across all layers**. By combining ImageNet pretraining with proven architectural innovations from V4 (attention mechanisms) and implementing discriminative learning rates, V2.1 aims to achieve the best of both worlds.

### Key Problem: V2.0 Overfitting

**V2.0 Performance (STM_CoordConvLDAM_preIN):**
- Test Macro F1: **0.8594**
- Best Val F1: 0.8505
- Early stopped at epoch 32
- Training loss: 1.0133 → 0.6800 (continuous decrease)
- Val loss: 1.5274 → 1.5349 (plateau/slight increase)
- **Diagnosis:** Model memorizing training data instead of learning generalizable features

**Why V2.0 Underperformed V4 (0.865):**
1. No attention mechanisms → Uniform feature treatment, no adaptive selection
2. Single learning rate → Pretrained layers updated too aggressively
3. Limited regularization → Only head dropout (0.3), no block dropout
4. Weak augmentation → Mixup α=0.3

---

## V2.1 Solution: Four-Pronged Enhancement Strategy

### 1. **Attention Mechanisms (from V4)**

**Added Components:**
- **Coordinate Attention (CA)** for layer1 + layer2
- **Squeeze-Excitation (SE)** for layer3 + layer4

**Why This Combination:**

#### Early Layers (1-2): Coordinate Attention
```
Position-aware attention for spatial patterns
├─ Layer1 (20×121): High-resolution STM features
│  ├─ CA captures: Row-wise (spectral) + Column-wise (temporal) dependencies
│  └─ Critical for: Position-dependent class distinctions
│
└─ Layer2 (10×61): Mid-resolution features
   ├─ CA maintains: Spatial structure awareness
   └─ Benefits: Tonal speech (specific frequency bands), Music vocal (harmonic patterns)
```

**Coordinate Attention Math:**
```
Input: X ∈ R^(B×C×H×W)

1. Spatial Pooling:
   X_h = AvgPool_W(X)  → (B, C, H, 1)  [pool along width]
   X_w = AvgPool_H(X)  → (B, C, 1, W)  [pool along height]

2. Shared Transform:
   Y = Conv1×1([X_h; X_w])  → (B, C/r, H+W, 1)
   Y = BatchNorm(Y)
   Y = ReLU(Y)

3. Split and Attend:
   A_h = σ(Conv1×1(Y[:,:,:H,:]))  → (B, C, H, 1)
   A_w = σ(Conv1×1(Y[:,:,H:,:]))  → (B, C, 1, W)

4. Output:
   Out = X ⊙ A_h ⊙ A_w  [element-wise multiplication]
```

**Why CA for STM:**
- **Position Dependence:** Low spectral modulation (ω≈0) = speech formants vs. high spectral modulation (ω>4) = music timbre
- **Asymmetric Attention:** STM is 20×121 (tall and narrow) → different importance for height vs. width
- **Long-Range Context:** Captures dependencies across entire frequency/time axis

#### Late Layers (3-4): Squeeze-and-Excitation
```
Channel-wise attention for semantic features
├─ Layer3 (5×31): Semantic feature extraction
│  ├─ SE learns: Which feature channels matter for classification
│  └─ Example: "Channel 42 encodes rhythmic patterns → boost for music"
│
└─ Layer4 (3×16): Abstract representations
   ├─ SE refines: Final feature selection before classification
   └─ Benefits: Music:non-vocal (high intra-class variance), Speech:tonal (minority class)
```

**Squeeze-and-Excitation Math:**
```
Input: X ∈ R^(B×C×H×W)

1. Squeeze: Global context
   Z = GlobalAvgPool(X)  → (B, C)

2. Excitation: Channel relationships
   S = σ(FC2(ReLU(FC1(Z))))  → (B, C)
   where FC1: C → C/r, FC2: C/r → C

3. Rescale:
   Out = X ⊙ S.view(B, C, 1, 1)
```

**Why SE for Late Layers:**
- **Channel Semantics:** Late layers encode abstract concepts (e.g., "rhythmicity", "tonality")
- **Low Spatial Info:** 3×16 features → position less important than channel identity
- **Variance Handling:** SE adapts per-sample → helps music:non-vocal diversity

**Combined Architecture:**
```
Input (B, 2, 20, 121)
    ↓
[Stem: CoordConv 4-ch]
    ↓
[Layer1: BasicBlock + CA] × 2  ← Position-aware
    ↓
[Layer2: BasicBlock + CA] × 2  ← Position-aware
    ↓
[Layer3: BasicBlock + SE] × 2  ← Channel selection
    ↓
[Layer4: BasicBlock + SE] × 2  ← Channel selection
    ↓
[GlobalPool → FC]
```

---

### 2. **Block Dropout (0.05)**

**Problem:** V2.0 had dropout only in the classification head, allowing residual blocks to overfit.

**Solution:** Add Dropout2D(0.05) after each residual block's attention module.

**Implementation:**
```python
class BasicBlockWithAttention(nn.Module):
    def forward(self, x):
        out = self.block(x)           # Pretrained residual block
        if self.attention:
            out = self.attention(out)  # CA or SE
        if self.dropout:
            out = self.dropout(out)    # Dropout2D(0.05)
        return out
```

**Why 0.05 (not 0.1 or 0.2):**
- Pretrained weights are valuable → minimal corruption
- Cumulative effect: 8 blocks × 0.05 = significant regularization
- Empirical: V4 used 0.05 successfully

**Expected Impact:**
- Reduces co-adaptation of feature maps
- Forces redundant representations
- Val-test gap reduction: Δ(Val F1 - Test F1) from 0.009 → ~0.003

---

### 3. **Discriminative Learning Rates**

**Problem:** V2.0 used uniform LR=1e-4 for all layers → pretrained filters updated too aggressively, losing ImageNet knowledge.

**Solution:** Layer-wise learning rates based on pretraining status:

| Layer Group | LR Multiplier | Rationale |
|-------------|---------------|-----------|
| **Stem + Layer1** | 0.1× | Heavily pretrained (ImageNet conv1 + early blocks), need minimal adaptation |
| **Layer2 + Layer3** | 0.5× | Pretrained but deeper, moderate adaptation for STM-specific features |
| **Layer4 + Head** | 1.0× | Late semantic features + new classifier, full adaptation needed |
| **Attention Modules** | 1.0× | Newly initialized (random), need full LR to train from scratch |
| **BatchNorm** | 1.0× | Statistics must adapt to STM data distribution |

**Implementation:**
```python
param_groups = [
    {'params': stem + layer1_conv_params, 'lr': lr * 0.1, 'name': 'stem_layer1'},
    {'params': layer2_conv + layer3_conv,  'lr': lr * 0.5, 'name': 'layer2_layer3'},
    {'params': layer4_conv + fc_params,    'lr': lr * 1.0, 'name': 'layer4_head'},
    {'params': all_attention_params,       'lr': lr * 1.0, 'name': 'attention'},
    {'params': all_batchnorm_params,       'lr': lr * 1.0, 'name': 'batchnorm'}
]
optimizer = AdamW(param_groups, weight_decay=5e-4)
```

**Theoretical Justification:**

**Transfer Learning Principle:** "Features learned on large datasets (ImageNet) are generic in early layers, task-specific in late layers"

- **Early layers (stem/layer1):** Detect edges, textures, Gabor-like filters
  - **ImageNet optimized these for 1000 classes** → broadly useful
  - **STM task:** Same low-level ripple patterns
  - **Strategy:** Freeze knowledge, fine-tune minimally

- **Middle layers (layer2-3):** Detect object parts, texture combinations
  - **Partially transferable:** ImageNet textures ≈ STM modulation patterns
  - **Strategy:** Moderate adaptation

- **Late layers (layer4):** Task-specific semantic features
  - **ImageNet:** "Cat face", "Car wheel"
  - **STM:** "Tonal sweep", "Rhythmic pulse"
  - **Strategy:** Full adaptation (but still better than random init)

**Expected Impact:**
- Prevents catastrophic forgetting of pretrained features
- Faster convergence (pretrained layers already near optimal)
- Better generalization (low-level features more robust)

**Comparison to V2.0:**
```
V2.0 (Uniform LR):
  Epoch 1: Layer1 updates Δθ₁ = -lr * ∇L  (too aggressive!)
  Result: Overwrites ImageNet filters, loses texture bias

V2.1 (Discriminative LR):
  Epoch 1: Layer1 updates Δθ₁ = -0.1*lr * ∇L  (conservative)
  Result: Preserves pretrained knowledge, adapts slowly
```

---

### 4. **Stronger Regularization**

#### Increased Head Dropout: 0.3 → 0.4
- **Rationale:** Classification head connects 512-dim features to 6 classes
- **Risk:** High capacity → memorization
- **Solution:** More aggressive dropout before FC layer
- **Trade-off:** Slightly slower convergence, but better generalization

#### Increased Weight Decay: 2e-4 → 5e-4
- **Rationale:** 11M parameters, large dataset (770k train samples)
- **Effect:** L2 penalty on weight magnitudes
  - Loss = LDAM_Loss + 5e-4 * Σ(θ²)
- **Benefits:** 
  - Prevents extreme weights
  - Smoother decision boundaries
  - Reduces overfitting to noisy labels

#### Stronger Mixup: α=0.3 → α=0.4
- **Mixup Equation:**
  ```
  λ ~ Beta(α, α)
  x_mixed = λ * x_i + (1-λ) * x_j
  y_mixed = λ * y_i + (1-λ) * y_j
  ```

- **Effect of α=0.4 (vs. 0.3):**
  - Higher α → More uniform λ distribution
  - More samples with λ ≈ 0.5 (equal mixing)
  - Smoother interpolation between classes

- **Why Beneficial:**
  - STM boundary classes (speech:tonal ↔ music:vocal) are ambiguous
  - Mixup creates synthetic examples in decision boundary
  - Forces model to learn smooth transitions

---

## Architecture Comparison

| Component | V2.0 (preIN) | V4 (Best) | V2.1 (preIN2) |
|-----------|--------------|-----------|---------------|
| **Backbone** | ImageNet ResNet-18 | From-scratch ResNet | ImageNet ResNet-18 |
| **Stem** | 4-ch CoordConv | 2-ch CoordConv | 4-ch CoordConv |
| **Attention** | None | CA (L1-2), SE (L3-4) | CA (L1-2), SE (L3-4) ✓ |
| **Block Dropout** | 0 | 0.05 | 0.05 ✓ |
| **Head Dropout** | 0.3 | 0.3 | 0.4 ✓ |
| **Learning Rate** | 1e-4 (uniform) | 1e-4 (uniform) | Discriminative ✓ |
| **Weight Decay** | 2e-4 | 2e-4 | 5e-4 ✓ |
| **Mixup α** | 0.3 | 0.3 | 0.4 ✓ |
| **Parameters** | ~11.2M | ~11.2M | ~11.5M (+3% for attention) |
| **Val F1** | 0.8505 | **0.8623** | Target: **0.870+** |
| **Test F1** | 0.8594 | **0.8650** | Target: **0.875-0.885** |

---

## Expected Performance Gains

### Quantitative Targets

**Overall:**
- Val F1: 0.8505 → **0.870-0.875** (+2.0-2.5%)
- Test F1: 0.8594 → **0.875-0.885** (+1.5-2.5%)
- Val-Test Gap: 0.009 → **0.003-0.005** (reduced overfitting)

**Per-Class Improvements (Test Set):**

| Class | V2.0 Recall | V2.1 Target | Δ | Mechanism |
|-------|-------------|-------------|---|-----------|
| **speech:tonal** | 0.65 | **0.70-0.72** | +5-7% | CA detects pitch modulation peaks, SE emphasizes tonal channels |
| **music:non-vocal** | 0.69 | **0.73-0.76** | +4-7% | SE handles high intra-class variance (jazz vs. EDM), stronger mixup smooths boundaries |
| **music:vocal** | 0.82 | **0.84-0.86** | +2-4% | Attention improves vocal vs. non-vocal distinction, reduced overfitting |
| **speech:non-tonal** | 0.99 | **0.99** | 0% | Already saturated (majority class) |
| **env:urban** | 0.97 | **0.97** | 0% | Already excellent (distinctive patterns) |
| **env:wildlife** | 0.97 | **0.97** | 0% | Already excellent |

### Qualitative Improvements

1. **Better Feature Hierarchy:**
   - Early layers: Preserve ImageNet texture bias (Gabor filters)
   - Late layers: Adapt to STM-specific semantics (tonality, rhythm)
   - **Result:** More interpretable, transferable representations

2. **Adaptive Per-Sample Processing:**
   - Attention dynamically weights features based on input
   - **Example:** For classical music (repetitive structure) → SE boosts structure-encoding channels
   - **Example:** For tonal speech (pitch variation) → CA emphasizes spectral modulation axis

3. **Reduced Overfitting:**
   - Block dropout: Forces distributed representations
   - Discriminative LR: Prevents catastrophic forgetting
   - Stronger mixup: Smooths decision boundaries
   - **Result:** Better generalization to unseen corpora/speakers

---

## Training Configuration

### Hyperparameters

```python
# Model
num_classes = 6
dropout = 0.4          # Head dropout (↑ from 0.3)
block_dropout = 0.05   # Block dropout (new)

# Optimizer
base_lr = 1e-4
weight_decay = 5e-4    # ↑ from 2e-4

# Discriminative LR
lr_stem_layer1 = 0.1 * base_lr  # 1e-5
lr_layer2_layer3 = 0.5 * base_lr  # 5e-5
lr_layer4_head = 1.0 * base_lr  # 1e-4
lr_attention = 1.0 * base_lr    # 1e-4
lr_batchnorm = 1.0 * base_lr    # 1e-4

# Scheduler
scheduler = ReduceLROnPlateau(mode='max', factor=0.5, patience=7, min_lr=1e-6)

# Loss
criterion = LDAMLoss(max_m=0.5, s=30, label_smooth=0.05)
drw_start_epoch = 50  # Deferred Reweighting

# Augmentation
mixup_alpha = 0.4      # ↑ from 0.3
mixup_prob = 0.3

# Training
batch_size = 256
max_epochs = 100
early_stop_patience = 20
grad_clip = 1.0
```

### Training Dynamics

**Expected Convergence Pattern:**

```
Epochs 1-10: Rapid adaptation
  - Attention modules learn from scratch
  - Layer4 + Head adapt to STM semantics
  - Layer1-3 fine-tune slowly (discriminative LR)
  - Train loss: 2.5 → 1.2
  - Val F1: 0.80 → 0.84

Epochs 10-30: Refinement
  - Attention stabilizes
  - Pretrained features adapt gradually
  - Train loss: 1.2 → 0.8
  - Val F1: 0.84 → 0.87

Epochs 30-50: Plateau
  - Marginal improvements
  - Scheduler reduces LR if plateau detected
  - Train loss: 0.8 → 0.65
  - Val F1: 0.87 → 0.873

Epochs 50+: DRW activated
  - Reweight loss for minority classes
  - Final refinement for speech:tonal, music:non-vocal
  - Train loss: 0.65 → 0.60
  - Val F1: 0.873 → 0.875

Early stopping: ~60-70 epochs (vs. V2.0's 32)
```

---

## Implementation Details

### Key Code Blocks

#### 1. BasicBlockWithAttention Wrapper
```python
class BasicBlockWithAttention(nn.Module):
    def __init__(self, block, attention_type='CA', dropout=0.05):
        super().__init__()
        self.block = block  # Pretrained BasicBlock
        
        out_channels = block.conv2.out_channels
        if attention_type == 'CA':
            self.attention = CoordinateAttention(out_channels)
        elif attention_type == 'SE':
            self.attention = SqueezeExcitation(out_channels)
        else:
            self.attention = None
        
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None
    
    def forward(self, x):
        out = self.block(x)  # Original ResNet block
        if self.attention:
            out = self.attention(out)  # Apply attention
        if self.dropout:
            out = self.dropout(out)  # Regularization
        return out
```

#### 2. Layer Construction with Attention
```python
# In PretrainedSTMResNet18.__init__:
self.layer1 = nn.Sequential(
    *[BasicBlockWithAttention(block, attention_type='CA', dropout=0.05) 
      for block in pretrained_model.layer1]
)

self.layer2 = nn.Sequential(
    *[BasicBlockWithAttention(block, attention_type='CA', dropout=0.05) 
      for block in pretrained_model.layer2]
)

self.layer3 = nn.Sequential(
    *[BasicBlockWithAttention(block, attention_type='SE', dropout=0.05) 
      for block in pretrained_model.layer3]
)

self.layer4 = nn.Sequential(
    *[BasicBlockWithAttention(block, attention_type='SE', dropout=0.05) 
      for block in pretrained_model.layer4]
)
```

#### 3. Discriminative LR Setup
```python
# In Trainer.__init__:
param_groups = [
    {
        'params': list(model.conv1.parameters()) + 
                  [p for block in model.layer1 for p in block.block.parameters()],
        'lr': lr * 0.1, 
        'name': 'stem_layer1'
    },
    {
        'params': [p for block in model.layer2 for p in block.block.parameters()] +
                  [p for block in model.layer3 for p in block.block.parameters()],
        'lr': lr * 0.5, 
        'name': 'layer2_layer3'
    },
    {
        'params': [p for block in model.layer4 for p in block.block.parameters()] +
                  list(model.fc.parameters()) + list(model.dropout.parameters()),
        'lr': lr * 1.0, 
        'name': 'layer4_head'
    },
    {
        'params': [p for block in model.layer1 for p in block.attention.parameters() if block.attention] +
                  [p for block in model.layer2 for p in block.attention.parameters() if block.attention] +
                  [p for block in model.layer3 for p in block.attention.parameters() if block.attention] +
                  [p for block in model.layer4 for p in block.attention.parameters() if block.attention],
        'lr': lr * 1.0, 
        'name': 'attention'
    },
    {
        'params': list(model.bn1.parameters()) + 
                  [p for layer in [model.layer1, model.layer2, model.layer3, model.layer4] 
                   for block in layer for m in block.block.modules() 
                   if isinstance(m, nn.BatchNorm2d) for p in m.parameters()],
        'lr': lr * 1.0, 
        'name': 'batchnorm'
    }
]

optimizer = torch.optim.AdamW(param_groups, weight_decay=5e-4)
```

---

## Risk Analysis & Mitigation

### Potential Issues

1. **Risk: Attention modules slow convergence**
   - **Cause:** Random initialization of CA/SE
   - **Mitigation:** Use full LR (1.0×) for attention params
   - **Fallback:** Initialize attention with identity (output = input)

2. **Risk: Discriminative LR too conservative for stem/layer1**
   - **Symptom:** Val F1 plateaus early (<0.85)
   - **Mitigation:** Increase LR multiplier 0.1 → 0.2
   - **Adjustment:** Monitor layer1 gradient magnitudes

3. **Risk: Increased parameter count (attention) causes overfitting**
   - **Current:** ~300K extra params (11.2M → 11.5M, +2.7%)
   - **Mitigation:** Block dropout (0.05) specifically targets this
   - **Monitor:** Val-test gap should decrease, not increase

4. **Risk: Stronger mixup (α=0.4) hurts minority classes**
   - **Concern:** Over-smoothing speech:tonal decision boundary
   - **Mitigation:** DRW at epoch 50 reweights minority classes
   - **Validation:** Check speech:tonal recall separately

### Debugging Strategy

If Val F1 < 0.87 after 50 epochs:

1. **Check attention activation:** Are CA/SE weights learning (not stuck at uniform)?
   ```python
   # Print attention statistics
   for name, module in model.named_modules():
       if isinstance(module, (CoordinateAttention, SqueezeExcitation)):
           print(f"{name}: mean={module.conv_h.weight.mean():.4f}")
   ```

2. **Check discriminative LR:** Are layer1 gradients too small?
   ```python
   # Log gradient norms per layer group
   for name, param in model.named_parameters():
       if param.grad is not None:
           print(f"{name}: grad_norm={param.grad.norm():.6f}")
   ```

3. **Ablation:** Remove one enhancement at a time
   - Run with attention but uniform LR
   - Run with discriminative LR but no attention
   - Isolate which component underperforms

---

## Comparison to Other Approaches

### Why Not Just Use V4 Architecture?

**V4 Advantages:**
- Proven architecture (0.865 F1)
- From-scratch training = no pretraining bias mismatch

**V4 Disadvantages:**
- **No transfer learning:** Learns texture filters from scratch
- **Longer convergence:** Needs more epochs to learn low-level features
- **Less robust:** ImageNet pretraining encodes billions of edge cases

**V2.1 Advantages:**
- **Best of both worlds:** ImageNet features + V4 architecture
- **Faster convergence:** Pretrained early layers
- **Better generalization:** Transfer learning regularization effect

### Why Not More Aggressive Transfer (e.g., Freeze Layer1)?

**Considered:** Freeze stem + layer1 completely (LR=0)

**Rejected Because:**
- STM features ≠ ImageNet images (different modality)
- Some adaptation needed even for low-level filters
- Discriminative LR (0.1×) is "soft freeze" → better than hard freeze

---

## Evaluation Plan

### Success Criteria

**Primary Metric (Macro F1):**
- **Minimum:** 0.870 (beat V4's 0.865)
- **Target:** 0.875-0.880
- **Stretch:** 0.885 (approach SOTA 0.89)

**Secondary Metrics:**
- **Val-Test Gap:** < 0.005 (reduced from V2.0's 0.009)
- **Speech:Tonal Recall:** > 0.70 (vs. V2.0's 0.65)
- **Music:Non-Vocal F1:** > 0.73 (vs. V2.0's 0.68)

**Qualitative:**
- Confusion matrix: Fewer speech:tonal → music:vocal errors
- Attention visualizations: CA focuses on discriminative regions
- Convergence: Smoother val curve, no wild oscillations

### Comparison Benchmarks

| Model | Macro F1 | speech:tonal | music:non-vocal | Params | Training Time |
|-------|----------|--------------|-----------------|--------|---------------|
| V2.0 (preIN) | 0.8594 | 0.77 (P:0.93, R:0.65) | 0.68 | 11.2M | 32 epochs |
| V4 (Best) | **0.8650** | 0.77 | 0.68 | 11.2M | ~80 epochs |
| **V2.1 (preIN2)** | **0.875-0.885** | **0.83-0.85** | **0.73-0.76** | 11.5M | ~60-70 epochs |
| SOTA (Target) | 0.89 | ? | ? | ? | ? |

---

## Conclusion

STM_CoordConvLDAM_preIN2 represents a carefully engineered fusion of:
1. **Transfer Learning** (ImageNet pretraining)
2. **Attention Mechanisms** (V4's proven architecture)
3. **Discriminative Optimization** (layer-wise learning rates)
4. **Strong Regularization** (dropout, weight decay, mixup)

By addressing V2.0's overfitting through multiple complementary strategies rather than a single silver bullet, V2.1 is positioned to achieve robust performance gains. The discriminative learning rate strategy, in particular, is novel in the STM classification context and leverages the unique advantage of pretraining.

**Expected Outcome:** A model that combines the texture bias of ImageNet with the adaptive feature selection of attention mechanisms, achieving **0.875-0.885 Macro F1** and establishing a new benchmark for STM-based audio classification.

---

## Quick Start

### Training
```bash
# Standard mode (full dataset)
python STM_CoordConvLDAM_preIN2.py 0

# Downsampled mode (balanced classes)
python STM_CoordConvLDAM_preIN2.py 1
```

### Monitoring
```python
# Check learning rates
tensorboard --logdir=model/STM/CoordConvLDAM_preIN2_corpora_categories/standard/ckpt

# Watch for:
# - Val F1 > 0.87 by epoch 30
# - Smooth convergence (no oscillations)
# - Attention modules learning (not stuck)
```

### Post-Training Analysis
```python
# Load best model
checkpoint = torch.load('best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Evaluate per-class
from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred, target_names=class_names))

# Visualize attention (future work)
# CA/SE weights can be extracted and visualized as heatmaps
```

---

**Document Version:** 1.0  
**Last Updated:** February 6, 2026  
**Author:** STM Research Team  
**Status:** Ready for Training
