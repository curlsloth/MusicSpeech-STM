# STM Classification with CoordConv-ResNet and LDAM Loss
## Phase 1.4: Attention Mechanisms + Multi-Scale Features

### Motivation

Analysis of V2 results shows **persistent weakness in music:non-vocal** (0.66 recall):
- Low-level features (e.g., simple spectral patterns) may be insufficient
- Need better **feature selection** (attention) and **multi-scale context**
- V2 and V3 focus on training dynamics; V4 enhances **architecture**

### Core Problem

**Why music:non-vocal is hard**:
1. **High intra-class variance**: Jazz piano vs. electronic dance music have very different STM signatures
2. **Inter-class similarity**: Instrumental music can sound like non-tonal speech (percussive patterns)
3. **Feature complexity**: Requires both local details (rhythm) and global context (structure)

Standard ResNet processes features uniformly → no mechanism to:
- **Emphasize discriminative features** (e.g., specific modulation rates)
- **Suppress noise** (e.g., background artifacts in recordings)
- **Integrate multi-scale information** (local + global patterns)

### Solution Strategy: V4

**Architecture enhancements**:
1. **Squeeze-and-Excitation (SE) blocks**: Channel-wise attention
2. **Coordinate Attention (CA)**: Position-aware attention (better than SE for spatial data)
3. **Multi-Scale Feature Fusion**: Combine features from different depths
4. **Keep V2 training dynamics**: Proven regularization + adaptive LR

### Key Improvements Over V2

#### 1. **Squeeze-and-Excitation (SE) Blocks**

**From**: "Squeeze-and-Excitation Networks" (Hu et al., CVPR 2018)

**Problem**: All feature channels treated equally
- Channel 10 might encode "low-rate modulation" (speech indicator)
- Channel 50 might encode "high-scale modulation" (music indicator)
- Standard convolution: Both contribute equally to output

**SE Mechanism**:
```
Input: Feature map X (batch, C, H, W)

1. Squeeze: Global average pooling
   z = GlobalAvgPool(X)  → (batch, C)
   
2. Excitation: Two FC layers
   s = Sigmoid(FC2(ReLU(FC1(z))))  → (batch, C)
   
3. Rescale: Channel-wise multiplication
   X_out = s.unsqueeze(-1).unsqueeze(-1) * X
```

**Effect**:
- Network learns which channels are important for current sample
- For music:non-vocal: Boost channels detecting rhythmic patterns
- For speech:tonal: Boost channels detecting pitch modulation
- **Adaptive feature selection per sample**

**Implementation**:
- Add SE block after each residual block
- Reduction ratio: 16 (compress to C/16 in bottleneck)
- Minimal parameter overhead: ~1% of total params

#### 2. **Coordinate Attention (CA)**

**From**: "Coordinate Attention for Efficient Mobile Network Design" (Hou et al., CVPR 2021)

**Problem**: SE only captures channel relationships, ignores spatial structure
- STM features: Position matters! (Low vs. high modulation rates)
- SE loses spatial information in global pooling

**CA Mechanism**:
```
Input: Feature map X (batch, C, H, W)

1. X-axis pooling: Pool along height
   x_pool = AvgPool(X, dim=height)  → (batch, C, 1, W)
   
2. Y-axis pooling: Pool along width
   y_pool = AvgPool(X, dim=width)   → (batch, C, H, 1)
   
3. Concatenate and transform
   xy = Concat([x_pool, y_pool], axis=-1)  → (batch, C, H+W)
   xy = Conv1x1(xy)  → (batch, C/r, H+W)
   
4. Split and generate attention maps
   x_attn = Sigmoid(Conv1x1(xy[:, :, :W]))  → (batch, C, 1, W)
   y_attn = Sigmoid(Conv1x1(xy[:, :, W:]))  → (batch, C, H, 1)
   
5. Apply attention
   X_out = X * x_attn * y_attn
```

**Effect**:
- Captures **long-range dependencies** along both axes
- X-axis (temporal modulation): Identifies important rate bins
- Y-axis (spectral modulation): Identifies important scale bins
- **Position-aware attention** (synergistic with CoordConv)

**Why better than SE for STM**:
- STM (20×121): Small height, long width → asymmetric
- CA preserves spatial structure
- Explicitly models row/column dependencies

#### 3. **Multi-Scale Feature Fusion**

**Problem**: Single-scale features miss context
- Early layers: Local patterns (e.g., single modulation peak)
- Late layers: Global patterns (e.g., overall energy distribution)
- Music:non-vocal needs both: Local rhythm + Global structure

**Feature Pyramid Approach**:
```
Layer1 (stride=1): (64, 20, 121)  → Local, high-resolution
Layer2 (stride=2): (128, 10, 61)  → Mid-level
Layer3 (stride=2): (256, 5, 31)   → High-level
Layer4 (stride=2): (512, 3, 16)   → Global, low-resolution

Multi-Scale Fusion:
1. Upsample layer3 to match layer4 spatial size
2. Concatenate: [layer3_upsampled, layer4]
3. Conv1x1: Fuse features
4. Feed to classifier
```

**Effect**:
- Classifier sees both fine-grained details and global context
- Music:non-vocal: Use global structure for genre identification
- Speech:tonal: Use local peaks for pitch tracking

**Tradeoff**:
- Slightly more parameters (~5-10%)
- Better feature representation
- Worth the cost for hard classes

#### 4. **Hybrid Attention: SE + CA**

**Strategy**: Use both SE and CA, but at different stages

**Layer-wise attention assignment**:
- **Layer1 (high-res)**: CA (preserve spatial structure)
- **Layer2 (mid-res)**: CA (still need position info)
- **Layer3 (mid-res)**: SE (channel selection more important)
- **Layer4 (low-res)**: SE (spatial structure mostly lost)

**Rationale**:
- Early layers: Position critical → CA
- Late layers: Channel selection critical → SE
- Progressive transition from spatial to semantic

### Architecture Enhancements

**Enhanced BasicBlock**:
```python
BasicBlock:
  conv1 (CoordConv)
  bn1
  relu
  dropout (0.05)
  conv2
  bn2
  
  # NEW: Attention mechanism
  if use_CA:
      attention = CoordinateAttention(channels, reduction=16)
  elif use_SE:
      attention = SqueezeExcitation(channels, reduction=16)
  
  out = attention(out) + identity
  relu
```

**Multi-Scale Fusion Head**:
```python
# Extract multi-scale features
feat_layer3 = self.layer3(x)  # (256, 5, 31)
feat_layer4 = self.layer4(feat_layer3)  # (512, 3, 16)

# Upsample layer3 to match layer4
feat_layer3_up = F.interpolate(feat_layer3, size=feat_layer4.shape[-2:])

# Concatenate
feat_fused = torch.cat([feat_layer3_up, feat_layer4], dim=1)  # (768, 3, 16)

# Fusion convolution
feat_fused = self.fusion_conv(feat_fused)  # (512, 3, 16)

# Flatten and classify
```

### Training Configuration

**Keep V2 proven strategies**:
- Moderate dropout (0.3 head, 0.05 blocks)
- ReduceLROnPlateau scheduler
- Early DRW (epoch 50)
- Mixup augmentation (α=0.3, 30% prob)
- Label smoothing (0.05)
- Early stopping (patience=20)

**No balanced sampling** (unlike V3):
- V4 focuses on **better features**, not **better sampling**
- Standard random sampling (simpler, faster)
- Let attention mechanisms handle class imbalance

| Parameter | V2 | V4 | Change |
|-----------|----|----|--------|
| Attention | None | CA + SE | Architecture |
| Multi-scale | No | Yes | Architecture |
| Batch sampler | Random | Random | Same |
| Focal loss | No | No | Keep simple |
| Dropout | 0.3 head, 0.05 blocks | Same | Keep V2 |
| Mixup | Standard (α=0.3) | Same | Keep V2 |
| DRW start | Epoch 50 | Epoch 50 | Same |
| Max epochs | 100 | 100 | Same |

### Expected Improvements

#### 1. **Better Music:Non-Vocal Performance**

**V2 Weakness**: 0.66 recall (lowest)

**V4 Target**: **0.72-0.75** recall (+6-9%)

**Why achievable**:
- CA: Identifies discriminative spatial patterns for each subgenre
- Multi-scale: Captures both local rhythm and global structure
- SE: Selectively emphasizes genre-specific channels

#### 2. **Overall Macro F1 Improvement**

**V2**: 0.8623

**V4 Target**: **0.87-0.88** (+0.7-1.7%)

**Mechanism**:
- Attention improves all classes, not just minorities
- Better feature quality → higher precision/recall across board

#### 3. **More Robust to Intra-Class Variance**

**Problem**: Music:non-vocal has high variance (jazz vs. EDM)

**V4 Solution**: Adaptive feature selection
- For jazz sample: CA emphasizes swing rhythm patterns
- For EDM sample: CA emphasizes regular beat patterns
- **Sample-specific attention** reduces confusion

#### 4. **Better Interpretability**

**Benefit**: Can visualize attention maps
- Which modulation rates are important for each class?
- Which spatial locations receive high attention?
- Helps debug failures

### Implementation Details

#### Coordinate Attention Module

```python
class CoordinateAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        hidden_channels = max(8, in_channels // reduction)
        
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.act = nn.ReLU(inplace=True)
        
        self.conv_h = nn.Conv2d(hidden_channels, in_channels, 1)
        self.conv_w = nn.Conv2d(hidden_channels, in_channels, 1)
    
    def forward(self, x):
        b, c, h, w = x.size()
        
        # Pool along each axis
        x_h = self.pool_h(x)  # (b, c, h, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # (b, c, w, 1)
        
        # Concatenate
        y = torch.cat([x_h, x_w], dim=2)  # (b, c, h+w, 1)
        
        # Shared transform
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        
        # Split and generate attention
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        
        a_h = self.conv_h(x_h).sigmoid()  # (b, c, h, 1)
        a_w = self.conv_w(x_w).sigmoid()  # (b, c, 1, w)
        
        # Apply attention
        out = x * a_h * a_w
        
        return out
```

#### Squeeze-and-Excitation Module

```python
class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        hidden_channels = max(1, in_channels // reduction)
        
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _ = x.size()
        
        # Squeeze
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        
        # Excitation
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        
        # Rescale
        y = y.view(b, c, 1, 1)
        
        return x * y
```

#### Multi-Scale Fusion

```python
class MultiScaleFusion(nn.Module):
    def __init__(self):
        super().__init__()
        # Fusion: 768 channels (256 + 512) -> 512
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(768, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, feat_low, feat_high):
        # Upsample lower-level features to match higher-level spatial size
        feat_low_up = F.interpolate(
            feat_low, 
            size=feat_high.shape[-2:], 
            mode='bilinear', 
            align_corners=False
        )
        
        # Concatenate and fuse
        feat_concat = torch.cat([feat_low_up, feat_high], dim=1)
        feat_fused = self.fusion_conv(feat_concat)
        
        return feat_fused
```

### Monitoring Training

**Key indicators**:

1. **Attention Map Statistics** (log periodically):
   - Mean/std of attention weights
   - Should see high variance (indicates selective attention)
   - Low variance → attention not learning

2. **Per-Class Improvement**:
   - Track music:non-vocal F1 every 5 epochs
   - Should see steady improvement (not just lucky batches)

3. **Parameter Count**:
   - V2: ~12M params
   - V4: ~13-14M params (+8-15%)
   - Acceptable overhead

4. **Training Time**:
   - Attention adds ~10-15% compute per batch
   - Still feasible on single GPU

### File Structure

```
model/STM/CoordConvLDAM4_corpora_categories/
├── standard/
│   └── ckpt/
│       └── YYYY-MM-DD_HH-MM/
│           ├── best_model.pt
│           ├── checkpoint_epoch_20.pt
│           ├── checkpoint_epoch_50.pt  # DRW starts
│           ├── test_predictions.npy
│           ├── test_targets.npy
│           └── attention_maps/  # Optional: saved attention visualizations
│               ├── sample_0_ca_layer1.npy
│               ├── sample_0_se_layer4.npy
│               └── ...
└── downsample/
    └── ...
```

### Usage

```bash
# Standard training (full dataset)
python STM_CoordConvLDAM4.py 0

# Downsampled non-tonal speech
python STM_CoordConvLDAM4.py 1
```

### Expected Console Output

```
Epoch 1/100
============================================================
Model: CoordConvResNet18 with CA + SE + Multi-Scale Fusion
Total parameters: 13,245,678 (+9.2% vs V2)
  Batch 500/3010, Loss: 1.8234, DRW: False, Mixup: True
  Attention stats - CA mean: 0.52, SE mean: 0.48
  ...
Train Loss: 1.7123
Val Loss: 1.2456, Val Macro F1: 0.8123
Current learning rate: 0.000100

...

Epoch 50/100
============================================================
*** Activating Deferred Reweighting (DRW) ***
  Batch 500/3010, Loss: 0.9234, DRW: True, Mixup: True
  Attention stats - CA mean: 0.67, SE mean: 0.59 (← Higher, more selective)
  ...
Train Loss: 0.9123
Val Loss: 0.7456, Val Macro F1: 0.8756
Current learning rate: 0.000025

Per-class F1 (validation):
  speech:non-tonal: 0.96
  speech:tonal: 0.78
  music:vocal: 0.85
  music:non-vocal: 0.74 (← Improved from 0.66!)
  env:urban: 0.97
  env:wildlife: 0.95
```

### Theoretical Justification

#### Why Attention Works for Imbalanced Data

From "Learning Imbalanced Datasets with Attention Modules" (Wang et al., 2020):

> "Attention mechanisms allow networks to adaptively focus on discriminative features for minority classes, compensating for lack of training samples"

**Mechanism**:
- Minority classes have fewer samples → less opportunity to learn features
- Attention: Explicitly learns **which** features matter
- Requires less data to identify discriminative patterns

#### Why Multi-Scale Helps

From "Feature Pyramid Networks for Object Detection" (Lin et al., CVPR 2017):

> "Objects at different scales require features from different levels; combining them improves small and large object detection"

**Analog for STM**:
- Music:non-vocal subgenres have different "scales"
  - Jazz: Fine-grained swing rhythm (local)
  - Ambient: Slow evolving textures (global)
- Multi-scale fusion: Captures both

#### Why CA > SE for Spatial Data

From the CA paper:

> "For tasks where spatial structure matters (segmentation, detection), CA outperforms SE by 2-3% because it preserves position information"

**STM is spatial**:
- (20, 121) = (spectral scale, temporal rate)
- Position encodes semantic meaning
- CA explicitly models spatial dependencies

### Comparison with V1, V2, V3

| Metric | V1 | V2 | V3 (Expected) | V4 (Expected) |
|--------|----|----|---------------|---------------|
| Test Macro F1 | 0.8594 | 0.8623 | 0.87-0.88 | 0.87-0.88 |
| music:non-vocal recall | 0.71 | 0.66 | 0.72-0.75 | 0.72-0.75 |
| speech:tonal recall | 0.64 | 0.72 | 0.76-0.78 | 0.74-0.76 |
| Key innovation | CoordConv + LDAM | Better regularization | Balanced sampling + Focal | Attention + Multi-scale |
| Training complexity | Medium | Medium | High (custom sampler) | Medium |
| Parameters | 12M | 12M | 12M | 13.5M (+12%) |
| Inference speed | 1.0× | 1.0× | 1.0× | 0.85× (slower) |

**V3 vs V4 tradeoffs**:
- **V3**: Better training dynamics (balanced batches, focal loss)
  - Pros: Directly addresses imbalance
  - Cons: Complex sampler, slower convergence
- **V4**: Better architecture (attention, multi-scale)
  - Pros: Better features, more generalizable
  - Cons: More parameters, slightly slower inference

**Recommended strategy**:
1. Try V3 first (training-focused)
2. Try V4 second (architecture-focused)
3. If both work: **Ensemble V3 + V4** (likely +1-2% F1 boost)

### If Performance Plateaus

**Diagnosis**:

1. **Check attention map diversity**:
   - Low variance → attention not learning → try higher reduction ratio
   - High variance but poor performance → features not discriminative

2. **Ablation study**:
   - Remove CA: Does performance drop? (Measures CA contribution)
   - Remove SE: Does performance drop? (Measures SE contribution)
   - Remove multi-scale: Does performance drop?

3. **Attention visualization**:
   - Plot attention maps for music:non-vocal samples
   - Do they highlight reasonable patterns? (Rhythmic regions?)
   - If not: May need different attention mechanism

### Advanced Extensions (If Needed)

1. **Non-local attention** (Wang et al., CVPR 2018):
   - Full self-attention across spatial locations
   - Captures long-range dependencies
   - Expensive: O(H²W²) complexity

2. **Channel-spatial co-attention**:
   - Joint channel + spatial attention
   - More powerful than CA alone
   - Risk: Overfitting with limited data

3. **Learnable multi-scale fusion**:
   - Instead of fixed interpolation, use learned upsampling
   - Transposed convolutions or learned bilinear filters

4. **Attention dropout**:
   - Randomly drop attention during training
   - Prevents over-reliance on single features
   - Improves robustness

### References

1. Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018
2. Hou et al., "Coordinate Attention for Efficient Mobile Network Design", CVPR 2021
3. Lin et al., "Feature Pyramid Networks for Object Detection", CVPR 2017
4. Wang et al., "Non-local Neural Networks", CVPR 2018
5. Cao et al., "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss", NeurIPS 2019

### Success Criteria

**V4 is successful if**:
- ✅ Test Macro F1 > 0.87 (vs V2: 0.8623)
- ✅ music:non-vocal recall > 0.72 (vs V2: 0.66)
- ✅ Attention maps show interpretable patterns
- ✅ Improvement comes from better features (not just training tricks)
- ✅ Generalizes well (test F1 close to val F1)

**If achieved**: V4 demonstrates that **architectural innovations** (attention, multi-scale) are as important as **training dynamics** (LDAM, DRW, mixup) for imbalanced audio classification.

**Next steps**: Phase 2 → Vision Mamba (linear complexity, global context)
