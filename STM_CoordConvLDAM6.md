# STM Classification with CoordConv-ResNet and LDAM Loss
## Phase 1.6: Advanced Regularization + DropBlock + CutMix

### Motivation

**V4 Results Analysis**:
- Test Macro F1: **0.8631** (only +0.0008 over V2)
- music:non-vocal recall: **0.71** (+0.05 improvement)
- **Early stopping at epoch 28** (training instability)

**V5 Strategy**: Combine balanced sampling + attention (hybrid approach)

**V6 Strategy**: Different direction - **Better regularization**
- Keep V4 architecture (attention proven useful)
- Add advanced regularization techniques
- Prevent overfitting with structured dropout
- Use CutMix instead of standard Mixup

### Problem Analysis

**Why V4 stopped training early (epoch 28)**:
1. **Overfitting**: Validation F1 plateaued while training loss decreased
2. **Insufficient regularization**: Standard dropout (0.3) not enough
3. **Mixup limitation**: Random pixel mixing doesn't exploit spatial structure

**Evidence from training log**:
- Epoch 8: Val F1 = 0.8508 (best)
- Epochs 9-28: Val F1 oscillates 0.84-0.85 (no improvement)
- Conclusion: Model capacity fine, but regularization insufficient

### Solution Strategy: V6

**Keep what works (from V4)**:
- ✅ CoordConv + Attention (CA/SE) architecture
- ✅ Multi-scale fusion
- ✅ LDAM loss with DRW
- ✅ Label smoothing (0.05)

**Add advanced regularization**:
1. **DropBlock** (spatial-aware dropout for feature maps)
2. **CutMix** (cut-and-paste augmentation, better than Mixup for 2D data)
3. **Stochastic Depth** (randomly drop residual blocks during training)
4. **Stronger weight decay** (2e-4 → 5e-4)
5. **Cosine Annealing LR** (replace ReduceLROnPlateau for smoother decay)

### Core Improvements

#### 1. **DropBlock**

**From**: "DropBlock: A regularization method for convolutional networks" (Ghiasi et al., NeurIPS 2018)

**Problem with standard Dropout**:
- Dropout2d: Randomly drops individual spatial locations
- For CNNs: Adjacent pixels highly correlated
- Network can still "see around" dropped pixels
- **Weak regularization for convolutional features**

**DropBlock solution**:
```
Standard Dropout2d:
  X X O X X    (randomly drop pixels)
  O X X X O
  X O X X X

DropBlock (block_size=3):
  X X O O O    (drop contiguous blocks)
  X X O O O
  O O O X X

Effect: Forces network to learn from partial/occluded features
→ Better generalization
```

**Implementation**:
```python
class DropBlock2d(nn.Module):
    def __init__(self, drop_prob=0.1, block_size=7):
        # drop_prob: Probability of dropping a block
        # block_size: Size of spatial blocks to drop (7x7 for STM)
        
    def forward(self, x):
        if not self.training:
            return x
        
        # 1. Sample block centers with probability gamma
        gamma = self._compute_gamma(x)
        mask = torch.bernoulli(torch.ones_like(x) * gamma)
        
        # 2. Expand sampled points to blocks
        mask = F.max_pool2d(mask, kernel_size=self.block_size, 
                            stride=1, padding=self.block_size // 2)
        
        # 3. Apply mask and normalize
        mask = 1 - mask
        out = x * mask * mask.numel() / mask.sum()
        
        return out
```

**DropBlock schedule**:
- Epochs 1-20: drop_prob = 0.0 (no DropBlock, learn basic features)
- Epochs 21-60: drop_prob linearly increases 0.0 → 0.15
- Epochs 61-100: drop_prob = 0.15 (constant)

**Where to apply**:
- After layer2, layer3, layer4 (not layer1, too early)
- Block size = 5 for layer2/3, 3 for layer4 (smaller spatial maps)

#### 2. **CutMix**

**From**: "CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features" (Yun et al., ICCV 2019)

**Problem with Mixup**:
- Mixup: `x_mixed = λ*x₁ + (1-λ)*x₂` (pixel-wise blending)
- Creates **unrealistic** samples (ghosting effect)
- For STM: Mixed modulation patterns may not exist in real audio

**CutMix solution**:
```
Sample 1:        Sample 2:        CutMix result:
AAAAA            BBBBB            AAAAA
AAAAA            BBBBB            ABBBA
AAAAA            BBBBB   →        ABBBA
AAAAA            BBBBB            ABBBA
AAAAA            BBBBB            AAAAA

- Cut a random box from Sample 2
- Paste into Sample 1
- Label: λ*y₁ + (1-λ)*y₂, where λ = area_kept / total_area
```

**Why better for STM**:
- Preserves local modulation structure
- Creates more realistic mixed samples
- Forces model to recognize partial patterns
- Improves localization (attention benefits more)

**Implementation**:
```python
def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    # Random box coordinates
    H, W = x.size(2), x.size(3)
    cut_ratio = np.sqrt(1 - lam)
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)
    
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    # Paste box from shuffled sample
    x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    
    # Adjust lambda based on actual box area
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (H * W))
    
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam
```

#### 3. **Stochastic Depth**

**From**: "Deep Networks with Stochastic Depth" (Huang et al., ECCV 2016)

**Idea**: Randomly skip entire residual blocks during training
```
Standard ResNet block:
  x → [Block] → out
      out = Block(x) + x  (always)

Stochastic Depth:
  x → [Block?] → out
      if coin_flip(drop_prob):
          out = x  (skip block)
      else:
          out = Block(x) + x  (apply block)
```

**Benefits**:
- Reduces effective network depth during training
- Prevents co-adaptation of layers
- Improves gradient flow (shorter paths)
- Acts as implicit ensemble (different depths per batch)

**Drop probability schedule**:
- Linear: layer1 (0.0) → layer4 (0.2)
- Shallower layers more important, drop less
- Deeper layers can be skipped more often

#### 4. **Cosine Annealing LR**

**From**: "SGDR: Stochastic Gradient Descent with Warm Restarts" (Loshchilov & Hutter, ICLR 2017)

**Problem with ReduceLROnPlateau**:
- Reactive: Only reduces LR after plateau
- Can get stuck in local minima
- Sharp LR drops can destabilize training

**Cosine Annealing**:
```
LR schedule:
  LR(epoch) = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(π * epoch / max_epochs))
  
  max_lr (1e-4) ────╮
                     ╲
                      ╲
                       ╲___________  min_lr (1e-6)
  epoch:  0        50           100
```

**Benefits**:
- Smooth, predictable decay
- Reaches low LR gradually (fine-tunes near end)
- No manual tuning of patience/factor
- Better for long training (100+ epochs)

#### 5. **Stronger Weight Decay**

**Increase from 2e-4 → 5e-4**:
- V4 used 2e-4 (standard for ImageNet)
- STM dataset smaller (~1M samples vs 1.2M ImageNet)
- Higher weight decay prevents overfitting

### Architecture (Same as V4)

**CoordConvResNet18_Attention** (no changes):
- Layer1, Layer2: Coordinate Attention
- Layer3, Layer4: Squeeze-and-Excitation
- Multi-scale fusion: layer3 + layer4
- Parameters: ~13.5M

**Modified BasicBlock**:
```python
class BasicBlock(nn.Module):
    def __init__(self, ..., drop_prob=0.0, stochastic_depth=0.0):
        # ... standard layers ...
        
        # NEW: DropBlock after conv2
        self.dropblock = DropBlock2d(drop_prob=drop_prob, block_size=5)
        
        # NEW: Stochastic depth probability
        self.stochastic_depth = stochastic_depth
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        if self.dropout is not None:
            out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # NEW: Apply DropBlock
        out = self.dropblock(out)
        
        # Apply attention
        if self.attention is not None:
            out = self.attention(out)
        
        # NEW: Stochastic depth
        if self.training and self.stochastic_depth > 0:
            if torch.rand(1).item() < self.stochastic_depth:
                return identity  # Skip this block
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out
```

### Training Configuration

| Parameter | V2 | V4 | V6 (Advanced Reg) |
|-----------|----|----|-------------------|
| Architecture | Basic | + Attention | + Attention |
| DropBlock | No | No | **Yes (0.15)** |
| Stochastic Depth | No | No | **Yes (0.0-0.2)** |
| Augmentation | Mixup | Mixup | **CutMix** |
| LR Scheduler | ReduceLROnPlateau | ReduceLROnPlateau | **CosineAnnealing** |
| Weight Decay | 2e-4 | 2e-4 | **5e-4** |
| Initial LR | 1e-4 | 1e-4 | 1e-4 |
| Min LR | 1e-6 | 1e-6 | 1e-6 |
| Max Epochs | 100 | 100 | **150** (longer for cosine) |
| DRW Start | Epoch 50 | Epoch 50 | **Epoch 75** (50% of 150) |
| Early Stopping | Patience 20 | Patience 20 | **Patience 30** |
| Batch Size | 256 | 256 | 256 |
| Label Smoothing | 0.05 | 0.05 | 0.05 |

### Expected Improvements

#### 1. **Longer Training Without Overfitting**

**V4**: Early stopped at epoch 28
**V6 Target**: Train for 100-120 epochs

**Why**:
- DropBlock prevents overfitting to spatial locations
- Stochastic depth prevents layer co-adaptation
- Stronger weight decay reduces parameter magnitudes
- CutMix creates more diverse training samples

#### 2. **Better Generalization**

**V4**: Test F1 (0.8631) ≈ Val F1 (0.8508) (good generalization, but low capacity usage)
**V6 Target**: Test F1 > 0.88, close to Val F1

**Mechanism**:
- DropBlock: Learns robust features (not location-dependent)
- CutMix: Learns partial pattern recognition
- Stochastic depth: Ensemble effect (implicit)

#### 3. **Music:Non-Vocal Improvement**

**V4**: 0.71 recall
**V6 Target**: **0.74-0.77** recall

**Why**:
- CutMix: Better for fine-grained discrimination
  - Mix jazz + EDM → Model learns both local patterns
- DropBlock: Forces attention to focus on salient features
  - Can't rely on entire spatial map
- Longer training: More gradient updates for minority class

#### 4. **Training Stability**

**V4**: Validation F1 fluctuated 0.84-0.85 (epochs 9-28)
**V6**: Smoother curves with cosine annealing

**Cosine benefits**:
- Gradual LR decay (no sudden drops)
- Fine-tuning phase at end (low LR)
- More predictable convergence

### Implementation Details

#### DropBlock2d Module

```python
class DropBlock2d(nn.Module):
    def __init__(self, drop_prob=0.1, block_size=7):
        super().__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size
    
    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            return x
        
        gamma = self._compute_gamma(x)
        
        # Sample mask
        mask = torch.bernoulli(torch.ones_like(x) * gamma)
        
        # Expand to blocks
        block_mask = F.max_pool2d(
            mask,
            kernel_size=(self.block_size, self.block_size),
            stride=(1, 1),
            padding=self.block_size // 2
        )
        
        # Invert and normalize
        block_mask = 1 - block_mask
        normalize_scale = block_mask.numel() / block_mask.sum()
        
        return x * block_mask * normalize_scale
    
    def _compute_gamma(self, x):
        # Adjust drop_prob for block size
        return self.drop_prob / (self.block_size ** 2)
```

#### DropBlock Schedule in Trainer

```python
class Trainer:
    def __init__(self, ...):
        # DropBlock schedule
        self.dropblock_epochs = [20, 60]  # [start, ramp_end]
        self.max_drop_prob = 0.15
        
    def update_dropblock(self, epoch, model):
        if epoch < self.dropblock_epochs[0]:
            drop_prob = 0.0
        elif epoch < self.dropblock_epochs[1]:
            # Linear ramp
            progress = (epoch - self.dropblock_epochs[0]) / (self.dropblock_epochs[1] - self.dropblock_epochs[0])
            drop_prob = progress * self.max_drop_prob
        else:
            drop_prob = self.max_drop_prob
        
        # Update all DropBlock modules in model
        for module in model.modules():
            if isinstance(module, DropBlock2d):
                module.drop_prob = drop_prob
        
        return drop_prob
```

#### CutMix Function

```python
def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    # Generate random box
    H, W = x.size(2), x.size(3)
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)
    
    # Random center point
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    # Apply cutmix
    x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    
    # Adjust lambda based on actual cut area
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (H * W))
    
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam
```

### Monitoring Training

**Key metrics**:

1. **DropBlock progression**:
   - Epoch 20: drop_prob = 0.0
   - Epoch 40: drop_prob = 0.075
   - Epoch 60+: drop_prob = 0.15
   - Log when drop_prob changes

2. **Stochastic depth statistics**:
   - Track % of blocks actually skipped per batch
   - Should increase from layer1 (0%) to layer4 (20%)

3. **Learning rate curve**:
   - Plot LR vs epoch (should be smooth cosine)
   - Verify reaches min_lr at epoch 150

4. **Training vs validation gap**:
   - V4: Gap minimal but both plateaued early
   - V6: Gap should stay <5% while both improve

5. **CutMix effectiveness**:
   - Visualize cutmix samples (first batch of epoch 1)
   - Verify boxes are reasonable size (20-60% of image)

### File Structure

```
model/STM/CoordConvLDAM6_corpora_categories/
├── standard/
│   └── ckpt/
│       └── YYYY-MM-DD_HH-MM/
│           ├── best_model.pt
│           ├── checkpoint_epoch_20.pt   # DropBlock starts
│           ├── checkpoint_epoch_60.pt   # DropBlock fully ramped
│           ├── checkpoint_epoch_75.pt   # DRW starts
│           ├── checkpoint_epoch_100.pt
│           ├── test_predictions.npy
│           ├── test_targets.npy
│           └── regularization_schedule.png  # Optional: plot drop schedules
└── downsample/
    └── ...
```

### Usage

```bash
# Standard training (full dataset)
python STM_CoordConvLDAM6.py 0

# Downsampled non-tonal speech
python STM_CoordConvLDAM6.py 1
```

### Expected Console Output

```
Epoch 1/150
============================================================
Model: CoordConvResNet18_Attention with DropBlock + CutMix
Total parameters: 13,267,890 (+0.2% for DropBlock)
DropBlock: prob=0.000 (warming up)
Stochastic Depth: layer1=0.00, layer2=0.05, layer3=0.10, layer4=0.20
Learning Rate: 0.000100 (Cosine schedule)
  Batch 500/3010, Loss: 2.1234, DRW: False, CutMix: True
  ...

Epoch 20/150
============================================================
*** DropBlock activated: prob=0.000 → 0.001 ***
...

Epoch 40/150
============================================================
DropBlock: prob=0.075 (ramping)
...

Epoch 60/150
============================================================
*** DropBlock fully ramped: prob=0.150 ***
...

Epoch 75/150
============================================================
*** Activating Deferred Reweighting (DRW) ***
...

Epoch 100/150
============================================================
Learning Rate: 0.000013 (cosine decay)
Train Loss: 0.8456
Val Loss: 1.4123, Val Macro F1: 0.8812
...
```

### Theoretical Justification

#### DropBlock vs Standard Dropout

From the DropBlock paper (Ghiasi et al., NeurIPS 2018):
> "DropBlock achieves 2.1% improvement in ImageNet classification over standard dropout, and 5.2% for object detection. The structured dropping is crucial for convolutional networks."

**Why it works**:
- CNNs have strong spatial correlations
- Dropping individual pixels is too weak
- Dropping blocks forces learning from partial information
- Better for 2D data like STM spectrograms

#### CutMix vs Mixup

From the CutMix paper (Yun et al., ICCV 2019):
> "CutMix improves ImageNet top-1 accuracy by 1.8% over Mixup. It better preserves natural image statistics and improves localization ability."

**For STM classification**:
- STM features have spatial structure (freq × time)
- Mixup blurs this structure (unrealistic)
- CutMix preserves local modulation patterns
- Attention benefits more (learns to focus on partial patterns)

#### Stochastic Depth

From the original paper (Huang et al., ECCV 2016):
> "Stochastic depth reduces ResNet-110 test error from 6.41% to 5.23% on CIFAR-10 by preventing layer co-adaptation."

**Application to STM**:
- Our ResNet-18: 8 residual blocks total
- Drop probability: 0.0 (layer1) → 0.2 (layer4)
- Effective ensemble of 2^8 = 256 network depths
- Improves generalization

### Comparison with Previous Versions

| Metric | V2 | V4 | V5 (Expected) | V6 (Expected) |
|--------|----|----|---------------|---------------|
| Test Macro F1 | 0.8623 | 0.8631 | 0.88-0.89 | **0.87-0.88** |
| music:non-vocal recall | 0.66 | 0.71 | 0.75-0.78 | **0.74-0.77** |
| Training epochs completed | ~100 | 28 (early stop) | 100-120 | **100-130** |
| Overfitting risk | Medium | High (stopped early) | Low (balanced sampling) | **Very low (strong reg)** |
| Regularization | Standard | Standard | Balanced batches | **DropBlock + Stochastic Depth** |
| Augmentation | Mixup | Mixup | Remix | **CutMix** |
| Key innovation | LDAM | Attention | Hybrid | **Advanced regularization** |

**V5 vs V6 tradeoffs**:
- **V5**: Addresses class imbalance (balanced sampling + focal loss)
  - Better for imbalanced data
  - More complex training (custom sampler)
- **V6**: Addresses overfitting (stronger regularization)
  - Better for model capacity utilization
  - Simpler training (standard random sampling)
  - Longer convergence (150 vs 120 epochs)

**When to use which**:
- V5: If class imbalance is the main issue
- V6: If overfitting is the main issue
- V4 result suggests **overfitting** (early stop) → **V6 more promising**

### Ablation Study

**Test each component's contribution**:
1. V6 (full): DropBlock + CutMix + Stochastic Depth + Cosine LR
2. V6 - DropBlock: Remove DropBlock
3. V6 - CutMix: Use standard Mixup
4. V6 - Stochastic Depth: No block skipping
5. V6 - Cosine: Use ReduceLROnPlateau

**Expected ranking**:
- V6 (full) > V6 - Stochastic Depth > V6 - CutMix > V6 - DropBlock > V6 - Cosine

**If DropBlock hurts**:
- May be too strong for STM (small spatial maps)
- Try smaller block_size (3 instead of 5)
- Or lower max drop_prob (0.10 instead of 0.15)

### Success Criteria

**V6 is successful if**:
- ✅ Test Macro F1 ≥ 0.87 (+2.4% over V2)
- ✅ music:non-vocal recall ≥ 0.74 (+8% over V2)
- ✅ Training stable for 100+ epochs (vs V4: 28)
- ✅ No significant overfitting (test F1 within 2% of val F1)
- ✅ CutMix samples look realistic (visualized)

**If achieved**:
- V6 demonstrates **regularization** more important than sampling strategies
- Provides alternative to V5's balanced sampling approach

**If not achieved**:
- Regularization may be too strong (underfitting)
- Try V6-lite: Only DropBlock + CutMix (no stochastic depth)
- Or reduce weight decay back to 2e-4

### Advanced Extensions

**If V6 underperforms expectations**:

1. **Adaptive DropBlock**: 
   - Per-class drop probabilities
   - Higher drop_prob for majority classes

2. **Multi-scale CutMix**:
   - Apply CutMix at multiple resolutions
   - Mix at both input and feature levels

3. **AutoAugment for audio**:
   - Learned augmentation policies
   - Requires search phase (expensive)

4. **Manifold Mixup**:
   - Apply mixup in hidden layers
   - Better than input-level mixup for some tasks

### References

1. Ghiasi et al., "DropBlock: A regularization method for convolutional networks", NeurIPS 2018
2. Yun et al., "CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features", ICCV 2019
3. Huang et al., "Deep Networks with Stochastic Depth", ECCV 2016
4. Loshchilov & Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts", ICLR 2017
5. Hou et al., "Coordinate Attention for Efficient Mobile Network Design", CVPR 2021

### Next Steps

1. **If V6 succeeds**:
   - Document best regularization practices for STM
   - Compare V5 vs V6 (balanced sampling vs regularization)
   - Consider ensemble V5 + V6

2. **If V6 fails**:
   - Analyze failure mode (overfitting vs underfitting?)
   - Try V6-lite (fewer regularization techniques)
   - Consider returning to simpler models (V2)

3. **Overall strategy**:
   - V5: Training dynamics approach
   - V6: Regularization approach
   - Both target ~0.87-0.88 F1 from different angles
   - Best of V5 and V6 → Baseline for Phase 2 (Vision Mamba)
