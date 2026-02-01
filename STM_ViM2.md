# STM Classification with Vision Mamba (Vim) - Variant 2
## Lighter/Faster Model for Rapid Experimentation

### Overview

This variant reduces model capacity to enable faster training cycles and establish a lighter baseline. The goal is to test whether the full capacity of the baseline model (STM_ViM.py) is necessary, or if a more compact model can achieve competitive performance with significantly reduced training time.

**Key hypothesis**: STM features may be sufficiently discriminative that a lighter model can capture the essential patterns without the full depth/width of the baseline.

**Status**: ✅ Script is fully functional and ready to run

### Recent Fixes (Latest Update)

1. **Added missing import**: Added `torch.nn.functional as F` for loss computation
2. **Fixed `train_epoch`**: Now returns both loss and F1 score as expected by training loop

### Hyperparameter Changes from Baseline

| Parameter | Baseline (ViM) | Variant 2 (ViM2) | Change |
|-----------|----------------|------------------|---------|
| d_model | 192 | 128 | ↓ 33% |
| depth | 12 | 8 | ↓ 33% |
| d_state | 16 | 12 | ↓ 25% |
| d_conv | 4 | 4 | Same |
| expand | 2 | 2 | Same |
| drop_path_rate | 0.1 | 0.1 | Same |
| dropout | 0.1 | 0.1 | Same |
| batch_size | 64 | 64 | Same |
| learning_rate | 1e-4 | 1e-4 | Same |
| Total params | ~8M | ~3.5M | ↓ 56% |

### Rationale

#### 1. Reduced Model Dimension (d_model: 192 → 128)

**Reason**: Lower embedding dimension reduces memory and computation per token
- Each token embedding: 192 → 128 values
- Attention-equivalent operations: O(128) vs O(192) per layer
- **Speedup**: ~1.5× per layer

**Trade-off**:
- Less representational capacity per token
- May struggle with subtle spectral patterns
- Should still capture major modulation features

#### 2. Fewer Layers (depth: 12 → 8)

**Reason**: Reduces total forward pass time
- 12 Vim blocks → 8 Vim blocks
- **Speedup**: ~1.5× total forward pass
- Combined with d_model reduction: **~2× overall speedup**

**Trade-off**:
- Shorter "information integration path"
- May miss long-range dependencies across frequency bands
- Still sufficient for local-to-medium range patterns

#### 3. Smaller SSM State (d_state: 16 → 12)

**Reason**: SSM hidden state size affects memory and computation
- State update operations: O(d_state) per token
- Modest reduction to maintain most capacity

**Trade-off**:
- Slightly less "memory" in the recurrent state
- Should be sufficient for 1220-token sequences

### Expected Performance

#### Training Efficiency

**Time per epoch**:
- Baseline: ~30-35 minutes
- Variant 2: ~**20-25 minutes**
- **Improvement**: ~35% faster

**Total training time** (50 epochs):
- Baseline: ~25-30 hours
- Variant 2: ~**17-21 hours**
- **Improvement**: Fits in overnight training window

**Memory usage**:
- Baseline: ~18-20 GB per batch
- Variant 2: ~**12-14 GB per batch**
- Can potentially increase batch size if needed

#### Classification Performance

**Expected macro-F1** (relative to baseline):
- **Best case**: -1 to -2% (minimal capacity loss)
- **Likely case**: -2 to -4% (acceptable tradeoff for speed)
- **Worst case**: -5 to -7% (capacity becomes limiting)

**Per-class expectations**:
- **Speech (majority)**: Similar to baseline (abundant training data)
- **Music**: Slight degradation (moderate data, complex patterns)
- **Environment (minority)**: May degrade more (limited samples + subtle patterns)

**Why this might still work well**:
1. Symmetric STM already reduces dimensionality (121 → 61 rates)
2. Core modulation patterns may be captured by 128-dim embeddings
3. 8 layers still provides reasonable depth for integration
4. Balanced Softmax compensates for minority class difficulty

### Use Cases

#### 1. Rapid Prototyping
- Fast iteration on data augmentation strategies
- Quick testing of different preprocessing approaches
- Initial hyperparameter exploration

#### 2. Baseline Establishment
- Demonstrates minimum viable architecture
- Shows performance floor for comparison
- Validates that complexity is beneficial (if baseline >> ViM2)

#### 3. Ensemble Component
- Can train multiple ViM2 models quickly for ensembling
- Complementary to deeper models like baseline ViM

#### 4. Resource-Constrained Deployment
- If performance is acceptable, enables deployment on smaller GPUs
- Faster inference time for real-time applications

### Training Strategy

Same as baseline:
```python
Optimizer: AdamW
  - Learning rate: 1e-4
  - Weight decay: 1e-4
  
Scheduler: CosineAnnealingLR (T_max=50)

Batch size: 64 (same as baseline)

Gradient clipping: max_norm=1.0
```

**Note**: Keeping lr and weight_decay same as baseline allows direct comparison.

### Potential Improvements if Performance Lags

If ViM2 underperforms significantly:

#### Option A: Selective Capacity Increase
```python
# Increase depth while keeping d_model low
depth = 10  # vs 8
# Maintains speed advantage while adding integration capacity
```

#### Option B: Bottleneck Architecture
```python
# Wider embedding, then compress
self.patch_embed = nn.Sequential(
    nn.Linear(1, 192),  # Wide initial representation
    nn.GELU(),
    nn.Linear(192, 128)  # Compress for processing
)
```

#### Option C: Attention Pooling
```python
# Replace global average pooling with learned attention
# Allows model to focus on most informative tokens
```

#### Option D: Increase Batch Size
```python
# With reduced memory, can increase batch size
batch_size = 96  # vs 64
# Better gradient estimates may compensate for lower capacity
```

### Ablation Study Value

By comparing ViM (baseline) vs ViM2:
- **Quantifies capacity requirements**: How much depth/width is needed?
- **Speed/accuracy tradeoff**: Is 35% faster worth X% F1 loss?
- **Overfitting check**: Does baseline overfit while ViM2 generalizes?

If ViM2 ≈ ViM in performance:
- Suggests baseline is over-parameterized
- ViM2 becomes the new baseline (faster, simpler)
- Motivates even lighter variants

If ViM >> ViM2 in performance:
- Validates the need for full capacity
- Suggests STM patterns are complex and benefit from depth
- Justifies the longer training time

### Expected Results Summary

| Metric | Baseline (ViM) | Variant 2 (ViM2) |
|--------|----------------|------------------|
| Parameters | ~8M | ~3.5M |
| Time/epoch | ~30-35 min | ~20-25 min |
| Memory/batch | ~18-20 GB | ~12-14 GB |
| Macro-F1 (expected) | X% | X-2 to X-4% |
| Speech F1 | High | Similar |
| Music F1 | Medium | Slightly lower |
| Env F1 | Low | Lower |

### Usage

```bash
# Standard training
python STM_ViM2.py 0

# Downsampled
python STM_ViM2.py 1
```

### File Structure

```
model/STM/ViM2_corpora_categories/
├── standard/
│   └── ckpt/
│       └── YYYY-MM-DD_HH-MM/
│           ├── best_model.pt
│           ├── test_predictions.npy
│           └── test_targets.npy
└── downsample/
    └── ...
```

### Next Steps After Training

1. **Compare with baseline**: 
   - Plot learning curves (train/val loss, val F1)
   - Per-class F1 comparison
   - Confusion matrix comparison

2. **Analyze failure cases**:
   - Which samples does ViM2 misclassify that baseline gets right?
   - Are they genuinely harder or just need more capacity?

3. **Decide on tradeoff**:
   - If performance gap is small (<3% F1): Use ViM2 as default
   - If gap is moderate (3-5% F1): Context-dependent choice
   - If gap is large (>5% F1): Baseline is necessary

### References

- Base architecture from STM_ViM.py (baseline model)
- Follows same principles: symmetric STM, bidirectional Mamba, Balanced Softmax
- Hyperparameter choices inspired by efficient vision models (EfficientNet, MobileNet concepts applied to Mamba)
