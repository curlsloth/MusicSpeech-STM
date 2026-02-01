# STM Classification with Vision Mamba (Vim) - Variant 3
## Deeper Model for Complex Long-Range Dependencies

### Overview

This variant increases model depth to test whether additional layers can capture more sophisticated modulation patterns and long-range dependencies across the STM spectrum. The hypothesis is that speech/music/environment classification may benefit from deeper feature hierarchies, similar to how deeper CNNs learn increasingly abstract visual representations.

**Key hypothesis**: Deeper integration paths enable the model to learn multi-hop reasoning like "IF 2Hz rate is high (token 100) AND 8Hz rate is low (token 400) AND 4cyc/oct is moderate (token 800), THEN predict music:vocal". Such complex rules may require >12 layers to learn effectively.

**Status**: ✅ Script is fully functional and ready to run

### Recent Fixes (Latest Update)

1. **Added missing import**: Added `torch.nn.functional as F` for loss computation
2. **Fixed `train_epoch`**: Now returns both loss and F1 score as expected by training loop

### Hyperparameter Changes from Baseline

| Parameter | Baseline (ViM) | Variant 3 (ViM3) | Change |
|-----------|----------------|------------------|---------|
| d_model | 192 | 192 | Same |
| depth | 12 | 16 | ↑ 33% |
| d_state | 16 | 20 | ↑ 25% |
| d_conv | 4 | 4 | Same |
| expand | 2 | 2 | Same |
| drop_path_rate | 0.1 | 0.15 | ↑ 50% |
| dropout | 0.1 | 0.1 | Same |
| batch_size | 64 | 64 | Same |
| learning_rate | 1e-4 | 1e-4 | Same |
| Total params | ~8M | ~10.5M | ↑ 31% |

### Rationale

#### 1. Increased Depth (depth: 12 → 16)

**Reason**: More layers = longer "information integration path"
- 16 layers allow 16-hop message passing across the sequence
- Enables hierarchical feature learning:
  - Layers 1-4: Local patterns (within 50-100 tokens)
  - Layers 5-8: Regional patterns (100-300 tokens)
  - Layers 9-12: Global patterns (300+ tokens)
  - Layers 13-16: Cross-spectrum integration (full 1220 tokens)

**Analogy to CNNs**:
- Early layers: edges/textures (local modulation patterns)
- Middle layers: object parts (spectral bands, rate ranges)
- Deep layers: semantic concepts (speech formants, music harmonics)

**Why it might help**:
- STM has 20 frequency bands × 61 rates = complex 2D structure
- Flattened to 1220 tokens, but semantics still 2D
- Deeper model can "unflatten" and learn 2D relationships

#### 2. Larger SSM State (d_state: 16 → 20)

**Reason**: SSM state is the "memory" that propagates through the sequence
- Larger state can remember more context
- Critical for long sequences (1220 tokens)
- State must encode: "what I've seen so far" + "what to look for next"

**Technical detail**:
```python
h_t = A·h_{t-1} + B·x_t  # State update in SSM
y_t = C·h_t               # Output generation
```
- d_state=20 means h_t ∈ R^20 (vs R^16 in baseline)
- More dimensions = richer representation of sequence history

**Why it might help**:
- Prevents "forgetting" of early tokens by the time we reach token 1220
- Enables better bidirectional integration (forward + backward scans)

#### 3. Increased DropPath (drop_path_rate: 0.1 → 0.15)

**Reason**: Deeper models need stronger regularization
- DropPath randomly drops entire layers during training
- Forces each layer to be independently useful
- Prevents over-reliance on specific layer combinations

**Stochastic depth schedule**:
```python
# drop_prob increases linearly from 0 to 0.15 across 16 layers
Layer 1: drop_prob = 0.00
Layer 4: drop_prob = 0.03
Layer 8: drop_prob = 0.07
Layer 12: drop_prob = 0.11
Layer 16: drop_prob = 0.15
```

**Why it's critical**:
- Without increased DropPath, 16 layers might overfit
- Especially important given limited training data for minority classes
- Maintains generalization while increasing capacity

### Expected Performance

#### Training Efficiency

**Time per epoch**:
- Baseline: ~30-35 minutes
- Variant 3: ~**40-45 minutes**
- **Cost**: ~30% slower (4 extra layers)

**Total training time** (50 epochs):
- Baseline: ~25-30 hours
- Variant 3: ~**33-38 hours**
- **Cost**: Additional 5-8 hours

**Memory usage**:
- Baseline: ~18-20 GB per batch
- Variant 3: ~**22-24 GB per batch**
- **Risk**: May need to reduce batch_size to 48 or 32 on some GPUs

#### Classification Performance

**Expected macro-F1** (relative to baseline):
- **Best case**: +3 to +5% (depth enables complex pattern learning)
- **Likely case**: +1 to +2% (modest improvement, worth the cost)
- **Worst case**: 0 to -1% (overfitting, regularization insufficient)

**Per-class expectations**:
- **Speech (majority)**: Similar or slightly better (more capacity for subtle distinctions)
- **Music**: **Likely biggest gain** (complex harmonic structures benefit from depth)
- **Environment (minority)**: May improve if deeper features generalize better

**Why deeper might win**:
1. **Compositional patterns**: "Music = formants + periodicity + spectral centroid clustering"
2. **Negation logic**: "NOT speech (lacks formants) AND high spectral flux → environment"
3. **Cross-band reasoning**: "IF low-freq energy high AND high-freq energy low → bass-heavy music"

### Architectural Insights

#### Information Flow Through 16 Layers

**Visualization** of what each depth range might learn:

```
Input: (batch, 1220 tokens, 192 dims)
↓
Layers 1-4: "Texture extraction"
  - Token 100 (2Hz rate): Is energy concentrated or diffuse?
  - Token 500 (7.5Hz rate): Periodic or noisy?
  - Output: (batch, 1220, 192) with local descriptors
↓
Layers 5-8: "Motif detection"
  - Cluster of tokens 200-300: Formant-like pattern?
  - Cluster of tokens 800-900: Harmonic series?
  - Output: (batch, 1220, 192) with regional summaries
↓
Layers 9-12: "Global structure"
  - Across all freq bands: Is there spectro-temporal coherence?
  - Across all rates: Dominant modulation frequency?
  - Output: (batch, 1220, 192) with global context
↓
Layers 13-16: "Decision-relevant abstraction"
  - Combine all cues: speech-ness, music-ness, env-ness
  - Output: (batch, 1220, 192) optimized for classification
↓
Global Average Pooling: (batch, 192)
↓
Classification Head: (batch, 6)
```

#### Why 16 is Not Too Deep

**Concern**: Will 16 layers cause vanishing gradients?

**Mitigations in our architecture**:
1. **Residual connections**: Every VimBlock has `x = x + block(x)`
2. **LayerNorm**: Stabilizes activations before each SSM
3. **Bidirectional**: Forward + backward scans provide redundant paths
4. **DropPath**: Forces gradient flow through multiple layer combinations

**Evidence from vision**:
- ViT (Vision Transformer): 24 layers standard, 32 layers for ViT-Large
- ResNet: 50-152 layers common
- Our 16 layers with residuals ≈ 32-layer CNN without residuals

### Comparison with Baseline

| Aspect | Baseline (12 layers) | Deeper (16 layers) |
|--------|----------------------|---------------------|
| Local patterns | ✓ | ✓✓ |
| Regional patterns | ✓ | ✓✓ |
| Global patterns | ✓ | ✓✓✓ |
| Complex reasoning | Limited | ✓✓✓ |
| Training time | 30-35 min/epoch | 40-45 min/epoch |
| Overfitting risk | Low | Medium (mitigated by DropPath) |
| Best for | Standard classification | Complex, subtle distinctions |

### Potential Failure Modes

#### 1. Overfitting

**Symptom**: Val F1 plateaus or decreases while train loss keeps decreasing

**Diagnosis**:
```python
# Check train vs val gap
if train_loss < 0.1 and val_loss > 0.5:
    print("Overfitting detected!")
```

**Solutions**:
- Increase DropPath to 0.2
- Add dropout to classification head (0.1 → 0.2)
- Reduce learning rate (1e-4 → 5e-5)
- Use mixup data augmentation

#### 2. Slow Convergence

**Symptom**: Val F1 at epoch 25 << Baseline's val F1 at epoch 25

**Diagnosis**: Deeper models may need more iterations to converge

**Solutions**:
- Increase num_epochs to 75
- Use learning rate warmup for first 5 epochs
- Lower initial learning rate (1e-4 → 5e-5)

#### 3. Memory Overflow

**Symptom**: CUDA out of memory error

**Solutions**:
```python
# Reduce batch size
batch_size = 48  # or 32

# Enable gradient accumulation
accumulation_steps = 2
# Effective batch size = 48 × 2 = 96
```

### Ablation Experiments

If ViM3 succeeds, further ablations to run:

#### A. Depth Ablation
- ViM3a: depth=14 (between baseline and ViM3)
- ViM3b: depth=18 (beyond ViM3)
- ViM3c: depth=20 (push limits)

**Goal**: Find optimal depth

#### B. State Size Ablation
- ViM3d: d_state=24 (larger than ViM3)
- ViM3e: d_state=16 (baseline state, 16 layers depth)

**Goal**: Separate depth vs state contributions

#### C. Width vs Depth Tradeoff
- ViM3f: depth=16, d_model=128 (narrow & deep)
- ViM3g: depth=8, d_model=256 (wide & shallow)

**Goal**: Is depth or width more important for STM?

### Expected Results Summary

| Metric | Baseline (ViM) | Variant 3 (ViM3) | Target Improvement |
|--------|----------------|------------------|---------------------|
| Parameters | ~8M | ~10.5M | +31% |
| Time/epoch | ~30-35 min | ~40-45 min | +30% slower |
| Memory/batch | ~18-20 GB | ~22-24 GB | +20% |
| Macro-F1 (expected) | X% | X+1 to X+2% | +1-2% |
| Music F1 (expected) | Y% | Y+2 to Y+4% | +2-4% (biggest gain) |
| Env F1 (expected) | Z% | Z+1 to Z+2% | +1-2% |

**Success criterion**: If ViM3 macro-F1 > baseline +1%, the added depth is justified.

### Usage

```bash
# Standard training
python STM_ViM3.py 0

# Downsampled
python STM_ViM3.py 1
```

### File Structure

```
model/STM/ViM3_corpora_categories/
├── standard/
│   └── ckpt/
│       └── YYYY-MM-DD_HH-MM/
│           ├── best_model.pt
│           ├── test_predictions.npy
│           └── test_targets.npy
└── downsample/
    └── ...
```

### Post-Training Analysis

After training, analyze:

1. **Layer-wise feature importance**:
   - Remove each layer, measure F1 drop
   - Identify which layers contribute most

2. **Attention visualization** (if we add attention):
   - Which tokens does layer 16 focus on?
   - Do deep layers look at different tokens than shallow layers?

3. **Gradient flow**:
   - Plot gradient magnitudes per layer
   - Ensure no vanishing/exploding gradients

### When to Use ViM3 vs Baseline

**Use ViM3 if**:
- Performance matters more than speed
- You have >24GB GPU memory
- Subtle classification boundaries (e.g., tonal vs non-tonal speech)
- Complex multi-cue integration needed

**Use Baseline if**:
- Fast iteration is critical
- Memory constrained
- Performance already satisfactory
- Overfitting is a concern

### References

- Base architecture from STM_ViM.py (baseline model)
- Depth scaling insights from "Scaling Vision Transformers" (Zhai et al., CVPR 2022)
- Stochastic depth from "Deep Networks with Stochastic Depth" (Huang et al., ECCV 2016)
- SSM depth analysis from "Mamba: Linear-Time Sequence Modeling" (Gu & Dao, 2023)
