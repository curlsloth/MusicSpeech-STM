# STM Classification with Vision Mamba (Vim) - Variant 4
## Wider Model for Richer Token Representations

### Overview

This variant increases the model width (d_model) while slightly reducing depth to test whether richer per-token representations are more important than deep integration for STM classification. The hypothesis is that with sufficient representation capacity, each token can encode complex features that require less cross-token communication.

**Key hypothesis**: "Width over depth" - If each token embedding is sufficiently expressive (256 dims vs 192), fewer layers may be needed to integrate information across the 1220-token sequence.

### Hyperparameter Changes from Baseline

| Parameter | Baseline (ViM) | Variant 4 (ViM4) | Change |
|-----------|----------------|------------------|---------|
| d_model | 192 | 256 | ↑ 33% |
| depth | 12 | 10 | ↓ 17% |
| d_state | 16 | 24 | ↑ 50% |
| d_conv | 4 | 4 | Same |
| expand | 2 | 3 | ↑ 50% |
| drop_path_rate | 0.1 | 0.1 | Same |
| dropout | 0.1 | 0.1 | Same |
| batch_size | 64 | 64 (may reduce to 48) | Same/Reduced |
| learning_rate | 1e-4 | 1e-4 | Same |
| Total params | ~8M | ~13M | ↑ 63% |

### Rationale

#### 1. Increased Model Dimension (d_model: 192 → 256)

**Reason**: Wider embeddings = more expressive token representations
- Each token goes from 192 → 256 dimensions
- Can encode more nuanced features per bin
- **Analogy**: Like increasing the color depth of an image (8-bit → 12-bit)

**What wider embeddings can capture**:
```python
# Token 500 (representing 7.5 Hz modulation rate at freq band 10)
# 192 dims (baseline): 
#   - Basic energy level
#   - Temporal structure
#   - Local context
# 256 dims (ViM4):
#   - ALL above, PLUS:
#   - Fine-grained spectral shape
#   - Multi-scale temporal patterns
#   - Interaction with neighboring bins
#   - Higher-order statistics
```

**Why this might help for STM**:
- Each (frequency, rate) bin contains rich spectro-temporal information
- 256 dims allows decomposing this into more orthogonal features
- Reduces need for deep integration if features are well-encoded initially

#### 2. Slightly Reduced Depth (depth: 12 → 10)

**Reason**: Parameter budget trade-off
- Increasing d_model from 192 → 256 already adds significant parameters
- Reducing depth 12 → 10 keeps total params reasonable (~13M vs ~16M if we kept 12 layers)
- **Test**: Is "wide & shallow" better than "narrow & deep" for STM?

**Information integration with 10 layers**:
- Still sufficient for global context (10 layers × bidirectional = 20 "hops")
- Majority of information integration happens in first 6-8 layers typically
- Layers 9-10 focus on fine-tuning decision boundaries

#### 3. Larger SSM State (d_state: 16 → 24)

**Reason**: SSM state needs to scale with d_model
- SSM operates on d_model-dimensional inputs
- Larger d_model → need proportionally larger state to maintain capacity
- d_state/d_model ratio: 16/192 ≈ 8.3% (baseline) vs 24/256 ≈ 9.4% (ViM4)

**State capacity**:
```
h_t ∈ R^{d_state} must summarize context for next token
Baseline: 16 dims to summarize 192-dim sequence history
ViM4: 24 dims to summarize 256-dim sequence history
```

#### 4. Increased Expand Factor (expand: 2 → 3)

**Reason**: Mamba's internal FFN needs wider hidden layer
- FFN hidden dim = d_model × expand
- Baseline: 192 × 2 = 384 hidden dims
- ViM4: 256 × 3 = 768 hidden dims (2× larger)

**Why expand=3**:
- With d_model=256, need more non-linear capacity
- Allows richer transformations of token representations
- Standard practice: larger models use larger expand factors

### Expected Performance

#### Training Efficiency

**Time per epoch**:
- Baseline: ~30-35 minutes
- Variant 4: ~**35-40 minutes**
- **Cost**: ~15% slower (wider embeddings but fewer layers)

**Memory usage**:
- Baseline: ~18-20 GB per batch (batch_size=64)
- Variant 4: ~**24-26 GB per batch** (batch_size=64)
- **Risk**: May need batch_size=48 on 24GB GPUs

**Total training time** (50 epochs):
- Baseline: ~25-30 hours
- Variant 4: ~**29-33 hours**
- **Cost**: Additional 4-5 hours (acceptable)

#### Classification Performance

**Expected macro-F1** (relative to baseline):
- **Best case**: +2 to +4% (width wins over depth for STM)
- **Likely case**: +0 to +1% (comparable to baseline)
- **Worst case**: -1 to 0% (width doesn't compensate for reduced depth)

**Per-class expectations**:
- **Speech (majority)**: Similar to baseline (both have sufficient capacity)
- **Music**: **Potential gain** (richer harmonic representations)
- **Environment**: **Potential gain** (subtle patterns benefit from wide embeddings)

**Why wider might win**:
1. **Feature quality**: Better initial embeddings reduce need for deep integration
2. **Efficiency**: Fewer layers = less gradient degradation
3. **Generalization**: Wider models sometimes generalize better (not overfit to layer patterns)

### Width vs Depth Trade-off

**Width-favoring scenarios**:
- Features are compositional (combine local patterns)
- Each token is semantically rich
- Integration is mostly local (within ~100-200 tokens)
- **Example**: If speech formants are localized in specific frequency bands

**Depth-favoring scenarios**:
- Features require complex reasoning across full sequence
- Hierarchical abstraction is critical
- Long-range dependencies (>500 tokens)
- **Example**: If classification depends on "token 100 AND NOT token 800"

**STM characteristics**:
- 20 freq bands × 61 rates = semi-local structure
- Spectral patterns are band-specific (favors width)
- BUT temporal patterns cross bands (favors depth)
- **Verdict**: Unclear a priori → experiment needed!

### Architectural Insights

#### Token Representation Capacity

**Comparison**:
```
Baseline: 192 dims per token
  → Can represent ~192 independent features
  → Sufficient for basic modulation spectrum
  
ViM4: 256 dims per token
  → Can represent ~256 independent features
  → Room for subtle distinctions:
    * Peak sharpness
    * Harmonic spacing
    * Temporal envelope
    * Cross-band coherence
```

**Visualization**: Imagine each token as a "snapshot" of one STM bin
- 192 dims: Standard photo resolution
- 256 dims: High-res photo with more detail
- Question: Does STM have that much detail per bin?

#### Parameter Distribution

**Baseline (8M params)**:
```
Embedding layer: ~234K
Position embedding: ~234K
12 × Vim blocks: ~7.2M (600K per block)
Classification head: ~58K
```

**ViM4 (13M params)**:
```
Embedding layer: ~256K
Position embedding: ~312K
10 × Vim blocks: ~12M (1.2M per block - 2× larger!)
Classification head: ~130K
```

**Key insight**: 10 wider layers (1.2M each) ≈ 12 narrower layers (600K each) in total params, but compute pattern is different.

### Comparison with Other Variants

| Aspect | ViM2 (Lighter) | ViM (Baseline) | ViM3 (Deeper) | ViM4 (Wider) | ViM5 (Compact) |
|--------|----------------|----------------|---------------|--------------|----------------|
| d_model | 128 | 192 | 192 | 256 | 96 |
| depth | 8 | 12 | 16 | 10 | 6 |
| Params | ~3.5M | ~8M | ~10.5M | ~13M | ~1.2M |
| Time/epoch | ~20-25 min | ~30-35 min | ~40-45 min | ~35-40 min | ~12-15 min |
| Best for | Speed | Balance | Complexity | Rich features | Ultra-fast |

**ViM4 niche**: Maximum representational capacity per token while maintaining reasonable depth.

### Potential Failure Modes

#### 1. Memory Overflow

**Symptom**: CUDA OOM error with batch_size=64

**Solution**:
```python
# Reduce batch size
batch_size = 48  # or 32

# OR use gradient accumulation
accumulation_steps = 2
effective_batch_size = 48 × 2 = 96
```

#### 2. Underfitting (Insufficient Depth)

**Symptom**: Val F1 lower than baseline despite more parameters

**Diagnosis**: 10 layers insufficient for global integration

**Solution**:
```python
# Ablation: Test depth=12 with d_model=256
# This creates ViM4b with ~15.6M params
depth = 12  # vs 10 in ViM4
```

#### 3. Overfitting (Too Wide for Data)

**Symptom**: Train F1 >> Val F1

**Solution**:
- Increase dropout: 0.1 → 0.15
- Increase weight_decay: 1e-4 → 2e-4
- Add mixup/cutmix augmentation

### Ablation Experiments

If ViM4 performs well, test:

#### A. Width Ablation
- ViM4a: d_model=224, depth=10 (intermediate width)
- ViM4b: d_model=288, depth=10 (even wider)

**Goal**: Find optimal width for STM

#### B. Depth Recovery
- ViM4c: d_model=256, depth=12 (restore baseline depth)
- ViM4d: d_model=256, depth=14 (wider + deeper)

**Goal**: Test if width + depth synergize

#### C. Expand Factor
- ViM4e: expand=2 (baseline expand, despite d_model=256)
- ViM4f: expand=4 (even larger FFN)

**Goal**: Isolate expand's contribution

### Expected Results Summary

| Metric | Baseline (ViM) | Variant 4 (ViM4) | Target |
|--------|----------------|------------------|---------|
| Parameters | ~8M | ~13M | +63% |
| Time/epoch | ~30-35 min | ~35-40 min | +15% |
| Memory/batch | ~18-20 GB | ~24-26 GB | +30% |
| Macro-F1 (expected) | X% | X to X+1% | +0-1% |
| Music F1 (expected) | Y% | Y+1 to Y+3% | +1-3% (potential) |
| Env F1 (expected) | Z% | Z+1 to Z+2% | +1-2% (potential) |

**Success criterion**: If ViM4 matches or exceeds baseline with fewer layers, "width > depth" hypothesis confirmed.

### Usage

```bash
# Standard training (may need batch_size=48)
python STM_ViM4.py 0

# Downsampled
python STM_ViM4.py 1
```

### File Structure

```
model/STM/ViM4_corpora_categories/
├── standard/
│   └── ckpt/
│       └── YYYY-MM-DD_HH-MM/
│           ├── best_model.pt
│           ├── test_predictions.npy
│           └── test_targets.npy
└── downsample/
    └── ...
```

### When to Use ViM4 vs Baseline

**Use ViM4 if**:
- You believe STM bins contain rich, independent information
- Memory is available (24-26 GB)
- You want to test "width over depth" hypothesis
- Feature quality matters more than integration

**Use Baseline if**:
- Memory constrained (<24 GB)
- Depth-first philosophy (ResNet-style)
- Baseline already performing well
- Want faster training than ViM4

### References

- Base architecture from STM_ViM.py (baseline model)
- Width scaling insights from "Scaling Vision Transformers" (Zhai et al., CVPR 2022)
- Width vs depth analysis from "Wide Residual Networks" (Zagoruyko & Komodakis, BMVC 2016)
- SSM capacity from "Mamba: Linear-Time Sequence Modeling" (Gu & Dao, 2023)
