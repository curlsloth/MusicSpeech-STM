# STM Classification with Vision Mamba (Vim) - Variant 5
## Compact/Minimal Model for Ultra-Fast Experimentation

### Overview

This variant represents a minimal viable architecture, reducing both width and depth by 50% to establish a performance floor and enable rapid experimentation. The goal is to answer: "How much model capacity is actually necessary for STM classification?"

**Key hypothesis**: STM features may be sufficiently discriminative that even a compact model can achieve reasonable performance, especially for the well-represented speech/music classes. This variant prioritizes speed over capacity.

### Hyperparameter Changes from Baseline

| Parameter | Baseline (ViM) | Variant 5 (ViM5) | Change |
|-----------|----------------|------------------|---------|
| d_model | 192 | 96 | ↓ 50% |
| depth | 12 | 6 | ↓ 50% |
| d_state | 16 | 8 | ↓ 50% |
| d_conv | 4 | 4 | Same |
| expand | 2 | 2 | Same |
| drop_path_rate | 0.1 | 0.05 | ↓ 50% |
| dropout | 0.1 | 0.1 | Same |
| batch_size | 64 | 128 | ↑ 2× |
| learning_rate | 1e-4 | 1e-4 | Same |
| Total params | ~8M | ~1.2M | ↓ 85% |

### Rationale

#### 1. Minimal Model Dimension (d_model: 192 → 96)

**Reason**: Test minimum embedding size for STM
- 96 dims = half the baseline capacity
- Still provides 96 independent feature dimensions per token
- Sufficient for basic pattern recognition

**What 96 dims can capture**:
- Energy distribution across modulation rates
- Basic spectral shape
- Primary temporal patterns
- Coarse frequency band information

**What 96 dims cannot capture well**:
- Fine-grained spectral details
- Complex harmonic relationships
- Subtle cross-band interactions
- Higher-order statistics

**Why we test this**:
- Establishes performance floor
- If ViM5 ≈ Baseline: suggests over-parameterization
- If ViM5 << Baseline: validates need for capacity

#### 2. Minimal Depth (depth: 12 → 6)

**Reason**: Fastest possible training while maintaining architecture
- 6 layers = 6 bidirectional integrations
- Still allows 12 "hops" of information flow (forward + backward)
- Sufficient for local-to-regional patterns (not full global context)

**Integration capacity**:
```
Layer 1-2: Local context (±50 tokens)
Layer 3-4: Regional context (±200 tokens)
Layer 5-6: Limited global context (±500 tokens)
```

**Trade-off**: Cannot model full 1220-token dependencies effectively

#### 3. Minimal SSM State (d_state: 16 → 8)

**Reason**: Match reduced model dimension
- d_state/d_model ratio: 8/96 ≈ 8.3% (same as baseline)
- Maintains proportional SSM capacity
- 8 dims sufficient for basic sequence memory

#### 4. Reduced DropPath (drop_path_rate: 0.1 → 0.05)

**Reason**: Shallow models need less regularization
- With only 6 layers, dropping layers too aggressively hurts learning
- 0.05 provides light regularization without crippling capacity
- Linear schedule: 0.00 (layer 1) → 0.05 (layer 6)

#### 5. Increased Batch Size (batch_size: 64 → 128)

**Reason**: Compact model uses far less memory
- Memory per sample: ~1.2M params vs ~8M (baseline)
- Can fit 2× more samples per batch
- **Benefit**: Better gradient estimates compensate for reduced capacity

**Why larger batch helps**:
```python
Baseline (8M params, batch=64):
  Gradient variance: σ²/64
  Parameter updates: High precision

ViM5 (1.2M params, batch=128):
  Gradient variance: σ²/128 (more stable!)
  Parameter updates: Lower capacity but more reliable direction
```

### Expected Performance

#### Training Efficiency

**Time per epoch**:
- Baseline: ~30-35 minutes
- Variant 5: ~**12-15 minutes**
- **Speedup**: ~**2.5× faster**

**Total training time** (50 epochs):
- Baseline: ~25-30 hours
- Variant 5: ~**10-12 hours**
- **Benefit**: Can train overnight or in a few hours

**Memory usage**:
- Baseline: ~18-20 GB per batch (batch_size=64)
- Variant 5: ~**10-12 GB per batch** (batch_size=128)
- **Benefit**: Leaves headroom for larger batches or multi-GPU

**Throughput**:
- Baseline: ~64 samples / 2 sec ≈ 32 samples/sec
- ViM5: ~128 samples / 1.2 sec ≈ 107 samples/sec
- **Speedup**: 3.3× samples per second

#### Classification Performance

**Expected macro-F1** (relative to baseline):
- **Best case**: -3 to -5% (minimal capacity surprisingly effective)
- **Likely case**: -5 to -8% (capacity limitation visible but not catastrophic)
- **Worst case**: -8 to -12% (too minimal for complex patterns)

**Per-class expectations**:
- **Speech:non-tonal (majority)**: -2 to -4% (abundant data helps small model)
- **Speech:tonal**: -4 to -6% (subtle distinctions need more capacity)
- **Music:vocal**: -5 to -8% (complex harmonics suffer)
- **Music:non-vocal**: -6 to -10% (diverse patterns need capacity)
- **Environment (minority)**: -8 to -15% (limited data + low capacity = struggle)

**Why performance degrades**:
1. **Limited feature capacity**: 96 dims cannot represent subtle patterns
2. **Shallow integration**: 6 layers insufficient for global context
3. **Minority class difficulty**: Small model struggles with rare classes

**Why performance isn't catastrophic**:
1. **Symmetric STM**: Already reduces dimensionality (1220 vs 2420 tokens)
2. **Balanced Softmax**: Compensates for class imbalance
3. **Core patterns**: Main speech/music distinctions are robust
4. **Large batch**: Stable gradients enable effective learning

### Use Cases

#### 1. Rapid Prototyping
- Test data preprocessing variants in hours, not days
- Quick sanity checks before committing to full training
- Initial hyperparameter exploration

#### 2. Ablation Baseline
- Establishes performance floor for comparison
- Shows value of additional capacity (baseline vs ViM5)
- Quantifies capacity requirements

#### 3. Ensemble Component
- Train many ViM5 variants quickly (different seeds, augmentations)
- Ensemble of 5× ViM5 models in same time as 1× baseline
- Diversity may compensate for individual model weakness

#### 4. Feature Extractor
- Use ViM5 as fast feature encoder
- Freeze ViM5, train lightweight classifier on top
- Transfer learning scenario

#### 5. Educational/Debugging
- Fast enough to run on laptops (if scaled down further)
- Understand architecture without waiting hours
- Debugging code changes

### Architectural Insights

#### Minimal Viable Architecture (MVA)

**What ViM5 tests**:
- Can Vision Mamba work at all with minimal resources?
- Is the bottleneck capacity or architecture?
- How much performance scales with parameters?

**Comparison to linear baselines**:
```python
ViM5: 1.2M params, 6 layers, O(L) complexity
Logistic Regression: ~7K params, 0 layers, O(1) complexity
MLP (2-layer): ~250K params, 2 layers, O(1) complexity
```

ViM5 should significantly outperform linear/MLP baselines despite being "minimal"

#### Parameter Efficiency

**Parameter distribution**:
```
Embedding layer: ~96K (96 × 1 × 1000)
Position embedding: ~117K (1220 × 96)
6 × Vim blocks: ~900K (150K per block)
Classification head: ~50K
Total: ~1.2M
```

**Compared to baseline (8M)**:
```
Baseline: 8M params for 12 layers @ d_model=192
ViM5: 1.2M params for 6 layers @ d_model=96
Ratio: 8M / 1.2M = 6.67× parameter reduction
Depth ratio: 12 / 6 = 2×
Width ratio: 192 / 96 = 2×
Combined: 2 × 2 = 4× (close to observed 6.67× due to head scaling)
```

### Experimental Value

#### Performance vs Capacity Curve

By comparing all variants, we can plot:
```
Variant   | Params | Macro-F1 (expected)
----------|--------|--------------------
ViM5      | 1.2M   | X - 7%
ViM2      | 3.5M   | X - 3%
ViM       | 8M     | X
ViM3      | 10.5M  | X + 2%
ViM4      | 13M    | X + 1%
```

**Insights**:
- Diminishing returns above ~8M params?
- Sweet spot around 3.5-8M params?
- Is performance linear, logarithmic, or sigmoid with capacity?

#### Speed vs Accuracy Trade-off

```
Variant   | Time/epoch | Macro-F1 | Efficiency Score
----------|------------|----------|------------------
ViM5      | 15 min     | X - 7%   | (X-7) / 15 = ?
ViM2      | 23 min     | X - 3%   | (X-3) / 23 = ?
ViM       | 33 min     | X        | X / 33 = ?
ViM3      | 43 min     | X + 2%   | (X+2) / 43 = ?
ViM4      | 38 min     | X + 1%   | (X+1) / 38 = ?
```

Calculate: Which variant maximizes performance per minute?

### Potential Improvements if Performance is Acceptable

If ViM5 achieves, say, only -5% macro-F1 vs baseline:

#### Option A: Knowledge Distillation
```python
# Train ViM (teacher) first
# Train ViM5 (student) to match ViM's logits

loss = α × CE(student_logits, labels) + (1-α) × KD(student_logits, teacher_logits)
```
**Benefit**: ViM5 learns to mimic ViM's decision boundaries

#### Option B: Ensemble of ViM5 Models
```python
# Train 5 different ViM5 models
# Ensemble predictions

ensemble_pred = majority_vote([vim5_1, vim5_2, ..., vim5_5])
```
**Cost**: 5 × 12 = 60 hours (still 2× faster than 1× baseline)
**Benefit**: Diversity improves robustness

#### Option C: Hybrid Architecture
```python
# Use ViM5 for initial features
# Add task-specific heads

vim5_features = vim5.extract_features(x)  # (batch, 96)
speech_head = SpeechClassifier(vim5_features)
music_head = MusicClassifier(vim5_features)
```

### Expected Results Summary

| Metric | Baseline (ViM) | Variant 5 (ViM5) | Speedup/Reduction |
|--------|----------------|------------------|-------------------|
| Parameters | ~8M | ~1.2M | 6.7× smaller |
| Time/epoch | ~30-35 min | ~12-15 min | 2.5× faster |
| Memory/batch | ~18-20 GB | ~10-12 GB | 1.7× less |
| Throughput | ~32 samples/s | ~107 samples/s | 3.3× faster |
| Macro-F1 (expected) | X% | X-5 to X-8% | -5 to -8% |
| Speech F1 | High | Medium-High | -2 to -4% |
| Music F1 | Medium | Low-Medium | -6 to -10% |
| Env F1 | Low | Very Low | -10 to -15% |

**Success criterion**: If ViM5 achieves >60% of baseline performance with <20% of parameters, the architecture is highly parameter-efficient.

### Usage

```bash
# Standard training (faster batch_size=128)
python STM_ViM5.py 0

# Downsampled
python STM_ViM5.py 1
```

### File Structure

```
model/STM/ViM5_corpora_categories/
├── standard/
│   └── ckpt/
│       └── YYYY-MM-DD_HH-MM/
│           ├── best_model.pt
│           ├── test_predictions.npy
│           └── test_targets.npy
└── downsample/
    └── ...
```

### When to Use ViM5 vs Others

**Use ViM5 if**:
- Extremely time-constrained (need results today)
- Running multiple experiments simultaneously
- Testing hypotheses quickly
- Resource-constrained environment
- Educational purposes
- Don't need state-of-the-art performance

**Don't use ViM5 if**:
- Performance is critical
- Working on minority classes specifically
- Need publication-quality results
- Have sufficient compute budget

### Comparison Summary Table

| Variant | Philosophy | Params | Speed | F1 (expected) | Best For |
|---------|-----------|--------|-------|---------------|----------|
| ViM5 | Minimal | 1.2M | 2.5× | X-7% | Speed, prototyping |
| ViM2 | Light | 3.5M | 1.5× | X-3% | Fast + decent perf |
| ViM | Balanced | 8M | 1× | X | Default choice |
| ViM3 | Deep | 10.5M | 0.75× | X+2% | Complex patterns |
| ViM4 | Wide | 13M | 0.85× | X+1% | Rich features |

**Strategic use**: Start with ViM5 for rapid iteration → Move to ViM2 for better results → Use ViM/ViM3/ViM4 for final model.

### References

- Base architecture from STM_ViM.py (baseline model)
- Minimal model design from "MobileNets" (Howard et al., 2017)
- Efficiency analysis from "EfficientNet: Rethinking Model Scaling" (Tan & Le, ICML 2019)
- Fast training insights from "Mixed Precision Training" (Micikevicius et al., ICLR 2018)
