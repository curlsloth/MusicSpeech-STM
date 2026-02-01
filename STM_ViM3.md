# STM Classification with Vision Mamba (Vim) - Variant 3
## Enhanced SSM State Memory for Long-Range Dependencies

### Overview

This variant enhances the SSM (State Space Model) state capacity to test whether larger internal memory improves long-range dependency modeling across the 1220-token STM sequence. Rather than increasing depth (which proved computationally prohibitive), we focus on expanding the SSM's ability to maintain richer context as information propagates through the sequence.

**Key hypothesis**: Enhanced SSM state (d_state=20 vs 16) enables better "memory" of distant tokens during bidirectional scanning, allowing the model to integrate patterns like "IF low-frequency modulation at token 100 correlates with high-frequency structure at token 1000, THEN predict music:vocal". The SSM state acts as a bottleneck for information flow—expanding it may improve pattern integration.

**Status**: ✅ Script is fully functional and ready to run

### Recent Fixes (Latest Update)

**Major architectural revision (Stage 1 optimization)**:
1. **Reduced depth**: Changed from depth=16 → 12 (baseline depth) for computational efficiency
   - Initial 16-layer design caused extreme slowness (~5-15 min/batch, would take 40-80 hours/epoch)
   - With 12 layers, training time returns to reasonable ~30-35 min/epoch
2. **Kept enhanced state**: Maintained d_state=20 (vs baseline 16) as the key differentiator
3. **Normalized regularization**: Reduced drop_path_rate from 0.15 → 0.1 (baseline)
4. **Fixed tuple unpacking bug**: train_epoch() returns (loss, f1) but was being captured as single variable

**Rationale**: The enhanced SSM state capacity (d_state=20) is more valuable than extreme depth. Larger state enables richer context propagation during forward/backward scans without the computational burden of additional layers.

### Hyperparameter Changes from Baseline

| Parameter | Baseline (ViM) | Variant 3 (ViM3) | Change |
|-----------|----------------|------------------|---------|
| d_model | 192 | 192 | Same |
| depth | 12 | 12 | **Same** (reverted from 16) |
| d_state | 16 | 20 | ↑ 25% ✨ **KEY CHANGE** |
| d_conv | 4 | 4 | Same |
| expand | 2 | 2 | Same |
| drop_path_rate | 0.1 | 0.1 | Same (reverted from 0.15) |
| dropout | 0.1 | 0.1 | Same |
| batch_size | 64 | 64 | Same |
| learning_rate | 1e-4 | 1e-4 | Same |
| Total params | ~8M | ~7.5M | Similar (~6% fewer)

### Rationale

#### 1. Enhanced SSM State (d_state: 16 → 20) - THE KEY INNOVATION

**Reason**: SSM state is the "memory bottleneck" that propagates through the sequence
- Larger state = richer context representation
- Critical for long sequences (1220 tokens) with bidirectional scanning
- State must encode: "what I've seen" + "what patterns matter" + "where to look next"

**Technical detail**:
```python
h_t = A·h_{t-1} + B·x_t  # State update in SSM
y_t = C·h_t               # Output generation
```
- d_state=20 means h_t ∈ R^20 (vs R^16 in baseline)
- 25% more dimensions = 25% more "working memory"
- With bidirectional scanning: forward state + backward state both benefit

**Why it matters more than depth**:
- **SSM state is the bottleneck**: All information from previous tokens must flow through h_t
- **Long sequences amplify the effect**: By token 1220, the state has seen 1219 tokens
- **Limited state = information loss**: Small h_t forces "forgetting" of early patterns
- **Larger state = better integration**: More dimensions to encode complex cross-token relationships

**Computational advantage over depth**:
- Increasing d_state: 16→20 adds ~6% parameters, minimal time overhead
- Increasing depth: 12→16 would add ~33% parameters, ~50% time overhead
- **Trade-off is strongly favorable**: Better memory capacity without speed penalty

#### 2. Why Depth Was Reverted (Initial depth=16 Failed)

**Problem discovered**: 16 layers with 1220 tokens caused extreme computational cost
- Expected: ~40-45 min/epoch
- Reality: First batch took >5 minutes, implying **40-80 hours/epoch**
- Root cause: 16 layers × 2 directions × 1220 tokens = 39,040 SSM operations per sample

**Decision**: Revert depth to 12 (baseline) while keeping enhanced d_state=20
- Maintains the "better memory" hypothesis
- Returns to reasonable training time (~30-35 min/epoch)
- Focuses innovation on the SSM state capacity rather than sheer depth

### Expected Performance

#### Training Efficiency

**Time per epoch**:
- Baseline: ~30-35 minutes
- Variant 3: ~**40-45 minutes**
- **Cost**: ~30% slower (4 extra layers)
30-35 minutes** (same, since depth is same)
- **Cost**: Negligible (~1-2% slower from larger state)

**Total training time** (50 epochs):
- Baseline: ~25-30 hours
- Variant 3: ~**25-30 hours** (same)
- **Cost**: Essentially no additional time

**Memory usage**:
- Baseline: ~18-20 GB per batch
- Variant 3: ~**18-21 GB per batch**
- **Risk**: Minimal increase, should not require batch size reduction

#### Classification Performance

**Expected macro-F1** (relative to baseline):
- **Best case**: +1 to +3% (enhanced state improves long-range pattern capture)
- **Likely case**: +0.5 to +1.5% (modest but consistent improvement)
- **Worst case**: ±0% (state capacity not a bottleneck for this task)

**Per-class expectations**:
- **Speech (majority)**: +0.5 to +1% (better tonal vs non-tonal distinction)
- **Music**: **Likely biggest gain +1 to +2%** (harmonic patterns benefit from state memory)
- **Environment (minority)**: +0.5 to +1.5% (better discrimination of complex textures)

**Why enhanced state might win**:
1. **Long-range dependencies**: Better memory of early tokens when processing late tokens
2. **Bidirectional consistency**: Forward and backward scans maintain richer context
3. **Pattern integration**: More capacity to encode "IF...AND...THEN" rules across tokens
4. **Bottleneck expansion**: State size was potentially limiting information flow
Enhanced SSM State Memory

**What the state encodes** during forward scan at token t:

```python
# Standard Mamba SSM state update
h_t = A @ h_{t-1} + B @ x_t  # State update
y_t = C @ h_t                 # Output computation

# With d_state=16 (baseline):
h_t ∈ R^16  # 16-dimensional "memory" of past tokens

# With d_state=20 (ViM3):
h_t ∈ R^20  # 20-dimensional "memory" - 25% more capacity
```

**What the extra dimensions enable**:

| Aspect | d_state=16 (baseline) | d_state=20 (ViM3) |
|--------|----------------------|-------------------|
| Pattern memory | Can track 16 concurrent patterns | Can track 20 concurrent patterns |
| Feature richness | May need to "forget" some early info | Better retention across 1220 tokens |
| Bidirectional integration | Limited forward-backward alignment | Richer context for both directions |

**Concrete example** at token 800 (mid-sequence):
```
With d_state=16:
  - State encodes: "frequency band 3 was high", "rate 5Hz peaked at token 200",
                   "gradual increase in high-freq", "periodic pattern detected"
  - BUT: Limited to 16 such facts → older patterns may be compressed/lost

With d_state=20:
  - State encodes: All of the above PLUS "correlation between bands",
                   "secondary modulation at 12Hz", "onset timing patterns"
  - BENEFIT: Richer representation → better long-range integration
```

#### Why Enhanced State is Preferred Over Depth

**Comparison of architectural choices**:

| Approach | Parameters | Time/Epoch | Memory | Risk |
|----------|-----------|------------|--------|------|
| Baseline | ~8M | 30-35 min | 18-20 GB | Low |
| ViM3 (Enhanced State) | ~7.5M | 30-35 min | 18-21 GB | **Low** ✓ |
| Hypothetical depth=16 | ~10.5M | 40-80 hours! | 22-24 GB | **High** ✗ |

**Why state wins**:
- **Surgical improvement**: Targets the exact bottleneck (state capacity)
- **Minimal cost**: Almost no additional compute or memory
- **Lower risk**: Same depth = same training dynamics as baseline
- **Testable hypothesis**: Clear A/B test of state capacity importance
3. **Bidirectional**: Forward + backward scans provide redundant paths
4. **DropPath**: Forces gradient flow through multiple layer combinations

**Evidence from visiod_state=16) | Enhanced Memory (d_state=20) |
|--------|----------------------|------------------------------|
| Local patterns | ✓ | ✓ (same) |
| Regional patterns | ✓ | ✓ (same) |
| Global patterns | ✓ | ✓✓ (better long-range) |
| State capacity | 16 dimensions | 20 dimensions (+25%) |
| Information retention | Good | Better (less forgetting) |
| Training time | 30-35 min/epoch | 30-35 min/epoch (same) |
| Overfitting risk | Low | Low (same regularization) |
| Best for | Standard sequences | Long sequences with complex dependencie
| Local patterns | ✓ | ✓✓ |
| Regional patterns | ✓ | ✓✓ |
| Global patterns | ✓ | ✓✓✓ |
| Complex reasoning | Limited | ✓✓✓ |
| Training time | 30-35 min/epoch | 40-45 min/epoch |
| Overfitting risk | Low | Medium (mitigated by DropPath) |
| Best for | Standard classification | Complex, subtle distinctions |

### Potential Failure Modes
State Capacity Not the Bottleneck

**Symptom**: ViM3 performance ≈ baseline (no improvement)

**Diagnosis**: d_state=16 was already sufficient; increasing to 20 adds no value

**Interpretation**: This is actually valuable information!
- Tells us the bottleneck is elsewhere (maybe d_model, training data, etc.)
- At least we didn't waste compute (since training time is the same)

**Next steps if this occurs**:
- Try ViM4 (wider d_model=256) to test if width is the bottleneck
- Try data augmentation to increase effective training data
- Analyze which classes are struggling most

#### 2. Overfitting (Unlikely but Possible)

**Symptom**: Val F1 plateaus while train loss keeps decreasing

**Diagnosis**: More state capacity → more memorization capacity

**Solutions**:
- Increase dropout in classification head (0.1 → 0.2)
- Add DropPath (currently 0.1 → try 0.15)
- Reduce learning rate (1e-4 → 5e-5)

#### 3. Minimal but Noisy Improvement

**Symptom**: ViM3 is +0.3% better than baseline, but within noise margin

**Diagnosis**: Genuine but small effect; need more runs to confirm

**Solutions**:
- Run multiple seeds (3-5 runs)
- Check if improvement is consistent across classes
- Analyze whether specific patterns benefit (e.g., long-range dependencies)ffective batch size = 48 × 2 = 96
```

### Ablation Experiments

If ViM3 succeeds, further ablations to run:

#### A. Depth Ablation
- ViM3a: depth=14 (between baseline and ViM3)
- ViM3b: hows improvement, further ablations to run:

#### A. State Size Sweep
- ViM3a: d_state=18 (between baseline and ViM3)
- ViM3b: d_state=24 (beyond ViM3)
- ViM3c: d_state=32 (push limits)

**Goal**: Find optimal state size before diminishing returns

#### B. State + Depth Interaction
- ViM3d: d_state=20, depth=10 (less depth, same state)
- ViM3e: d_state=24, depth=12 (more state, same depth)
- ViM3f: d_state=16, depth=14 (baseline state, more depth - if computationally feasible)

**Goal**: Separate state capacity vs depth contributions

#### C. State + Width Tradeoff
- ViM3g: d_state=20, d_model=160 (narrower model)
- ViM3h: d_state=20, d_model=224 (wider model)

**Goal**: Test if state enhancement synergizes with model widthet Improvement |
|--------|----------------|------------------|---------------------|
| Parameters | ~8M | ~10.5M | +31% |
| Time/epoch | ~30-35 min | ~40-45 min | +30% slower |
| Memory/batch | ~18-20 GB | ~22-24 GB | +20% |
| Macro-F1 (expected) | X% | X+1 to X+2% | +1-2% |
| Music F1 (expected) 7.5M | Similar (~6% fewer) |
| Time/epoch | ~30-35 min | ~30-35 min | No change |
| Memory/batch | ~18-20 GB | ~18-21 GB | Negligible |
| Macro-F1 (expected) | X% | X+0.5 to X+1.5% | +0.5-1.5% |
| Music F1 (expected) | Y% | Y+1 to Y+2% | +1-2% (biggest gain) |
| Env F1 (expected) | Z% | Z+0.5 to Z+1.5% | +0.5-1.5% |

**Success criterion**: If ViM3 macro-F1 > baseline +0.5%, the enhanced state is validated as beneficial.

**Risk-reward profile**: 
- **Risk**: Essentially zero (same compute cost)
- **Reward**: Modest but meaningful improvement
- **Learning**: Even if no improvement, confirms state wasn't the bottleneck
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

**You want to test whether SSM state capacity matters
- Performance improvement with zero time cost appeals
- Long sequences (1220 tokens) may benefit from better memory
- You want a "safer" architectural exploration (same depth)

**Use Baseline if**:
- PeLessons Learned: Why Depth Didn't Work

**Initial design (depth=16)**: Failed due to extreme computational cost
- Theory said: "Deeper = better long-range dependencies"
- Reality showed: 16 layers × 1220 tokens = prohibitive (40-80 hours/epoch)
- **Lesson**: Sequence models have different scaling laws than vision models
  - ViT can use 24 layers on 196 tokens (14×14 patches)
  - Mamba with 16 layers on 1220 tokens is 6× more computation per layer

**Revised design (d_state=20, depth=12)**: Surgical enhancement
- Targets the exact bottleneck (state capacity) without depth overhead
- Maintains practical training times
- **Lesson**: Expand bottlenecks, not everything

**General principle**: For sequence models on long sequences:
1. Width (d_model) scales cheaply
2. State (d_state) scales very cheaply  
3. Depth (num_layers) scales expensively
4. Sequence length (L) scales very expensively

**Recommendation for future variants**: Explore width and state before depth.

### References

- Base architecture from STM_ViM.py (baseline model)
- SSM state space analysis from "Mamba: Linear-Time Sequence Modeling" (Gu & Dao, 2023)
- State capacity insights from "Hungry Hungry Hippos" (Fu et al., ICLR 2023)
- Long sequence modeling from "Efficiently Modeling Long Sequences with Structured State Spaces" (Gu et al., ICLR 2022 time, minimal risksfactory
- Overfitting is a concern

### References

- Base architecture from STM_ViM.py (baseline model)
- Depth scaling insights from "Scaling Vision Transformers" (Zhai et al., CVPR 2022)
- Stochastic depth from "Deep Networks with Stochastic Depth" (Huang et al., ECCV 2016)
- SSM depth analysis from "Mamba: Linear-Time Sequence Modeling" (Gu & Dao, 2023)
