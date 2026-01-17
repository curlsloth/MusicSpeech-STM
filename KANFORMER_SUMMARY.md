# Kanformer for STM Audio Classification

## Overview

This document describes the implementation of a **Kanformer** (KAN-integrated Conformer) architecture for classifying audio based on Spectrotemporal Modulation (STM) features. The Kanformer replaces the standard Feed-Forward Networks (FFNs) in the Conformer architecture with **Group-Rational Kolmogorov-Arnold Network (GR-KAN)** layers.

## Theoretical Motivation

### The Problem with Standard Activations

Traditional neural networks (including Conformers) use **fixed activation functions** like ReLU, GELU, or Swish. These functions are the same across all neurons and all layers. While the Universal Approximation Theorem guarantees that MLPs with fixed activations can approximate any continuous function, this requires:

1. **Many layers** to build up complex non-linearities
2. **Many parameters** to compensate for the rigidity of fixed activations
3. **Large datasets** to learn the right combinations

### The KAN Solution

Kolmogorov-Arnold Networks are based on the **Kolmogorov-Arnold representation theorem**, which states that any multivariate continuous function can be represented as a composition of univariate functions and addition.

**Key Innovation**: Instead of fixed activations on nodes with learnable weights on edges, KANs have:
- **Learnable univariate functions on edges** (parameterized as splines or rational functions)
- **Simple addition operations at nodes**

For STM classification, this means:
- The network can learn **custom activation curves** for different modulation frequencies
- It can be **highly sensitive** to specific "sweet spots" (e.g., 4 Hz for speech syllables)
- It can **suppress noise** in irrelevant frequency bands with learned non-linearities

### Why Rational Functions?

We use **rational functions** (ratios of polynomials) instead of B-splines because:

1. **Faster computation**: Polynomial evaluation is very fast on GPUs
2. **Better stability**: Less prone to numerical issues than splines
3. **Compact representation**: Fewer parameters than high-degree splines
4. **Proven effectiveness**: Achieved 4x parameter efficiency in speech tasks

## Architecture Details

### High-Level Structure

```
Input: (batch, 20, 121) STM grid
  ↓
Convolutional Stem (local feature extraction)
  ↓
Add 2D Positional Embeddings (anisotropic)
  ↓
Kanformer Block × 4
  ├─ KAN-FFN (half-step residual)
  ├─ Multi-Head Self-Attention
  ├─ Convolution Module
  └─ KAN-FFN (half-step residual)
  ↓
Global Average Pooling
  ↓
KAN Classifier → 6 classes
```

### Core Components

#### 1. RationalFunction Layer

Implements learnable activation: `P(x) / Q(x)`

- **Numerator P(x)**: Polynomial of degree 5
- **Denominator Q(x)**: Polynomial of degree 4
- **Initialization**: Approximates ReLU, then learns custom curve
- **Safety**: Denominator clamped to avoid division by zero

#### 2. GroupRationalKANLayer

Replaces standard FFN (Linear → ReLU → Linear) with:

```
LayerNorm → Linear → [Group-wise Rational Functions] → Dropout → Linear
```

**Grouping**: Features divided into 8 groups, each with its own rational function
- Reduces parameters (shares within groups)
- Preserves expressivity (different functions per group)

#### 3. Kanformer Block

Follows Conformer's "half-step" residual pattern:

```python
x = x + 0.5 * KAN_FFN1(x)      # Pre-processing
x = x + MHSA(x)                # Global context
x = x + ConvModule(x)          # Local smoothness
x = x + 0.5 * KAN_FFN2(x)      # Post-processing
```

**Why this works for STM**:
- **KAN-FFN**: Learns non-linear mappings specific to modulation patterns
- **MHSA**: Captures long-range dependencies (e.g., Rate ↔ Scale correlations)
- **ConvModule**: Exploits local continuity in the modulation grid

### Key Design Choices

#### Anisotropic Positional Encoding

Since the Rate (time) and Scale (frequency) axes have **different physical meanings**, we use:
- **Learnable 2D embeddings** (not sinusoidal)
- **Different scales** for each axis
- This respects the fact that "diagonal" relationships in STM are semantically complex

#### Focal Loss

To handle class imbalance (speech >> music >> environment):
- Standard cross-entropy down-weights minority classes
- **Focal Loss**: `FL = -(1 - p)^γ * log(p)`
- Focuses learning on hard examples
- Reduces overfitting to majority class

#### Gradient Clipping

KANs can have **unstable gradients** during early training because:
- Rational functions can produce large values if denominator is small
- Polynomial gradients can explode

**Solution**: Clip gradients to max norm of 1.0

#### Learning Rate Warmup

KAN layers need time to "learn" their activation shapes:
- First 5 epochs: Linear warmup from `0 → 1e-4`
- Then: Cosine annealing to `1e-6`

## Implementation Highlights

### File: `STMkanformer_model.py`

**Total Lines**: ~800

**Main Classes**:
1. `RationalFunction`: Learnable activation (P(x)/Q(x))
2. `GroupRationalKANLayer`: KAN-based FFN replacement
3. `MultiHeadSelfAttention`: Standard MHSA with relative positions
4. `ConvolutionModule`: Depthwise separable convolution
5. `KanformerBlock`: Full Conformer block with KAN-FFN
6. `KanformerClassifier`: Complete model
7. `prepData_STM_Kanformer`: Data loading (same as Conformer baseline)
8. `FocalLoss`: Class-balanced loss
9. `KanformerTrainer`: Training loop with warmup

### Model Configuration

```python
KanformerClassifier(
    input_dim=20,          # Spectral bins
    num_classes=6,         # Speech/Music/Env categories
    d_model=128,           # Hidden dimension
    num_heads=4,           # Attention heads
    ffn_dim=512,           # KAN intermediate dimension
    num_layers=4,          # Kanformer blocks
    kernel_size=31,        # Convolution kernel
    dropout=0.1,           # Dropout rate
    num_kan_groups=8       # KAN feature groups
)
```

### Parameter Count

Approximately **1.5-2M parameters** (slightly more than Conformer due to rational function coefficients)

## Usage

### Training

```bash
# Standard training (fresh start)
python STMkanformer_model.py 0

# With non-tonal speech downsampling
python STMkanformer_model.py 1

# Resume from cancelled training
python STMkanformer_model.py 0 --resume model/STM/Kanformer_corpora_categories/standard/ckpt/2025-01-15_14-30
```

### Resuming Training

If training is interrupted, you can resume from the last checkpoint:

```bash
# Find your checkpoint directory (e.g., model/STM/Kanformer_corpora_categories/standard/ckpt/2025-01-15_14-30)
# Resume training
python STMkanformer_model.py 0 --resume <path_to_checkpoint_dir>
```

The script will:
- Load model weights and optimizer state
- Resume from the next epoch after the last checkpoint
- Preserve training history (losses, F1 scores)
- Continue with the same learning rate schedule

**Checkpoint files**:
- `latest_checkpoint.pt`: Saved every epoch (for easy resumption)
- `checkpoint_epoch_X.pt`: Saved every 5 epochs
- `best_model.pt`: Best validation F1 model (used for final testing)

### Output

```
model/STM/Kanformer_corpora_categories/
├── standard/
│   └── ckpt/
│       └── 2025-01-XX_HH-MM/
│           ├── best_model.pt
│           ├── checkpoint_epoch_5.pt
│           ├── test_predictions.npy
│           └── test_targets.npy
└── downsample/
    └── ckpt/...
```

### Loading Trained Model

```python
checkpoint = torch.load('model/.../best_model.pt')
model = KanformerClassifier(...)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

## Expected Performance

Based on analogous tasks (ASVspoof synthetic speech detection):

| Metric | Baseline Conformer | Kanformer (Expected) |
|--------|-------------------|---------------------|
| Macro F1 | X% (baseline) | **(X + 5-8)%** |
| Test Accuracy | Y% | **(Y + 4-7)%** |
| Parameters | ~1.2M | ~1.8M |

**Key Advantage**: Kanformer should excel at:
- **Speech vs. Vocal Music** (most difficult boundary)
- **Tonal vs. Non-tonal Speech**
- Low-data regimes (learns better features with fewer samples)

## Comparison to Baseline Conformer

| Component | Conformer | Kanformer |
|-----------|-----------|-----------|
| Input Encoding | Conv1D | Conv1D + 2D Positional |
| Attention | MHSA | MHSA (same) |
| Convolution | Depthwise | Depthwise (same) |
| **FFN** | **Linear + Swish** | **KAN (Rational Functions)** |
| Loss | Cross-Entropy | Focal Loss |
| LR Schedule | ReduceLROnPlateau | Warmup + Cosine |

## Ablation Studies (Recommended)

To validate the KAN contribution, run:

1. **Kanformer vs. Conformer**: Direct comparison on same data
2. **Rational vs. Spline KAN**: Compare function types
3. **Group Size**: Test `num_kan_groups` ∈ {4, 8, 16}
4. **Focal Loss Impact**: Compare with standard CE loss

## Theoretical Justification (From Research Document)

1. **Manifold Approximation**: STM decision boundaries are **non-linear high-dimensional manifolds**. KANs approximate these with fewer parameters than MLPs (Kolmogorov-Arnold theorem).

2. **ASVspoof Evidence**: Kanformer achieved **60% relative improvement** in deepfake detection (similar modulation analysis). If it can detect subtle synthetic artifacts, it should excel at coarser speech/music distinctions.

3. **Anisotropic Inductive Bias**: Standard CNNs assume **translational invariance**. STM features violate this (location = meaning). Kanformer uses:
   - **Local convolution** (for smoothness)
   - **Global attention** (for context)
   - **KAN** (for non-linear boundaries)
   - No assumption that patterns repeat across the grid

## Troubleshooting

### Resuming After Cancellation
If training was cancelled:
1. Locate checkpoint directory: `model/STM/Kanformer_corpora_categories/{standard|downsample}/ckpt/YYYY-MM-DD_HH-MM/`
2. Check for `latest_checkpoint.pt` or `checkpoint_epoch_X.pt`
3. Run: `python STMkanformer_model.py <mode> --resume <checkpoint_dir>`

### Memory Issues
- Reduce `batch_size` (try 64 or 32)
- Reduce `d_model` (try 96)
- Use mixed precision: `torch.cuda.amp.autocast()`

### Unstable Training
- Check gradient norms (should be < 10 after clipping)
- Increase warmup epochs (try 10)
- Reduce learning rate (try 5e-5)

### Slow Convergence
- Increase `num_kan_groups` (reduces parameters, may train faster)
- Verify data normalization (STD should be ~1.0)

## Future Extensions

1. **GraphKAN**: Replace grid structure with graph (harmonic edge connections)
2. **Temporal Kanformer**: For time-varying STM sequences (use Mamba backbone)
3. **Ensemble**: Combine Kanformer predictions with Conformer
4. **Interpretability**: Visualize learned rational functions per modulation bin

## References

1. **KAN Original Paper**: Liu et al., "KAN: Kolmogorov-Arnold Networks" (2024)
2. **Kanformer for Deepfakes**: "XLSR-Kanformer" ASVspoof 2021 (60% EER improvement)
3. **GR-KAN**: Group-Rational KANs for speech enhancement (4x parameter efficiency)
4. **STM Baseline**: User's previous work on MusicSpeech-STM corpus

---

**Contact**: For questions or issues with this implementation, refer to the original research document (`optimizing.txt`) or check model training logs in the checkpoint directory.
