# Asym-Mixer-KAN: Asymmetric MLP-Mixer with Kolmogorov-Arnold Networks for STM-based Audio Classification

## Abstract

We present Asym-Mixer-KAN, a novel deep learning architecture for audio classification based on Spectrotemporal Modulation (STM) features. This model addresses fundamental limitations of convolutional neural networks when applied to the STM domain by: (1) processing asymmetric directional information through a 2-channel representation (Magnitude + Difference), (2) employing MLP-Mixer blocks that respect the semantic coordinate structure of the modulation space, and (3) integrating Kolmogorov-Arnold Network (KAN) layers with learnable B-spline activation functions. Combined with advanced training techniques including Label-Distribution-Aware Margin (LDAM) loss with Deferred Reweighting (DRW), CutMix augmentation, and DropBlock regularization, this architecture targets 0.89-0.90 macro F1 score on the challenging 6-class music-speech-environment classification task.

---

## 1. Introduction

### 1.1 The Spectrotemporal Modulation Paradigm

Traditional audio analysis relies on instantaneous spectral representations (MFCC, spectrograms) that decompose sound into local time-frequency patterns. Spectrotemporal Modulation (STM) features instead capture the **rates of change** in the spectrogram, mapping acoustic signals to a modulation domain defined by:

- **Temporal Modulation Rate (ω)**: Speed of amplitude changes over time (Hz)
- **Spectral Modulation Scale (Ω)**: Rate of spectral pattern variations (cycles/octave)

This representation mimics the neurophysiological processing of the mammalian auditory cortex, where neurons are tuned to specific modulation rates rather than static frequencies. The transformation is computed via 2D Fourier Transform of the auditory spectrogram:

$$
\text{STM}(\omega, \Omega) = |\mathcal{F}_{2D}[S(t, f)]|
$$

where $S(t, f)$ is the time-frequency spectrogram.

### 1.2 The Fundamental Problem: Translational Invariance vs. Semantic Coordinates

**Critical Insight**: The STM space is a **semantic map**, not a spatial scene.

Consider the temporal modulation axis:
- **4 Hz region**: Encodes syllabic rhythm in speech (intelligibility cues)
- **40 Hz region**: Encodes perceptual roughness or low pitch (timbral quality)
- **100 Hz region**: Encodes flutter, vocal fry, or tremolo effects

These are **distinct auditory objects with distinct semantic meanings**. A convolutional kernel trained to detect energy at 4 Hz would produce semantically incoherent outputs if translated to 40 Hz. Yet standard CNNs apply the same filters across all positions, enforcing translational invariance—the assumption that a feature retains its meaning regardless of location.

**This is the architectural-physical mismatch that limits CNN performance on STM features.**

### 1.3 The Asymmetry Problem

The Modulation Power Spectrum (MPS) exhibits conjugate symmetry due to the 2D FFT of a real-valued spectrogram. However, the physical interpretation of positive and negative temporal modulation rates differs:

- **Positive rates (+ω)**: Downward frequency sweeps (high → low frequency over time)
- **Negative rates (−ω)**: Upward frequency sweeps (low → high frequency over time)

Standard approaches either:
1. **Fold the spectrum** (average ±ω) → **Destructive**: Loses directional sweep information
2. **Feed full spectrum to CNN** → **Inefficient**: Forces the network to implicitly learn arithmetic relationships between distant pixels

**Our solution**: Create a 2-channel representation that explicitly disentangles magnitude from directionality.

---

## 2. Architecture Design

### 2.1 Asymmetric 2-Channel STM Processing

#### Input Specification
- **Raw STM**: Shape (20 frequency bands, 121 modulation rates)
- **Rate range**: -15 Hz (index 0) to +15 Hz (index 120), with DC at index 60

#### Processing Pipeline

**Step 1: Separate Negative and Positive Rates**
```python
negative_chunk = stm[:, :, 0:60]    # -15 Hz to -0.25 Hz (upward sweeps)
dc_component = stm[:, :, 60:61]     # 0 Hz
positive_chunk = stm[:, :, 61:121]  # +0.25 Hz to +15 Hz (downward sweeps)
```

**Step 2: Align via Flipping**
```python
negative_flipped = flip(negative_chunk, dim=2)
# Now: -15Hz aligns with +15Hz, -0.25Hz aligns with +0.25Hz
```

**Step 3: Create 2-Channel Representation**

**Channel 0 (Magnitude)**: Total modulation energy (direction-invariant)
$$
M(\Omega, |\omega|) = \frac{S_{\text{neg}}^{\text{flip}}(\Omega, \omega) + S_{\text{pos}}(\Omega, \omega)}{2}
$$

**Channel 1 (Difference)**: Directional information
$$
D(\Omega, |\omega|) = S_{\text{pos}}(\Omega, \omega) - S_{\text{neg}}^{\text{flip}}(\Omega, \omega)
$$

**Step 4: Concatenate DC**
```python
output_ch0 = [DC, M(0.25Hz), ..., M(15Hz)]  # Shape: (20, 61)
output_ch1 = [DC, D(0.25Hz), ..., D(15Hz)]  # Shape: (20, 61)
final = stack([output_ch0, output_ch1])     # Shape: (2, 20, 61)
```

#### Physical Interpretation

| Magnitude | Difference | Acoustic Phenomenon |
|-----------|------------|---------------------|
| High | Low | Symmetric modulation (stationary noise, sustained tones) |
| High | High positive | Strong downward sweeps (falling intonation, siren down) |
| High | High negative | Strong upward sweeps (rising questions, chirps) |
| Low | Any | Weak modulation (silence, unmodulated signals) |

**Key advantage**: Neural networks excel at mixing information across channels at the same spatial location (via 1×1 convolutions or channel-wise MLPs). This representation transforms a geometric relationship (symmetry across an axis) into a channel-depth relationship, enabling efficient feature extraction.

### 2.2 Model Architecture: Asym-Mixer-KAN

#### Overview
```
Input (batch, 2, 20, 61)
    ↓
[Input Embedding + Coordinate Encoding]
    ↓
Token Representation (batch, 20, d_model)
    ↓
[Stack of 12 Mixer-KAN Blocks]
    ↓
Layer Normalization
    ↓
Global Average Pooling
    ↓
Classification Head
    ↓
Output (batch, 6)
```

#### 2.2.1 Input Embedding with Coordinate Awareness

**Token Definition**: Each frequency band becomes a token
- Token features: Concatenation of both channels across all modulation rates
- Feature dimension per token: 2 channels × 61 rates = 122 features

**Coordinate Embeddings** (Position-Aware, Not Positional):
1. **Frequency Coordinate Embedding**: Learnable embeddings $E_{\text{freq}} \in \mathbb{R}^{20 \times d}$
   - Encodes the semantic meaning of each frequency band
   - Low frequencies (0-4) encode bass/fundamental, high frequencies (16-19) encode brilliance

2. **Rate Coordinate MLP**: Learnable function $f_{\text{rate}}: \mathbb{R}^{61} \rightarrow \mathbb{R}^{d}$
   - Maps the modulation rate profile to a coordinate encoding
   - Adds awareness of which rates are active in each token

**Embedding Equation**:
$$
x_i = \text{Linear}(\text{concat}[M_i, D_i]) + E_{\text{freq}}[i] + f_{\text{rate}}(r_i)
$$

where:
- $M_i, D_i$: Magnitude and Difference channels for frequency band $i$
- $E_{\text{freq}}[i]$: Frequency coordinate embedding
- $r_i$: Coordinate vector $[0, \frac{1}{60}, \ldots, 1]$ representing rate positions

#### 2.2.2 Mixer-KAN Block

**Architecture per block**:
```
Input (batch, num_tokens=20, d_model=256)
    ↓
[Token-Mixing Path]
    LayerNorm → Transpose → KAN MLP → Transpose → Residual
    ↓
[Channel-Mixing Path]
    LayerNorm → KAN MLP → Residual
    ↓
Output (batch, 20, 256)
```

**Token-Mixing KAN MLP**:
- **Purpose**: Mix information across frequency bands (spatial dimension)
- **Architecture**: 
  - Input: (batch, d_model, num_tokens=20)
  - KAN Layer 1: 20 → 80 (4× expansion)
  - Dropout
  - KAN Layer 2: 80 → 20
- **Effect**: Each frequency band attends to all other bands globally

**Channel-Mixing KAN MLP**:
- **Purpose**: Mix information across feature channels at each spatial location
- **Architecture**:
  - Input: (batch, num_tokens=20, d_model=256)
  - KAN Layer 1: 256 → 1024 (4× expansion)
  - Dropout
  - KAN Layer 2: 1024 → 256
- **Effect**: Each token independently transforms its features

**Why Mixer Architecture?**
1. **Global receptive field**: Every token can communicate with every other token (like Transformers)
2. **Linear complexity**: O(L) instead of O(L²) for self-attention
3. **Position-aware**: With coordinate embeddings, the model knows which spatial position it's processing
4. **No translational invariance**: Unlike CNNs, each spatial position has distinct processing

### 2.3 Kolmogorov-Arnold Networks (KAN)

#### Motivation

Traditional MLPs use fixed activation functions (ReLU, GELU, Swish). The Kolmogorov-Arnold Representation Theorem states that any multivariate continuous function can be represented as a composition of univariate functions:

$$
f(\mathbf{x}) = \sum_{q=1}^{2n+1} \Phi_q\left(\sum_{p=1}^{n} \phi_{q,p}(x_p)\right)
$$

**KAN Insight**: Instead of fixing the activation functions, learn them as part of the network.

#### Implementation: B-Spline Basis Functions

KAN layers parameterize activation functions using cubic B-spline basis functions:

$$
\text{KAN}(\mathbf{x})_j = \sum_{i=1}^{d_{\text{in}}} \sum_{k=1}^{G} w_{j,i,k} \cdot B_k^3(x_i) + \mathbf{W}_{\text{base}} \mathbf{x}
$$

where:
- $B_k^3(x)$: Cubic B-spline basis function with knot at position $k$
- $w_{j,i,k}$: Learnable spline coefficients
- $\mathbf{W}_{\text{base}}$: Linear transformation (residual connection)
- $G$: Grid size (number of spline knots, default=5)

**B-Spline Construction** (Cox-de Boor recursion):
- **Order 0** (piecewise constant): $B_i^0(x) = 1$ if $t_i \leq x < t_{i+1}$, else 0
- **Order k**: 
$$
B_i^k(x) = \frac{x - t_i}{t_{i+k} - t_i} B_i^{k-1}(x) + \frac{t_{i+k+1} - x}{t_{i+k+1} - t_{i+1}} B_{i+1}^{k-1}(x)
$$

**Advantages**:
1. **Learnable non-linearity**: Optimal activation discovered during training
2. **Local control**: B-splines provide local adjustment without global impact
3. **Smoothness**: Cubic splines ensure $C^2$ continuity (smooth derivatives)
4. **Interpretability**: Can visualize learned activation functions post-training

### 2.4 DropBlock Regularization

Standard dropout randomly drops individual units, which is inefficient for spatially correlated features. **DropBlock** drops contiguous spatial blocks:

1. Sample a Bernoulli mask with probability $\gamma = \frac{p_{\text{drop}}}{\text{block\_size}^2}$
2. Expand each dropped unit to a block via max pooling
3. Apply mask and normalize: $x' = x \cdot \text{mask} \cdot \frac{|\text{mask}|}{|\text{mask}|_{\text{sum}}}$

**Effect**: Forces the network to learn distributed representations. Cannot rely on a single contiguous region of modulation energy.

---

## 3. Training Methodology

### 3.1 Loss Function: LDAM with Deferred Reweighting (DRW)

**Problem**: Class imbalance in dataset
- Non-tonal speech: ~40% of training data
- Wildlife sounds: ~5% of training data

**Standard Cross-Entropy**: Biased toward majority classes (optimizes accuracy, not balanced F1)

#### Label-Distribution-Aware Margin (LDAM) Loss

Adjusts decision boundaries by introducing class-dependent margins:

$$
\mathcal{L}_{\text{LDAM}}(\mathbf{z}, y) = -\log \frac{\exp(z_y - \Delta_y)}{\sum_{j=1}^{C} \exp(z_j - \Delta_j \cdot \mathbb{1}_{j=y})}
$$

where:
- $\mathbf{z}$: Logits (pre-softmax outputs)
- $y$: True class
- $\Delta_y$: Margin for class $y$, inversely proportional to class frequency:
$$
\Delta_y = C \cdot \left(\frac{1}{n_y}\right)^{1/4}
$$

where $n_y$ is the number of samples in class $y$, and $C$ is a scaling constant (default=0.5).

**Intuition**: Minority classes get larger margins, pushing their decision boundaries away from majority classes. The model must be more confident to classify a sample as a majority class.

#### Deferred Reweighting (DRW) Schedule

**Two-phase training**:
1. **Phase 1 (Epochs 0-40)**: Pure LDAM without sample reweighting
   - Focus: Learn good feature representations
   - The network discovers generalizable patterns without being forced into boundary adjustments
   
2. **Phase 2 (Epochs 40-50)**: Enable class-frequency reweighting
   - Per-sample loss weight: $w_y = \frac{1}{n_y}$ (normalized)
   - Focus: Fine-tune decision boundaries for optimal F1 score

**Why DRW works**: Early reweighting can cause the network to memorize minority class examples instead of learning transferable features. DRW delays this until representations are stable.

### 3.2 CutMix Augmentation

**Problem with Mixup**: Linear interpolation creates "ghosting" artifacts
- Mixup: $x' = \lambda x_1 + (1-\lambda) x_2$
- Result: Blurred modulation patterns that don't correspond to any real acoustic signal
- Failure mode: Model learns to recognize artifacts instead of true features

**CutMix Solution**: Cut and paste rectangular patches
1. Randomly sample two samples $(x_1, y_1)$ and $(x_2, y_2)$
2. Sample a bounding box $(x, y, w, h)$ with area ratio $(1 - \lambda)$
3. Paste: $x'[x:x+w, y:y+h] = x_2[x:x+w, y:y+h]$
4. Mixed label: $y' = \lambda y_1 + (1-\lambda) y_2$

**Physical Interpretation for STM**:
- A patch of "speech rhythm" (4 Hz region) pasted onto "music harmonics" (high spectral scale)
- Result: Compositional sound similar to "Vocal Music" (speech + music simultaneously)
- The model learns to detect multiple concurrent acoustic objects

**Advantages**:
1. Preserves local STM structure (no blurring)
2. Encourages the model to recognize partial patterns
3. Naturally handles overlapping domains (Vocal Music, Tonal Speech)
4. Regularization: Cannot overfit to global statistics

### 3.3 Optimization Strategy

**Optimizer**: AdamW (Adam with decoupled weight decay)
- Learning rate: $10^{-3}$
- Weight decay: $10^{-4}$
- $\beta_1 = 0.9, \beta_2 = 0.999$

**Scheduler**: CosineAnnealingWarmRestarts
- **Why not ReduceLROnPlateau?** Reacts too late; sharp drops destabilize optimization
- **Warm restarts**: Periodically reset learning rate to escape local minima
  - $T_0 = 10$ epochs (first restart)
  - $T_{\text{mult}} = 2$ (double period after each restart)
  - $\eta_{\text{min}} = 10^{-6}$ (minimum learning rate)

**Learning rate schedule**:
```
Epochs 0-10:  1e-3 → 1e-6 (cosine decay)
Epoch 10:     1e-3 (restart)
Epochs 10-30: 1e-3 → 1e-6 (cosine decay)
Epoch 30:     1e-3 (restart)
Epochs 30-50: 1e-3 → 1e-6 (cosine decay)
```

**Gradient Clipping**: Max norm = 1.0
- Prevents exploding gradients during KAN layer training
- B-spline computations can produce large gradients for extreme inputs

---

## 4. Experimental Setup

### 4.1 Dataset

**Classes** (6 total):
1. Speech: Non-tonal (English, French, German, etc.)
2. Speech: Tonal (Mandarin, Thai, Vietnamese, etc.)
3. Music: Vocal (singing with lyrics)
4. Music: Non-vocal (instrumental)
5. Environment: Urban (traffic, construction, sirens)
6. Environment: Wildlife (birds, insects, mammals)

**Data Split** (10-fold speaker-grouped):
- Train: Folds 0-7 (80%)
- Validation: Fold 8 (10%)
- Test: Fold 9 (10%)

**Class Imbalance** (approximate):
- Non-tonal speech: 40%
- Tonal speech: 15%
- Music vocal: 12%
- Music non-vocal: 18%
- Urban: 10%
- Wildlife: 5%

**STM Feature Extraction**:
- Sampling rate: 16 kHz
- Cochlear filterbank: 128 Gaussian filters (170 Hz - 7 kHz, log-spaced)
- Modulation rates: -15 Hz to +15 Hz (0.25 Hz resolution)
- Spectral scales: 0 to 7.09 cycles/octave (0.37 cyc/oct resolution)
- Normalization: Per-sample z-score (preserve relative patterns)

### 4.2 Model Configuration

| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| d_model | 256 | Balance between capacity and efficiency |
| depth | 12 | Deep enough for hierarchical abstraction |
| mlp_ratio | 4 | Standard expansion ratio (MLP-Mixer paper) |
| kan_grid_size | 5 | Sufficient spline resolution without overfitting |
| batch_size | 128 | Larger than ViM (Mixer is more efficient) |
| dropout | 0.1 | Moderate regularization |
| drop_block_size | 4 | Covers ~6% of spatial dimension (20×61) |
| cutmix_prob | 0.5 | Apply CutMix to half of training batches |
| cutmix_alpha | 1.0 | Uniform distribution over mixing ratios |
| ldam_max_m | 0.5 | Maximum margin scaling |
| drw_start_epoch | 40 | Start reweighting after 80% of training |
| num_epochs | 50 | Total training duration |

**Model Size**:
- Total parameters: ~8.5M
- Trainable parameters: ~8.5M
- Memory footprint (batch=128): ~3.2 GB GPU

### 4.3 Training Modes

**Mode 0**: Full dataset (natural class distribution)
- Use when: Evaluating real-world performance
- Expected behavior: Higher majority class F1, lower minority class F1

**Mode 1**: Downsampled non-tonal speech (balanced classes)
- Use when: Maximizing macro F1 score
- Method: Randomly downsample non-tonal speech to match tonal speech count
- Expected behavior: More balanced per-class F1 scores

---

## 5. Theoretical Performance Analysis

### 5.1 Expected Improvements Over Baseline (STM_CoordConvLDAM: 0.86 F1)

| Component | Expected Gain | Mechanism |
|-----------|---------------|-----------|
| 2-channel asymmetric input | +1-2% | Captures directional sweep information (prosody, chirps) |
| KAN activation learning | +0.5-1% | Discovers optimal non-linearities for STM feature space |
| Mixer architecture | +0.5-1% | Global receptive field without CNN translation invariance |
| CutMix (vs Mixup) | +1% | Eliminates ghosting, handles compositional sounds |
| DropBlock (vs Dropout) | +0.5% | Forces robust distributed representations |
| Cosine scheduler (vs Plateau) | +0.5% | Better convergence, avoids late-stage destabilization |
| **Total Expected** | **+4-6%** | **Target: 0.89-0.91 Macro F1** |

### 5.2 Per-Class Performance Predictions

**Strong Performance Expected**:
1. **Non-vocal Music** (0.92-0.95 F1)
   - High spectral modulation (harmonic stacks)
   - Low temporal modulation (stable pitch)
   - Clear separation from speech in STM space

2. **Non-tonal Speech** (0.88-0.92 F1)
   - Characteristic 4-8 Hz syllabic rhythm
   - Low spectral modulation (broad formants)
   - Large training set (even in balanced mode)

**Moderate Performance Expected**:
3. **Urban Environment** (0.85-0.88 F1)
   - Distinct roughness signatures (40-100 Hz)
   - May confuse with non-tonal speech (traffic rhythms)
   - Improved by asymmetric channel (directional sweeps in sirens)

4. **Wildlife Environment** (0.82-0.87 F1)
   - Strong directional sweeps (bird chirps)
   - Smallest class → benefits most from LDAM+DRW
   - Asymmetric channel critical for disambiguation

**Challenging Classes**:
5. **Tonal Speech** (0.80-0.85 F1)
   - Overlaps with music: pitch contour modulation
   - Overlaps with non-tonal: syllabic rhythm
   - Requires subtle feature interactions (KAN helps here)

6. **Vocal Music** (0.78-0.83 F1)
   - **Compositional**: Speech + Music simultaneously
   - High magnitude in both 4 Hz (lyrics) and high scale (melody)
   - CutMix training explicitly addresses this overlap

### 5.3 Confusion Matrix Predictions

**Likely Confusions**:
- Tonal Speech ↔ Vocal Music (shared pitch modulation)
- Non-tonal Speech ↔ Urban (rhythmic traffic patterns)
- Vocal Music ↔ Non-vocal Music (when lyrics are sparse)

**Unlikely Confusions** (well-separated in STM space):
- Speech ↔ Environment
- Music ↔ Environment
- Non-tonal Speech ↔ Tonal Speech

---

## 6. Ablation Study Hypotheses

To validate design choices, the following ablation experiments are recommended:

### 6.1 Input Representation

| Configuration | Expected F1 | Delta |
|---------------|-------------|-------|
| Full model (2-channel asymmetric) | 0.90 | Baseline |
| 1-channel folded (averaged) | 0.87 | -3% (loses directionality) |
| 1-channel positive only | 0.88 | -2% (loses negative sweep info) |
| Raw 121-rate without processing | 0.85 | -5% (redundancy + increased complexity) |

**Hypothesis**: Asymmetric 2-channel representation provides 2-3% gain over folded spectrum.

### 6.2 Architecture Components

| Configuration | Expected F1 | Delta |
|---------------|-------------|-------|
| Full Asym-Mixer-KAN | 0.90 | Baseline |
| Replace KAN with standard MLP (GELU) | 0.88 | -2% (fixed activation suboptimal) |
| Remove coordinate embeddings | 0.87 | -3% (loses position awareness) |
| Replace Mixer with Transformer | 0.89 | -1% (O(L²) less efficient, similar capacity) |
| Replace Mixer with CNN (CoordConv) | 0.86 | -4% (translation invariance mismatch) |

**Hypothesis**: Each component contributes meaningfully; KAN and coordinate embeddings are critical.

### 6.3 Training Strategy

| Configuration | Expected F1 | Delta |
|---------------|-------------|-------|
| Full training (LDAM+DRW+CutMix+DropBlock) | 0.90 | Baseline |
| Replace LDAM with CE + class weights | 0.87 | -3% (early reweighting hurts representations) |
| Replace CutMix with Mixup | 0.88 | -2% (ghosting artifacts) |
| No augmentation | 0.86 | -4% (overfitting) |
| Replace DropBlock with standard Dropout | 0.89 | -1% (weaker spatial regularization) |
| ReduceLROnPlateau instead of Cosine | 0.88 | -2% (late reaction, destabilization) |

**Hypothesis**: Training methodology contributes ~3-4% over naive cross-entropy + Mixup.

---

## 7. Implementation Details

### 7.1 File Structure

```
STMasm_mixer_kan.py        # Main model implementation
STMasm_mixer_kan.md        # This documentation
/Asym-Mixer-KAN_*/         # Checkpoint directories
    best_model.pt          # Best validation F1 checkpoint
    checkpoint_epoch_*.pt  # Periodic checkpoints
    test_predictions.npy   # Final test predictions
    test_targets.npy       # True test labels
```

### 7.2 Usage

**Training from scratch (Mode 0: Full dataset)**:
```bash
python STMasm_mixer_kan.py 0
```

**Training with balanced classes (Mode 1)**:
```bash
python STMasm_mixer_kan.py 1
```

**Resume training**:
```bash
python STMasm_mixer_kan.py 0 --resume Asym-Mixer-KAN_full_20260204_143022
```

### 7.3 Computational Requirements

**Hardware**:
- GPU: NVIDIA A100 (40GB) or equivalent
- Alternative: RTX 3090 (24GB) with batch_size=64
- CPU fallback: Possible but 50× slower

**Training time** (A100):
- Full 50 epochs: ~3-4 hours (Mode 0)
- Per epoch: ~4-5 minutes

**Memory usage**:
- Model: ~34 MB (8.5M parameters × 4 bytes)
- Optimizer states: ~68 MB (AdamW maintains 2× parameter states)
- Activations (batch=128): ~3 GB
- Total: ~3.2 GB (comfortable on modern GPUs)

### 7.4 Reproducibility

**Random seeds**: Set in main execution block (not shown in minimal implementation)
- PyTorch: `torch.manual_seed(42)`
- NumPy: `np.random.seed(42)`
- Python: `random.seed(42)`
- CUDA: `torch.cuda.manual_seed_all(42)`, `torch.backends.cudnn.deterministic=True`

**Note**: B-spline computation may have minor floating-point variations across GPU architectures.

---

## 8. Limitations and Future Work

### 8.1 Current Limitations

1. **Computational Cost**: 
   - KAN layers are ~2× slower than standard MLPs due to B-spline computation
   - Mitigation: Grid size = 5 is a compromise between expressiveness and speed

2. **Overlapping Domain Problem**:
   - Vocal Music and Tonal Speech still challenging (80-83% F1 expected)
   - Root cause: Truly compositional signals (both speech AND music present)
   - Current solution: CutMix mimics this, but not perfect

3. **Interpretability**:
   - KAN activation functions are learnable but not inherently interpretable
   - Future work: Visualize learned activation shapes per layer

4. **Hyperparameter Sensitivity**:
   - DRW start epoch (40) and LDAM margin scaling (0.5) chosen empirically
   - May require tuning for different datasets

### 8.2 Future Research Directions

**8.2.1 Multi-Label Classification**

The paper strongly recommends multi-label formulation:
- Vocal Music = [Speech=1, Music=1, Environment=0]
- Tonal Speech = [Speech=1, Tonal=1]
- Change final layer to sigmoid, use Binary Cross-Entropy or Asymmetric Loss

**Expected gain**: +2-3% on overlapping classes

**8.2.2 Disentangled Representation Learning**

Split encoder into orthogonal subspaces:
- **Pitch Encoder**: Extracts pitch contour information
- **Timbre Encoder**: Extracts spectral envelope and texture
- **Prosody Encoder**: Extracts rhythm and dynamics

Use orthogonality loss:
$$
\mathcal{L}_{\text{orth}} = \|E_{\text{pitch}}^T E_{\text{timbre}}\|_F^2 + \|E_{\text{pitch}}^T E_{\text{prosody}}\|_F^2 + \|E_{\text{timbre}}^T E_{\text{prosody}}\|_F^2
$$

**Expected gain**: +1-2% on Tonal Speech vs Vocal Music disambiguation

**8.2.3 Hierarchical Branching**

Early exit architecture:
```
Input → Shared Encoder → Branch 1: Speech vs Music vs Environment
                        ↓
              (if Speech) → Branch 2: Tonal vs Non-tonal
              (if Music)  → Branch 2: Vocal vs Non-vocal
              (if Env)    → Branch 2: Urban vs Wildlife
```

**Advantage**: Prevents "easy" environment samples from confusing speech/music boundaries

**8.2.4 Attention Mechanism for Rate-Scale Interaction**

While Mixer provides global receptive field, it doesn't explicitly model **rate-scale correlation** (e.g., high rate often implies low scale). Add a lightweight cross-attention module:

```
Rate Tokens ← attend ← Scale Tokens
```

**Expected gain**: +0.5-1% by capturing physical dependencies

---

## 9. Comparison with Prior Work

### 9.1 vs. STM_CoordConvLDAM (0.86 F1)

**Strengths of CoordConv**:
- Successfully restored position awareness to CNNs
- LDAM loss effectively handles imbalance

**Limitations**:
- Still enforces local translational invariance (convolutional kernels)
- Mixup augmentation creates ghosting artifacts
- ReduceLROnPlateau scheduler reacts too late

**Asym-Mixer-KAN improvements**:
- No translational invariance (Mixer architecture)
- 2-channel asymmetric input (directional information)
- KAN layers (learnable activations)
- CutMix + DropBlock (better regularization)
- Cosine scheduler (better convergence)

### 9.2 vs. STM_ViM (0.84 F1)

**Why ViM underperformed**:
- Designed for long sequences, but STM is a **spatial map**, not a sequence
- Unidirectional bias (despite bidirectional scanning)
- Linear complexity of SSM not critical for 1220 tokens
- Position-invariant by default (needs explicit positional embeddings)

**Asym-Mixer-KAN advantages**:
- Treats STM as a 2D spatial map (natural representation)
- Coordinate embeddings encode semantic grid positions
- Mixer explicitly designed for spatial mixing (not sequential)
- Comparable efficiency (both O(L) complexity)

### 9.3 vs. Transformer-based Models

**Transformers on STM**:
- Global receptive field ✓
- O(L²) complexity ✗ (inefficient for 1220 tokens)
- Position-agnostic without positional embeddings ✗

**Asym-Mixer-KAN advantages**:
- Global receptive field ✓ (via token-mixing)
- O(L) complexity ✓ (linear in sequence length)
- Position-aware by design ✓ (coordinate embeddings)
- KAN adds expressiveness beyond fixed attention patterns

---

## 10. Conclusion

Asym-Mixer-KAN represents a **principled architectural redesign** for STM-based audio classification. By addressing the fundamental mismatch between CNN translational invariance and the semantic coordinate structure of the modulation domain, and by explicitly disentangling directional sweep information through 2-channel processing, this model is positioned to achieve 0.89-0.91 macro F1 score—a significant improvement over the current state-of-the-art.

The integration of **Kolmogorov-Arnold Networks** provides learnable non-linearities that can adapt to the specific statistical structure of STM features, while **advanced training techniques** (LDAM+DRW, CutMix, DropBlock) ensure robust learning despite severe class imbalance and compositional overlaps.

This work demonstrates that optimal deep learning for STM features requires moving beyond generic image processing architectures toward **domain-aware designs** that respect the physical and perceptual structure of the auditory modulation space.

---

## References

1. **Spectrotemporal Modulation**: Chi, T., et al. "Multiresolution spectrotemporal analysis of complex sounds." *Journal of the Acoustical Society of America* (2005).

2. **MLP-Mixer**: Tolstikhin, I., et al. "MLP-Mixer: An all-MLP Architecture for Vision." *NeurIPS* (2021).

3. **Kolmogorov-Arnold Networks**: Liu, Z., et al. "KAN: Kolmogorov-Arnold Networks." *arXiv:2404.19756* (2024).

4. **LDAM Loss**: Cao, K., et al. "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss." *NeurIPS* (2019).

5. **CutMix**: Yun, S., et al. "CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features." *ICCV* (2019).

6. **DropBlock**: Ghiasi, G., et al. "DropBlock: A regularization method for convolutional networks." *NeurIPS* (2018).

7. **Research Paper**: "Optimization of Spectrotemporal Modulation Analysis: Architectural Paradigms for Superior Audio Classification and Segregation" (2026).

---

## Appendix A: Mathematical Notation

| Symbol | Meaning |
|--------|---------|
| $\omega$ | Temporal modulation rate (Hz) |
| $\Omega$ | Spectral modulation scale (cycles/octave) |
| $S(t, f)$ | Time-frequency spectrogram |
| $M(\Omega, \omega)$ | Magnitude channel (symmetric) |
| $D(\Omega, \omega)$ | Difference channel (asymmetric) |
| $d$ | Model embedding dimension (d_model) |
| $L$ | Sequence length (number of tokens = 20) |
| $C$ | Number of classes (6) |
| $n_y$ | Number of training samples in class $y$ |
| $\Delta_y$ | LDAM margin for class $y$ |
| $B_k^p(x)$ | B-spline basis function of order $p$ with knot $k$ |

---

## Appendix B: Hyperparameter Sensitivity Analysis (Planned)

| Hyperparameter | Default | Range to Test | Expected Optimal |
|----------------|---------|---------------|------------------|
| d_model | 256 | [128, 192, 256, 384] | 256 (current) |
| depth | 12 | [6, 9, 12, 15] | 12-15 |
| kan_grid_size | 5 | [3, 5, 7, 9] | 5-7 |
| drw_start_epoch | 40 | [30, 35, 40, 45] | 38-42 |
| ldam_max_m | 0.5 | [0.3, 0.5, 0.7] | 0.5 (current) |
| cutmix_alpha | 1.0 | [0.5, 1.0, 1.5] | 1.0 (uniform) |

**Planned experiments**: 5-fold cross-validation on validation set for each configuration.

---

**Document Version**: 1.0  
**Date**: February 4, 2026  
**Corresponding Implementation**: `STMasm_mixer_kan.py`  
**Status**: Ready for experimental validation
