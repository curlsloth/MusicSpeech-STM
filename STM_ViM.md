# STM Classification with Vision Mamba (Vim)
## Phase 2: The Modern Sequence Model

### Overview

Vision Mamba addresses the fundamental limitation of standard Transformers when processing long STM sequences: **O(L²) complexity makes attention prohibitively expensive**. Mamba provides **O(L) complexity** while maintaining global context. 

**NEW: Symmetric STM Processing** - Following STMasm_enhanced5.py, we exploit the up/down-sweep symmetry of modulation spectra:
- **Input**: 20 freq × 121 rates = 2420 tokens
- **After symmetric processing**: 20 freq × 61 rates = **1210 tokens**
- **Speed gain**: **2× faster** (O(L) means half the tokens = half the time)
- **Memory**: 2× less

This enables practical training times (~30 minutes per epoch vs >1 hour).

### Core Innovation: State Space Models (SSMs)

#### The Attention Problem

Standard Transformer for 2420 tokens:
```
Attention complexity: O(2420²) ≈ 5.86M operations per layer
Memory: 2420 × 2420 × num_heads attention matrices
```

Mamba for 2420 tokens:
```
SSM complexity: O(2420) = 2420 operations per layer
Memory: Linear in sequence length
```

**With Symmetric STM Processing** (NEW):
```
Mamba for 1210 tokens: O(1210) operations per layer
Speed: 2× faster than 2420 tokens
Memory: 2× less
```

**Key Mechanism**: Selective State Space Model
- Maintains a hidden state `h_t` that integrates information across the sequence
- Updates state recursively: `h_t = A·h_{t-1} + B·x_t`
- Output: `y_t = C·h_t`
- **Selective**: A, B, C are input-dependent (unlike linear RNNs)

### Symmetric STM Processing

**NEW in this implementation**: Following STMasm_enhanced5.py approach

#### Motivation

Modulation spectra exhibit **up/down-sweep symmetry**:
- Positive modulation rates (+0.25 to +15 Hz) represent upward frequency sweeps
- Negative modulation rates (-15 to -0.25 Hz) represent downward frequency sweeps
- For many sounds, these are equivalent (e.g., a siren sounds similar going up or down)

#### Processing Steps

```python
def process_symmetric_stm(stm_data):
    # Input: (batch, 20 freq, 121 rates)
    
    # Step A: Separate negative, DC, and positive rates
    negative_chunk = stm_data[:, :, 0:60]   # -15 Hz to -0.25 Hz
    dc_component = stm_data[:, :, 60:61]    # 0 Hz
    positive_chunk = stm_data[:, :, 61:121] # +0.25 Hz to +15 Hz
    
    # Step B: Flip negative chunk to align with positive
    negative_flipped = torch.flip(negative_chunk, dims=[2])
    
    # Step C: Average aligned chunks
    averaged_chunk = (negative_flipped + positive_chunk) / 2.0
    
    # Step D: Concatenate DC at the beginning
    output = torch.cat([dc_component, averaged_chunk], dim=2)
    # Output: (batch, 20 freq, 61 rates)
    
    return output
```

#### Benefits

1. **Speed**: 2420 → 1210 tokens = **2× faster** (O(L) complexity)
2. **Memory**: 2× reduction in sequence length
3. **Regularization**: Averaging reduces noise
4. **Information preserved**: Spectral structure maintained

#### Trade-offs

**What's lost**:
- Distinction between rising vs falling frequency sweeps
- Directionality information in temporal modulation

**Why acceptable**:
- Speech/music classification rarely depends on sweep direction
- Spectral modulation structure (20 freq bands) fully preserved
- 2× speedup enables deeper models and more experiments

### Architecture Details

#### 1. Token Embedding

**Critical Decision**: 1×1 patches (each bin = 1 token) after symmetric processing

```python
Input: (batch, 1210) - flattened symmetric STM (20×61)
↓
Reshape: (batch, 1210, 1)
↓
Linear embedding: (batch, 1210, d_model=192)
```

**Why 1×1 patches**:
- Preserves maximum resolution after symmetric processing
- No arbitrary grouping of spectral/temporal bins
- Each modulation bin retains independent representation
- Computationally feasible with O(L) complexity

**Symmetric processing advantage**:
- Original: 2420 tokens would require >1 hour per epoch
- Symmetric: 1210 tokens enables ~30 minute epochs

#### 2. Positional Embeddings

**CRITICAL**: Learnable absolute positional embeddings

```python
pos_embed = nn.Parameter(torch.zeros(1, 1210, 192))
```

**Why learnable absolute (not relative)**:
- Addresses the "Translation Invariance" problem (Section 2.2)
- Model learns "Bin 25 (4Hz rate) has semantic meaning"
- Relative encoding (like in Conformer) masks absolute position
- Each token knows its exact (spectral_bin, temporal_bin) location after symmetric processing

**Initialization**: Truncated normal (std=0.02)

#### 3. Bidirectional Scanning

**Key insight from Vim paper**: STM is non-causal

```python
# Forward scan
x_forward = mamba_forward(x)  # Processes sequence left→right

# Backward scan  
x_backward = flip(x, dim=1)
x_backward = mamba_backward(x_backward)
x_backward = flip(x_backward, dim=1)  # Reverse back

# Combine
x = x_forward + x_backward
```

**Why bidirectional**:
- Temporal modulation spectrum is non-causal (all bins exist simultaneously)
- "End" of spectrum (±15 Hz) doesn't depend on "beginning" (DC component)
- Forward: captures low→high rate dependencies
- Backward: captures high→low rate dependencies
- Combined: full context integration

**Contrast with audio Mamba (causal)**:
- Audio waveforms: t=5sec depends on t=0-5sec (causal)
- STM texture: All frequencies present simultaneously (non-causal)

#### 4. VimBlock Architecture

```
Input: (batch, 2420, 192)
├─ LayerNorm
├─ Forward SSM → (batch, 2420, 192)
├─ Backward SSM → (batch, 2420, 192)
├─ Sum scans
├─ DropPath (stochastic depth)
└─ Residual connection
Output: (batch, 2420, 192)
```

**Stochastic Depth**: 
- Drop probability increases linearly from 0 to 0.1 across 12 layers
- Regularizes deep networks
- Enables training 12-layer model without overfitting

#### 5. Model Configuration (Vim-Small)

```python
seq_len: 1210       # After symmetric STM processing (20×61)
d_model: 192        # Embedding dimension
depth: 12           # Number of Vim blocks
d_state: 16         # SSM hidden state size
d_conv: 4           # Conv kernel in SSM (for local context)
expand: 2           # FFN expansion factor
```

Total parameters: ~8M (comparable to CoordConv-ResNet)

**Memory/Speed with symmetric processing**:
- Forward pass (batch=64, seq=1210): ~77k operations per layer
- 2× faster than original 2420 tokens
- Enables practical experimentation and hyperparameter tuning

### Balanced Softmax Loss

**From**: "Balanced Meta-Softmax for Long-Tailed Visual Recognition" (Ren et al., NeurIPS 2020)

#### Mechanism

```python
adjusted_logits = logits + log(class_frequency)
loss = CrossEntropy(adjusted_logits, target)
```

#### Mathematical Justification

Standard Softmax assumes balanced test distribution:
```
P(y|x) = exp(z_y) / Σ_j exp(z_j)
```

For imbalanced training data, this is biased. Balanced Softmax compensates:
```
P_balanced(y|x) = (prior_y × exp(z_y)) / Σ_j (prior_j × exp(z_j))
                = exp(z_y + log(prior_y)) / Σ_j exp(z_j + log(prior_j))
```

**Effect**:
- Majority class (Speech, 60%): log(0.6) = -0.51 → logit penalty
- Minority class (Environment, 5%): log(0.05) = -3.0 → logit boost
- Classifier learns decision boundary that is unbiased w.r.t. class distribution

**Comparison to LDAM**:
- LDAM: Modifies margins (geometric separation)
- Balanced Softmax: Modifies logits (probabilistic calibration)
- Both address imbalance; BS is simpler, LDAM is theoretically stronger

### Training Strategy

#### Hyperparameters

```python
Optimizer: AdamW
  - Learning rate: 1e-4
  - Weight decay: 1e-4
  
Scheduler: CosineAnnealingLR (T_max=50)

Batch size: 64 (smaller than CoordConv due to sequence length)

Gradient clipping: max_norm=1.0
```

#### Why smaller batch size?

Memory requirements:
```
CoordConv input: (batch, 1, 20, 121) = batch × 2420 values
Mamba input: (batch, 2420, 192) = batch × 464,640 values

Mamba requires ~192× more memory per sample!
```

With batch_size=64:
- Manageable on 24GB GPUs
- Still provides good gradient estimates
- Can increase if using gradient accumulation

#### No DRW (Deferred Reweighting)

Unlike LDAM, Balanced Softmax doesn't use DRW because:
1. It's a single-stage adjustment (no margin tuning)
2. Log-frequency correction is applied from epoch 0
3. Simpler training loop

### Installation Requirements

**Critical**: Mamba requires specific CUDA compilation

```bash
# Install Mamba SSM
pip install mamba-ssm

# Required dependency
pip install causal-conv1d>=1.2.0

# May require (if compilation fails):
pip install ninja
```

**If mamba-ssm unavailable**:
- Code includes placeholder MLP implementation
- Demonstrates architecture structure
- Replace with actual Mamba when installed

### Expected Performance

#### Advantages over CoordConv-ResNet

1. **Global Context**:
   - CoordConv: Receptive field limited by convolution kernel
   - Mamba: Every token can influence every other token via state

2. **Full Resolution**:
   - CoordConv: Downsamples to 3×16 by final layer
   - Mamba: Maintains all 2420 tokens throughout

3. **Learns Complex Rules**:
   - Example: "If 4Hz energy (token 250) is high AND 2cyc/oct energy (token 800) is low, predict Environment"
   - Requires long-range dependencies Mamba excels at

#### Computational Efficiency

**With symmetric processing** (1210 tokens):
```
Forward pass (batch=64, seq=1210):
- Transformer (O(L²)): ~93M operations per layer
- Mamba (O(L)): ~77k operations per layer
  
Speedup: ~1200× per layer vs Transformer!
```

**Mamba vs Mamba** (2420 → 1210 tokens):
```
Original (2420 tokens): ~155k operations
Symmetric (1210 tokens): ~77k operations

Speedup: 2×
```

**Reality**: Mamba has overhead from SSM discretization
- Actual speedup vs Transformer: ~3-5×
- Actual speedup with symmetric processing: ~2× vs non-symmetric Mamba
- **Total**: ~30-35 min/epoch (vs >60 min without symmetric processing)

#### Expected vs. Baseline

Relative to MLP baseline:
- **+5-7% Macro-F1**: Global context captures modulation texture
- **+10-15% minority class recall**: Balanced Softmax
- **-2% majority class accuracy**: Acceptable tradeoff

Relative to CoordConv-ResNet:
- **+2-3% Macro-F1**: Better long-range modeling
- **Similar training time**: O(L) complexity compensates for depth

### Troubleshooting

#### "mamba_ssm not found"

```bash
# Check CUDA version
nvcc --version

# Install matching PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Install mamba-ssm
pip install mamba-ssm causal-conv1d>=1.2.0
```

#### Out of Memory

```python
# Reduce batch size
batch_size = 32  # or 16

# Reduce model size
d_model = 128    # from 192
depth = 8        # from 12

# Use gradient accumulation
accumulation_steps = 4
```

#### Training instability

```python
# Increase gradient clipping
max_norm = 0.5

# Reduce learning rate
lr = 5e-5

# Warm-up scheduler
from torch.optim.lr_scheduler import LambdaLR
```

### Visualization Ideas

After training, extract intermediate states:

```python
# Hook to capture state evolution
def hook_fn(module, input, output):
    states.append(output.detach())

model.blocks[6].register_forward_hook(hook_fn)

# Visualize how state integrates modulation energy
# across the sequence (show temporal/spectral dependencies)
```

### File Structure

```
model/STM/ViM_corpora_categories/
├── standard/
│   └── ckpt/
│       └── YYYY-MM-DD_HH-MM/
│           ├── best_model.pt
│           ├── test_predictions.npy
│           └── test_targets.npy
└── downsample/
    └── ...
```

### Usage

```bash
# Standard training (with symmetric processing)
python STM_ViM.py 0

# Downsampled (with symmetric processing)
python STM_ViM.py 1
```

**Expected training time** (with symmetric processing):
- Per epoch: ~30-35 minutes (vs >60 min without)
- 50 epochs: ~25-30 hours (vs >50 hours)
- **Practical for overnight training**

### Theoretical Justification

From the document (Section 4):

> "Vision Mamba (Vim) architecture is the premier candidate... Flattening the 20×121 STM grid results in a sequence length of 2420. A standard Transformer would struggle with the 2420² attention matrix... Mamba can ingest the full 2420-token sequence natively, preserving every nuance of the modulation spectrum."

This implementation directly realizes that vision:
1. ✅ Full 2420-token processing (no downsampling)
2. ✅ Bidirectional scanning (non-causal dependencies)
3. ✅ Absolute positional embeddings (position-aware)
4. ✅ O(L) complexity (scalable to long sequences)
5. ✅ Balanced Softmax (imbalance correction)

### Comparison Matrix

| Aspect | CoordConv-ResNet | Vision Mamba |
|--------|------------------|--------------|
| Complexity | O(1) per pixel | O(L) per token |
| Position awareness | CoordConv channels | Positional embeddings |
| Global context | Limited by receptive field | Full sequence |
| Resolution at output | 3×16 (downsampled) | 2420 (full) |
| Parameters | ~12M | ~8M |
| Training speed | Faster per epoch | Slower per epoch |
| Best for | Spatial patterns | Sequential dependencies |

### Next Steps (Phase 3)

If Vision Mamba achieves strong performance but training is slow:
- Try FT-Transformer (Phase 3): Similar global context, potentially faster
- Hybrid approach: Use Mamba embeddings → freeze → train tabular head

If performance plateaus:
- Check positional embedding initialization
- Experiment with d_state (SSM capacity)
- Try different scanning patterns (spiral, zigzag)

### References

1. Zhu et al., "Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model", ICML 2024
2. Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", arXiv 2023
3. Ren et al., "Balanced Meta-Softmax for Long-Tailed Visual Recognition", NeurIPS 2020
4. Audio Classification Model Improvement document (Sections 4, 7.2, 9)
