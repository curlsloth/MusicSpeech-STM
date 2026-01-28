# STM Classification with Vision Mamba (Vim)
## Phase 2: The Modern Sequence Model

### Overview

Vision Mamba addresses the fundamental limitation of standard Transformers when processing the full 2420-dimensional STM feature vector: **O(L²) complexity makes attention prohibitively expensive for long sequences**. Mamba provides **O(L) complexity** while maintaining global context, enabling us to process every modulation bin without downsampling or PCA.

### Core Innovation: State Space Models (SSMs)

#### The Attention Problem

Standard Transformer for 2420 tokens:
```
Attention complexity: O(2420²) ≈ 5.86M operations per layer
Memory: 2420 × 2420 × num_heads attention matrices
```

This forces typical approaches to:
1. **Downsample**: Lose fine-grained modulation details
2. **Apply PCA**: Destroy topological structure (Section 10 warns against this)
3. **Patchify**: Arbitrary grouping of bins that may not respect harmonic relationships

#### The Mamba Solution

```
SSM complexity: O(2420) = 2420 operations per layer
Memory: Linear in sequence length
```

**Key Mechanism**: Selective State Space Model
- Maintains a hidden state `h_t` that integrates information across the sequence
- Updates state recursively: `h_t = A·h_{t-1} + B·x_t`
- Output: `y_t = C·h_t`
- **Selective**: A, B, C are input-dependent (unlike linear RNNs)

### Architecture Details

#### 1. Token Embedding

**Critical Decision**: 1×1 patches (each bin = 1 token)

```python
Input: (batch, 2420) - flattened STM
↓
Reshape: (batch, 2420, 1)
↓
Linear embedding: (batch, 2420, d_model=192)
```

**Why 1×1 patches**:
- Preserves maximum resolution
- No arbitrary grouping of spectral/temporal bins
- Each modulation bin retains independent representation
- Computationally feasible with O(L) complexity

**Comparison to standard Vim (images)**:
- Images: 16×16 patches to reduce 224×224 → 196 tokens
- STM: 1×1 patches, 20×121 → 2420 tokens (feasible with Mamba!)

#### 2. Positional Embeddings

**CRITICAL**: Learnable absolute positional embeddings

```python
pos_embed = nn.Parameter(torch.zeros(1, 2420, 192))
```

**Why learnable absolute (not relative)**:
- Addresses the "Translation Invariance" problem (Section 2.2)
- Model learns "Bin 50 (4Hz rate) has semantic meaning"
- Relative encoding (like in Conformer) masks absolute position
- Each token knows its exact (spectral_bin, temporal_bin) location

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
d_model: 192        # Embedding dimension
depth: 12           # Number of Vim blocks
d_state: 16         # SSM hidden state size
d_conv: 4           # Conv kernel in SSM (for local context)
expand: 2           # FFN expansion factor
```

Total parameters: ~8M (comparable to CoordConv-ResNet)

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

```
Forward pass (batch=64, seq=2420):
- Transformer (O(L²)): ~375M operations
- Mamba (O(L)): ~155k operations
  
Speedup: ~2400× per layer!
```

**Reality**: Mamba has overhead from SSM discretization
- Actual speedup: ~3-5× over Transformer
- Still enables processing full 2420 sequence

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
# Standard training
python STM_ViM.py 0

# Downsampled
python STM_ViM.py 1
```

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
