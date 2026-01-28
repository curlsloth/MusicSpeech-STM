# STM Classification with FT-Transformer
## Phase 3: The Tabular Approach

### Overview

FT-Transformer treats STM features as **tabular data** - 2420 continuous variables that can be analyzed through **feature-wise attention**. This perspective sidesteps the spatial inductive biases of CNNs and the sequential assumptions of RNNs/SSMs, directly discovering which modulation bins co-occur for each class.

### Core Philosophy: Tabular Deep Learning

#### Why Tabular?

STM features resemble structured data:
- **Row**: One audio sample
- **Columns**: 2420 features (modulation bins)
- **Values**: Continuous energy values

Unlike images (where neighboring pixels are spatially related) or sequences (where tokens have temporal order), STM bins can be viewed as **independent measurements** that happen to have a 2D layout.

**Analogy to medical diagnostics**:
- Feature 250 (4Hz temporal modulation) = "Heart rate"
- Feature 800 (2 cyc/oct spectral modulation) = "Blood pressure"
- **Question**: Which combinations predict "Speech" vs "Music"?

FT-Transformer answers this by learning attention weights between features.

### Architecture Details

#### 1. Feature Tokenizer

**Problem with standard MLPs**:
```python
# Standard MLP layer 1
hidden = W @ features  # (d_hidden, 2420) @ (2420, 1)
```
This immediately mixes all features! The model loses track of "which feature contributed what."

**FT-Transformer solution**:
```python
# Each feature gets a unique learnable embedding
feature_embeddings: (2420, d_model)

# Project feature value
value_proj = Linear(feature_value)  # (1,) → (d_model,)

# Combine
token = value_proj + feature_embedding
```

**Result**: Each feature remains **distinct** before attention.

**Concrete example**:
```
Feature 250 (4Hz rate):
  - Embedding vector: [0.31, -0.52, 0.18, ...] (learned from data)
  - Value: 0.87 (high energy at 4Hz)
  - Token: value_projection(0.87) + embedding_250
  
Feature 800 (2 cyc/oct scale):
  - Embedding vector: [-0.12, 0.43, 0.91, ...]
  - Value: 0.65
  - Token: value_projection(0.65) + embedding_800
```

#### 2. Attention Mechanism

**Standard Transformer attention**:
```python
Q, K, V = tokens @ W_Q, tokens @ W_K, tokens @ W_V
attention_weights = softmax(Q @ K^T / √d_model)
output = attention_weights @ V
```

**For STM, this discovers**:
- **Which features attend to which**: If Feature 250 attends strongly to Feature 800, 
  the model learns "4Hz temporal modulation co-occurs with 2cyc/oct spectral for this class"
  
- **Class-specific patterns**: Attention maps differ for Speech vs Music
  - Speech: High attention between low-rate (4Hz) and low-scale (smooth) bins
  - Music: High attention between harmonic rate patterns and high-scale (textured) bins

#### 3. Architecture Flow

```
Input: (batch, 2420) - flat feature vector

↓ Feature Tokenizer
Tokens: (batch, 2420, d_model=192)

↓ Add CLS token (optional)
(batch, 2421, d_model)

↓ Transformer Block 1
  ├─ Multi-head attention (8 heads)
  ├─ Add & Norm
  ├─ Feed-forward (d_model → 512 → d_model)
  └─ Add & Norm

↓ Transformer Blocks 2-6 (repeat)

↓ Extract CLS token
(batch, d_model)

↓ Classification Head
  ├─ Linear(192 → 96)
  ├─ GELU
  ├─ Dropout(0.1)
  └─ Linear(96 → 6)

Output: (batch, 6) - class logits
```

#### 4. CLS Token

**Inspiration**: BERT uses a special [CLS] token for classification

**For FT-Transformer**:
```python
cls_token = nn.Parameter(torch.randn(1, 1, d_model))
tokens = torch.cat([cls_token, feature_tokens], dim=1)
```

**Purpose**:
- CLS token aggregates information from all features via attention
- Final hidden state of CLS → classification decision
- Alternative: Mean pool all feature tokens

**Why CLS works**:
- Learns to "ask questions" of features via attention
- Example: "Is there high energy at 4Hz?" → attends to Feature 250
- Then: "Is there low energy at 8 cyc/oct?" → attends to Feature 1500
- Combines answers → "Speech"

### Comparison to Other Approaches

| Aspect | MLP | FT-Transformer |
|--------|-----|----------------|
| Feature mixing | Immediate (Layer 1) | Gradual (via attention) |
| Feature interaction | Fixed (weight matrix) | Dynamic (attention) |
| Interpretability | Black box | Attention weights |
| Parameters | ~10M (fully connected) | ~5M (shared attention) |

| Aspect | CoordConv-ResNet | FT-Transformer |
|--------|------------------|----------------|
| Inductive bias | Spatial (convolution) | None (pure attention) |
| Position awareness | Coordinate channels | Feature embeddings |
| Global context | Limited (receptive field) | Full (attention) |
| Best for | Spatial patterns | Feature correlations |

| Aspect | Vision Mamba | FT-Transformer |
|--------|--------------|----------------|
| Sequence assumption | Yes (SSM) | No (bag of features) |
| Complexity | O(L) | O(L²) |
| Global context | Full | Full |
| Bidirectional | Explicit (2 scans) | Implicit (attention) |

### Hyperparameter Choices

#### Model Configuration

```python
d_model: 192       # Embedding per feature
n_heads: 8         # Multi-head attention
depth: 6           # Transformer blocks
d_ff: 512          # Feed-forward hidden dim
use_gradient_checkpointing: True  # CRITICAL for memory management
```

**Rationale**:
- `d_model=192`: Each feature gets a rich 192-dim representation
- `n_heads=8`: 8 different attention patterns to capture diverse correlations
- `depth=6`: Moderately deep (vs 12 in Vim) - tabular data needs less depth
- `d_ff=512`: Standard 2-3× expansion in FFN
- **`use_gradient_checkpointing=True`**: Essential for GPU memory with 2421 tokens

**Parameter count**: ~5M (efficient for tabular data)

**⚠️ Memory Note**: With 2421 tokens, attention matrices are huge (24GB per layer). Gradient checkpointing is mandatory.

#### Training

```python
Optimizer: AdamW
  - lr: 1e-4
  - weight_decay: 1e-5 (lighter than CoordConv)

Scheduler: CosineAnnealingLR

Batch size: 32 (REDUCED from 128 to prevent OOM)

Loss: Balanced Softmax
```

**Batch size rationale**: O(L²) attention creates (batch × heads × 2421²) tensor. Batch=32 keeps memory <45GB on A100.

### Expected Performance

#### Strengths

1. **Discovers feature interactions**:
   - Model learns "4Hz + 2cyc/oct = Speech" without spatial bias
   - Attention weights are interpretable (see which bins matter)

2. **No PCA needed**:
   - Processes all 2420 features directly
   - Feature Tokenizer acts as learned dimensionality reduction

3. **Flexible**:
   - No assumptions about 2D structure
   - Works even if bins were shuffled (though embeddings would adjust)

#### Potential Weaknesses

1. **O(L²) complexity**:
   - 2420² ≈ 5.86M attention operations per layer
   - Slower than Mamba's O(L)
   - But: Only 6 layers (vs 12 in Mamba) compensates

2. **May underutilize topology**:
   - Doesn't explicitly know "Feature 250 and 251 are adjacent"
   - Relies on embeddings to learn spatial relationships
   - CoordConv explicitly provides this

3. **Potential overfitting**:
   - High expressivity (attention can fit anything)
   - Requires good regularization (dropout, weight decay)

#### Performance Predictions

Relative to MLP baseline:
- **+4-6% Macro-F1**: Explicit feature interactions
- **+8-12% minority recall**: Balanced Softmax
- **Similar training time to CoordConv**: Both O(L²), different architectures

Relative to CoordConv-ResNet:
- **+1-2% if spatial patterns weak**: Pure feature correlations matter more
- **-1-2% if spatial patterns strong**: Misses local structure

Relative to Vision Mamba:
- **Faster per epoch**: Fewer layers, simpler operations
- **Similar final F1**: Both achieve global context
- **Better interpretability**: Attention weights

### Interpretability: Attention Visualization

After training, extract attention weights:

```python
# Hook into Transformer block
attentions = []

def hook_fn(module, input, output):
    # output[1] contains attention weights
    attentions.append(output[1])

model.blocks[3].attention.register_forward_hook(hook_fn)

# Forward pass
model.eval()
with torch.no_grad():
    _ = model(sample_input)

# Analyze attention
attn_weights = attentions[0]  # (batch, n_heads, n_features+1, n_features+1)

# Example: Which features does CLS attend to for "Speech" samples?
cls_attention = attn_weights[speech_indices, :, 0, 1:]  # (n_speech, n_heads, 2420)

# Average across heads and samples
avg_attention = cls_attention.mean(dim=(0, 1))  # (2420,)

# Reshape to 2D
attn_map = avg_attention.reshape(20, 121)

# Visualize
plt.imshow(attn_map, cmap='hot')
plt.xlabel('Temporal Modulation (Rate)')
plt.ylabel('Spectral Modulation (Scale)')
plt.title('CLS Token Attention for Speech Class')
```

**Expected pattern for Speech**:
- High attention around (low rate, low scale) region
- Corresponds to syllabic rhythm (4Hz) and smooth spectrum

### Advanced Techniques

#### 1. Feature Importance

```python
# Compute feature importance via gradient
model.eval()
input.requires_grad = True
output = model(input)
output[:, class_idx].sum().backward()

importance = input.grad.abs().mean(dim=0)  # (2420,)
```

#### 2. Attention Rollout

Combine attention across layers:
```python
attn_rollout = torch.eye(2421)
for layer in range(6):
    attn = attention_weights[layer]  # (n_heads, 2421, 2421)
    attn = attn.mean(dim=0)  # Average heads
    attn_rollout = attn @ attn_rollout

# Final CLS attention to each feature
final_attention = attn_rollout[0, 1:]  # (2420,)
```

#### 3. Attention Gating

Sparsify attention for efficiency:
```python
# In TransformerBlock
attn_weights = attention(Q, K, V)
gate = (attn_weights > threshold).float()
sparse_attn = attn_weights * gate
```

### File Structure

```
model/STM/FTTransformer_corpora_categories/
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
python STM_FTtransformer.py 0

# Downsampled
python STM_FTtransformer.py 1
```

### Troubleshooting

#### **CRITICAL: GPU Out of Memory (OOM)**

The FT-Transformer has O(L²) attention complexity with L=2421 tokens. This creates large attention matrices:

**Memory usage per batch**:
- Attention matrix size: (batch_size, n_heads, 2421, 2421)
- With batch=128, heads=8: **~24GB per layer** (6 layers total!)

**Solutions implemented**:

1. **Reduced batch size** (DONE):
   ```python
   batch_size = 32  # Reduced from 128
   ```

2. **Gradient checkpointing** (DONE):
   ```python
   model = FTTransformer(
       ...,
       use_gradient_checkpointing=True  # Trades compute for memory
   )
   ```
   This recomputes activations during backward pass instead of storing them.
   - **Memory savings**: ~40-50%
   - **Speed cost**: ~20% slower training

3. **If still OOM, reduce model size**:
   ```python
   # Option A: Fewer attention heads
   n_heads = 4  # From 8
   
   # Option B: Smaller embedding
   d_model = 128  # From 192
   
   # Option C: Fewer layers
   depth = 4  # From 6
   ```

4. **Advanced: Flash Attention** (if PyTorch ≥ 2.0):
   ```python
   # In TransformerBlock.forward(), replace:
   attn_out, _ = self.attention(x, x, x)
   
   # With scaled_dot_product_attention (automatically uses Flash Attention):
   from torch.nn.functional import scaled_dot_product_attention
   Q, K, V = self.attention._get_qkv(x)  # Custom method needed
   attn_out = scaled_dot_product_attention(Q, K, V)
   ```
   Flash Attention provides 2-4× memory reduction with no quality loss.

**Memory usage breakdown**:
```
Component                    Memory (batch=32, fp32)
--------------------------------------------------
Input features (32×2420)     ~311 KB
Feature embeddings           ~1.8 MB
Attention matrices (6 layers) ~21 GB  ← BOTTLENECK!
Gradients (2x model size)    ~20 GB
Total estimated:             ~42 GB (fits in 48GB GPU)
```

#### Slow Training

```python
# Reduce model size
d_model = 128
depth = 4

# Or mixed precision (if not already enabled)
torch.cuda.amp.autocast()
```

#### Overfitting

```python
# Increase regularization
dropout = 0.2
weight_decay = 1e-4

# Or reduce model capacity
n_heads = 4
depth = 4
```

#### Underfitting

```python
# Increase capacity (only if GPU memory allows!)
d_model = 256
depth = 8
d_ff = 1024

# Or reduce regularization
dropout = 0.05
```

### Theoretical Justification

From the document (Section 6.1):

> "The FT-Transformer offers a middle ground between the MLP and the Transformer. It assigns a learnable embedding to each *feature*... It applies attention *between features*. The model learns which modulation bins typically co-occur for specific classes... Why it beats MLP: MLPs multiply all features by a weight matrix in the first layer, mixing them immediately. FT-Transformer allows features to remain distinct and interact dynamically via attention before being mixed."

This implementation realizes that principle:
1. ✅ Per-feature embeddings (2420 unique vectors)
2. ✅ Feature-wise attention (discovers correlations)
3. ✅ Dynamic mixing (attention is input-dependent)
4. ✅ No spatial assumptions (works for any feature layout)
5. ✅ Balanced Softmax (handles imbalance)

### When to Use FT-Transformer

**Use FT-Transformer if**:
- Interpretability is important (attention weights)
- You suspect feature interactions matter more than spatial structure
- You want to compare against pure attention (no conv or SSM biases)

**Use CoordConv-ResNet if**:
- Spatial patterns dominate (e.g., energy "blobs" in specific regions)
- You need fast inference (convolution is highly optimized)

**Use Vision Mamba if**:
- Long-range dependencies are critical
- You need to process full resolution (2420 bins)
- Memory is constrained (O(L) vs O(L²))

### Summary Table

| Phase | Architecture | Key Innovation | Complexity | Best For |
|-------|-------------|----------------|------------|----------|
| 1 | CoordConv-ResNet | Position-aware convolution | O(1) per pixel | Spatial patterns |
| 2 | Vision Mamba | SSM with bidirectional scanning | O(L) | Long sequences |
| 3 | FT-Transformer | Per-feature embeddings + attention | O(L²) | Feature correlations |

All three:
- ✅ No PCA (preserve full 2420 dimensions)
- ✅ Address class imbalance (LDAM or Balanced Softmax)
- ✅ Respect position semantics (CoordConv / Embeddings / Embeddings)
- ✅ Capture global context (to varying degrees)

### Conclusion

FT-Transformer completes the three-phase exploration:
1. **Phase 1**: Fixed the spatial problem (translation invariance)
2. **Phase 2**: Fixed the complexity problem (long sequences)
3. **Phase 3**: Fixed the mixing problem (feature independence)

The best model depends on which problem dominates in STM data. Run all three and compare!

### References

1. Gorishniy et al., "Revisiting Deep Learning Models for Tabular Data", NeurIPS 2021
2. Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", NAACL 2019 (CLS token concept)
3. Ren et al., "Balanced Meta-Softmax for Long-Tailed Visual Recognition", NeurIPS 2020
4. Audio Classification Model Improvement document (Sections 6, 9)
