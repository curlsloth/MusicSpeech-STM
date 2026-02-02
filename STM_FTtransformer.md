# STM Classification with FT-Transformer
## Phase 3: The Tabular Approach

### Overview

FT-Transformer treats STM features as **tabular data** - continuous variables that can be analyzed through **feature-wise attention**. This perspective sidesteps the spatial inductive biases of CNNs and the sequential assumptions of RNNs/SSMs, directly discovering which modulation bins co-occur for each class.

**NEW: Symmetric STM Processing** - Following STMasm_enhanced5.py, we exploit the up/down-sweep symmetry of modulation spectra:
- **Input**: 20 freq × 121 rates = 2420 features
- **After symmetric processing**: 20 freq × 61 rates = **1220 features**
- **Speed gain**: **4× faster** (O(L²) means 1220²/2420² ≈ 0.25× operations)
- **Memory**: Attention matrices are 4× smaller

This enables practical training times (~10-15 minutes per epoch vs >1 hour).

### Core Philosophy: Tabular Deep Learning

#### Why Tabular?

STM features resemble structured data:
- **Row**: One audio sample
- **Columns**: 1220 features (modulation bins after symmetric processing)
- **Values**: Continuous energy values

Unlike images (where neighboring pixels are spatially related) or sequences (where tokens have temporal order), STM bins can be viewed as **independent measurements** that happen to have a 2D layout.

**Symmetric processing**: Average positive and negative modulation rates
- Original: 121 rates (-15 Hz to +15 Hz)
- Symmetric: 61 rates (0 Hz to +15 Hz)
- Rationale: Up/down sweeps convey similar information for classification

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
feature_embeddings: (1220, d_model)

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
Input: (batch, 1220) - flat feature vector after symmetric processing

↓ Feature Tokenizer
Tokens: (batch, 1220, d_model=192)

↓ Add CLS token (optional)
(batch, 1221, d_model)

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

**Attention matrix size**: (batch, 8 heads, 1221, 1221) ≈ 1.49M elements
- Original (2421 tokens): 5.86M elements
- **4× memory reduction!**

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
n_features: 1210   # After symmetric STM processing (20×61)
d_model: 192       # Embedding per feature
n_heads: 8         # Multi-head attention
depth: 6           # Transformer blocks
d_ff: 512          # Feed-forward hidden dim
use_gradient_checkpointing: True  # CRITICAL for memory management
```

**Rationale**:
- `n_features=1220`: Reduced from 2420 via symmetric processing
- `d_model=192`: Each feature gets a rich 192-dim representation
- `n_heads=8`: 8 different attention patterns to capture diverse correlations
- `depth=6`: Moderately deep (vs 12 in Vim) - tabular data needs less depth
- `d_ff=512`: Standard 2-3× expansion in FFN
- **`use_gradient_checkpointing=True`**: Still needed despite 1220 tokens

**Parameter count**: ~5M (efficient for tabular data)

**⚠️ Memory Note**: Even with 1220 tokens, attention matrices are large (~1.5M per layer). Gradient checkpointing remains essential.

**Speed improvement with symmetric processing**:
- Attention operations: 1220² vs 2420² = **4× faster**
- Per epoch: ~10-15 min (vs >60 min without symmetric processing)
- 50 epochs: ~8-12 hours (vs >50 hours)

#### Training

```python
Optimizer: AdamW
  - lr: 1e-4
  - weight_decay: 1e-5 (lighter than CoordConv)

Scheduler: CosineAnnealingLR

Batch size: 64 (optimized for 1220 tokens)

Loss: Balanced Softmax
```

**Batch size rationale**: O(L²) attention creates (batch × heads × 1221²) tensor. With 1220 features (not 2420), batch=64 is safe on 48GB GPUs.

### Expected Performance

#### Strengths

1. **Discovers feature interactions**:
   - Model learns "4Hz + 2cyc/oct = Speech" without spatial bias
   - Attention weights are interpretable (see which bins matter)

2. **No PCA needed**:
   - Processes all 1220 features after symmetric processing
   - Feature Tokenizer acts as learned dimensionality reduction

3. **Flexible**:
   - No assumptions about 2D structure
   - Works even if bins were shuffled (though embeddings would adjust)

#### Potential Weaknesses

1. **O(L²) complexity** (mitigated by symmetric processing):
   - Original: 2420² ≈ 5.86M attention operations per layer
   - Symmetric: 1220² ≈ 1.49M attention operations per layer
   - **4× speedup** makes training practical

2. **May underutilize topology**:
   - Doesn't explicitly know "Feature 125 and 126 are adjacent"
   - Relies on embeddings to learn spatial relationships
   - CoordConv explicitly provides this

3. **Symmetric processing trade-off**:
   - Loses distinction between rising vs falling frequency sweeps
   - Acceptable for most audio classification tasks
   - Can be disabled if directionality is critical

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
cls_attention = attn_weights[speech_indices, :, 0, 1:]  # (n_speech, n_heads, 1220)

# Average across heads and samples
avg_attention = cls_attention.mean(dim=(0, 1))  # (1220,)

# Reshape to 2D (after symmetric processing: 20 freq x 61 rates)
attn_map = avg_attention.reshape(20, 61)

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

importance = input.grad.abs().mean(dim=0)  # (1220,)
```

#### 2. Attention Rollout

Combine attention across layers:
```python
attn_rollout = torch.eye(1221)
for layer in range(6):
    attn = attention_weights[layer]  # (n_heads, 1221, 1221)
    attn = attn.mean(dim=0)  # Average heads
    attn_rollout = attn @ attn_rollout

# Final CLS attention to each feature
final_attention = attn_rollout[0, 1:]  # (1220,)
    attn = attn.mean(dim=0)  # Average heads
    attn_rollout = attn @ attn_rollout

# Final CLS attention to each feature
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
│           ├── latest_checkpoint.pt
│           ├── checkpoint_epoch_5.pt
│           ├── checkpoint_epoch_10.pt
│           ├── ...
│           ├── test_predictions.npy
│           └── test_targets.npy
└── downsample/
    └── ...
```

### Usage

```bash
# Standard training (with symmetric processing)
python STM_FTtransformer.py 0

# Downsampled (with symmetric processing)
python STM_FTtransformer.py 1

# Resume training from a checkpoint directory
python STM_FTtransformer.py 0 --resume model/STM/FTTransformer_corpora_categories/standard/ckpt/2026-02-01_10-30

# Resume will automatically load from latest_checkpoint.pt or the most recent checkpoint_epoch_*.pt
```

**Expected training time** (with symmetric processing):
- Per epoch: ~10-15 minutes (vs >60 min without)
- 50 epochs: ~8-12 hours (vs >50 hours)
- **Practical for same-day experimentation**

**Checkpoint strategy**:
- `best_model.pt`: Model with highest validation F1 score
- `latest_checkpoint.pt`: Most recent epoch (updated every epoch) - use for resume
- `checkpoint_epoch_N.pt`: Saved every 5 epochs for backup

### Troubleshooting

#### **CRITICAL: GPU Out of Memory (OOM)**

The FT-Transformer has O(L²) attention complexity. With symmetric processing, this is much improved:

**Memory usage per batch** (after symmetric processing):
- Attention matrix size: (batch_size, n_heads, 1221, 1221)
- With batch=64, heads=8: **~4.7GB per layer** (vs 19GB without symmetric processing)
- 6 layers total = manageable on 48GB GPUs

**Solutions already implemented**:

1. **Symmetric STM processing** (CRITICAL):
   ```python
   # Automatically applied in data loading
   # 2420 features → 1220 features
   # Memory: 4× reduction
   # Speed: 4× faster
   ```

2. **Optimized batch size**:
   ```python
   batch_size = 64  # Optimized for 1220 tokens
   ```

3. **Gradient checkpointing**:
   ```python
   model = FTTransformer(
       ...,
       use_gradient_checkpointing=True
   )
   ```
   - **Memory savings**: ~40-50%
   - **Speed cost**: ~20% slower training

**If still OOM, further reduce model size**:
   ```python
   # Option A: Fewer attention heads
   n_heads = 4  # From 8
   
   # Option B: Smaller embedding
   d_model = 128  # From 192
   
   # Option C: Fewer layers
   depth = 4  # From 6
   ```

**Memory usage breakdown** (WITH symmetric processing):
```
Component                    Memory (batch=64, fp32)
--------------------------------------------------
Input features (64×1220)     ~312 KB
Feature embeddings           ~900 KB
Attention matrices (6 layers) ~4.7 GB  (vs 19GB without symmetric!)
Gradients (2x model size)    ~10 GB
Total estimated:             ~15 GB (fits in 48GB GPU comfortably)
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
1. ✅ Per-feature embeddings (1220 unique vectors)
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
- ✅ No PCA (preserve full features: 2420 or 1220 after symmetric processing)
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
