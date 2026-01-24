# ASM Modified Features Summary

## Executive Summary

**ASM Modified Features** implements custom STM preprocessing inspired by auditory neuroscience:
- **Root Hypothesis**: Raw STM features treat all frequencies equally, but auditory systems have 1/f response characteristics
- **Solution**: Apply 1/f normalization + symmetric decomposition to enhance energy representation
- **Innovation**: Feature engineering meets neural architecture optimization
- **Target**: Match or exceed Enhanced ASM v3 (0.87+) through better input representation

## Motivation: Why Modify STM Features?

### Problem with Raw STM Features

**Current Processing** (STM06 → ASM v3):
```python
# Load STM: (150, 500) power spectrum
# Crop to region of interest
# dB transform: 10 × log10(power)
# Normalize to [0, 1]: (x - min) / (max - min)
# Reshape to (121, 20): rate × scale
# Feed to model
```

**Issues**:
1. **Uniform frequency treatment**: All rates weighted equally
2. **Loss of directionality**: Positive vs negative rates treated independently
3. **1/f natural bias**: Low frequencies dominate, high frequencies underrepresented
4. **Missing biological priors**: Auditory system has known frequency response curves

### Biological Inspiration

**Auditory System Characteristics**:
- **1/f response**: Sensitivity decreases with frequency
- **Temporal asymmetry**: Direction of frequency sweeps matters (up-sweep vs down-sweep)
- **Opponent coding**: Neural populations encode "difference" between complementary stimuli

**Translation to STM**:
- **Rate axis** (-15 Hz to +15 Hz): Temporal modulation frequency
- **1/f normalization**: Counter low-frequency dominance
- **Symmetric map**: Total energy (up + down sweeps)
- **Asymmetric map**: Directional preference (up - down sweeps)

## Feature Preprocessing Pipeline

### Step 1: 1/f Normalization

**Mathematical Formulation**:
```
For each rate bin ω at index i:
    P_normalized(ω, s) = P_raw(ω, s) × |ω|

Where:
    ω = rate (frequency in Hz)
    s = scale (cyc/oct)
    P_raw = original STM power
    P_normalized = normalized power
```

**Implementation**:
```python
def preprocess_stm_features(stm_2d):
    # stm_2d: (batch, 121, 20)
    
    # Generate frequency vector
    frequency_vector = torch.linspace(-15.0, 15.0, 121)
    # [-15.00, -14.75, ..., -0.25, 0.00, 0.25, ..., 14.75, 15.00]
    
    # Compute absolute frequency
    abs_freq = torch.abs(frequency_vector)  # (121,)
    # [15.00, 14.75, ..., 0.25, 0.00, 0.25, ..., 14.75, 15.00]
    
    # Apply 1/f normalization
    stm_normalized = stm_2d * abs_freq.view(1, 121, 1)
    
    # Preserve DC component (index 60, ω=0 Hz)
    dc_index = 60
    stm_normalized[:, dc_index, :] = stm_2d[:, dc_index, :]
```

**Effect**:
- **Low frequencies** (e.g., 0.25 Hz): Multiplied by 0.25 → reduced
- **Mid frequencies** (e.g., 5.0 Hz): Multiplied by 5.0 → boosted
- **High frequencies** (e.g., 15.0 Hz): Multiplied by 15.0 → strongly boosted
- **DC (0 Hz)**: Preserved unchanged (would become 0 otherwise)

**Rationale**:
- Raw STM has natural 1/f bias (low freq = high power)
- Multiplication by |ω| flattens the spectrum
- Helps model attend to high-frequency modulations
- Biologically plausible: compensates for 1/f sensitivity drop

### Step 2: Symmetric Decomposition (UPDATED - SYMMETRIC ONLY)

**Conceptual Framework**:
```
Rate Axis Structure:
    Negative rates (0-59):   Up-sweeps    (-15 Hz to -0.25 Hz)
    DC component (60):       No modulation (0 Hz)
    Positive rates (61-120): Down-sweeps  (+0.25 Hz to +15 Hz)

Hypothesis:
    - Symmetric information: Average energy regardless of direction
    - Simpler representation focuses on magnitude, not direction
```

**Mathematical Formulation (SYMMETRIC ONLY)**:
```
Let:
    P⁻(ω, s) = power at negative rate ω (indices 0-59)
    P⁺(ω, s) = power at positive rate ω (indices 61-120)

After flipping negative chunk to align:
    P⁻_flipped(i, s) = P⁻(59-i, s)  for i ∈ [0, 59]

Compute:
    M_sym(i, s) = (|P⁺(i, s)| + |P⁻_flipped(i, s)|) / 2  (Averaged Energy)

Output structure:
    Shape: (61, 20)
    Dimension 0: Frequency (0-15 Hz) = [DC, 0.25Hz, 0.5Hz, ..., 15Hz]
    Dimension 1: Scale (0-7.09 cyc/oct)
```

**Implementation (SYMMETRIC ONLY)**:
```python
def preprocess_stm_features(stm_2d):
    # ...after 1/f normalization...
    
    # Separate components
    negative_chunk = stm_normalized[:, :60, :]      # (batch, 60, 20)
    positive_chunk = stm_normalized[:, 61:, :]      # (batch, 60, 20)
    dc_component = stm_normalized[:, 60:61, :]      # (batch, 1, 20)
    
    # Flip negative to align with positive
    negative_flipped = torch.flip(negative_chunk, dims=[1])
    
    # Compute symmetric map only: AVERAGE
    symmetric_map = (torch.abs(positive_chunk) + torch.abs(negative_flipped)) / 2.0
    
    # Concatenate DC with symmetric map
    processed = torch.cat([dc_component, symmetric_map], dim=1)  # (batch, 61, 20)
    
    return processed
```

**Output Interpretation (SYMMETRIC ONLY)**:
```
Shape: (batch, 61, 20)

Dimension 0 (Frequency, 0-15 Hz):
    Index 0:    0.00 Hz (DC)
    Index 1:    0.25 Hz
    Index 2:    0.50 Hz
    ...
    Index 60:   15.0 Hz

Dimension 1 (Scale, 0-7.09 cyc/oct):
    20 bins representing spectral modulation

Symmetric map properties:
    - Averaged energy: (|P+| + |P-|) / 2
    - Always positive
    - Represents total modulation strength regardless of direction
    - More stable than raw power (averaging reduces noise)
```

**Key Design Decisions**:
1. **Single channel**: Simpler than 2-channel approach
2. **Symmetric averaging**: Robust energy estimate
3. **No directional info**: Trades directional detail for simplicity
4. **Increased model capacity**: dim=144 vs 128 to compensate for reduced features

**Rationale (SYMMETRIC ONLY)**:
- **Simplicity**: Single-channel input, standard architecture
- **Averaging**: More stable energy estimates than raw power
- **Efficiency**: 61×20 = 1,220 features vs original 2,420
- **Model compensation**: Slightly increased capacity (dim=144) offsets information loss

### Step 3: Per-Sample Normalization

**After preprocessing, normalize across spatial dimensions**:
```python
# STM_processed: (batch, 61, 20)
means = STM_processed.mean(dim=(1, 2), keepdim=True)  # (batch, 1, 1)
stds = STM_processed.std(dim=(1, 2), keepdim=True)    # (batch, 1, 1)
STM_processed = (STM_processed - means) / (stds + 1e-8)
```

**Effect**:
- Centers each sample to mean=0, std=1
- Applied AFTER symmetric averaging
- Ensures both channels have comparable scale after z-normalization
- Prevents any remaining magnitude imbalance

## Integration with ASM Architecture

### Data Flow (SYMMETRIC ONLY)

```
Raw STM (flattened) → Reshape to 2D (121, 20)
                   ↓
         1/f Normalization (by rate)
                   ↓
         Separate Negative/Positive Rates
                   ↓
         Flip & Align
                   ↓
         Compute Symmetric Map (averaged energy)
                   ↓
         Concatenate DC with symmetric map
                   ↓
         Single-channel tensor (61, 20)
                   ↓
         Per-sample Z-normalization
                   ↓
         (batch, 61, 20) → ASM Model
```

### Model Architecture (SYMMETRIC ONLY)

**Key changes for single-channel input with increased capacity**:
```python
ModifiedFeatureASMClassifier(
    time_steps=20,      # Scale dimension
    freq_steps=61,      # Frequency dimension (0-15 Hz)
    num_classes=6,
    dim=144,           # Increased from 128
    num_blocks=4,
    shift_range=2,
    expansion_factor=2,
    dropout=0.1
)
```

**Input Processing**:
```python
def forward(self, x, return_features=False):
    # x: (batch, 61, 20) - modified STM features
    
    # Add channel dimension: (batch, 1, 61, 20)
    x = x.unsqueeze(1)
    
    x = self.spec_augment(x)
    
    # Input projection: 1 input channel, increased first layer
    x = self.input_proj(x)  # Conv2d(1, dim//3, ...) → Conv2d(dim//3, dim, ...)
    
    # ...rest unchanged...
```

**Updated Input Projection Layer**:
```python
self.input_proj = nn.Sequential(
    nn.Conv2d(1, dim // 3, kernel_size=3, padding=1),  # 1 input channel, dim//3 = 48
    nn.BatchNorm2d(dim // 3),
    nn.GELU(),
    nn.Conv2d(dim // 3, dim, kernel_size=3, padding=1),
    nn.BatchNorm2d(dim),
    nn.GELU()
)
```

**Parameter count**: ~900K (slightly more than v3 due to dim=144)

## Training Configuration

### Same as ASM v3 (Proven Recipe)

**Loss Function**:
```python
ContrastiveFocalLoss(
    alpha=confusion_aware_weights,  # sqrt-based
    gamma=2.0,
    label_smoothing=0.01,
    contrastive_weight=0.1,
    similar_pairs=[(0, 1), (2, 3)]
)
```

**Optimizer & Scheduler**:
```python
AdamW(lr=1e-3, weight_decay=1e-4)
CosineAnnealingWarmRestarts(T_0=10, T_mult=2, eta_min=1e-6)
Warmup: 5 epochs
```

**Hyperparameters**:
```python
batch_size = 128
num_epochs = 50
gradient_clip_norm = 1.0
```

## Expected Performance Improvements

### Hypothesis 1: Better High-Frequency Discrimination

**Prediction**:
- Classes with high-rate modulations benefit from 1/f correction
- **Music classes** (2, 3): Often have rapid modulations → should improve
- **Speech classes** (0, 1): Slower modulations → less impact

**Target**:
```
Class 2 (music: vocal):     0.84 → 0.86 (+2 points)
Class 3 (music: non-vocal): 0.74 → 0.77 (+3 points)
```

### Hypothesis 2: Directional Discrimination (UPDATED)

**Prediction**:
- **Rescaled asymmetric channel** ensures equal contribution to learning
- Averaging symmetric map prevents magnitude imbalance
- 2-channel structure allows Conv2d to learn cross-channel patterns

**Expected benefits**:
```
Symmetric channel (averaged energy):
    - More stable estimates (averaging reduces noise)
    - Comparable magnitude to asymmetric channel
    - Better for classes with strong overall modulation

Asymmetric channel (rescaled direction):
    - Normalized importance across samples
    - Prevents outlier samples from dominating gradients
    - Better for classes with directional preferences

Combined:
    - Conv2d can learn complementary features from both channels
    - Cross-channel interactions captured by spatial convolutions
    - More balanced feature learning
```

**Target**:
```
Speech classes (0, 1):  Better separation via prosodic directionality
Music classes (2, 3):   Better separation via melodic energy+direction
Env classes (4, 5):     Maintain performance (less directional)
```

### Hypothesis 3: Overall Performance (UPDATED)

**Conservative Target**:
```
Test Macro F1: 0.87-0.88 (match or slightly exceed ASM v3)

Per-Class F1:
  Class 0: 0.95-0.96 (maintain)
  Class 1: 0.82-0.85 (+1-4 points over v3) ← Improved via rescaling
  Class 2: 0.85-0.87 (+1-3 points over v3) ← Improved via 2-channel
  Class 3: 0.76-0.79 (+2-5 points over v3) ← KEY IMPROVEMENT
  Class 4: 0.92-0.94 (maintain)
  Class 5: 0.91-0.93 (maintain)
```

**Optimistic Target**:
```
Test Macro F1: 0.88-0.90 (beat ASM v3 by 1-3 points)
Class 3 F1 ≥ 0.78 (break through 0.75 barrier)
All classes F1 ≥ 0.80 (uniform improvement)
```

**Why this version might perform better**:
1. **Balanced channels**: Rescaling prevents asymmetric channel from being ignored
2. **Reduced redundancy**: Single-sided spectrum (0-15 Hz) is more efficient
3. **Better Conv2d input**: 2 channels vs 1 channel with artificial concatenation
4. **Stable gradients**: Normalized ranges prevent gradient issues

## Comparison with ASM v3

### Key Differences (SYMMETRIC ONLY)

| Aspect | ASM v3 | Modified Features (Symmetric) |
|--------|--------|-------------------------------|
| **Preprocessing** | dB + normalize | dB + normalize + 1/f + symmetric |
| **Feature Dim** | (121, 20) | (61, 20) |
| **Channels** | 1 | 1 |
| **Frequency Range** | -15 to +15 Hz | 0 to +15 Hz |
| **Symmetric** | N/A | Averaged: (|P+|+|P-|)/2 |
| **Asymmetric** | N/A | Not used |
| **Feature Count** | 2,420 | 1,220 (50% reduction) |
| **Model Dim** | 128 | 144 (+12.5%) |
| **Params** | ~800K | ~900K |
| **Bio-Inspired** | No | Yes (1/f, symmetric) |
| **Expected F1** | 0.87+ | **0.87-0.89** |

**Why symmetric-only might work**:
1. **1/f normalization is key**: Spectral balancing may be main driver
2. **Simpler is better**: Reduced complexity, easier optimization
3. **Averaging reduces noise**: More stable energy estimates
4. **Directional info redundant**: Energy sufficient for classification
5. **Model compensation**: Increased capacity offsets information loss

### Complementary Approaches

**ASM v3 Strategy**: "Better learning algorithm"
- Softer class weighting
- Contrastive loss for separation
- Minimal label smoothing

**Modified Features Strategy**: "Better input representation"
- 1/f normalization for spectral balance
- Symmetric/asymmetric decomposition for directionality
- Biologically-inspired preprocessing

**Combined Power**:
- If both work independently → potentially 0.88-0.89 F1
- If synergistic → could reach 0.90 F1
- If redundant → only incremental gain

## Ablation Study Design

### Experiments to Run

**1. Baseline Comparison**:
```bash
# ASM v3 (no modified features)
python STMasm_enhanced3.py 0

# Modified Features (this script)
python STMasm_feateng.py 0
```

**2. Feature Component Ablation**:
```python
# Experiment A: Only 1/f normalization
def preprocess_stm_features_1f_only(stm_2d):
    # Apply 1/f, but don't decompose
    frequency_vector = torch.linspace(-15.0, 15.0, 121)
    abs_freq = torch.abs(frequency_vector).view(1, 121, 1)
    return stm_2d * abs_freq

# Experiment B: Only symmetric/asymmetric (no 1/f)
def preprocess_stm_features_decomp_only(stm_2d):
    # Decompose without 1/f normalization
    # ...decomposition logic...
    
# Experiment C: Both (full pipeline)
# → Main script
```

**3. Component Weight Analysis**:
```python
# Try different concatenation orders
# [Asym | DC | Sym] vs [Sym | DC | Asym]
# Weight channels differently
# Zero out one channel to measure contribution
```

### Expected Ablation Results

**If 1/f normalization is key**:
```
Baseline:         0.8566 (v1)
Only 1/f:         0.87-0.88
Only decomp:      0.86-0.87
Both:             0.88-0.89
```

**If decomposition is key**:
```
Baseline:         0.8566
Only 1/f:         0.86-0.87
Only decomp:      0.87-0.88
Both:             0.88-0.89
```

**If both are redundant**:
```
Baseline:         0.8566
Only 1/f:         0.86-0.87
Only decomp:      0.86-0.87
Both:             0.86-0.87 (no synergy)
```

## Implementation Details

### Frequency Vector Generation

**Critical for Correctness**:
```python
# STM parameters from STM06_STMpreproc.py
xmin, xmax = 190, 310  # Indices in original 500-bin array
xrange = [-15, 15]     # Hz range
x_ds_factor = 1        # No downsampling on time axis

# After cropping: 121 bins
n_time = (xmax - xmin + 1) // x_ds_factor  # = 121

# Frequency vector
rate_min = -15.0
rate_max = 15.0
frequency_vector = torch.linspace(rate_min, rate_max, n_time)

# Resolution check
rate_resolution = (rate_max - rate_min) / (n_time - 1)
# = 30 / 120 = 0.25 Hz ✓ Correct
```

**Index Mapping (UPDATED)**:
```python
Original (121 bins):
    Index    Frequency (Hz)
    0        -15.00
    1        -14.75
    ...
    59       -0.25
    60       0.00          ← DC component
    61       +0.25
    ...
    120      +15.00

After Processing (61 bins, 0-15 Hz):
    Index    Frequency (Hz)
    0        0.00           ← DC
    1        0.25
    2        0.50
    ...
    60       15.00
```

### Asymmetric Map Rescaling (REMOVED)

**Previous approach** (now removed):
```python
# OLD: Per-sample rescaling to [-1, 1]
max_abs_asym = torch.abs(asymmetric_map).reshape(batch, -1).max(dim=1)[0]
asymmetric_map_rescaled = asymmetric_map / max_abs_asym
```

**New approach** (current):
```python
# NEW: Keep raw asymmetric differences
asymmetric_map = torch.abs(positive_chunk) - torch.abs(negative_flipped)
# No rescaling - let z-normalization handle scaling
```

**Rationale for change**:
1. **Preserve natural signal strength**: Rescaling removes information about magnitude
2. **Z-normalization is sufficient**: Per-sample normalization handles scaling
3. **Avoid artificial equalization**: Different samples naturally have different directional strengths
4. **More informative gradients**: Raw differences provide clearer learning signal

**Benefits**:
- Simpler preprocessing pipeline
- Preserves relative magnitude information across samples
- Z-normalization (mean=0, std=1) provides adequate scaling
- More natural representation of directional preference

### Channel Stacking (NEW)

**Implementation**:
```python
# After computing both maps with DC prepended
symmetric_with_dc = torch.cat([dc_component, symmetric_map], dim=1)     # (batch, 61, 20)
asymmetric_with_dc = torch.cat([dc_component, asymmetric_map], dim=1)  # (batch, 61, 20)

# Stack into 2-channel tensor
processed = torch.stack([symmetric_with_dc, asymmetric_with_dc], dim=-1)  # (batch, 61, 20, 2)
```

**Dimension semantics**:
```
processed[b, f, s, 0] = Symmetric map at frequency f, scale s, sample b
processed[b, f, s, 1] = Asymmetric map at frequency f, scale s, sample b
```

**Why stack instead of concatenate**:
```python
# Option A: Concatenate along frequency (NOT USED)
concat_freq = torch.cat([symmetric_with_dc, asymmetric_with_dc], dim=1)  # (batch, 122, 20)
# Issues:
#   - Treats channels as additional frequencies (semantically wrong)
#   - Harder for Conv2d to learn cross-channel interactions
#   - Loses explicit channel structure

# Option B: Stack as separate channel dimension (USED) ✓
stack_channels = torch.stack([symmetric_with_dc, asymmetric_with_dc], dim=-1)  # (batch, 61, 20, 2)
# Benefits:
#   - Natural input for Conv2d: (batch, channels, height, width)
#   - Explicit channel separation
#   - Conv2d learns cross-channel patterns naturally
#   - Standard in computer vision (RGB images, etc.)
```

### Per-Sample Normalization (UPDATED)

**After preprocessing, normalize across all dimensions**:
```python
# STM_processed: (batch, 61, 20)
means = STM_processed.mean(dim=(1, 2), keepdim=True)  # (batch, 1, 1)
stds = STM_processed.std(dim=(1, 2), keepdim=True)    # (batch, 1, 1)
STM_processed = (STM_processed - means) / (stds + 1e-8)
```

**Effect**:
- Centers each sample to mean=0, std=1
- Applied AFTER symmetric averaging
- Ensures both channels have comparable scale after z-normalization
- Prevents any remaining magnitude imbalance

## Integration with ASM Architecture

### Data Flow (SYMMETRIC ONLY)

```
Raw STM (flattened) → Reshape to 2D (121, 20)
                   ↓
         1/f Normalization (by rate)
                   ↓
         Separate Negative/Positive Rates
                   ↓
         Flip & Align
                   ↓
         Compute Symmetric Map (averaged energy)
                   ↓
         Concatenate DC with symmetric map
                   ↓
         Single-channel tensor (61, 20)
                   ↓
         Per-sample Z-normalization
                   ↓
         (batch, 61, 20) → ASM Model
```

### Model Architecture (SYMMETRIC ONLY)

**Key changes for single-channel input with increased capacity**:
```python
ModifiedFeatureASMClassifier(
    time_steps=20,      # Scale dimension
    freq_steps=61,      # Frequency dimension (0-15 Hz)
    num_classes=6,
    dim=144,           # Increased from 128
    num_blocks=4,
    shift_range=2,
    expansion_factor=2,
    dropout=0.1
)
```

**Input Processing**:
```python
def forward(self, x, return_features=False):
    # x: (batch, 61, 20) - modified STM features
    
    # Add channel dimension: (batch, 1, 61, 20)
    x = x.unsqueeze(1)
    
    x = self.spec_augment(x)
    
    # Input projection: 1 input channel, increased first layer
    x = self.input_proj(x)  # Conv2d(1, dim//3, ...) → Conv2d(dim//3, dim, ...)
    
    # ...rest unchanged...
```

**Updated Input Projection Layer**:
```python
self.input_proj = nn.Sequential(
    nn.Conv2d(1, dim // 3, kernel_size=3, padding=1),  # 1 input channel, dim//3 = 48
    nn.BatchNorm2d(dim // 3),
    nn.GELU(),
    nn.Conv2d(dim // 3, dim, kernel_size=3, padding=1),
    nn.BatchNorm2d(dim),
    nn.GELU()
)
```

**Parameter count**: ~900K (slightly more than v3 due to dim=144)

## Training Configuration

### Same as ASM v3 (Proven Recipe)

**Loss Function**:
```python
ContrastiveFocalLoss(
    alpha=confusion_aware_weights,  # sqrt-based
    gamma=2.0,
    label_smoothing=0.01,
    contrastive_weight=0.1,
    similar_pairs=[(0, 1), (2, 3)]
)
```

**Optimizer & Scheduler**:
```python
AdamW(lr=1e-3, weight_decay=1e-4)
CosineAnnealingWarmRestarts(T_0=10, T_mult=2, eta_min=1e-6)
Warmup: 5 epochs
```

**Hyperparameters**:
```python
batch_size = 128
num_epochs = 50
gradient_clip_norm = 1.0
```

## Expected Performance Improvements

### Hypothesis 1: Better High-Frequency Discrimination

**Prediction**:
- Classes with high-rate modulations benefit from 1/f correction
- **Music classes** (2, 3): Often have rapid modulations → should improve
- **Speech classes** (0, 1): Slower modulations → less impact

**Target**:
```
Class 2 (music: vocal):     0.84 → 0.86 (+2 points)
Class 3 (music: non-vocal): 0.74 → 0.77 (+3 points)
```

### Hypothesis 2: Directional Discrimination (UPDATED)

**Prediction**:
- **Rescaled asymmetric channel** ensures equal contribution to learning
- Averaging symmetric map prevents magnitude imbalance
- 2-channel structure allows Conv2d to learn cross-channel patterns

**Expected benefits**:
```
Symmetric channel (averaged energy):
    - More stable estimates (averaging reduces noise)
    - Comparable magnitude to asymmetric channel
    - Better for classes with strong overall modulation

Asymmetric channel (rescaled direction):
    - Normalized importance across samples
    - Prevents outlier samples from dominating gradients
    - Better for classes with directional preferences

Combined:
    - Conv2d can learn complementary features from both channels
    - Cross-channel interactions captured by spatial convolutions
    - More balanced feature learning
```

**Target**:
```
Speech classes (0, 1):  Better separation via prosodic directionality
Music classes (2, 3):   Better separation via melodic energy+direction
Env classes (4, 5):     Maintain performance (less directional)
```

### Hypothesis 3: Overall Performance (UPDATED)

**Conservative Target**:
```
Test Macro F1: 0.87-0.88 (match or slightly exceed ASM v3)

Per-Class F1:
  Class 0: 0.95-0.96 (maintain)
  Class 1: 0.82-0.85 (+1-4 points over v3) ← Improved via rescaling
  Class 2: 0.85-0.87 (+1-3 points over v3) ← Improved via 2-channel
  Class 3: 0.76-0.79 (+2-5 points over v3) ← KEY IMPROVEMENT
  Class 4: 0.92-0.94 (maintain)
  Class 5: 0.91-0.93 (maintain)
```

**Optimistic Target**:
```
Test Macro F1: 0.88-0.90 (beat ASM v3 by 1-3 points)
Class 3 F1 ≥ 0.78 (break through 0.75 barrier)
All classes F1 ≥ 0.80 (uniform improvement)
```

**Why this version might perform better**:
1. **Balanced channels**: Rescaling prevents asymmetric channel from being ignored
2. **Reduced redundancy**: Single-sided spectrum (0-15 Hz) is more efficient
3. **Better Conv2d input**: 2 channels vs 1 channel with artificial concatenation
4. **Stable gradients**: Normalized ranges prevent gradient issues

## Comparison with ASM v3

### Key Differences (SYMMETRIC ONLY)

| Aspect | ASM v3 | Modified Features (Symmetric) |
|--------|--------|-------------------------------|
| **Preprocessing** | dB + normalize | dB + normalize + 1/f + symmetric |
| **Feature Dim** | (121, 20) | (61, 20) |
| **Channels** | 1 | 1 |
| **Frequency Range** | -15 to +15 Hz | 0 to +15 Hz |
| **Symmetric** | N/A | Averaged: (|P+|+|P-|)/2 |
| **Asymmetric** | N/A | Not used |
| **Feature Count** | 2,420 | 1,220 (50% reduction) |
| **Model Dim** | 128 | 144 (+12.5%) |
| **Params** | ~800K | ~900K |
| **Bio-Inspired** | No | Yes (1/f, symmetric) |
| **Expected F1** | 0.87+ | **0.87-0.89** |

**Why symmetric-only might work**:
1. **1/f normalization is key**: Spectral balancing may be main driver
2. **Simpler is better**: Reduced complexity, easier optimization
3. **Averaging reduces noise**: More stable energy estimates
4. **Directional info redundant**: Energy sufficient for classification
5. **Model compensation**: Increased capacity offsets information loss

### Complementary Approaches

**ASM v3 Strategy**: "Better learning algorithm"
- Softer class weighting
- Contrastive loss for separation
- Minimal label smoothing

**Modified Features Strategy**: "Better input representation"
- 1/f normalization for spectral balance
- Symmetric/asymmetric decomposition for directionality
- Biologically-inspired preprocessing

**Combined Power**:
- If both work independently → potentially 0.88-0.89 F1
- If synergistic → could reach 0.90 F1
- If redundant → only incremental gain

## Ablation Study Design

### Experiments to Run

**1. Baseline Comparison**:
```bash
# ASM v3 (no modified features)
python STMasm_enhanced3.py 0

# Modified Features (this script)
python STMasm_feateng.py 0
```

**2. Feature Component Ablation**:
```python
# Experiment A: Only 1/f normalization
def preprocess_stm_features_1f_only(stm_2d):
    # Apply 1/f, but don't decompose
    frequency_vector = torch.linspace(-15.0, 15.0, 121)
    abs_freq = torch.abs(frequency_vector).view(1, 121, 1)
    return stm_2d * abs_freq

# Experiment B: Only symmetric/asymmetric (no 1/f)
def preprocess_stm_features_decomp_only(stm_2d):
    # Decompose without 1/f normalization
    # ...decomposition logic...
    
# Experiment C: Both (full pipeline)
# → Main script
```

**3. Component Weight Analysis**:
```python
# Try different concatenation orders
# [Asym | DC | Sym] vs [Sym | DC | Asym]
# Weight channels differently
# Zero out one channel to measure contribution
```

### Expected Ablation Results

**If 1/f normalization is key**:
```
Baseline:         0.8566 (v1)
Only 1/f:         0.87-0.88
Only decomp:      0.86-0.87
Both:             0.88-0.89
```

**If decomposition is key**:
```
Baseline:         0.8566
Only 1/f:         0.86-0.87
Only decomp:      0.87-0.88
Both:             0.88-0.89
```

**If both are redundant**:
```
Baseline:         0.8566
Only 1/f:         0.86-0.87
Only decomp:      0.86-0.87
Both:             0.86-0.87 (no synergy)
```

## Implementation Details

### Frequency Vector Generation

**Critical for Correctness**:
```python
# STM parameters from STM06_STMpreproc.py
xmin, xmax = 190, 310  # Indices in original 500-bin array
xrange = [-15, 15]     # Hz range
x_ds_factor = 1        # No downsampling on time axis

# After cropping: 121 bins
n_time = (xmax - xmin + 1) // x_ds_factor  # = 121

# Frequency vector
rate_min = -15.0
rate_max = 15.0
frequency_vector = torch.linspace(rate_min, rate_max, n_time)

# Resolution check
rate_resolution = (rate_max - rate_min) / (n_time - 1)
# = 30 / 120 = 0.25 Hz ✓ Correct
```

**Index Mapping (UPDATED)**:
```python
Original (121 bins):
    Index    Frequency (Hz)
    0        -15.00
    1        -14.75
    ...
    59       -0.25
    60       0.00          ← DC component
    61       +0.25
    ...
    120      +15.00

After Processing (61 bins, 0-15 Hz):
    Index    Frequency (Hz)
    0        0.00           ← DC
    1        0.25
    2        0.50
    ...
    60       15.00
```

### Asymmetric Map Rescaling (REMOVED)

**Previous approach** (now removed):
```python
# OLD: Per-sample rescaling to [-1, 1]
max_abs_asym = torch.abs(asymmetric_map).reshape(batch, -1).max(dim=1)[0]
asymmetric_map_rescaled = asymmetric_map / max_abs_asym
```

**New approach** (current):
```python
# NEW: Keep raw asymmetric differences
asymmetric_map = torch.abs(positive_chunk) - torch.abs(negative_flipped)
# No rescaling - let z-normalization handle scaling
```

**Rationale for change**:
1. **Preserve natural signal strength**: Rescaling removes information about magnitude
2. **Z-normalization is sufficient**: Per-sample normalization handles scaling
3. **Avoid artificial equalization**: Different samples naturally have different directional strengths
4. **More informative gradients**: Raw differences provide clearer learning signal

**Benefits**:
- Simpler preprocessing pipeline
- Preserves relative magnitude information across samples
- Z-normalization (mean=0, std=1) provides adequate scaling
- More natural representation of directional preference

### Channel Stacking (NEW)

**Implementation**:
```python
# After computing both maps with DC prepended
symmetric_with_dc = torch.cat([dc_component, symmetric_map], dim=1)     # (batch, 61, 20)
asymmetric_with_dc = torch.cat([dc_component, asymmetric_map], dim=1)  # (batch, 61, 20)

# Stack into 2-channel tensor
processed = torch.stack([symmetric_with_dc, asymmetric_with_dc], dim=-1)  # (batch, 61, 20, 2)
```

**Dimension semantics**:
```
processed[b, f, s, 0] = Symmetric map at frequency f, scale s, sample b
processed[b, f, s, 1] = Asymmetric map at frequency f, scale s, sample b
```

**Why stack instead of concatenate**:
```python
# Option A: Concatenate along frequency (NOT USED)
concat_freq = torch.cat([symmetric_with_dc, asymmetric_with_dc], dim=1)  # (batch, 122, 20)
# Issues:
#   - Treats channels as additional frequencies (semantically wrong)
#   - Harder for Conv2d to learn cross-channel interactions
#   - Loses explicit channel structure

# Option B: Stack as separate channel dimension (USED) ✓
stack_channels = torch.stack([symmetric_with_dc, asymmetric_with_dc], dim=-1)  # (batch, 61, 20, 2)
# Benefits:
#   - Natural input for Conv2d: (batch, channels, height, width)
#   - Explicit channel separation
#   - Conv2d learns cross-channel patterns naturally
#   - Standard in computer vision (RGB images, etc.)
```

### Per-Sample Normalization (UPDATED)

**After preprocessing, normalize across all dimensions**:
```python
# STM_processed: (batch, 61, 20)
means = STM_processed.mean(dim=(1, 2), keepdim=True)  # (batch, 1, 1)
stds = STM_processed.std(dim=(1, 2), keepdim=True)    # (batch, 1, 1)
STM_processed = (STM_processed - means) / (stds + 1e-8)
```

**Effect**:
- Centers each sample to mean=0, std=1
- Applied AFTER symmetric averaging
- Ensures both channels have comparable scale after z-normalization
- Prevents any remaining magnitude imbalance

## Integration with ASM Architecture

### Data Flow (SYMMETRIC ONLY)

```
Raw STM (flattened) → Reshape to 2D (121, 20)
                   ↓
         1/f Normalization (by rate)
                   ↓
         Separate Negative/Positive Rates
                   ↓
         Flip & Align
                   ↓
         Compute Symmetric Map (averaged energy)
                   ↓
         Concatenate DC with symmetric map
                   ↓
         Single-channel tensor (61, 20)
                   ↓
         Per-sample Z-normalization
                   ↓
         (batch, 61, 20) → ASM Model
```

### Model Architecture (SYMMETRIC ONLY)

**Key changes for single-channel input with increased capacity**:
```python
ModifiedFeatureASMClassifier(
    time_steps=20,      # Scale dimension
    freq_steps=61,      # Frequency dimension (0-15 Hz)
    num_classes=6,
    dim=144,           # Increased from 128
    num_blocks=4,
    shift_range=2,
    expansion_factor=2,
    dropout=0.1
)
```

**Input Processing**:
```python
def forward(self, x, return_features=False):
    # x: (batch, 61, 20) - modified STM features
    
    # Add channel dimension: (batch, 1, 61, 20)
    x = x.unsqueeze(1)
    
    x = self.spec_augment(x)
    
    # Input projection: 1 input channel, increased first layer
    x = self.input_proj(x)  # Conv2d(1, dim//3, ...) → Conv2d(dim//3, dim, ...)
    
    # ...rest unchanged...
```

**Updated Input Projection Layer**:
```python
self.input_proj = nn.Sequential(
    nn.Conv2d(1, dim // 3, kernel_size=3, padding=1),  # 1 input channel, dim//3 = 48
    nn.BatchNorm2d(dim // 3),
    nn.GELU(),
    nn.Conv2d(dim // 3, dim, kernel_size=3, padding=1),
    nn.BatchNorm2d(dim),
    nn.GELU()
)
```

**Parameter count**: ~900K (slightly more than v3 due to dim=144)

## Training Configuration

### Same as ASM v3 (Proven Recipe)

**Loss Function**:
```python
ContrastiveFocalLoss(
    alpha=confusion_aware_weights,  # sqrt-based
    gamma=2.0,
    label_smoothing=0.01,
    contrastive_weight=0.1,
    similar_pairs=[(0, 1), (2, 3)]
)
```

**Optimizer & Scheduler**:
```python
AdamW(lr=1e-3, weight_decay=1e-4)
CosineAnnealingWarmRestarts(T_0=10, T_mult=2, eta_min=1e-6)
Warmup: 5 epochs
```

**Hyperparameters**:
```python
batch_size = 128
num_epochs = 50
gradient_clip_norm = 1.0
```

## Expected Performance Improvements

### Hypothesis 1: Better High-Frequency Discrimination

**Prediction**:
- Classes with high-rate modulations benefit from 1/f correction
- **Music classes** (2, 3): Often have rapid modulations → should improve
- **Speech classes** (0, 1): Slower modulations → less impact

**Target**:
```
Class 2 (music: vocal):     0.84 → 0.86 (+2 points)
Class 3 (music: non-vocal): 0.74 → 0.77 (+3 points)
```

### Hypothesis 2: Directional Discrimination (UPDATED)

**Prediction**:
- **Rescaled asymmetric channel** ensures equal contribution to learning
- Averaging symmetric map prevents magnitude imbalance
- 2-channel structure allows Conv2d to learn cross-channel patterns

**Expected benefits**:
```
Symmetric channel (averaged energy):
    - More stable estimates (averaging reduces noise)
    - Comparable magnitude to asymmetric channel
    - Better for classes with strong overall modulation

Asymmetric channel (rescaled direction):
    - Normalized importance across samples
    - Prevents outlier samples from dominating gradients
    - Better for classes with directional preferences

Combined:
    - Conv2d can learn complementary features from both channels
    - Cross-channel interactions captured by spatial convolutions
    - More balanced feature learning
```

**Target**:
```
Speech classes (0, 1):  Better separation via prosodic directionality
Music classes (2, 3):   Better separation via melodic energy+direction
Env classes (4, 5):     Maintain performance (less directional)
```

### Hypothesis 3: Overall Performance (UPDATED)

**Conservative Target**:
```
Test Macro F1: 0.87-0.88 (match or slightly exceed ASM v3)

Per-Class F1:
  Class 0: 0.95-0.96 (maintain)
  Class 1: 0.82-0.85 (+1-4 points over v3) ← Improved via rescaling
  Class 2: 0.85-0.87 (+1-3 points over v3) ← Improved via 2-channel
  Class 3: 0.76-0.79 (+2-5 points over v3) ← KEY IMPROVEMENT
  Class 4: 0.92-0.94 (maintain)
  Class 5: 0.91-0.93 (maintain)
```

**Optimistic Target**:
```
Test Macro F1: 0.88-0.90 (beat ASM v3 by 1-3 points)
Class 3 F1 ≥ 0.78 (break through 0.75 barrier)
All classes F1 ≥ 0.80 (uniform improvement)
```

**Why this version might perform better**:
1. **Balanced channels**: Rescaling prevents asymmetric channel from being ignored
2. **Reduced redundancy**: Single-sided spectrum (0-15 Hz) is more efficient
3. **Better Conv2d input**: 2 channels vs 1 channel with artificial concatenation
4. **Stable gradients**: Normalized ranges prevent gradient issues

## Comparison with ASM v3

### Key Differences (SYMMETRIC ONLY)

| Aspect | ASM v3 | Modified Features (Symmetric) |
|--------|--------|-------------------------------|
| **Preprocessing** | dB + normalize | dB + normalize + 1/f + symmetric |
| **Feature Dim** | (121, 20) | (61, 20) |
| **Channels** | 1 | 1 |
| **Frequency Range** | -15 to +15 Hz | 0 to +15 Hz |
| **Symmetric** | N/A | Averaged: (|P+|+|P-|)/2 |
| **Asymmetric** | N/A | Not used |
| **Feature Count** | 2,420 | 1,220 (50% reduction) |
| **Model Dim** | 128 | 144 (+12.5%) |
| **Params** | ~800K | ~900K |
| **Bio-Inspired** | No | Yes (1/f, symmetric) |
| **Expected F1** | 0.87+ | **0.87-0.89** |

**Why symmetric-only might work**:
1. **1/f normalization is key**: Spectral balancing may be main driver
2. **Simpler is better**: Reduced complexity, easier optimization
3. **Averaging reduces noise**: More stable energy estimates
4. **Directional info redundant**: Energy sufficient for classification
5. **Model compensation**: Increased capacity offsets information loss

### Complementary Approaches

**ASM v3 Strategy**: "Better learning algorithm"
- Softer class weighting
- Contrastive loss for separation
- Minimal label smoothing

**Modified Features Strategy**: "Better input representation"
- 1/f normalization for spectral balance
- Symmetric/asymmetric decomposition for directionality
- Biologically-inspired preprocessing

**Combined Power**:
- If both work independently → potentially 0.88-0.89 F1
- If synergistic → could reach 0.90 F1
- If redundant → only incremental gain

## Ablation Study Design

### Experiments to Run

**1. Baseline Comparison**:
```bash
# ASM v3 (no modified features)
python STMasm_enhanced3.py 0

# Modified Features (this script)
python STMasm_feateng.py 0
```

**2. Feature Component Ablation**:
```python
# Experiment A: Only 1/f normalization
def preprocess_stm_features_1f_only(stm_2d):
    # Apply 1/f, but don't decompose
    frequency_vector = torch.linspace(-15.0, 15.0, 121)
    abs_freq = torch.abs(frequency_vector).view(1, 121, 1)
    return stm_2d * abs_freq

# Experiment B: Only symmetric/asymmetric (no 1/f)
def preprocess_stm_features_decomp_only(stm_2d):
    # Decompose without 1/f normalization
    # ...decomposition logic...
    
# Experiment C: Both (full pipeline)
# → Main script
```

**3. Component Weight Analysis**:
```python
# Try different concatenation orders
# [Asym | DC | Sym] vs [Sym | DC | Asym]
# Weight channels differently
# Zero out one channel to measure contribution
```

### Expected Ablation Results

**If 1/f normalization is key**:
```
Baseline:         0.8566 (v1)
Only 1/f:         0.87-0.88
Only decomp:      0.86-0.87
Both:             0.88-0.89
```

**If decomposition is key**:
```
Baseline:         0.8566
Only 1/f:         0.86-0.87
Only decomp:      0.87-0.88
Both:             0.88-0.89
```

**If both are redundant**:
```
Baseline:         0.8566
Only 1/f:         0.86-0.87
Only decomp:      0.86-0.87
Both:             0.86-0.87 (no synergy)
```

## Implementation Details

### Frequency Vector Generation

**Critical for Correctness**:
```python
# STM parameters from STM06_STMpreproc.py
xmin, xmax = 190, 310  # Indices in original 500-bin array
xrange = [-15, 15]     # Hz range
x_ds_factor = 1        # No downsampling on time axis

# After cropping: 121 bins
n_time = (xmax - xmin + 1) // x_ds_factor  # = 121

# Frequency vector
rate_min = -15.0
rate_max = 15.0
frequency_vector = torch.linspace(rate_min, rate_max, n_time)

# Resolution check
rate_resolution = (rate_max - rate_min) / (n_time - 1)
# = 30 / 120 = 0.25 Hz ✓ Correct
```

**Index Mapping (UPDATED)**:
```python
Original (121 bins):
    Index    Frequency (Hz)
    0        -15.00
    1        -14.75
    ...
    59       -0.25
    60       0.00          ← DC component
    61       +0.25
    ...
    120      +15.00

After Processing (61 bins, 0-15 Hz):
    Index    Frequency (Hz)
    0        0.00           ← DC
    1        0.25
    2        0.50
    ...
    60       15.00
```

### Asymmetric Map Rescaling (REMOVED)

**Previous approach** (now removed):
```python
# OLD: Per-sample rescaling to [-1, 1]
max_abs_asym = torch.abs(asymmetric_map).reshape(batch, -1).max(dim=1)[0]
asymmetric_map_rescaled = asymmetric_map / max_abs_asym
```

**New approach** (current):
```python
# NEW: Keep raw asymmetric differences
asymmetric_map = torch.abs(positive_chunk) - torch.abs(negative_flipped)
# No rescaling - let z-normalization handle scaling
```

**Rationale for change**:
1. **Preserve natural signal strength**: Rescaling removes information about magnitude
2. **Z-normalization is sufficient**: Per-sample normalization handles scaling
3. **Avoid artificial equalization**: Different samples naturally have different directional strengths
4. **More informative gradients**: Raw differences provide clearer learning signal

**Benefits**:
- Simpler preprocessing pipeline
- Preserves relative magnitude information across samples
- Z-normalization (mean=0, std=1) provides adequate scaling
- More natural representation of directional preference

### Channel Stacking (NEW)

**Implementation**:
```python
# After computing both maps with DC prepended
symmetric_with_dc = torch.cat([dc_component, symmetric_map], dim=1)     # (batch, 61, 20)
asymmetric_with_dc = torch.cat([dc_component, asymmetric_map], dim=1)  # (batch, 61, 20)

# Stack into 2-channel tensor
processed = torch.stack([symmetric_with_dc, asymmetric_with_dc], dim=-1)  # (batch, 61, 20, 2)
```

**Dimension semantics**:
```
processed[b, f, s, 0] = Symmetric map at frequency f, scale s, sample b
processed[b, f, s, 1] = Asymmetric map at frequency f, scale s, sample b
```

**Why stack instead of concatenate**:
```python
# Option A: Concatenate along frequency (NOT USED)
concat_freq = torch.cat([symmetric_with_dc, asymmetric_with_dc], dim=1)  # (batch, 122, 20)
# Issues:
#   - Treats channels as additional frequencies (semantically wrong)
#   - Harder for Conv2d to learn cross-channel interactions
#   - Loses explicit channel structure

# Option B: Stack as separate channel dimension (USED) ✓
stack_channels = torch.stack([symmetric_with_dc, asymmetric_with_dc], dim=-1)  # (batch, 61, 20, 2)
# Benefits:
#   - Natural input for Conv2d: (batch, channels, height, width)
#   - Explicit channel separation
#   - Conv2d learns cross-channel patterns naturally
#   - Standard in computer vision (RGB images, etc.)
```

### Per-Sample Normalization (UPDATED)

**After preprocessing, normalize across all dimensions**:
```python
# STM_processed: (batch, 61, 20)
means = STM_processed.mean(dim=(1, 2), keepdim=True)  # (batch, 1, 1)
stds = STM_processed.std(dim=(1, 2), keepdim=True)    # (batch, 1, 1)
STM_processed = (STM_processed - means) / (stds + 1e-8)
```

**Effect**:
- Centers each sample to mean=0, std=1
- Applied AFTER symmetric averaging
- Ensures both channels have comparable scale after z-normalization
- Prevents any remaining magnitude imbalance

## Integration with ASM Architecture

### Data Flow (SYMMETRIC ONLY)

```
Raw STM (flattened) → Reshape to 2D (121, 20)
                   ↓
         1/f Normalization (by rate)
                   ↓
         Separate Negative/Positive Rates
                   ↓
         Flip & Align
                   ↓
         Compute Symmetric Map (averaged energy)
                   ↓
         Concatenate DC with symmetric map
                   ↓
         Single-channel tensor (61, 20)
                   ↓
         Per-sample Z-normalization
                   ↓
         (batch, 61, 20) → ASM Model
```

### Model Architecture (SYMMETRIC ONLY)

**Key changes for single-channel input with increased capacity**:
```python
ModifiedFeatureASMClassifier(
    time_steps=20,      # Scale dimension
    freq_steps=61,      # Frequency dimension (0-15 Hz)
    num_classes=6,
    dim=144,           # Increased from 128
    num_blocks=4,
    shift_range=2,
    expansion_factor=2,
    dropout=0.1
)
```

**Input Processing**:
```python
def forward(self, x, return_features=False):
    # x: (batch, 61, 20) - modified STM features
    
    # Add channel dimension: (batch, 1, 61, 20)
    x = x.unsqueeze(1)
    
    x = self.spec_augment(x)
    
    # Input projection: 1 input channel, increased first layer
    x = self.input_proj(x)  # Conv2d(1, dim//3, ...) → Conv2d(dim//3, dim, ...)
    
    # ...rest unchanged...
```

**Updated Input Projection Layer**:
```python
self.input_proj = nn.Sequential(
    nn.Conv2d(1, dim // 3, kernel_size=3, padding=1),  # 1 input channel, dim//3 = 48
    nn.BatchNorm2d(dim // 3),
    nn.GELU(),
    nn.Conv2d(dim // 3, dim, kernel_size=3, padding=1),
    nn.BatchNorm2d(dim),
    nn.GELU()
)
```

**Parameter count**: ~900K (slightly more than v3 due to dim=144)

## Training Configuration

### Same as ASM v3 (Proven Recipe)

**Loss Function**:
```python
ContrastiveFocalLoss(
    alpha=confusion_aware_weights,  # sqrt-based
    gamma=2.0,
    label_smoothing=0.01,
    contrastive_weight=0.1,
    similar_pairs=[(0, 1), (2, 3)]
)
```

**Optimizer & Scheduler**:
```python
AdamW(lr=1e-3, weight_decay=1e-4)
CosineAnnealingWarmRestarts(T_0=10, T_mult=2, eta_min=1e-6)
Warmup: 5 epochs
```

**Hyperparameters**:
```python
batch_size = 128
num_epochs = 50
gradient_clip_norm = 1.0
```

## Expected Performance Improvements

### Hypothesis 1: Better High-Frequency Discrimination

**Prediction**:
- Classes with high-rate modulations benefit from 1/f correction
- **Music classes** (2, 3): Often have rapid modulations → should improve
- **Speech classes** (0, 1): Slower modulations → less impact

**Target**:
```
Class 2 (music: vocal):     0.84 → 0.86 (+2 points)
Class 3 (music: non-vocal): 0.74 → 0.77 (+3 points)
```

### Hypothesis 2: Directional Discrimination (UPDATED)

**Prediction**:
- **Rescaled asymmetric channel** ensures equal contribution to learning
- Averaging symmetric map prevents magnitude imbalance
- 2-channel structure allows Conv2d to learn cross-channel patterns

**Expected benefits**:
```
Symmetric channel (averaged energy):
    - More stable estimates (averaging reduces noise)
    - Comparable magnitude to asymmetric channel
    - Better for classes with strong overall modulation

Asymmetric channel (rescaled direction):
    - Normalized importance across samples
    - Prevents outlier samples from dominating gradients
    - Better for classes with directional preferences

Combined:
    - Conv2d can learn complementary features from both channels
    - Cross-channel interactions captured by spatial convolutions
    - More balanced feature learning
```

**Target**:
```
Speech classes (0, 1):  Better separation via prosodic directionality
Music classes (2, 3):   Better separation via melodic energy+direction
Env classes (4, 5):     Maintain performance (less directional)
```

### Hypothesis 3: Overall Performance (UPDATED)

**Conservative Target**:
```
Test Macro F1: 0.87-0.88 (match or slightly exceed ASM v3)

Per-Class F1:
  Class 0: 0.95-0.96 (maintain)
  Class 1: 0.82-0.85 (+1-4 points over v3) ← Improved via rescaling
  Class 2: 0.85-0.87 (+1-3 points over v3) ← Improved via 2-channel
  Class 3: 0.76-0.79 (+2-5 points over v3) ← KEY IMPROVEMENT
  Class 4: 0.92-0.94 (maintain)
  Class 5: 0.91-0.93 (maintain)
```

**Optimistic Target**:
```
Test Macro F1: 0.88-0.90 (beat ASM v3 by 1-3 points)
Class 3 F1 ≥ 0.78 (break through 0.75 barrier)
All classes F1 ≥ 0.80 (uniform improvement)
```

**Why this version might perform better**:
1. **Balanced channels**: Rescaling prevents asymmetric channel from being ignored
2. **Reduced redundancy**: Single-sided spectrum (0-15 Hz) is more efficient
3. **Better Conv2d input**: 2 channels vs 1 channel with artificial concatenation
4. **Stable gradients**: Normalized ranges prevent gradient issues

## Comparison with ASM v3

### Key Differences (SYMMETRIC ONLY)

| Aspect | ASM v3 | Modified Features (Symmetric) |
|--------|--------|-------------------------------|
| **Preprocessing** | dB + normalize | dB + normalize + 1/f + symmetric |
| **Feature Dim** | (121, 20) | (61, 20) |
| **Channels** | 1 | 1 |
| **Frequency Range** | -15 to +15 Hz | 0 to +15 Hz |
| **Symmetric** | N/A | Averaged: (|P+|+|P-|)/2 |
| **Asymmetric** | N/A | Not used |
| **Feature Count** | 2,420 | 1,220 (50% reduction) |
| **Model Dim** | 128 | 144 (+12.5%) |
| **Params** | ~800K | ~900K |
| **Bio-Inspired** | No | Yes (1/f, symmetric) |
| **Expected F1** | 0.87+ | **0.87-0.89** |

**Why symmetric-only might work**:
1. **1/f normalization is key**: Spectral balancing may be main driver
2. **Simpler is better**: Reduced complexity, easier optimization
3. **Averaging reduces noise**: More stable energy estimates
4. **Directional info redundant**: Energy sufficient for classification
5. **Model compensation**: Increased capacity offsets information loss

### Complementary Approaches

**ASM v3 Strategy**: "Better learning algorithm"
- Softer class weighting
- Contrastive loss for separation
- Minimal label smoothing

**Modified Features Strategy**: "Better input representation"
- 1/f normalization for spectral balance
- Symmetric/asymmetric decomposition for directionality
- Biologically-inspired preprocessing

**Combined Power**:
- If both work independently → potentially 0.88-0.89 F1
- If synergistic → could reach 0.90 F1
- If redundant → only incremental gain

## Ablation Study Design

### Experiments to Run

**1. Baseline Comparison**:
```bash
# ASM v3 (no modified features)
python STMasm_enhanced3.py 0

# Modified Features (this script)
python STMasm_feateng.py 0
```

**2. Feature Component Ablation**:
```python
# Experiment A: Only 1/f normalization
def preprocess_stm_features_1f_only(stm_2d):
    # Apply 1/f, but don't decompose
    frequency_vector = torch.linspace(-15.0, 15.0, 121)
    abs_freq = torch.abs(frequency_vector).view(1, 121, 1)
    return stm_2d * abs_freq

# Experiment B: Only symmetric/asymmetric (no 1/f)
def preprocess_stm_features_decomp_only(stm_2d):
    # Decompose without 1/f normalization
    # ...decomposition logic...
    
# Experiment C: Both (full pipeline)
# → Main script
```

**3. Component Weight Analysis**:
```python
# Try different concatenation orders
# [Asym | DC | Sym] vs [Sym | DC | Asym]
# Weight channels differently
# Zero out one channel to measure contribution
```

### Expected Ablation Results

**If 1/f normalization is key**:
```
Baseline:         0.8566 (v1)
Only 1/f:         0.87-0.88
Only decomp:      0.86-0.87
Both:             0.88-0.89
```

**If decomposition is key**:
```
Baseline:         0.8566
Only 1/f:         0.86-0.87
Only decomp:      0.87-0.88
Both:             0.88-0.89
```

**If both are redundant**:
```
Baseline:         0.8566
Only 1/f:         0.86-0.87
Only decomp:      0.86-0.87
Both:             0.86-0.87 (no synergy)
```

## Implementation Details

### Frequency Vector Generation

**Critical for Correctness**:
```python
# STM parameters from STM06_STMpreproc.py
xmin, xmax = 190, 310  # Indices in original 500-bin array
xrange = [-15, 15]     # Hz range
x_ds_factor = 1        # No downsampling on time axis

# After cropping: 121 bins
n_time = (xmax - xmin + 1) // x_ds_factor  # = 121

# Frequency vector
rate_min = -15.0
rate_max = 15.0
frequency_vector = torch.linspace(rate_min, rate_max, n_time)

# Resolution check
rate_resolution = (rate_max - rate_min) / (n_time - 1)
# = 30 / 120 = 0.25 Hz ✓ Correct
```

**Index Mapping (UPDATED)**:
```python
Original (121 bins):
    Index    Frequency (Hz)
    0        -15.00
    1        -14.75
    ...
    59       -0.25
    60       0.00          ← DC component
    61       +0.25
    ...
    120      +15.00

After Processing (61 bins, 0-15 Hz):
    Index    Frequency (Hz)
    0        0.00           ← DC
    1        0.25
    2        0.50
    ...
    60       15.00
```

### Asymmetric Map Rescaling (REMOVED)

**Previous approach** (now removed):
```python
# OLD: Per-sample rescaling to [-1, 1]
max_abs_asym = torch.abs(asymmetric_map).reshape(batch, -1).max(dim=1)[0]
asymmetric_map_rescaled = asymmetric_map / max_abs_asym
```

**New approach** (current):
```python
# NEW: Keep raw asymmetric differences
asymmetric_map = torch.abs(positive_chunk) - torch.abs(negative_flipped)
# No rescaling - let z-normalization handle scaling
```

**Rationale for change**:
1. **Preserve natural signal strength**: Rescaling removes information about magnitude
2. **Z-normalization is sufficient**: Per-sample normalization handles scaling
3. **Avoid artificial equalization**: Different samples naturally have different directional strengths
4. **More informative gradients**: Raw differences provide clearer learning signal

**Benefits**:
- Simpler preprocessing pipeline
- Preserves relative magnitude information across samples
- Z-normalization (mean=0, std=1) provides adequate scaling
- More natural representation of directional preference

### Channel Stacking (NEW)

**Implementation**:
```python
# After computing both maps with DC prepended
symmetric_with_dc = torch.cat([dc_component, symmetric_map], dim=1)     # (batch, 61, 20)
asymmetric_with_dc = torch.cat([dc_component, asymmetric_map], dim=1)  # (batch, 61, 20)

# Stack into 2-channel tensor
processed = torch.stack([symmetric_with_dc, asymmetric_with_dc], dim=-1)  # (batch, 61, 20, 2)
```

**Dimension semantics**:
```
processed[b, f, s, 0] = Symmetric map at frequency f, scale s, sample b
processed[b, f, s, 1] = Asymmetric map at frequency f, scale s, sample b
```

**Why stack instead of concatenate**:
```python
# Option A: Concatenate along frequency (NOT USED)
concat_freq = torch.cat([symmetric_with_dc, asymmetric_with_dc], dim=1)  # (batch, 122, 20)
# Issues:
#   - Treats channels as additional frequencies (semantically wrong)
#   - Harder for Conv2d to learn cross-channel interactions
#   - Loses explicit channel structure

# Option B: Stack as separate channel dimension (USED) ✓
stack_channels = torch.stack([symmetric_with_dc, asymmetric_with_dc], dim=-1)  # (batch, 61, 20, 2)
# Benefits:
#   - Natural input for Conv2d: (batch, channels, height, width)
#   - Explicit channel separation
#   - Conv2d learns cross-channel patterns naturally
#   - Standard in computer vision (RGB images, etc.)
```

### Per-Sample Normalization (UPDATED)

**After preprocessing, normalize across all dimensions**:
```python
# STM_processed: (batch, 61, 20)
means = STM_processed.mean(dim=(1, 2), keepdim=True)  # (batch, 1, 1)
stds = STM_processed.std(dim=(1, 2), keepdim=True)    # (batch, 1, 1)
STM_processed = (STM_processed - means) / (stds + 1e-8)
```

**Effect**:
- Centers each sample to mean=0, std=1
- Applied AFTER symmetric averaging
- Ensures both channels have comparable scale after z-normalization
- Prevents any remaining magnitude imbalance

## Integration with ASM Architecture

### Data Flow (SYMMETRIC ONLY)

```
Raw STM (flattened) → Reshape to 2D (121, 20)
                   ↓
         1/f Normalization (by rate)
                   ↓
         Separate Negative/Positive Rates
                   ↓
         Flip & Align
                   ↓
         Compute Symmetric Map (averaged energy)
                   ↓
         Concatenate DC with symmetric map
                   ↓
         Single-channel tensor (61, 20)
                   ↓
         Per-sample Z-normalization
                   ↓
         (batch, 61, 20) → ASM Model
```

### Model Architecture (SYMMETRIC ONLY)

**Key changes for single-channel input with increased capacity**:
```python
ModifiedFeatureASMClassifier(
    time_steps=20,      # Scale dimension
    freq_steps=61,      # Frequency dimension (0-15 Hz)
    num_classes=6,
    dim=144,           # Increased from 128
    num_blocks=4,
    shift_range=2,
    expansion_factor=2,
    dropout=0.1
)
```

**Input Processing**:
```python
def forward(self, x, return_features=False):
    # x: (batch, 61, 20) - modified STM features
    
    # Add channel dimension: (batch, 1, 61, 20)
    x = x.unsqueeze(1)
    
    x = self.spec_augment(x)
    
    # Input projection: 1 input channel, increased first layer
    x = self.input_proj(x)  # Conv2d(1, dim//3, ...) → Conv2d(dim//3, dim, ...)
    
    # ...rest unchanged...
```

**Updated Input Projection Layer**:
```python
self.input_proj = nn.Sequential(
    nn.Conv2d(1, dim // 3, kernel_size=3, padding=1),  # 1 input channel, dim//3 = 48
    nn.BatchNorm2d(dim // 3),
    nn.GELU(),
    nn.Conv2d(dim // 3, dim, kernel_size=3, padding=1),
    nn.BatchNorm2d(dim),
    nn.GELU()
)
```

**Parameter count**: ~900K (slightly more than v3 due to dim=144)

## Training Configuration

### Same as ASM v3 (Proven Recipe)

**Loss Function**:
```python
ContrastiveFocalLoss(
    alpha=confusion_aware_weights,  # sqrt-based
    gamma=2.0,
    label_smoothing=0.01,
    contrastive_weight=0.1,
    similar_pairs=[(0, 1), (2, 3)]
)
```

**Optimizer & Scheduler**:
```python
AdamW(lr=1e-3, weight_decay=1e-4)
CosineAnnealingWarmRestarts(T_0=10, T_mult=2, eta_min=1e-6)
Warmup: 5 epochs
```

**Hyperparameters**:
```python
batch_size = 128
num_epochs = 50
gradient_clip_norm = 1.0
```

## Expected Performance Improvements

### Hypothesis 1: Better High-Frequency Discrimination

**Prediction**:
- Classes with high-rate modulations benefit from 1/f correction
- **Music classes** (2, 3): Often have rapid modulations → should improve
- **Speech classes** (0, 1): Slower modulations → less impact

**Target**:
```
Class 2 (music: vocal):     0.84 → 0.86 (+2 points)
Class 3 (music: non-vocal): 0.74 → 0.77 (+3 points)
```

### Hypothesis 2: Directional Discrimination (UPDATED)

**Prediction**:
- **Rescaled asymmetric channel** ensures equal contribution to learning
- Averaging symmetric map prevents magnitude imbalance
- 2-channel structure allows Conv2d to learn cross-channel patterns

**Expected benefits**:
```
Symmetric channel (averaged energy):
    - More stable estimates (averaging reduces noise)
    - Comparable magnitude to asymmetric channel
    - Better for classes with strong overall modulation

Asymmetric channel (rescaled direction):
    - Normalized importance across samples
    - Prevents outlier samples from dominating gradients
    - Better for classes with directional preferences

Combined:
    - Conv2d can learn complementary features from both channels
    - Cross-channel interactions captured by spatial convolutions
    - More balanced feature learning
```

**Target**:
```
Speech classes (0, 1):  Better separation via prosodic directionality
Music classes (2, 3):   Better separation via melodic energy+direction
Env classes (4, 5):     Maintain performance (less directional)
```

### Hypothesis 3: Overall Performance (UPDATED)

**Conservative Target**:
```
Test Macro F1: 0.87-0.88 (match or slightly exceed ASM v3)

Per-Class F1:
  Class 0: 0.95-0.96 (maintain)
  Class 1: 0.82-0.85 (+1-4 points over v3) ← Improved via rescaling
  Class 2: 0.85-0.87 (+1-3 points over v3) ← Improved via 2-channel
  Class 3: 0.76-0.79 (+2-5 points over v3) ← KEY IMPROVEMENT
  Class 4: 0.92-0.94 (maintain)
  Class 5: 0.91-0.93 (maintain)
```

**Optimistic Target**:
```
Test Macro F1: 0.88-0.90 (beat ASM v3 by 1-3 points)
Class 3 F1 ≥ 0.78 (break through 0.75 barrier)
All classes F1 ≥ 0.80 (uniform improvement)
```

**Why this version might perform better**:
1. **Balanced channels**: Rescaling prevents asymmetric channel from being ignored
2. **Reduced redundancy**: Single-sided spectrum (0-15 Hz) is more efficient
3. **Better Conv2d input**: 2 channels vs 1 channel with artificial concatenation
4. **Stable gradients**: Normalized ranges prevent gradient issues

## Comparison with ASM v3

### Key Differences (SYMMETRIC ONLY)

| Aspect | ASM v3 | Modified Features (Symmetric) |
|--------|--------|-------------------------------|
| **Preprocessing** | dB + normalize | dB + normalize + 1/f + symmetric |
| **Feature Dim** | (121, 20) | (61, 20) |
| **Channels** | 1 | 1 |
| **Frequency Range** | -15 to +15 Hz | 0 to +15 Hz |
| **Symmetric** | N/A | Averaged: (|P+|+|P-|)/2 |
| **Asymmetric** | N/A | Not used |
| **Feature Count** | 2,420 | 1,220 (50% reduction) |
| **Model Dim** | 128 | 144 (+12.5%) |
| **Params** | ~800K | ~900K |
| **Bio-Inspired** | No | Yes (1/f, symmetric) |
| **Expected F1** | 0.87+ | **0.87-0.89** |

**Why symmetric-only might work**:
1. **1/f normalization is key**: Spectral balancing may be main driver
2. **Simpler is better**: Reduced complexity, easier optimization
3. **Averaging reduces noise**: More stable energy estimates
4. **Directional info redundant**: Energy sufficient for classification
5. **Model compensation**: Increased capacity offsets information loss

### Complementary Approaches

**ASM v3 Strategy**: "Better learning algorithm"
- Softer class weighting
- Contrastive loss for separation
- Minimal label smoothing

**Modified Features Strategy**: "Better input representation"
- 1/f normalization for spectral balance
- Symmetric/asymmetric decomposition for directionality
- Biologically-inspired preprocessing

**Combined Power**:
- If both work independently → potentially 0.88-0.89 F1
- If synergistic → could reach 0.90 F1
- If redundant → only incremental gain

## Ablation Study Design

### Experiments to Run

**1. Baseline Comparison**:
```bash
# ASM v3 (no modified features)
python STMasm_enhanced3.py 0

# Modified Features (this script)
python STMasm_feateng.py 0
```

**2. Feature Component Ablation**:
```python
# Experiment A: Only 1/f normalization
def preprocess_stm_features_1f_only(stm_2d):
    # Apply 1/f, but don't decompose
    frequency_vector = torch.linspace(-15.0, 15.0, 121)
    abs_freq = torch.abs(frequency_vector).view(1, 121, 1)
    return stm_2d * abs_freq

# Experiment B: Only symmetric/asymmetric (no 1/f)
def preprocess_stm_features_decomp_only(stm_2d):
    # Decompose without 1/f normalization
    # ...decomposition logic...
    
# Experiment C: Both (full pipeline)
# → Main script
```

**3. Component Weight Analysis**:
```python
# Try different concatenation orders
# [Asym | DC | Sym] vs [Sym | DC | Asym]
# Weight channels differently
# Zero out one channel to measure contribution
```

### Expected Ablation Results

**If 1/f normalization is key**:
```
Baseline:         0.8566 (v1)
Only 1/f:         0.87-0.88
Only decomp:      0.86-0.87
Both:             0.88-0.89
```

**If decomposition is key**:
```
Baseline:         0.8566
Only 1/f:         0.86-0.87
Only decomp:      0.87-0.88
Both:             0.88-0.89
```

**If both are redundant**:
```
Baseline:         0.8566
Only 1/f:         0.86-0.87
Only decomp:      0.86-0.87
Both:             0.86-0.87 (no synergy)
```

## Implementation Details

### Frequency Vector Generation

**Critical for Correctness**:
```python
# STM parameters from STM06_STMpreproc.py
xmin, xmax = 190, 310  # Indices in original 500-bin array
xrange = [-15, 15]     # Hz range
x_ds_factor = 1        # No downsampling on time axis

# After cropping: 121 bins
n_time = (xmax - xmin + 1) // x_ds_factor  # = 121

# Frequency vector
rate_min = -15.0
rate_max = 15.0
frequency_vector = torch.linspace(rate_min, rate_max, n_time)

# Resolution check
rate_resolution = (rate_max - rate_min) / (n_time - 1)
# = 30 / 120 = 0.25 Hz ✓ Correct
```

**Index Mapping (UPDATED)**:
```python
Original (121 bins):
    Index    Frequency (Hz)
    0        -15.00
    1        -14.75
    ...
    59       -0.25
    60       0.00          ← DC component
    61       +0.25
    ...
    120      +15.00

After Processing (61 bins, 0-15 Hz):
    Index    Frequency (Hz)
    0        0.00           ← DC
    1        0.25
    2        0.50
    ...
    60       15.00
```

### Asymmetric Map Rescaling (REMOVED)

**Previous approach** (now removed):
```python
# OLD: Per-sample rescaling to [-1, 1]
max_abs_asym = torch.abs(asymmetric_map).reshape(batch, -1).max(dim=1)[0]
asymmetric_map_rescaled = asymmetric_map / max_abs_asym
```

**New approach** (current):
```python
# NEW: Keep raw asymmetric differences
asymmetric_map = torch.abs(positive_chunk) - torch.abs(negative_flipped)
# No rescaling - let z-normalization handle scaling
```

**Rationale for change**:
1. **Preserve natural signal strength**: Rescaling removes information about magnitude
2. **Z-normalization is sufficient**: Per-sample normalization handles scaling
3. **Avoid artificial equalization**: Different samples naturally have different directional strengths
4. **More informative gradients**: Raw differences provide clearer learning signal

**Benefits**:
- Simpler preprocessing pipeline
- Preserves relative magnitude information across samples
- Z-normalization (mean=0, std=1) provides adequate scaling
- More natural representation of directional preference

### Channel Stacking (NEW)

**Implementation**:
```python
# After computing both maps with DC prepended
symmetric_with_dc = torch.cat([dc_component, symmetric_map], dim=1)     # (batch, 61, 20)
asymmetric_with_dc = torch.cat([dc_component, asymmetric_map], dim=1)  # (batch, 61, 20)

# Stack into 2-channel tensor
processed = torch.stack([symmetric_with_dc, asymmetric_with_dc], dim=-1)  # (batch, 61, 20, 2)
```

**Dimension semantics**:
```
processed[b, f, s, 0] = Symmetric map at frequency f, scale s, sample b
processed[b, f, s, 1] = Asymmetric map at frequency f, scale s, sample b
```

**Why stack instead of concatenate**:
```python
# Option A: Concatenate along frequency (NOT USED)
concat_freq = torch.cat([symmetric_with_dc, asymmetric_with_dc], dim=1)  # (batch, 122, 20)
# Issues:
#   - Treats channels as additional frequencies (semantically wrong)
#   - Harder for Conv2d to learn cross-channel interactions
#   - Loses explicit channel structure

# Option B: Stack as separate channel dimension (USED) ✓
stack_channels = torch.stack([symmetric_with_dc, asymmetric_with_dc], dim=-1)  # (batch, 61, 20, 2)
# Benefits:
#   - Natural input for Conv2d: (batch, channels, height, width)
#   - Explicit channel separation
#   - Conv2d learns cross-channel patterns naturally
#   - Standard in computer vision (RGB images, etc.)
```

### Per-Sample Normalization (UPDATED)

**After preprocessing, normalize across all dimensions**:
```python
# STM_processed: (batch, 61, 20)
means = STM_processed.mean(dim=(1, 2), keepdim=True)  # (batch, 1, 1)
stds = STM_processed.std(dim=(1, 2), keepdim=True)    # (batch, 1, 1)
STM_processed = (STM_processed - means) / (stds + 1e-8)
```

**Effect**:
- Centers each sample to mean=0, std=1
- Applied AFTER symmetric averaging
- Ensures both channels have comparable scale after z-normalization
- Prevents any remaining magnitude imbalance

## Integration with ASM Architecture

### Data Flow (SYMMETRIC ONLY)

```
Raw STM (flattened) → Reshape to 2D (121, 20)
                   ↓
         1/f Normalization (by rate)
                   ↓
         Separate Negative/Positive Rates
                   ↓
         Flip & Align
                   ↓
         Compute Symmetric Map (averaged energy)
                   ↓
         Concatenate DC with symmetric map
                   ↓
         Single-channel tensor (61, 20)
                   ↓
         Per-sample Z-normalization
                   ↓
         (batch, 61, 20) → ASM Model
```

### Model Architecture (SYMMETRIC ONLY)

**Key changes for single-channel input with increased capacity**:
```python
ModifiedFeatureASMClassifier(
    time_steps=20,      # Scale dimension
    freq_steps=61,      # Frequency dimension (0-15 Hz)
    num_classes=6,
    dim=144,           # Increased from 128
    num_blocks=4,
    shift_range=2,
    expansion_factor=2,
    dropout=0.1
)
```

**Input Processing**:
```python
def forward(self, x, return_features=False):
    # x: (batch, 61, 20) - modified STM features
    
    # Add channel dimension: (batch, 1, 61, 20)
    x = x.unsqueeze(1)
    
    x = self.spec_augment(x)
    
    # Input projection: 1 input channel, increased first layer
    x = self.input_proj(x)  # Conv2d(1, dim//3, ...) → Conv2d(dim//3, dim, ...)
    
    # ...rest unchanged...
```

**Updated Input Projection Layer**:
```python
self.input_proj = nn.Sequential(
    nn.Conv2d(1, dim // 3, kernel_size=3, padding=1),  # 1 input channel, dim//3 = 48
    nn.BatchNorm2d(dim // 3),
    nn.GELU(),
    nn.Conv2d(dim // 3, dim, kernel_size=3, padding=1),
    nn.BatchNorm2d(dim),
    nn.GELU()
)
```

**Parameter count**: ~900K (slightly more than v3 due to dim=144)

## Training Configuration

### Same as ASM v3 (Proven Recipe)

**Loss Function**:
```python
ContrastiveFocalLoss(
    alpha=confusion_aware_weights,  # sqrt-based
    gamma=2.0,
    label_smoothing=0.01,
    contrastive_weight=0.1,
    similar_pairs=[(0, 1), (2, 3)]
)
```

**Optimizer & Scheduler**:
```python
AdamW(lr=1e-3, weight_decay=1e-4)
CosineAnnealingWarmRestarts(T_0=10, T_mult=2, eta_min=1e-6)
Warmup: 5 epochs
```

**Hyperparameters**:
```python
batch_size = 128
num_epochs = 50
gradient_clip_norm = 1.0
```

## Expected Performance Improvements

### Hypothesis 1: Better High-Frequency Discrimination

**Prediction**:
- Classes with high-rate modulations benefit from 1/f correction
- **Music classes** (2, 3): Often have rapid modulations → should improve
- **Speech classes** (0, 1): Slower modulations → less impact

**Target**:
```
Class 2 (music: vocal):     0.84 → 0.86 (+2 points)
Class 3 (music: non-vocal): 0.74 → 0.77 (+3 points)
```

### Hypothesis 2: Directional Discrimination (UPDATED)

**Prediction**:
- **Rescaled asymmetric channel** ensures equal contribution to learning
- Averaging symmetric map prevents magnitude imbalance
- 2-channel structure allows Conv2d to learn cross-channel patterns

**Expected benefits**:
```
Symmetric channel (averaged energy):
    - More stable estimates (averaging reduces noise)
    - Comparable magnitude to asymmetric channel
    - Better for classes with strong overall modulation

Asymmetric channel (rescaled direction):
    - Normalized importance across samples
    - Prevents outlier samples from dominating gradients
    - Better for classes with directional preferences

Combined:
    - Conv2d can learn complementary features from both channels
    - Cross-channel interactions captured by spatial convolutions
    - More balanced feature learning
```

**Target**:
```
Speech classes (0, 1):  Better separation via prosodic directionality
Music classes (2, 3):   Better separation via melodic energy+direction
Env classes (4, 5):     Maintain performance (less directional)
```

### Hypothesis 3: Overall Performance (UPDATED)

**Conservative Target**:
```
Test Macro F1: 0.87-0.88 (match or slightly exceed ASM v3)

Per-Class F1:
  Class 0: 0.95-0.96 (maintain)
  Class 1: 0.82-0.85 (+1-4 points over v3) ← Improved via rescaling
  Class 2: 0.85-0.87 (+1-3 points over v3) ← Improved via 2-channel
  Class 3: 0.76-0.79 (+2-5 points over v3) ← KEY IMPROVEMENT
  Class 4: 0.92-0.94 (maintain)
  Class 5: 0.91-0.93 (maintain)
```

**Optimistic Target**:
```
Test Macro F1: 0.88-0.90 (beat ASM v3 by 1-3 points)
Class 3 F1 ≥ 0.78 (break through 0.75 barrier)
All classes F1 ≥ 0.80 (uniform improvement)
```

**Why this version might perform better**:
1. **Balanced channels**: Rescaling prevents asymmetric channel from being ignored
2. **Reduced redundancy**: Single-sided spectrum (0-15 Hz) is more efficient
3. **Better Conv2d input**: 2 channels vs 1 channel with artificial concatenation
4. **Stable gradients**: Normalized ranges prevent gradient issues

## Comparison with ASM v3

### Key Differences (SYMMETRIC ONLY)

| Aspect | ASM v3 | Modified Features (Symmetric) |
|--------|--------|-------------------------------|
| **Preprocessing** | dB + normalize | dB + normalize + 1/f + symmetric |
| **Feature Dim** | (121, 20) | (61, 20) |
| **Channels** | 1 | 1 |
| **Frequency Range** | -15 to +15 Hz | 0 to +15 Hz |
| **Symmetric** | N/A | Averaged: (|P+|+|P-|)/2 |
| **Asymmetric** | N/A | Not used |
| **Feature Count** | 2,420 | 1,220 (50% reduction) |
| **Model Dim** | 128 | 144 (+12.5%) |
| **Params** | ~800K | ~900K |
| **Bio-Inspired** | No | Yes (1/f, symmetric) |
| **Expected F1** | 0.87+ | **0.87-0.89** |

**Why symmetric-only might work**:
1. **1/f normalization is key**: Spectral balancing may be main driver
2. **Simpler is better**: Reduced complexity, easier optimization
3. **Averaging reduces noise**: More stable energy estimates
4. **Directional info redundant**: Energy sufficient for classification
5. **Model compensation**: Increased capacity offsets information loss

### Complementary Approaches

**ASM v3 Strategy**: "Better learning algorithm"
- Softer class weighting
- Contrastive loss for separation
- Minimal label smoothing

**Modified Features Strategy**: "Better input representation"
- 1/f normalization for spectral balance
- Symmetric/asymmetric decomposition for directionality
- Biologically-inspired preprocessing

**Combined Power**:
- If both work independently → potentially 0.88-0.89 F1
- If synergistic → could reach 0.90 F1
- If redundant → only incremental gain

## Ablation Study Design

### Experiments to Run

**1. Baseline Comparison**:
```bash
# ASM v3 (no modified features)
python STMasm_enhanced3.py 0

# Modified Features (this script)
python STMasm_feateng.py 0
```

**2. Feature Component Ablation**:
```python
# Experiment A: Only 1/f normalization
def preprocess_stm_features_1f_only(stm_2d):
    # Apply 1/f, but don't decompose
    frequency_vector = torch.linspace(-15.0, 15.0, 121)
    abs_freq = torch.abs(frequency_vector).view(1, 121, 1)
    return stm_2d * abs_freq

# Experiment B: Only symmetric/asymmetric (no 1/f)
def preprocess_stm_features_decomp_only(stm_2d):
    # Decompose without 1/f normalization
    # ...decomposition logic...
    
# Experiment C: Both (full pipeline)
# → Main script
```

**3. Component Weight Analysis**:
```python
# Try different concatenation orders
# [Asym | DC | Sym] vs [Sym | DC | Asym]
# Weight channels differently
# Zero out one channel to measure contribution
```

### Expected Ablation Results

**If 1/f normalization is key**:
```
Baseline:         0.8566 (v1)
Only 1/f:         0.87-0.88
Only decomp:      0.86-0.87
Both:             0.88-0.89
```

**If decomposition is key**:
```
Baseline:         0.8566
Only 1/f:         0.86-0.87
Only decomp:      0.87-0.88
Both:             0.88-0.89
```

**If both are redundant**:
```
Baseline:         0.8566
Only 1/f:         0.86-0.87
Only decomp:      0.86-0.87
Both:             0.86-0.87 (no synergy)
```

## Implementation Details

### Frequency Vector Generation

**Critical for Correctness**:
```python
# STM parameters from STM06_STMpreproc.py
xmin, xmax = 190, 310  # Indices in original 500-bin array
xrange = [-15, 15]     # Hz range
x_ds_factor = 1        # No downsampling on time axis

# After cropping: 121 bins
n_time = (xmax - xmin + 1) // x_ds_factor  # = 121

# Frequency vector
rate_min = -15.0
rate_max = 15.0
frequency_vector = torch.linspace(rate_min, rate_max, n_time)

# Resolution check
rate_resolution = (rate_max - rate_min) / (n_time - 1)
# = 30 / 120 = 0.25 Hz ✓ Correct
```

**Index Mapping (UPDATED)**:
```python
Original (121 bins):
    Index    Frequency (Hz)
    0        -15.00
    1        -14.75
    ...
    59       -0.25
    60       0.00          ← DC component
    61       +0.25
    ...
    120      +15.00

After Processing (61 bins, 0-15 Hz):
    Index    Frequency (Hz)
    0        0.00           ← DC
    1        0.25
    2        0.50
    ...
    60       15.00
```

### Asymmetric Map Rescaling (REMOVED)

**Previous approach** (now removed):
```python
# OLD: Per-sample rescaling to [-1, 1]
max_abs_asym = torch.abs(asymmetric_map).reshape(batch, -1).max(dim=1)[0]
asymmetric_map_rescaled = asymmetric_map / max_abs_asym
```

**New approach** (current):
```python
# NEW: Keep raw asymmetric differences
asymmetric_map = torch.abs(positive_chunk) - torch.abs(negative_flipped)
# No rescaling - let z-normalization handle scaling
```

**Rationale for change**:
1. **Preserve natural signal strength**: Rescaling removes information about magnitude
2. **Z-normalization is sufficient**: Per-sample normalization handles scaling
3. **Avoid artificial equalization**: Different samples naturally have different directional strengths
4. **More informative gradients**: Raw differences provide clearer learning signal

**Benefits**:
- Simpler preprocessing pipeline
- Preserves relative magnitude information across samples
- Z-normalization (mean=0, std=1) provides adequate scaling
- More natural representation of directional preference

### Channel Stacking (NEW)

**Implementation**:
```python
# After computing both maps with DC prepended
symmetric_with_dc = torch.cat([dc_component, symmetric_map], dim=1)     # (batch, 61, 20)
asymmetric_with_dc = torch.cat([dc_component, asymmetric_map], dim=1)  # (batch, 61, 20)

# Stack into 2-channel tensor
processed = torch.stack([symmetric_with_dc, asymmetric_with_dc], dim=-1)  # (batch, 61, 20, 2)
```

**Dimension semantics**:
```
processed[b, f, s, 0] = Symmetric map at frequency f, scale s, sample b
processed[b, f, s, 1] = Asymmetric map at frequency f, scale s, sample b
```

**Why stack instead of concatenate**:
```python
# Option A: Concatenate along frequency (NOT USED)
concat_freq = torch.cat([symmetric_with_dc, asymmetric_with_dc], dim=1)  # (batch, 122, 20)
# Issues:
#   - Treats channels as additional frequencies (semantically wrong)
#   - Harder for Conv2d to learn cross-channel interactions
#   - Loses explicit channel structure

# Option B: Stack as separate channel dimension (USED) ✓
stack_channels = torch.stack([symmetric_with_dc, asymmetric_with_dc], dim=-1)  # (batch, 61, 20, 2)
# Benefits:
#   - Natural input for Conv2d: (batch, channels, height, width)
#   - Explicit channel separation
#   - Conv2d learns cross-channel patterns naturally
#   - Standard in computer vision (RGB images, etc.)
```

### Per-Sample Normalization (UPDATED)

**After preprocessing, normalize across all dimensions**:
```python
# STM_processed: (batch, 61, 20)
means = STM_processed.mean(dim=(1, 2), keepdim=True)  # (batch, 1, 1)
stds = STM_processed.std(dim=(1, 2), keepdim=True)    # (batch, 1, 1)
STM_processed = (STM_processed - means) / (stds + 1e-8)
```

**Effect**:
- Centers each sample to mean=0, std=1
- Applied AFTER symmetric averaging
- Ensures both channels have comparable scale after z-normalization
- Prevents any remaining magnitude imbalance

## Integration with ASM Architecture

### Data Flow (SYMMETRIC ONLY)

```
Raw STM (flattened) → Reshape to 2D (121, 20)
                   ↓
         1/f Normalization (by rate)
                   ↓
         Separate Negative/Positive Rates
                   ↓
         Flip & Align
                   ↓
         Compute Symmetric Map (averaged energy)
                   ↓
         Concatenate DC with symmetric map
                   ↓
         Single-channel tensor (61, 20)
                   ↓
         Per-sample Z-normalization
                   ↓
         (batch, 61, 20) → ASM Model
```

### Model Architecture (SYMMETRIC ONLY)

**Key changes for single-channel input with increased capacity**:
```python
ModifiedFeatureASMClassifier(
    time_steps=20,      # Scale dimension
    freq_steps=61,      # Frequency dimension (0-15 Hz)
    num_classes=6,
    dim=144,           # Increased from 128
    num_blocks=4,
    shift_range=2,
    expansion_factor=2,
    dropout=0.1
)
```

**Input Processing**:
```python
def forward(self, x, return_features=False):
    # x: (batch, 61, 20) - modified STM features
    
    # Add channel dimension: (batch, 1, 61, 20)
    x = x.unsqueeze(1)
    
    x = self.spec_augment(x)
    
    # Input projection: 1 input channel, increased first layer
    x = self.input_proj(x)  # Conv2d(1, dim//3, ...) → Conv2d(dim//3, dim, ...)
    
    # ...rest unchanged...
```

**Updated Input Projection Layer**:
```python
self.input_proj = nn.Sequential(
    nn.Conv2d(1, dim // 3, kernel_size=3, padding=1),  # 1 input channel, dim//3 = 48
    nn.BatchNorm2d(dim // 3),
    nn.GELU(),
    nn.Conv2d(dim // 3, dim, kernel_size=3, padding=1),
    nn.BatchNorm2d(dim),
    nn.GELU()
)
```

**Parameter count**: ~900K (slightly more than v3 due to dim=144)

## Training Configuration

### Same as ASM v3 (Proven Recipe)

**Loss Function**:
```python
ContrastiveFocalLoss(
    alpha=confusion_aware_weights,  # sqrt-based
    gamma=2.0,
    label_smoothing=0.01,
    contrastive_weight=0.1,
    similar_pairs=[(0, 1), (2, 3)]
)
```

**Optimizer & Scheduler**:
```python
AdamW(lr=1e-3, weight_decay=1e-4)
CosineAnnealingWarmRestarts(T_0=10, T_mult=2, eta_min=1e-6)
Warmup: 5 epochs
```

**Hyperparameters**:
```python
batch_size = 128
num_epochs = 50
gradient_clip_norm = 1.0
```

## Expected Performance Improvements

### Hypothesis 1: Better High-Frequency Discrimination

**Prediction**:
- Classes with high-rate modulations benefit from 1/f correction
- **Music classes** (2, 3): Often have rapid modulations → should improve
- **Speech classes** (0, 1): Slower modulations → less impact

**Target**:
```
Class 2 (music: vocal):     0.84 → 0.86 (+2 points)
Class 3 (music: non-vocal): 0.74 → 0.77 (+3 points)
```

### Hypothesis 2: Directional Discrimination (UPDATED)

**Prediction**:
- **Rescaled asymmetric channel** ensures equal contribution to learning
- Averaging symmetric map prevents magnitude imbalance
- 2-channel structure allows Conv2d to learn cross-channel patterns

**Expected benefits**:
```
Symmetric channel (averaged energy):
    - More stable estimates (averaging reduces noise)
    - Comparable magnitude to asymmetric channel
    - Better for classes with strong overall modulation

Asymmetric channel (rescaled direction):
    - Normalized importance across samples
    - Prevents outlier samples from dominating gradients
    - Better for classes with directional preferences

Combined:
    - Conv2d can learn complementary features from both channels
    - Cross-channel interactions captured by spatial convolutions
    - More balanced feature learning
```

**Target**:
```
Speech classes (0, 1):  Better separation via prosodic directionality
Music classes (2, 3):   Better separation via melodic energy+direction
Env classes (4, 5):     Maintain performance (less directional)
```

### Hypothesis 3: Overall Performance (UPDATED)

**Conservative Target**:
```
Test Macro F1: 0.87-0.88 (match or slightly exceed ASM v3)

Per-Class F1:
  Class 0: 0.95-0.96 (maintain)
  Class 1: 0.82-0.85 (+1-4 points over v3) ← Improved via rescaling
  Class 2: 0.85-0.87 (+1-3 points over v3) ← Improved via 2-channel
  Class 3: 0.76-0.79 (+2-5 points over v3) ← KEY IMPROVEMENT
  Class 4: 0.92-0.94 (maintain)
  Class 5: 0.91-0.93 (maintain)
```

**Optimistic Target**:
```
Test Macro F1: 0.88-0.90 (beat ASM v3 by 1-3 points)
Class 3 F1 ≥ 0.78 (break through 0.75 barrier)
All classes F1 ≥ 0.80 (uniform improvement)
```

**Why this version might perform better**:
1. **Balanced channels**: Rescaling prevents asymmetric channel from being ignored
2. **Reduced redundancy**: Single-sided spectrum (0-15 Hz) is more efficient
3. **Better Conv2d input**: 2 channels vs 1 channel with artificial concatenation
4. **Stable gradients**: Normalized ranges prevent gradient issues

## Comparison with ASM v3

### Key Differences (SYMMETRIC ONLY)

| Aspect | ASM v3 | Modified Features (Symmetric) |
|--------|--------|-------------------------------|
| **Preprocessing** | dB + normalize | dB + normalize + 1/f + symmetric |
| **Feature Dim** | (121, 20) | (61, 20) |
| **Channels** | 1 | 1 |
| **Frequency Range** | -15 to +15 Hz | 0 to +15 Hz |
| **Symmetric** | N/A | Averaged: (|P+|+|P-|)/2 |
| **Asymmetric** | N/A | Not used |
| **Feature Count** | 2,420 | 1,220 (50% reduction) |
| **Model Dim** | 128 | 144 (+12.5%) |
| **Params** | ~800K | ~900K |
| **Bio-Inspired** | No | Yes (1/f, symmetric) |
| **Expected F1** | 0.87+ | **0.87-0.89** |

**Why symmetric-only might work**:
1. **1/f normalization is key**: Spectral balancing may be main driver
2. **Simpler is better**: Reduced complexity, easier optimization
3. **Averaging reduces noise**: More stable energy estimates
4. **Directional info redundant**: Energy sufficient for classification
5. **Model compensation**: Increased capacity offsets information loss

### Complementary Approaches

**ASM v3 Strategy**: "Better learning algorithm"
- Softer class weighting
- Contrastive loss for separation
- Minimal label smoothing

**Modified Features Strategy**: "Better input representation"
- 1/f normalization for spectral balance
- Symmetric/asymmetric decomposition for directionality
- Biologically-inspired preprocessing

**Combined Power**:
- If both work independently → potentially 0.88-0.89 F1
- If synergistic → could reach 0.90 F1
- If redundant → only incremental gain

## Ablation Study Design

### Experiments to Run

**1. Baseline Comparison**:
```bash
# ASM v3 (no modified features)
python STMasm_enhanced3.py 0

# Modified Features (this script)
python STMasm_feateng.py 0
```

**2. Feature Component Ablation**:
```python
# Experiment A: Only 1/f normalization
def preprocess_stm_features_1f_only(stm_2d):
    # Apply 1/f, but don't decompose
    frequency_vector = torch.linspace(-15.0, 15.0, 121)
    abs_freq = torch.abs(frequency_vector).view(1, 121, 1)
    return stm_2d * abs_freq

# Experiment B: Only symmetric/asymmetric (no 1/f)
def preprocess_stm_features_decomp_only(stm_2d):
    # Decompose without 1/f normalization
    # ...decomposition logic...
    
# Experiment C: Both (full pipeline)
# → Main script
```

**3. Component Weight Analysis**:
```python
# Try different concatenation orders
# [Asym | DC | Sym] vs [Sym | DC | Asym]
# Weight channels differently
# Zero out one channel to measure contribution
```

### Expected Ablation Results

**If 1/f normalization is key**:
```
Baseline:         0.8566 (v1)
Only 1/f:         0.87-0.88
Only decomp:      0.86-0.87
Both:             0.88-0.89
```

**If decomposition is key**:
```
Baseline:         0.8566
Only 1/f:         0.86-0.87
Only decomp:      0.87-0.88
Both:             0.88-0.89
```

**If both are redundant**:
```
Baseline:         0.8566
Only 1/f:         0.86-0.87
Only decomp:      0.86-0.87
Both:             0.86-0.87 (no synergy)
```

## Implementation Details

### Frequency Vector Generation

**Critical for Correctness**:
```python
# STM parameters from STM06_STMpreproc.py
xmin, xmax = 190, 310  # Indices in original 500-bin array
xrange = [-15, 15]     # Hz range
x_ds_factor = 1        # No downsampling on time axis

# After cropping: 121 bins
n_time = (xmax - xmin + 1) // x_ds_factor  # = 121

# Frequency vector
rate_min = -15.0
rate_max = 15.0
frequency_vector = torch.linspace(rate_min, rate_max, n_time)

# Resolution check
rate_resolution = (rate_max - rate_min) / (n_time - 1)
# = 30 / 120 = 0.25 Hz ✓ Correct
```

**Index Mapping (UPDATED)**:
```python
Original (121 bins):
    Index    Frequency (Hz)
    0        -15.00
    1        -14.75
    ...
    59       -0.25
    60       0.00          ← DC component
    61       +0.25
    ...
    120      +15.00

After Processing (61 bins, 0-15 Hz):
    Index    Frequency (Hz)
    0        0.00           ← DC
    1        0.25
    2        0.50
    ...
    60       15.00
```

### Asymmetric Map Rescaling (REMOVED)

**Previous approach** (now removed):
```python
# OLD: Per-sample rescaling to [-1, 1]
max_abs_asym = torch.abs(asymmetric_map).reshape(batch, -1).max(dim=1)[0]
asymmetric_map_rescaled = asymmetric_map / max_abs_asym
```

**New approach** (current):
```python
# NEW: Keep raw asymmetric differences
asymmetric_map = torch.abs(positive_chunk) - torch.abs(negative_flipped)
# No rescaling - let z-normalization handle scaling
```

**Rationale for change**:
1. **Preserve natural signal strength**: Rescaling removes information about magnitude
2. **Z-normalization is sufficient**: Per-sample normalization handles scaling
3. **Avoid artificial equalization**: Different samples naturally have different directional strengths
4. **More informative gradients**: Raw differences provide clearer learning signal

**Benefits**:
- Simpler preprocessing pipeline
- Preserves relative magnitude information across samples
- Z-normalization (mean=0, std=1) provides adequate scaling
- More natural representation of directional preference

### Channel Stacking (NEW)

**Implementation**:
```python
# After computing both maps with DC prepended
symmetric_with_dc = torch.cat([dc_component, symmetric_map], dim=1)     # (batch, 61, 20)
asymmetric_with_dc = torch.cat([dc_component, asymmetric_map], dim=1)  # (batch, 61, 20)

# Stack into 2-channel tensor
processed = torch.stack([symmetric_with_dc, asymmetric_with_dc], dim=-1)  # (batch, 61, 20, 2)
```

**Dimension semantics**:
```
processed[b, f, s, 0] = Symmetric map at frequency f, scale s, sample b
processed[b, f, s, 1] = Asymmetric map at frequency f, scale s, sample b
```

**Why stack instead of concatenate**:
```python
# Option A: Concatenate along frequency (NOT USED)
concat_freq = torch.cat([symmetric_with_dc, asymmetric_with_dc], dim=1)  # (batch, 122, 20)
# Issues:
#   - Treats channels as additional frequencies (semantically wrong)
#   - Harder for Conv2d to learn cross-channel interactions
#   - Loses explicit channel structure

# Option B: Stack as separate channel dimension (USED) ✓
stack_channels = torch.stack([symmetric_with_dc, asymmetric_with_dc], dim=-1)  # (batch, 61, 20, 2)
# Benefits:
#   - Natural input for Conv2d: (batch, channels, height, width)
#   - Explicit channel separation
#   - Conv2d learns cross-channel patterns naturally
#   - Standard in computer vision (RGB images, etc.)
```

### Per-Sample Normalization (UPDATED)

**After preprocessing, normalize across all dimensions**:
```python
# STM_processed: (batch, 61, 20)
means = STM_processed.mean(dim=(1, 2), keepdim=True)  # (batch, 1, 1)
stds = STM_processed.std(dim=(1, 2), keepdim=True)    # (batch, 1, 1)
STM_processed = (STM_processed - means) / (stds + 1e-8)
```

**Effect**:
- Centers each sample to mean=0, std=1
- Applied AFTER symmetric averaging
- Ensures both channels have comparable scale after z-normalization
- Prevents any remaining magnitude imbalance

## Integration with ASM Architecture

### Data Flow (SYMMETRIC ONLY)

```
Raw STM (flattened) → Reshape to 2D (121, 20)
                   ↓
         1/f Normalization (by rate)
                   ↓
         Separate Negative/Positive Rates
                   ↓
         Flip & Align
                   ↓
         Compute Symmetric Map (averaged energy)
                   ↓
         Concatenate DC with symmetric map
                   ↓
         Single-channel tensor (61, 20)
                   ↓
         Per-sample Z-normalization
                   ↓
         (batch, 61, 20) → ASM Model
```

### Model Architecture (SYMMETRIC ONLY)

**Key changes for single-channel input with increased capacity**:
```python
ModifiedFeatureASMClassifier(
    time_steps=20,      # Scale dimension
    freq_steps=61,      # Frequency dimension (0-15 Hz)
    num_classes=6,
    dim=144,           # Increased from 128
    num_blocks=4,
    shift_range=2,
    expansion_factor=2,
    dropout=0.1
)
```

**Input Processing**:
```python
def forward(self, x, return_features=False):
    # x: (batch, 61, 20) - modified STM features
    
    # Add channel dimension: (batch, 1, 61, 20)
    x = x.unsqueeze(1)
    
    x = self.spec_augment(x)
    
    # Input projection: 1 input channel, increased first layer
    x = self.input_proj(x)  # Conv2d(1, dim//3, ...) → Conv2d(dim//3, dim, ...)
    
    # ...rest unchanged...
```

**Updated Input Projection Layer**:
```python
self.input_proj = nn.Sequential(
    nn.Conv2d(1, dim // 3, kernel_size=3, padding=1),  # 1 input channel, dim//3 = 48
    nn.BatchNorm2d(dim // 3),
    nn.GELU(),
    nn.Conv2d(dim // 3, dim, kernel_size=3, padding=1),
    nn.BatchNorm2d(dim),
    nn.GELU()
)
```

**Parameter count**: ~900K (slightly more than v3 due to dim=144)

## Training Configuration

### Same as ASM v3 (Proven Recipe)

**Loss Function**:
```python
ContrastiveFocalLoss(
    alpha=confusion_aware_weights,  # sqrt-based
    gamma=2.0,
    label_smoothing=0.01,
    contrastive_weight=0.1,
    similar_pairs=[(0, 1), (2, 3)]
)
```

**Optimizer & Scheduler**:
```python
AdamW(lr=1e-3, weight_decay=1e-4)
CosineAnnealingWarmRestarts(T_0=10, T_mult=2, eta_min=1e-6)
Warmup: 5 epochs
```

**Hyperparameters**:
```python
batch_size = 128
num_epochs = 50
gradient_clip_norm = 1.0
```

## Expected Performance Improvements

### Hypothesis 1: Better High-Frequency Discrimination

**Prediction**:
- Classes with high-rate modulations benefit from 1/f correction
- **Music classes** (2, 3): Often have rapid modulations → should improve
- **Speech classes** (0, 1): Slower modulations → less impact

**Target**:
```
Class 2 (music: vocal):     0.84 → 0.86 (+2 points)
Class 3 (music: non-vocal): 0.74 → 0.77 (+3 points)
```

### Hypothesis 2: Directional Discrimination (UPDATED)

**Prediction**:
- **Rescaled asymmetric channel** ensures equal contribution to learning
- Averaging symmetric map prevents magnitude imbalance
- 2-channel structure allows Conv2d to learn cross-channel patterns

**Expected benefits**:
```
Symmetric channel (averaged energy):
    - More stable estimates (averaging reduces noise)
    - Comparable magnitude to asymmetric channel
    - Better for classes with strong overall modulation

Asymmetric channel (rescaled direction):
    - Normalized importance across samples
    - Prevents outlier samples from dominating gradients
    - Better for classes with directional preferences

Combined:
    - Conv2d can learn complementary features from both channels
    - Cross-channel interactions captured by spatial convolutions
    - More balanced feature learning
```

**Target**:
```
Speech classes (0, 1):  Better separation via prosodic directionality
Music classes (2, 3):   Better separation via melodic energy+direction
Env classes (4, 5):     Maintain performance (less directional)
```

### Hypothesis 3: Overall Performance (UPDATED)

**Conservative Target**:
```
Test Macro F1: 0.87-0.88 (match or slightly exceed ASM v3)

Per-Class F1:
  Class 0: 0.95-0.96 (maintain)
  Class 1: 0.82-0.85 (+1-4 points over v3) ← Improved via rescaling
  Class 2: 0.85-0.87 (+1-3 points over v3) ← Improved via 2-channel
  Class 3: 0.76-0.79 (+2-5 points over v3) ← KEY IMPROVEMENT
  Class 4: 0.92-0.94 (maintain)
  Class 5: 0.91-0.93 (maintain)
```

**Optimistic Target**:
```
Test Macro F1: 0.88-0.90 (beat ASM v3 by 1-3 points)
Class 3 F1 ≥ 0.78 (break through 0.75 barrier)
All classes F1 ≥ 0.80 (uniform improvement)
```

**Why this version might perform better**:
1. **Balanced channels**: Rescaling prevents asymmetric channel from being ignored
2. **Reduced redundancy**: Single-sided spectrum (0-15 Hz) is more efficient
3. **Better Conv2d input**: 2 channels vs 1 channel with artificial concatenation
4. **Stable gradients**: Normalized ranges prevent gradient issues

## Comparison with ASM v3

### Key Differences (SYMMETRIC ONLY)

| Aspect | ASM v3 | Modified Features (Symmetric) |
|--------|--------|-------------------------------|
| **Preprocessing** | dB + normalize | dB + normalize + 1/f + symmetric |
| **Feature Dim** | (121, 20) | (61, 20) |
| **Channels** | 1 | 1 |
| **Frequency Range** | -15 to +15 Hz | 0 to +15 Hz |
| **Symmetric** | N/A | Averaged: (|P+|+|P-|)/2 |
| **Asymmetric** | N/A | Not used |
| **Feature Count** | 2,420 | 1,220 (50% reduction) |
| **Model Dim** | 128 | 144 (+12.5%) |
| **Params** | ~800K | ~900K |
| **Bio-Inspired** | No | Yes (1/f, symmetric) |
| **Expected F1** | 0.87+ | **0.87-0.89** |

**Why symmetric-only might work**:
1. **1/f normalization is key**: Spectral balancing may be main driver
2. **Simpler is better**: Reduced complexity, easier optimization
3. **Averaging reduces noise**: More stable energy estimates
4. **Directional info redundant**: Energy sufficient for classification
5. **Model compensation**: Increased capacity offsets information loss

### Complementary Approaches

**ASM v3 Strategy**: "Better learning algorithm"
- Softer class weighting
- Contrastive loss for separation
- Minimal label smoothing

**Modified Features Strategy**: "Better input representation"
- 1/f normalization for spectral balance
- Symmetric/asymmetric decomposition for directionality
- Biologically-inspired preprocessing

**Combined Power**:
- If both work independently → potentially 0.88-0.89 F1
- If synergistic → could reach 0.90 F1
- If redundant → only incremental gain

## Ablation Study Design

### Experiments to Run

**1. Baseline Comparison**:
```bash
# ASM v3 (no modified features)
python STMasm_enhanced3.py 0

# Modified Features (this script)
python STMasm_feateng.py 0
```

**2. Feature Component Ablation**:
```python
# Experiment A: Only 1/f normalization
def preprocess_stm_features_1f_only(stm_2d):
    # Apply 1/f, but don't decompose
    frequency_vector = torch.linspace(-15.0, 15.0, 121)
    abs_freq = torch.abs(frequency_vector).view(1, 121, 1)
    return stm_2d * abs_freq

# Experiment B: Only symmetric/asymmetric (no 1/f)
def preprocess_stm_features_decomp_only(stm_2d):
    # Decompose without 1/f normalization
    # ...decomposition logic...
    
# Experiment C: Both (full pipeline)
# → Main script
```

**3. Component Weight Analysis**:
```python
# Try different concatenation orders
# [Asym | DC | Sym] vs [Sym | DC | Asym]
# Weight channels differently
# Zero out one channel to measure contribution
```

### Expected Ablation Results

**If 1/f normalization is key**:
```
Baseline:         0.8566 (v1)
Only 1/f:         0.87-0.88
Only decomp:      0.86-0.87
Both:             0.88-0.89
```

**If decomposition is key**:
```
Baseline:         0.8566
Only 1/f:         0.86-0.87
Only decomp:      0.87-0.88
Both:             0.88-0.89
```

**If both are redundant**:
```
Baseline:         0.8566
Only 1/f:         0.86-0.87
Only decomp:      0.86-0.87
Both:             0.86-0.87 (no synergy)
```

## Implementation Details

### Frequency Vector Generation

**Critical for Correctness**:
```python
# STM parameters from STM06_STMpreproc.py
xmin, xmax = 190, 310  # Indices in original 500-bin array
xrange = [-15, 15]     # Hz range
x_ds_factor = 1        # No downsampling on time axis

# After cropping: 121 bins
n_time = (xmax - xmin + 1) // x_ds_factor  # = 121

# Frequency vector
rate_min = -15.0
rate_max = 15.0
frequency_vector = torch.linspace(rate_min, rate_max, n_time)

# Resolution check
rate_resolution = (rate_max - rate_min) / (n_time - 1)
# = 30 / 120 = 0.25 Hz ✓ Correct
```

**Index Mapping (UPDATED)**:
```python
Original (121 bins):
    Index    Frequency (Hz)
    0        -15.00
    1        -14.75
    ...
    59       -0.25
    60       0.00          ← DC component
    61       +0.25
    ...
    120      +15.00

After Processing (61 bins, 0-15 Hz):
    Index    Frequency (Hz)
    0        0.00           ← DC
    1        0.25
    2        0.50
    ...
    60       15.00
```

### Asymmetric Map Rescaling (REMOVED)

**Previous approach** (now removed):
```python
# OLD: Per-sample rescaling to [-1, 1]
max_abs_asym = torch.abs(asymmetric_map).reshape(batch, -1).max(dim=1)[0]
asymmetric_map_rescaled = asymmetric_map / max_abs_asym
```

**New approach** (current):
```python
# NEW: Keep raw asymmetric differences
asymmetric_map = torch.abs(positive_chunk) - torch.abs(negative_flipped)
# No rescaling - let z-normalization handle scaling
```

**Rationale for change**:
1. **Preserve natural signal strength**: Rescaling removes information about magnitude
2. **Z-normalization is sufficient**: Per-sample normalization handles scaling
3. **Avoid artificial equalization**: Different samples naturally have different directional strengths
4. **More informative gradients**: Raw differences provide clearer learning signal

**Benefits**:
- Simpler preprocessing pipeline
- Preserves relative magnitude information across samples
- Z-normalization (mean=0, std=1) provides adequate scaling
- More natural representation of directional preference

### Channel Stacking (NEW)

**Implementation**:
```python
# After computing both maps with DC prepended
symmetric_with_dc = torch.cat([dc_component, symmetric_map], dim=1)     # (batch, 61, 20)
asymmetric_with_dc = torch.cat([dc_component, asymmetric_map], dim=1)  # (batch, 61, 20)

# Stack into 2-channel tensor
processed = torch.stack([symmetric_with_dc, asymmetric_with_dc], dim=-1)  # (batch, 61, 20, 2)
```

**Dimension semantics**:
```
processed[b, f, s, 0] = Symmetric map at frequency f, scale s, sample b
processed[b, f, s, 1] = Asymmetric map at frequency f, scale s, sample b
```

**Why stack instead of concatenate**:
```python
# Option A: Concatenate along frequency (NOT USED)
concat_freq = torch.cat([symmetric_with_dc, asymmetric_with_dc], dim=1)  # (batch, 122, 20)
# Issues:
#   - Treats channels as additional frequencies (semantically wrong)
#   - Harder for Conv2d to learn cross-channel interactions
#   - Loses explicit channel structure

# Option B: Stack as separate channel dimension (USED) ✓
stack_channels = torch.stack([symmetric_with_dc, asymmetric_with_dc], dim=-1)  # (batch, 61, 20, 2)
# Benefits:
#   - Natural input for Conv2d: (batch, channels, height, width)
#   - Explicit channel separation
#   - Conv2d learns cross-channel patterns naturally
#   - Standard in computer vision (RGB images, etc.)
```

### Per-Sample Normalization (UPDATED)

**After preprocessing, normalize across all dimensions**:
```python
# STM_processed: (batch, 61, 20)
means = STM_processed.mean(dim=(1, 2), keepdim=True)  # (batch, 1, 1)
stds = STM_processed.std(dim=(1, 2), keepdim=True)    # (batch, 1, 1)
STM_processed = (STM_processed - means) / (stds + 1e-8)
```

**Effect**:
- Centers each sample to mean=0, std=1
- Applied AFTER symmetric averaging
- Ensures both channels have comparable scale after z-normalization
- Prevents any remaining magnitude imbalance

## Integration with ASM Architecture

### Data Flow (SYMMETRIC ONLY)

```
Raw STM (flattened) → Reshape to 2D (121, 20)
                   ↓
         1/f Normalization (by rate)
                   ↓
         Separate Negative/Positive Rates
                   ↓
         Flip & Align
                   ↓
         Compute Symmetric Map (averaged energy)
                   ↓
         Concatenate DC with symmetric map
                   ↓
         Single-channel tensor (61, 20)
                   ↓
         Per-sample Z-normalization
                   ↓
         (batch, 61, 20) → ASM Model
```

### Model Architecture (SYMMETRIC ONLY)

**Key changes for single-channel input with increased capacity**:
```python
ModifiedFeatureASMClassifier(
    time_steps=20,      # Scale dimension
    freq_steps=61,      # Frequency dimension (0-15 Hz)
    num_classes=6,
    dim=144,           # Increased from 128
    num_blocks=4,
    shift_range=2,
    expansion_factor=2,
    dropout=0.1
)
```

**Input Processing**:
```python
def forward(self, x, return_features=False):
    # x: (batch, 61, 20) - modified STM features
    
    # Add channel dimension: (batch, 1, 61, 20)
    x = x.unsqueeze(1)
    
    x = self.spec_augment(x)
    
    # Input projection: 1 input channel, increased first layer
    x = self.input_proj(x)  # Conv2d(1, dim//3, ...) → Conv2d(dim//3, dim, ...)
    
    # ...rest unchanged...
```

**Updated Input Projection Layer**:
```python
self.input_proj = nn.Sequential(
    nn.Conv2d(1, dim // 3, kernel_size=3, padding=1),  # 1 input channel, dim//3 = 48
    nn.BatchNorm2d(dim // 3),
    nn.GELU(),
    nn.Conv2d(dim // 3, dim, kernel_size=3, padding=1),
    nn.BatchNorm2d(dim),
    nn.GELU()
)
```

**Parameter count**: ~900K (slightly more than v3 due to dim=144)

## Training Configuration

### Same as ASM v3 (Proven Recipe)

**Loss Function**:
```python
ContrastiveFocalLoss(
    alpha=confusion_aware_weights,  # sqrt-based
    gamma=2.0,
    label_smoothing=0.01,
    contrastive_weight=0.1,
    similar_pairs=[(0, 1), (2, 3)]
)
```

**Optimizer & Scheduler**:
```python
AdamW(lr=1e-3, weight_decay=1e-4)
CosineAnnealingWarmRestarts(T_0=10, T_mult=2, eta_min=1e-6)
Warmup: 5 epochs
```

**Hyperparameters**:
```python
batch_size = 128
num_epochs = 50
gradient_clip_norm = 1.0
```

## Expected Performance Improvements

### Hypothesis 1: Better High-Frequency Discrimination

**Prediction**:
- Classes with high-rate modulations benefit from 1/f correction
- **Music classes** (2, 3): Often have rapid modulations → should improve
- **Speech classes** (0, 1): Slower modulations → less impact

**Target**:
```
Class 2 (music: vocal):     0.84 → 0.86 (+2 points)
Class 3 (music: non-vocal): 0.74 → 0.77 (+3 points)
```

### Hypothesis 2: Directional Discrimination (UPDATED)

**Prediction**:
- **Rescaled asymmetric channel** ensures equal contribution to learning
- Averaging symmetric map prevents magnitude imbalance
- 2-channel structure allows Conv2d to learn cross-channel patterns

**Expected benefits**:
```
Symmetric channel (averaged energy):
    - More stable estimates (averaging reduces noise)
    - Comparable magnitude to asymmetric channel
    - Better for classes with strong overall modulation

Asymmetric channel (rescaled direction):
    - Normalized importance across samples
    - Prevents outlier samples from dominating gradients
    - Better for classes with directional preferences

Combined:
    - Conv2d can learn complementary features from both channels
    - Cross-channel interactions captured by spatial convolutions
    - More balanced feature learning
```

**Target**:
```
Speech classes (0, 1):  Better separation via prosodic directionality
Music classes (2, 3):   Better separation via melodic energy+direction
Env classes (4, 5):     Maintain performance (less directional)
```

### Hypothesis 3: Overall Performance (UPDATED)

**Conservative Target**:
```
Test Macro F1: 0.87-0.88 (match or slightly exceed ASM v3)

Per-Class F1:
  Class 0: 0.95-0.96 (maintain)
  Class 1: 0.82-0.85 (+1-4 points over v3) ← Improved via rescaling
  Class 2: 0.85-0.87 (+1-3 points over v3) ← Improved via 2-channel
  Class 3: 0.76-0.79 (+2-5 points over v3) ← KEY IMPROVEMENT
  Class 4: 0.92-0.94 (maintain)
  Class 5: 0.91-0.93 (maintain)
```

**Optimistic Target**:
```
Test Macro F1: 0.88-0.90 (beat ASM v3 by 1-3 points)
Class 3 F1 ≥ 0.78 (break through 0.75 barrier)
All classes F1 ≥ 0.80 (uniform improvement)
```

**Why this version might perform better**:
1. **Balanced channels**: Rescaling prevents asymmetric channel from being ignored
2. **Reduced redundancy**: Single-sided spectrum (0-15 Hz) is more efficient
3. **Better Conv2d input**: 2 channels vs 1 channel with artificial concatenation
4. **Stable gradients**: Normalized ranges prevent gradient issues

## Comparison with ASM v3

### Key Differences (SYMMETRIC ONLY)

| Aspect | ASM v3 | Modified Features (Symmetric) |
|--------|--------|-------------------------------|
| **Preprocessing** | dB + normalize | dB + normalize + 1/f + symmetric |
| **Feature Dim** | (121, 20) | (61, 20) |
| **Channels** | 1 | 1 |
| **Frequency Range** | -15 to +15 Hz | 0 to +15 Hz |
| **Symmetric** | N/A | Averaged: (|P+|+|P-|)/2 |
| **Asymmetric** | N/A | Not used |
| **Feature Count** | 2,420 | 1,220 (50% reduction) |
| **Model Dim** | 128 | 144 (+12.5%) |
| **Params** | ~800K | ~900K |
| **Bio-Inspired** | No | Yes (1/f, symmetric) |
| **Expected F1** | 0.87+ | **0.87-0.89** |

**Why symmetric-only might work**:
1. **1/f normalization is key**: Spectral balancing may be main driver
2. **Simpler is better**: Reduced complexity, easier optimization
3. **Averaging reduces noise**: More stable energy estimates
4. **Directional info redundant**: Energy sufficient for classification
5. **Model compensation**: Increased capacity offsets information loss

### Complementary Approaches

**ASM v3 Strategy**: "Better learning algorithm"
- Softer class weighting
- Contrastive loss for separation
- Minimal label smoothing

**Modified Features Strategy**: "Better input representation"
- 1/f normalization for spectral balance
- Symmetric/asymmetric decomposition for directionality
- Biologically-inspired preprocessing

**Combined Power**:
- If both work independently → potentially 0.88-0.89 F1
- If synergistic → could reach 0.90 F1
- If redundant → only incremental gain

## Ablation Study Design

### Experiments to Run

**1. Baseline Comparison**:
```bash
# ASM v3 (no modified features)
python STMasm_enhanced3.py 0

# Modified Features (this script)
python STMasm_feateng.py 0
```

**2. Feature Component Ablation**:
```python
# Experiment A: Only 1/f normalization
def preprocess_stm_features_1f_only(stm_2d):
    # Apply 1/f, but don't decompose
    frequency_vector = torch.linspace(-15.0, 15.0, 121)
    abs_freq = torch.abs(frequency_vector).view(1, 121, 1)
    return stm_2d * abs_freq

# Experiment B: Only symmetric/asymmetric (no 1/f)
def preprocess_stm_features_decomp_only(stm_2d):
    # Decompose without 1/f normalization
    # ...decomposition logic...
    
# Experiment C: Both (full pipeline)
# → Main script
```

**3. Component Weight Analysis**:
```python
# Try different concatenation orders
# [Asym | DC | Sym] vs [Sym | DC | Asym]
# Weight channels differently
# Zero out one channel to measure contribution
```

### Expected Ablation Results

**If 1/f normalization is key**:
```
Baseline:         0.8566 (v1)
Only 1/f:         0.87-0.88
Only decomp:      0.86-0.87
Both:             0.88-0.89
```

**If decomposition is key**:
```
Baseline:         0.8566
Only 1/f:         0.86-0.87
Only decomp:      0.87-0.88
Both:             0.88-0.89
```

**If both are redundant**:
```
Baseline:         0.8566
Only 1/f:         0.86-0.87
Only decomp:      0.86-0.87
Both:             0.86-0.87 (no synergy)
```

## Implementation Details

### Frequency Vector Generation

**Critical for Correctness**:
```python
# STM parameters from STM06_STMpreproc.py
xmin, xmax = 190, 310  # Indices in original 500-bin array
xrange = [-15, 15]     # Hz range
x_ds_factor = 1        # No downsampling on time axis

# After cropping: 121 bins
n_time = (xmax - xmin + 1) // x_ds_factor  # = 121

# Frequency vector
rate_min = -15.0
rate_max = 15.0
frequency_vector = torch.linspace(rate_min, rate_max, n_time)

# Resolution check
rate_resolution = (rate_max - rate_min) / (n_time - 1)
# = 30 / 120 = 0.25 Hz ✓ Correct
```

**Index Mapping (UPDATED)**:
```python
Original (121 bins):
    Index    Frequency (Hz)
    0        -15.00
    1        -14.75
    ...
    59       -0.25
    60       0.00          ← DC component
    61       +0.25
    ...
    120      +15.00

After Processing (61 bins, 0-15 Hz):
    Index    Frequency (Hz)
    0        0.00           ← DC
    1        0.25
    2        0.50
    ...
    60       15.00
```

### Asymmetric Map Rescaling (REMOVED)

**Previous approach** (now removed):
```python
# OLD: Per-sample rescaling to [-1, 1]
max_abs_asym = torch.abs(asymmetric_map).reshape(batch, -1).max(dim=1)[0]
asymmetric_map_rescaled = asymmetric_map / max_abs_asym
```

**New approach** (current):
```python
# NEW: Keep raw asymmetric differences
asymmetric_map = torch.abs(positive_chunk) - torch.abs(negative_flipped)
# No rescaling - let z-normalization handle scaling
```

**Rationale for change**:
1. **Preserve natural signal strength**: Rescaling removes information about magnitude
2. **Z-normalization is sufficient**: Per-sample normalization handles scaling
3. **Avoid artificial equalization**: Different samples naturally have different directional strengths
4. **More informative gradients**: Raw differences provide clearer learning signal

**Benefits**:
- Simpler preprocessing pipeline
- Preserves relative magnitude information across samples
- Z-normalization (mean=0, std=1) provides adequate scaling
- More natural representation of directional preference

### Channel Stacking (NEW)

**Implementation**:
```python
# After computing both maps with DC prepended
symmetric_with_dc = torch.cat([dc_component, symmetric_map], dim=1)     # (batch, 61, 20)
asymmetric_with_dc = torch.cat([dc_component, asymmetric_map], dim=1)  # (batch, 61, 20)

# Stack into 2-channel tensor
processed = torch.stack([symmetric_with_dc, asymmetric_with_dc], dim=-1)  # (batch, 61, 20, 2)
```

**Dimension semantics**:
```
processed[b, f, s, 0] = Symmetric map at frequency f, scale s, sample b
processed[b, f, s, 1] = Asymmetric map at frequency f, scale s, sample b
```

**Why stack instead of concatenate**:
```python
# Option A: Concatenate along frequency (NOT USED)
concat_freq = torch.cat([symmetric_with_dc, asymmetric_with_dc], dim=1)  # (batch, 122, 20)
# Issues:
#   - Treats channels as additional frequencies (semantically wrong)
#   - Harder for Conv2d to learn cross-channel interactions
#   - Loses explicit channel structure

# Option B: Stack as separate channel dimension (USED) ✓
stack_channels = torch.stack([symmetric_with_dc, asymmetric_with_dc], dim=-1)  # (batch, 61, 20, 2)
# Benefits:
#   - Natural input for Conv2d: (batch, channels, height, width)
#   - Explicit channel separation
#   - Conv2d learns cross-channel patterns naturally
#   - Standard in computer vision (RGB images, etc.)
```

### Per-Sample Normalization (UPDATED)

**After preprocessing, normalize across all dimensions**:
```python
# STM_processed: (batch, 61, 20)
means = STM_processed.mean(dim=(1, 2), keepdim=True)  # (batch, 1, 1)
stds = STM_processed.std(dim=(1, 2), keepdim=True)    # (batch, 1, 1)
STM_processed = (STM_processed - means) / (stds + 1e-8)
```

**Effect**:
- Centers each sample to mean=0, std=1
- Applied AFTER symmetric averaging
- Ensures both channels have comparable scale after z-normalization
- Prevents any remaining magnitude imbalance

## Integration with ASM Architecture

### Data Flow (SYMMETRIC ONLY)

```
Raw STM (flattened) → Reshape to 2D (121, 20)
                   ↓
         1/f Normalization (by rate)
                   ↓
         Separate Negative/Positive Rates
                   ↓
         Flip & Align
                   ↓
         Compute Symmetric Map (averaged energy)
                   ↓
         Concatenate DC with symmetric map
                   ↓
         Single-channel tensor (61, 20)
                   ↓
         Per-sample Z-normalization
                   ↓
         (batch, 61, 20) → ASM Model
```

### Model Architecture (SYMMETRIC ONLY)

**Key changes for single-channel input with increased capacity**:
```python
ModifiedFeatureASMClassifier(
    time_steps=20,      # Scale dimension
    freq_steps=61,      # Frequency dimension (0-15 Hz)
    num_classes=6,
    dim=144,           # Increased from 128
    num_blocks=4,
    shift_range=2,
    expansion_factor=2,
    dropout=0.1
)
```

**Input Processing**:
```python
def forward(self, x, return_features=False):
    # x: (batch, 61, 20) - modified STM features
    
    # Add channel dimension: (batch, 1, 61, 20)
    x = x.unsqueeze(1)
    
    x = self.spec_augment(x)
    
    # Input projection: 1 input channel, increased first layer
    x = self.input_proj(x)  # Conv2d(1, dim//3, ...) → Conv2d(dim//3, dim, ...)
    
    # ...rest unchanged...
```

**Updated Input Projection Layer**:
```python
self.input_proj = nn.Sequential(
    nn.Conv2d(1, dim // 3, kernel_size=3, padding=1),  # 1 input channel, dim//3 = 48
    nn.BatchNorm2d(dim // 3),
    nn.GELU(),
    nn.Conv2d(dim // 3, dim, kernel_size=3, padding=1),
    nn.BatchNorm2d(dim),
    nn.GELU()
)
```

**Parameter count**: ~900K (slightly more than v3 due to dim=144)

## Training Configuration

### Same as ASM v3 (Proven Recipe)

**Loss Function**:
```python
ContrastiveFocalLoss(
    alpha=confusion_aware_weights,  # sqrt-based
    gamma=2.0,
    label_smoothing=0.01,
    contrastive_weight=0.1,
    similar_pairs=[(0, 1), (2, 3)]
)
```

**Optimizer & Scheduler**:
```python
AdamW(lr=1e-3, weight_decay=1e-4)
CosineAnnealingWarmRestarts(T_0=10, T_mult=2, eta_min=1e-6)
Warmup: 5 epochs
```

**Hyperparameters**:
```python
batch_size = 128
num_epochs = 50
gradient_clip_norm = 1.0
```

## Expected Performance Improvements

### Hypothesis 1: Better High-Frequency Discrimination

**Prediction**:
- Classes with high-rate modulations benefit from 1/f correction
- **Music classes** (2, 3): Often have rapid modulations → should improve
- **Speech classes** (0, 1): Slower modulations → less impact

**Target**:
```
Class 2 (music: vocal):     0.84 → 0.86 (+2 points)
Class 3 (music: non-vocal): 0.74 → 0.77 (+3 points)
```

### Hypothesis 2: Directional Discrimination (UPDATED)

**Prediction**:
- **Rescaled asymmetric channel** ensures equal contribution to learning
- Averaging symmetric map prevents magnitude imbalance
- 2-channel structure allows Conv2d to learn cross-channel patterns

**Expected benefits**:
```
Symmetric channel (averaged energy):
    - More stable estimates (averaging reduces noise)
    - Comparable magnitude to asymmetric channel
    - Better for classes with strong overall modulation

Asymmetric channel (rescaled direction):
    - Normalized importance across samples
    - Prevents outlier samples from dominating gradients
    - Better for classes with directional preferences

Combined:
    - Conv2d can learn complementary features from both channels
    - Cross-channel interactions captured by spatial convolutions
    - More balanced feature learning
```

**Target**:
```
Speech classes (0, 1):  Better separation via prosodic directionality
Music classes (2, 3):   Better separation via melodic energy+direction
Env classes (4, 5):     Maintain performance (less directional)
```

### Hypothesis 3: Overall Performance (UPDATED)

**Conservative Target**:
```
Test Macro F1: 0.87-0.88 (match or slightly exceed ASM v3)

Per-Class F1:
  Class 0: 0.95-0.96 (maintain)
  Class 1: 0.82-0.85 (+1-4 points over v3) ← Improved via rescaling
  Class 2: 0.85-0.87 (+1-3 points over v3) ← Improved via 2-channel
  Class 3: 0.76-0.79 (+2-5 points over v3) ← KEY IMPROVEMENT
  Class 4: 0.92-0.94 (maintain)
  Class 5: 0.91-0.93 (maintain)
```

**Optimistic Target**:
```
Test Macro F1: 0.88-0.90 (beat ASM v3 by 1-3 points)
Class 3 F1 ≥ 0.78 (break through 0.75 barrier)
All classes F1 ≥ 0.80 (uniform improvement)
```

**Why this version might perform better**:
1. **Balanced channels**: Rescaling prevents asymmetric channel from being ignored
2. **Reduced redundancy**: Single-sided spectrum (0-15 Hz) is more efficient
3. **Better Conv2d input**: 2 channels vs 1 channel with artificial concatenation
4. **Stable gradients**: Normalized ranges prevent gradient issues

## Comparison with ASM v3

### Key Differences (SYMMETRIC ONLY)

| Aspect | ASM v3 | Modified Features (Symmetric) |
|--------|--------|-------------------------------|
| **Preprocessing** | dB + normalize | dB + normalize + 1/f + symmetric |
| **Feature Dim** | (121, 20) | (61, 20) |
| **Channels** | 1 | 1 |
| **Frequency Range** | -15 to +15 Hz | 0 to +15 Hz |
| **Symmetric** | N/A | Averaged: (|P+|+|P-|)/2 |
| **Asymmetric** | N/A | Not used |
| **Feature Count** | 2,420 | 1,220 (50% reduction) |
| **Model Dim** | 128 | 144 (+12.5%) |
| **Params** | ~800K | ~900K |
| **Bio-Inspired** | No | Yes (1/f, symmetric) |
| **Expected F1** | 0.87+ | **0.87-0.89** |

**Why symmetric-only might work**:
1. **1/f normalization is key**: Spectral balancing may be main driver
2. **Simpler is better**: Reduced complexity, easier optimization
3. **Averaging reduces noise**: More stable energy estimates
4. **Directional info redundant**: Energy sufficient for classification
5. **Model compensation**: Increased capacity offsets information loss

### Complementary Approaches

**ASM v3 Strategy**: "Better learning algorithm"
- Softer class weighting
- Contrastive loss for separation
- Minimal label smoothing

**Modified Features Strategy**: "Better input representation"
- 1/f normalization for spectral balance
- Symmetric/asymmetric decomposition for directionality
- Biologically-inspired preprocessing

**Combined Power**:
- If both work independently → potentially 0.88-0.89 F1
- If synergistic → could reach 0.90 F1
- If redundant → only incremental gain

## Ablation Study Design

### Experiments to Run

**1. Baseline Comparison**:
```bash
# ASM v3 (no modified features)
python STMasm_enhanced3.py 0

# Modified Features (this script)
python STMasm_feateng.py 0
```

**2. Feature Component Ablation**:
```python
# Experiment A: Only 1/f normalization
def preprocess_stm_features_1f_only(stm_2d):
    # Apply 1/f, but don't decompose
    frequency_vector = torch.linspace(-15.0, 15.0, 121)
    abs_freq = torch.abs(frequency_vector).view(1, 121, 1)
    return stm_2d * abs_freq

# Experiment B: Only symmetric/asymmetric (no 1/f)
def preprocess_stm_features_decomp_only(stm_2d):
    # Decompose without 1/f normalization
    # ...decomposition logic...
    
# Experiment C: Both (full pipeline)
# → Main script
```

**3. Component Weight Analysis**:
```python
# Try different concatenation orders
# [Asym | DC | Sym] vs [Sym | DC | Asym]
# Weight channels differently
# Zero out one channel to measure contribution
```

### Expected Ablation Results

**If 1/f normalization is key**:
```
Baseline:         0.8566 (v1)
Only 1/f:         0.87-0.88
Only decomp:      0.86-0.87
Both:             0.88-0.89
```

**If decomposition is key**:
```
Baseline:         0.8566
Only 1/f:         0.86-0.87
Only decomp:      0.87-0.88
Both:             0.88-0.89
```

**If both are redundant**:
```
Baseline:         0.8566
Only 1/f:         0.86-0.87
Only decomp:      0.86-0.87
Both:             0.86-0.87 (no synergy)
```

## Implementation Details

### Frequency Vector Generation

**Critical for Correctness**:
```python
# STM parameters from STM06_STMpreproc.py
xmin, xmax = 190, 310  # Indices in original 500-bin array
xrange = [-15, 15]     # Hz range
x_ds_factor = 1        # No downsampling on time axis

# After cropping: 121 bins
n_time = (xmax - xmin + 1) // x_ds_factor  # = 121

# Frequency vector
rate_min = -15.0
rate_max = 15.0
frequency_vector = torch.linspace(rate_min, rate_max, n_time)

# Resolution check
rate_resolution = (rate_max - rate_min) / (n_time - 1)
# = 30 / 120 = 0.25 Hz ✓ Correct
```

**Index Mapping (UPDATED)**:
```python
Original (121 bins):
    Index    Frequency (Hz)
    0        -15.00
    1        -14.75
    ...
    59       -0.25
    60       0.00          ← DC component
    61       +0.25
    ...
    120      +15.00

After Processing (61 bins, 0-15 Hz):
    Index    Frequency (Hz)
    0        0.00           ← DC
    1        0.25
    2        0.50
    ...
    60       15.00
```

### Asymmetric Map Rescaling (REMOVED)

**Previous approach** (now removed):
```python
# OLD: Per-sample rescaling to [-1, 1]
max_abs_asym = torch.abs(asymmetric_map).reshape(batch, -1).max(dim=1)[0]
asymmetric_map_rescaled = asymmetric_map / max_abs_asym
```

**New approach** (current):
```python
# NEW: Keep raw asymmetric differences
asymmetric_map = torch.abs(positive_chunk) - torch.abs(negative_flipped)
# No rescaling - let z-normalization handle scaling
```

**Rationale for change**:
1. **Preserve natural signal strength**: Rescaling removes information about magnitude
2. **Z-normalization is sufficient**: Per-sample normalization handles scaling
3. **Avoid artificial equalization**: Different samples naturally have different directional strengths
4. **More informative gradients**: Raw differences provide clearer learning signal

**Benefits**:
- Simpler preprocessing pipeline
- Preserves relative magnitude information across samples
- Z-normalization (mean=0, std=1) provides adequate scaling
- More natural representation of directional preference

### Channel Stacking (NEW)

**Implementation**:
```python
# After computing both maps with DC prepended
symmetric_with_dc = torch.cat([dc_component, symmetric_map], dim=1)     # (batch, 61, 20)
asymmetric_with_dc = torch.cat([dc_component, asymmetric_map], dim=1)  # (batch, 61, 20)

# Stack into 2-channel tensor
processed = torch.stack([symmetric_with_dc, asymmetric_with_dc], dim=-1)  # (batch, 61, 20, 2)
```

**Dimension semantics**:
```
processed[b, f, s, 0] = Symmetric map at frequency f, scale s, sample b
processed[b, f, s, 1] = Asymmetric map at frequency f, scale s, sample b
```

**Why stack instead of concatenate**:
```python
# Option A: Concatenate along frequency (NOT USED)
concat_freq = torch.cat([symmetric_with_dc, asymmetric_with_dc], dim=1)  # (batch, 122, 20)
# Issues:
#   - Treats channels as additional frequencies (semantically wrong)
#   - Harder for Conv2d to learn cross-channel interactions
#   - Loses explicit channel structure

# Option B: Stack as separate channel dimension (USED) ✓
stack_channels = torch.stack([symmetric_with_dc, asymmetric_with_dc], dim=-1)  # (batch, 61, 20, 2)
# Benefits:
#   - Natural input for Conv2d: (batch, channels, height, width)
#   - Explicit channel separation
#   - Conv2d learns cross-channel patterns naturally
#   - Standard in computer vision (RGB images, etc.)
```

### Per-Sample Normalization (UPDATED)

**After preprocessing, normalize across all dimensions**:
```python
# STM_processed: (batch, 61, 20)
means = STM_processed.mean(dim=(1, 2), keepdim=True)  # (batch, 1, 1)
stds = STM_processed.std(dim=(1, 2), keepdim=True)    # (batch, 1, 1)
STM_processed = (STM_processed - means) / (stds + 1e-8)
```

**Effect**:
- Centers each sample to mean=0, std=1
- Applied AFTER symmetric averaging
- Ensures both channels have comparable scale after z-normalization
- Prevents any remaining magnitude imbalance

## Integration with ASM Architecture

### Data Flow (SYMMETRIC ONLY)

```
Raw STM (flattened) → Reshape to 2D (121, 20)
                   ↓
         1/f Normalization (by rate)
                   ↓
         Separate Negative/Positive Rates
                   ↓
         Flip & Align
                   ↓
         Compute Symmetric Map (averaged energy)
                   ↓
         Concatenate DC with symmetric map
                   ↓
         Single-channel tensor (61, 20)
                   ↓
         Per-sample Z-normalization
                   ↓
         (batch, 61, 20) → ASM Model
```

### Model Architecture (SYMMETRIC ONLY)

**Key changes for single-channel input with increased capacity**:
```python
ModifiedFeatureASMClassifier(
    time_steps=20,      # Scale dimension
    freq_steps=61,      # Frequency dimension (0-15 Hz)
    num_classes=6,
    dim=144,           # Increased from 128
    num_blocks=4,
    shift_range=2,
    expansion_factor=2,
    dropout=0.1
)
```

**Input Processing**:
```python
def forward(self, x, return_features=False):
    # x: (batch, 61, 20) - modified STM features
    
    # Add channel dimension: (batch, 1, 61, 20)
    x = x.unsqueeze(1)
    
    x = self.spec_augment(x)
    
    # Input projection: 1 input channel, increased first layer
    x = self.input_proj(x)  # Conv2d(1, dim//3, ...) → Conv2d(dim//3, dim, ...)
    
    # ...rest unchanged...
```

**Updated Input Projection Layer**:
```python
self.input_proj = nn.Sequential(
    nn.Conv2d(1, dim // 3, kernel_size=3, padding=1),  # 1 input channel, dim//3 = 48
    nn.BatchNorm2d(dim // 3),
    nn.GELU(),
    nn.Conv2d(dim // 3, dim, kernel_size=3, padding=1),
    nn.BatchNorm2d(dim),
    nn.GELU()
)
```

**Parameter count**: ~900K (slightly more than v3 due to dim=144)

## Training Configuration

### Same as ASM v3 (Proven Recipe)

**Loss Function**:
```python
ContrastiveFocalLoss(
    alpha=confusion_aware_weights,  # sqrt-based
    gamma=2.0,
    label_smoothing=0.01,
    contrastive_weight=0.1,
    similar_pairs=[(0, 1), (2, 3)]
)
```

**Optimizer & Scheduler**:
```python
AdamW(lr=1e-3, weight_decay=1e-4)
CosineAnnealingWarmRestarts(T_0=10, T_mult=2, eta_min=1e-6)
Warmup: 5 epochs
```

**Hyperparameters**:
```python
batch_size = 128
num_epochs = 50
gradient_clip_norm = 1.0
```

## Expected Performance Improvements

### Hypothesis 1: Better High-Frequency Discrimination

**Prediction**:
- Classes with high-rate modulations benefit from 1/f correction
- **Music classes** (2, 3): Often have rapid modulations → should improve
- **Speech classes** (0, 1): Slower modulations → less impact

**Target**:
```
Class 2 (music: vocal):     0.84 → 0.86 (+2 points)
Class 3 (music: non-vocal): 0.74 → 0.77 (+3 points)
```

### Hypothesis 2: Directional Discrimination (UPDATED)

**Prediction**:
- **Rescaled asymmetric channel** ensures equal contribution to learning
- Averaging symmetric map prevents magnitude imbalance
- 2-channel structure allows Conv2d to learn cross-channel patterns

**Expected benefits**:
```
Symmetric channel (averaged energy):
    - More stable estimates (averaging reduces noise)
    - Comparable magnitude to asymmetric channel
    - Better for classes with strong overall modulation

Asymmetric channel (rescaled direction):
    - Normalized importance across samples
    - Prevents outlier samples from dominating gradients
    - Better for classes with directional preferences

Combined:
    - Conv2d can learn complementary features from both channels
    - Cross-channel interactions captured by spatial convolutions
    - More balanced feature learning
```

**Target**:
```
Speech classes (0, 1):  Better separation via prosodic directionality
Music classes (2, 3):   Better separation via melodic energy+direction
Env classes (4, 5):     Maintain performance (less directional)
```

### Hypothesis 3: Overall Performance (UPDATED)

**Conservative Target**:
```
Test Macro F1: 0.87-0.88 (match or slightly exceed ASM v3)

Per-Class F1:
  Class 0: 0.95-0.96 (maintain)
  Class 1: 0.82-0.85 (+1-4 points over v3) ← Improved via rescaling
  Class 2: 0.85-0.87 (+1-3 points over v3) ← Improved via 2-channel
  Class 3: 0.76-0.79 (+2-5 points over v3) ← KEY IMPROVEMENT
  Class 4: 0.92-0.94 (maintain)
  Class 5: 0.91-0.93 (maintain)
```

**Optimistic Target**:
```
Test Macro F1: 0.88-0.90 (beat ASM v3 by 1-3 points)
Class 3 F1 ≥ 0.78 (break through 0.75 barrier)
All classes F1 ≥ 0.80 (uniform improvement)
```

**Why this version might perform better**:
1. **Balanced channels**: Rescaling prevents asymmetric channel from being ignored
2. **Reduced redundancy**: Single-sided spectrum (0-15 Hz) is more efficient
3. **Better Conv2d input**: 2 channels vs 1 channel with artificial concatenation
4. **Stable gradients**: Normalized ranges prevent gradient issues

## Comparison with ASM v3

### Key Differences (SYMMETRIC ONLY)

| Aspect | ASM v3 | Modified Features (Symmetric) |
|--------|--------|-------------------------------|
| **Preprocessing** | dB + normalize | dB + normalize + 1/f + symmetric |
| **Feature Dim** | (121, 20) | (61, 20) |
| **Channels** | 1 | 1 |
| **Frequency Range** | -15 to +15 Hz | 0 to +15 Hz |
| **Symmetric** | N/A | Averaged: (|P+|+|P-|)/2 |
| **Asymmetric** | N/A | Not used |
| **Feature Count** | 2,420 | 1,220 (50% reduction) |
| **Model Dim** | 128 | 144 (+12.5%) |
| **Params** | ~800K | ~900K |
| **Bio-Inspired** | No | Yes (1/f, symmetric) |
| **Expected F1** | 0.87+ | **0.87-0.89** |

**Why symmetric-only might work**:
1. **1/f normalization is key**: Spectral balancing may be main driver
2. **Simpler is better**: Reduced complexity, easier optimization
3. **Averaging reduces noise**: More stable energy estimates
4. **Directional info redundant**: Energy sufficient for classification
5. **Model compensation**: Increased capacity offsets information loss

### Complementary Approaches

**ASM v3 Strategy**: "Better learning algorithm"
- Softer class weighting
- Contrastive loss for separation
- Minimal label smoothing

**Modified Features Strategy**: "Better input representation"
- 1/f normalization for spectral balance
- Symmetric/asymmetric decomposition for directionality
- Biologically-inspired preprocessing

**Combined Power**:
- If both work independently → potentially 0.88-0.89 F1
- If synergistic → could reach 0.90 F1
- If redundant → only incremental gain

## Ablation Study Design

### Experiments to Run

**1. Baseline Comparison**:
```bash
# ASM v3 (no modified features)
python STMasm_enhanced3.py 0

# Modified Features (this script)
python STMasm_feateng.py 0
```

**2. Feature Component Ablation**:
```python
# Experiment A: Only 1/f normalization
def preprocess_stm_features_1f_only(stm_2d):
    # Apply 1/f, but don't decompose
    frequency_vector = torch.linspace(-15.0, 15.0, 121)
    abs_freq = torch.abs(frequency_vector).view(1, 121, 1)
    return stm_2d * abs_freq

# Experiment B: Only symmetric/asymmetric (no 1/f)
def preprocess_stm_features_decomp_only(stm_2d):
    # Decompose without 1/f normalization
    # ...decomposition logic...
    
# Experiment C: Both (full pipeline)
# → Main script
```

**3. Component Weight Analysis**:
```python
# Try different concatenation orders
# [Asym | DC | Sym] vs [Sym | DC | Asym]
# Weight channels differently
# Zero out one channel to measure contribution
```

### Expected Ablation Results

**If 1/f normalization is key**:
```
Baseline:         0.8566 (v1)
Only 1/f:         0.87-0.88
Only decomp:      0.86-0.87
Both:             0.88-0.89
```

**If decomposition is key**:
```
Baseline:         0.8566
Only 1/f:         0.86-0.87
Only decomp:      0.87-0.88
Both:             0.88-0.89
```

**If both are redundant**:
```
Baseline:         0.8566
Only 1/f:         0.86-0.87
Only decomp:      0.86-0.87
Both:             0.86-0.87 (no synergy)
```

## Implementation Details

### Frequency Vector Generation

**Critical for Correctness**:
```python
# STM parameters from STM06_STMpreproc.py
xmin, xmax = 190, 310  # Indices in original 500-bin array
xrange = [-15, 15]     # Hz range
x_ds_factor = 1        # No downsampling on time axis

# After cropping: 121 bins
n_time = (xmax - xmin + 1) // x_ds_factor  # = 121

# Frequency vector
rate_min = -15.0
rate_max = 15.0
frequency_vector = torch.linspace(rate_min, rate_max, n_time)

# Resolution check
rate_resolution = (rate_max - rate_min) / (n_time - 1)
# = 30 / 120 = 0.25 Hz ✓ Correct
```

**Index Mapping (UPDATED)**:
```python
Original (121 bins):
    Index    Frequency (Hz)
    0        -15.00
    1        -14.75
    ...
    59       -0.25
    60       0.00          ← DC component
    61       +0.25
    ...
    120      +15.00

After Processing (61 bins, 0-15 Hz):
    Index    Frequency (Hz)
    0        0.00           ← DC
    1        0.25
    2        0.50
    ...
    60       15.00
```

### Asymmetric Map Rescaling (REMOVED)

**Previous approach** (now removed):
```python
# OLD: Per-sample rescaling to [-1, 1]
max_abs_asym = torch.abs(asymmetric_map).reshape(batch, -1).max(dim=1)[0]
asymmetric_map_rescaled = asymmetric_map / max_abs_asym
```

**New approach** (current):
```python
# NEW: Keep raw asymmetric differences
asymmetric_map = torch.abs(positive_chunk) - torch.abs(negative_flipped)
# No rescaling - let z-normalization handle scaling
```

**Rationale for change**:
1. **Preserve natural signal strength**: Rescaling removes information about magnitude
2. **Z-normalization is sufficient**: Per-sample normalization handles scaling
3. **Avoid artificial equalization**: Different samples naturally have different directional strengths
4. **More informative gradients**: Raw differences provide clearer learning signal

**Benefits**:
- Simpler preprocessing pipeline
- Preserves relative magnitude information across samples
- Z-normalization (mean=0, std=1) provides adequate scaling
- More natural representation of directional preference

### Channel Stacking (NEW)

**Implementation**:
```python
# After computing both maps with DC prepended
symmetric_with_dc = torch.cat([dc_component, symmetric_map], dim=1)     # (batch, 61, 20)
asymmetric_with_dc = torch.cat([dc_component, asymmetric_map], dim=1)  # (batch, 61, 20)

# Stack into 2-channel tensor
processed = torch.stack([symmetric_with_dc, asymmetric_with_dc], dim=-1)  # (batch, 61, 20, 2)
```

**Dimension semantics**:
```
processed[b, f, s, 0] = Symmetric map at frequency f, scale s, sample b
processed[b, f, s, 1] = Asymmetric map at frequency f, scale s, sample b
```

**Why stack instead of concatenate**:
```python
# Option A: Concatenate along frequency (NOT USED)
concat_freq = torch.cat([symmetric_with_dc, asymmetric_with_dc], dim=1)  # (batch, 122, 20)
# Issues:
#   - Treats channels as additional frequencies (semantically wrong)
#   - Harder for Conv2d to learn cross-channel interactions
#   - Loses explicit channel structure

# Option B: Stack as separate channel dimension (USED) ✓
stack_channels = torch.stack([symmetric_with_dc, asymmetric_with_dc], dim=-1)  # (batch, 61, 20, 2)
# Benefits:
#   - Natural input for Conv2d: (batch, channels, height, width)
#   - Explicit channel separation
#   - Conv2d learns cross-channel patterns naturally
#   - Standard in computer vision (RGB images, etc.)
```

### Per-Sample Normalization (UPDATED)

**After preprocessing, normalize across all dimensions**:
```python
# STM_processed: (batch, 61, 20)
means = STM_processed.mean(dim=(1, 2), keepdim=True)  # (batch, 1, 1)
stds = STM_processed.std(dim=(1, 2), keepdim=True)    # (batch, 1, 1)
STM_processed = (STM_processed - means) / (stds + 1e-8)
```

**Effect**:
- Centers each sample to mean=0, std=1
- Applied AFTER symmetric averaging
- Ensures both channels have comparable scale after z-normalization
- Prevents any remaining magnitude imbalance

## Integration with ASM Architecture

### Data Flow (SYMMETRIC ONLY)

```
Raw STM (flattened) → Reshape to 2D (121, 20)
                   ↓
         1/f Normalization (by rate)
                   ↓
         Separate Negative/Positive Rates
                   ↓
         Flip & Align
                   ↓
         Compute Symmetric Map (averaged energy)
                   ↓
         Concatenate DC with symmetric map
                   ↓
         Single-channel tensor (61, 20)
                   ↓
         Per-sample Z-normalization
                   ↓
         (batch, 61, 20) → ASM Model
```

### Model Architecture (SYMMETRIC ONLY)

**Key changes for single-channel input with increased capacity**:
```python
ModifiedFeatureASMClassifier(
    time_steps=20,      # Scale dimension
    freq_steps=61,      # Frequency dimension (0-15 Hz)
    num_classes=6,
    dim=144,           # Increased from 128
    num_blocks=4,
    shift_range=2,
    expansion_factor=2,
    dropout=0.1
)
```

**Input Processing**:
```python
def forward(self, x, return_features=False):
    # x: (batch, 61, 20) - modified STM features
    
    # Add channel dimension: (batch, 1, 61, 20)
    x = x.unsqueeze(1)
    
    x = self.spec_augment(x)
    
    # Input projection: 1 input channel, increased first layer
    x = self.input_proj(x)  # Conv2d(1, dim//3, ...) → Conv2d(dim//3, dim, ...)
    
    # ...rest unchanged...
```

**Updated Input Projection Layer**:
```python
self.input_proj = nn.Sequential(
    nn.Conv2d(1, dim // 3, kernel_size=3, padding=1),  # 1 input channel, dim//3 = 48
    nn.BatchNorm2d(dim // 3),
    nn.GELU(),
    nn.Conv2d(dim // 3, dim, kernel_size=3, padding=1),
    nn.BatchNorm2d(dim),
    nn.GELU()
)
```

**Parameter count**: ~900K (slightly more than v3 due to dim=144)

## Training Configuration

### Same as ASM v3 (Proven Recipe)

**Loss Function**:
```python
ContrastiveFocalLoss(
    alpha=confusion_aware_weights,  # sqrt-based
    gamma=2.0,
    label_smoothing=0.01,
    contrastive_weight=0.1,
    similar_pairs=[(0, 1), (2, 3)]
)
```

**Optimizer & Scheduler**:
```python
AdamW(lr=1e-3, weight_decay=1e-4)
CosineAnnealingWarmRestarts(T_0=10, T_mult=2, eta_min=1e-6)
Warmup: 5 epochs
```

**Hyperparameters**:
```python
batch_size = 128
num_epochs = 50
gradient_clip_norm = 1.0
```

## Expected Performance Improvements

### Hypothesis 1: Better High-Frequency Discrimination

**Prediction**:
- Classes with high-rate modulations benefit from 1/f correction
- **Music classes** (2, 3): Often have rapid modulations → should improve
- **Speech classes** (0, 1): Slower modulations → less impact

**Target**:
```
Class 2 (music: vocal):     0.84 → 0.86 (+2 points)
Class 3 (music: non-vocal): 0.74 → 0.77 (+3 points)
```

### Hypothesis 2: Directional Discrimination (UPDATED)

**Prediction**:
- **Rescaled asymmetric channel** ensures equal contribution to learning
- Averaging symmetric map prevents magnitude imbalance
- 2-channel structure allows Conv2d to learn cross-channel patterns

**Expected benefits**:
```
Symmetric channel (averaged energy):
    - More stable estimates (averaging reduces noise)
    - Comparable magnitude to asymmetric channel
    - Better for classes with strong overall modulation

Asymmetric channel (rescaled direction):
    - Normalized importance across samples
    - Prevents outlier samples from dominating gradients
    - Better for classes with directional preferences

Combined:
    - Conv2d can learn complementary features from both channels
    - Cross-channel interactions captured by spatial convolutions
    - More balanced feature learning
```

**Target**:
```
Speech classes (0, 1):  Better separation via prosodic directionality
Music classes (2, 3):   Better separation via melodic energy+direction
Env classes (4, 5):     Maintain performance (less directional)
```

### Hypothesis 3: Overall Performance (UPDATED)

**Conservative Target**:
```
Test Macro F1: 0.87-0.88 (match or slightly exceed ASM v3)

Per-Class F1:
  Class 0: 0.95-0.96 (maintain)
  Class 1: 0.82-0.85 (+1-4 points over v3) ← Improved via rescaling
  Class 2: 0.85-0.87 (+1-3 points over v3) ← Improved via 2-channel
  Class 3: 0.76-0.79 (+2-5 points over v3) ← KEY IMPROVEMENT
  Class 4: 0.92-0.94 (maintain)
  Class 5: 0.91-0.93 (maintain)
```

**Optimistic Target**:
```
Test Macro F1: 0.88-0.90 (beat ASM v3 by 1-3 points)
Class 3 F1 ≥ 0.78 (break through 0.75 barrier)
All classes F1 ≥ 0.80 (uniform improvement)
```

**Why this version might perform better**:
1. **Balanced channels**: Rescaling prevents asymmetric channel from being ignored
2. **Reduced redundancy**: Single-sided spectrum (0-15 Hz) is more efficient
3. **Better Conv2d input**: 2 channels vs 1 channel with artificial concatenation
4. **Stable gradients**: Normalized ranges prevent gradient issues

## Comparison with ASM v3

### Key Differences (SYMMETRIC ONLY)

| Aspect | ASM v3 | Modified Features (Symmetric) |
|--------|--------|-------------------------------|
| **Preprocessing** | dB + normalize | dB + normalize + 1/f + symmetric |
| **Feature Dim** | (121, 20) | (61, 20) |
| **Channels** | 1 | 1 |
| **Frequency Range** | -15 to +15 Hz | 0 to +15 Hz |
| **Symmetric** | N/A | Averaged: (|P+|+|P-|)/2 |
| **Asymmetric** | N/A | Not used |
| **Feature Count** | 2,420 | 1,220 (50% reduction) |
| **Model Dim** | 128 | 144 (+12.5%) |
| **Params** | ~800K | ~900K |
| **Bio-Inspired** | No | Yes (1/f, symmetric) |
| **Expected F1** | 0.87+ | **0.87-0.89** |

**Why symmetric-only might work**:
1. **1/f normalization is key**: Spectral balancing may be main driver
2. **Simpler is better**: Reduced complexity, easier optimization
3. **Averaging reduces noise**: More stable energy estimates
4. **Directional info redundant**: Energy sufficient for classification
5. **Model compensation**: Increased capacity offsets information loss

### Complementary Approaches

**ASM v3 Strategy**: "Better learning algorithm"
- Softer class weighting
- Contrastive loss for separation
- Minimal label smoothing

**Modified Features Strategy**: "Better input representation"
- 1/f normalization for spectral balance
- Symmetric/asymmetric decomposition for directionality
- Biologically-inspired preprocessing

**Combined Power**:
- If both work independently → potentially 0.88-0.89 F1
- If synergistic → could reach 0.90 F1
- If redundant → only incremental gain

## Ablation Study Design

### Experiments to Run

**1. Baseline Comparison**:
```bash
# ASM v3 (no modified features)
python STMasm_enhanced3.py 0

# Modified Features (this script)
python STMasm_feateng.py 0
```

**2. Feature Component Ablation**:
```python
# Experiment A: Only 1/f normalization
def preprocess_stm_features_1f_only(stm_2d):
    # Apply 1/f, but don't decompose
    frequency_vector = torch.linspace(-15.0, 15.0, 121)
    abs_freq = torch.abs(frequency_vector).view(1, 121, 1)
    return stm_2d * abs_freq

# Experiment B: Only symmetric/asymmetric (no 1/f)
def preprocess_stm_features_decomp_only(stm_2d):
    # Decompose without 1/f normalization
    # ...decomposition logic...
    
# Experiment C: Both (full pipeline)
# → Main script
```

**3. Component Weight Analysis**:
```python
# Try different concatenation orders
# [Asym | DC | Sym] vs [Sym | DC | Asym]
# Weight channels differently
# Zero out one channel to measure contribution
```

### Expected Ablation Results

**If 1/f normalization is key**:
```
Baseline:         0.8566 (v1)
Only 1/f:         0.87-0.88
Only decomp:      0.86-0.87
Both:             0.88-0.89
```

**If decomposition is key**:
```
Baseline:         0.8566
Only 1/f:         0.86-0.87
Only decomp:      0.87-0.88
Both:             0.88-0.89
```

**If both are redundant**:
```
Baseline:         0.8566
Only 1/f:         0.86-0.87
Only decomp:      0.86-0.87
Both:             0.86-0.87 (no synergy)
```

## Implementation Details

### Frequency Vector Generation

**Critical for Correctness**:
```python
# STM parameters from STM06_STMpreproc.py
xmin, xmax = 190, 310  # Indices in original 500-bin array
xrange = [-15, 15]     # Hz range
x_ds_factor = 1        # No downsampling on time axis

# After cropping: 121 bins
n_time = (xmax - xmin + 1) // x_ds_factor  # = 121

# Frequency vector
rate_min = -15.0
rate_max = 15.0
frequency_vector = torch.linspace(rate_min, rate_max, n_time)

# Resolution check
rate_resolution = (rate_max - rate_min) / (n_time - 1)
# = 30 / 120 = 0.25 Hz ✓ Correct
```

**Index Mapping (UPDATED)**:
```python
Original (121 bins):
    Index    Frequency (Hz)
    0        -15.00
    1        -14.75
    ...
    59       -0.25
    60       0.00          ← DC component
    61       +0.25
    ...
    120      +15.00

After Processing (61 bins, 0-15 Hz):
    Index    Frequency (Hz)
    0        0.00           ← DC
    1        0.25
    2        0.50
    ...
    60       15.00
```

### Asymmetric Map Rescaling (REMOVED)

**Previous approach** (now removed):
```python
# OLD: Per-sample rescaling to [-1, 1]
max_abs_asym = torch.abs(asymmetric_map).reshape(batch, -1).max(dim=1)[0]
asymmetric_map_rescaled = asymmetric_map / max_abs_asym
```

**New approach** (current):
```python
# NEW: Keep raw asymmetric differences
asymmetric_map = torch.abs(positive_chunk) - torch.abs(negative_flipped)
# No rescaling - let z-normalization handle scaling
```

**Rationale for change**:
1. **Preserve natural signal strength**: Rescaling removes information about magnitude
2. **Z-normalization is sufficient**: Per-sample normalization handles scaling
3. **Avoid artificial equalization**: Different samples naturally have different directional strengths
4. **More informative gradients**: Raw differences provide clearer learning signal

**Benefits**:
- Simpler preprocessing pipeline
- Preserves relative magnitude information across samples
- Z-normalization (mean=0, std=1) provides adequate scaling
- More natural representation of directional preference

### Channel Stacking (NEW)

**Implementation**:
```python
# After computing both maps with DC prepended
symmetric_with_dc = torch.cat([dc_component, symmetric_map], dim=1)     # (batch, 61, 20)
asymmetric_with_dc = torch.cat([dc_component, asymmetric_map], dim=1)  # (batch, 61, 20)

# Stack into 2-channel tensor
processed = torch.stack([symmetric_with_dc, asymmetric_with_dc], dim=-1)  # (batch, 61, 20, 2)
```

**Dimension semantics**:
```
processed[b, f, s, 0] = Symmetric map at frequency f, scale s, sample b
processed[b, f, s, 1] = Asymmetric map at frequency f, scale s, sample b
```

**Why stack instead of concatenate**:
```python
# Option A: Concatenate along frequency (NOT USED)
concat_freq = torch.cat([symmetric_with_dc, asymmetric_with_dc], dim=1)  # (batch, 122, 20)
# Issues:
#   - Treats channels as additional frequencies (semantically wrong)
#   - Harder for Conv2d to learn cross-channel interactions
#   - Loses explicit channel structure

# Option B: Stack as separate channel dimension (USED) ✓
stack_channels = torch.stack([symmetric_with_dc, asymmetric_with_dc], dim=-1)  # (batch, 61, 20, 2)
# Benefits:
#   - Natural input for Conv2d: (batch, channels, height, width)
#   - Explicit channel separation
#   - Conv2d learns cross-channel patterns naturally
#   - Standard in computer vision (RGB images, etc.)
```

### Per-Sample Normalization (UPDATED)

**After preprocessing, normalize across all dimensions**:
```python
# STM_processed: (batch, 61, 20)
means = STM_processed.mean(dim=(1, 2), keepdim=True)  # (batch, 1, 1)
stds = STM_processed.std(dim=(1, 2), keepdim=True)    # (batch, 1, 1)
STM_processed = (STM_processed - means) / (stds + 1e-8)
```

**Effect**:
- Centers each sample to mean=0, std=1
- Applied AFTER symmetric averaging
- Ensures both channels have comparable scale after z-normalization
- Prevents any remaining magnitude imbalance

## Integration with ASM Architecture

### Data Flow (SYMMETRIC ONLY)

```
Raw STM (flattened) → Reshape to 2D (121, 20)
                   ↓
         1/f Normalization (by rate)
                   ↓
         Separate Negative/Positive Rates
                   ↓
         Flip & Align
                   ↓
         Compute Symmetric Map (averaged energy)
                   ↓
         Concatenate DC with symmetric map
                   ↓
         Single-channel tensor (61, 20)
                   ↓
         Per-sample Z-normalization
                   ↓
         (batch, 61, 20) → ASM Model
```

### Model Architecture (SYMMETRIC ONLY)

**Key changes for single-channel input with increased capacity**:
```python
ModifiedFeatureASMClassifier(
    time_steps=20,      # Scale dimension
    freq_steps=61,      # Frequency dimension (0-15 Hz)
    num_classes=6,
    dim=144,           # Increased from 128
    num_blocks=4,
    shift_range=2,
    expansion_factor=2,
    dropout=0.1
)
```

**Input Processing**:
```python
def forward(self, x, return_features=False):
    # x: (batch, 61, 20) - modified STM features
    
    # Add channel dimension: (batch, 1, 61, 20)
    x = x.unsqueeze(1)
    
    x = self.spec_augment(x)
    
    # Input projection: 1 input channel, increased first layer
    x = self.input_proj(x)  # Conv2d(1, dim//3, ...) → Conv2d(dim//3, dim, ...)
    
    # ...rest unchanged...
```

**Updated Input Projection Layer**:
```python
self.input_proj = nn.Sequential(
    nn.Conv2d(1, dim // 3, kernel_size=3, padding=1),  # 1 input channel, dim//3 = 48
    nn.BatchNorm2d(dim // 3),
    nn.GELU(),
    nn.Conv2d(dim // 3, dim, kernel_size=3, padding=1),
    nn.BatchNorm2d(dim),
    nn.GELU()
)
```

**Parameter count**: ~900K (slightly more than v3 due to dim=144)

## Training Configuration

### Same as ASM v3 (Proven Recipe)

**Loss Function**:
```python
ContrastiveFocalLoss(
    alpha=confusion_aware_weights,  # sqrt-based
    gamma=2.0,
    label_smoothing=0.01,
    contrastive_weight=0.1,
    similar_pairs=[(0, 1), (2, 3)]
)
```

**Optimizer & Scheduler**:
```python
AdamW(lr=1e-3, weight_decay=1e-4)
CosineAnnealingWarmRestarts(T_0=10, T_mult=2, eta_min=1e-6)
Warmup: 5 epochs
```

**Hyperparameters**:
```python
batch_size = 128
num_epochs = 50
gradient_clip_norm = 1.0
```

## Expected Performance Improvements

### Hypothesis 1: Better High-Frequency Discrimination

**Prediction**:
- Classes with high-rate modulations benefit from 1/f correction
- **Music classes** (2, 3): Often have rapid modulations → should improve
- **Speech classes** (0, 1): Slower modulations → less impact

**Target**:
```
Class 2 (music: vocal):     0.84 → 0.86 (+2 points)
Class 3 (music: non-vocal): 0.74 → 0.77 (+3 points)
```

### Hypothesis 2: Directional Discrimination (UPDATED)

**Prediction**:
- **Rescaled asymmetric channel** ensures equal contribution to learning
- Averaging symmetric map prevents magnitude imbalance
- 2-channel structure allows Conv2d to learn cross-channel patterns

**Expected benefits**:
```
Symmetric channel (averaged energy):
    - More stable estimates (averaging reduces noise)
    - Comparable magnitude to asymmetric channel
    - Better for classes with strong overall modulation

Asymmetric channel (rescaled direction):
    - Normalized importance across samples
    - Prevents outlier samples from dominating gradients
    - Better for classes with directional preferences

Combined:
    - Conv2d can learn complementary features from both channels
    - Cross-channel interactions captured by spatial convolutions
    - More balanced feature learning
```

**Target**:
```
Speech classes (0, 1):  Better separation via prosodic directionality
Music classes (2, 3):   Better separation via melodic energy+direction
Env classes (4, 5):     Maintain performance (less directional)
```

### Hypothesis 3: Overall Performance (UPDATED)

**Conservative Target**:
```
Test Macro F1: 0.87-0.88 (match or slightly exceed ASM v3)

Per-Class F1:
  Class 0: 0.95-0.96 (maintain)
  Class 1: 0.82-0.85 (+1-4 points over v3) ← Improved via rescaling
  Class 2: 0.85-0.87 (+1-3 points over v3) ← Improved via 2-channel
  Class 3: 0.76-0.79 (+2-5 points over v3) ← KEY IMPROVEMENT
  Class 4: 0.92-0.94 (maintain)
  Class 5: 0.91-0.93 (maintain)
```

**Optimistic Target**:
```
Test Macro F1: 0.88-0.90 (beat ASM v3 by 1-3 points)
Class 3 F1 ≥ 0.78 (break through 0.75 barrier)
All classes F1 ≥ 0.80 (uniform improvement)
```

**Why this version might perform better**:
1. **Balanced channels**: Rescaling prevents asymmetric channel from being ignored
2. **Reduced redundancy**: Single-sided spectrum (0-15 Hz) is more efficient
3. **Better Conv2d input**: 2 channels vs 1 channel with artificial concatenation
4. **Stable gradients**: Normalized ranges prevent gradient issues

## Comparison with ASM v3

### Key Differences (SYMMETRIC ONLY)

| Aspect | ASM v3 | Modified Features (Symmetric) |
|--------|--------|-------------------------------|
| **Preprocessing** | dB + normalize | dB + normalize + 1/f + symmetric |
| **Feature Dim** | (121, 20) | (61, 20) |
| **Channels** | 1 | 1 |
| **Frequency Range** | -15 to +15 Hz | 0 to +15 Hz |
| **Symmetric** | N/A | Averaged: (|P+|+|P-|)/2 |
| **Asymmetric** | N/A | Not used |
| **Feature Count** | 2,420 | 1,220 (50% reduction) |
| **Model Dim** | 128 | 144 (+12.5%) |
| **Params** | ~800K | ~900K |
| **Bio-Inspired** | No | Yes (1/f, symmetric) |
| **Expected F1** | 0.87+ | **0.87-0.89** |

**Why symmetric-only might work**:
1. **1/f normalization is key**: Spectral balancing may be main driver
2. **Simpler is better**: Reduced complexity, easier optimization
3. **Averaging reduces noise**: More stable energy estimates
4. **Directional info redundant**: Energy sufficient for classification
5. **Model compensation**: Increased capacity offsets information loss

### Complementary Approaches

**ASM v3 Strategy**: "Better learning algorithm"
- Softer class weighting
- Contrastive loss for separation
- Minimal label smoothing

**Modified Features Strategy**: "Better input representation"
- 1/f normalization for spectral balance
- Symmetric/asymmetric decomposition for directionality
- Biologically-inspired preprocessing

**Combined Power**:
- If both work independently → potentially 0.88-0.89 F1
- If synergistic → could reach 0.90 F1
- If redundant → only incremental gain

## Ablation Study Design

### Experiments to Run

**1. Baseline Comparison**:
```bash
# ASM v3 (no modified features)
python STMasm_enhanced3.py 0

# Modified Features (this script)
python STMasm_feateng.py 0
```

**2. Feature Component Ablation**:
```python
# Experiment A: Only 1/f normalization
def preprocess_stm_features_1f_only(stm_2d):
    # Apply 1/f, but don't decompose
    frequency_vector = torch.linspace(-15.0, 15.0, 121)
    abs_freq = torch.abs(frequency_vector).view(1, 121, 1)
    return stm_2d * abs_freq

# Experiment B: Only symmetric/asymmetric (no 1/f)
def preprocess_stm_features_decomp_only(stm_2d):
    # Decompose without 1/f normalization
    # ...decomposition logic...
    
# Experiment C: Both (full pipeline)
# → Main script
```

**3. Component Weight Analysis**:
```python
# Try different concatenation orders
# [Asym | DC | Sym] vs [Sym | DC | Asym]
# Weight channels differently
# Zero out one channel to measure contribution
```

### Expected Ablation Results

**If 1/f normalization is key**:
```
Baseline:         0.8566 (v1)
Only 1/f:         0.87-0.88
Only decomp:      0.86-0.87
Both:             0.88-0.89
```

**If decomposition is key**:
```
Baseline:         0.8566
Only 1/f:         0.86-0.87
Only decomp:      0.87-0.88
Both:             0.88-0.89
```

**If both are redundant**:
```
Baseline:         0.8566
Only 1/f:         0.86-0.87
Only decomp:      0.86-0.87
Both:             0.86-0.87 (no synergy)
```

## Implementation Details

### Frequency Vector Generation

**Critical for Correctness**:
```python
# STM parameters from STM06_STMpreproc.py
xmin, xmax = 190, 310  # Indices in original 500-bin array
xrange = [-15, 15]     # Hz range
x_ds_factor = 1        # No downsampling on time axis

# After cropping: 121 bins
n_time = (xmax - xmin + 1) // x_ds_factor  # = 121

# Frequency vector
rate_min = -15.0
rate_max = 15.0
frequency_vector = torch.linspace(rate_min, rate_max, n_time)

# Resolution check
rate_resolution = (rate_max - rate_min) / (n_time - 1)
# = 30 / 120 = 0.25 Hz ✓ Correct
```

**Index Mapping (UPDATED)**:
```python
Original (121 bins):
    Index    Frequency (Hz)
    0        -15.00
    1        -14.75
    ...
    59       -0.25
    60       0.00          ← DC component
    61       +0.25
    ...
    120      +15.00

After Processing (61 bins, 0-15 Hz):
    Index    Frequency (Hz)
    0        0.00           ← DC
    1        0.25
    2        0.50
    ...
    60       15.00
```

### Asymmetric Map Rescaling (REMOVED)

**Previous approach** (now removed):
```python
# OLD: Per-sample rescaling to [-1, 1]
max_abs_asym = torch.abs(asymmetric_map).reshape(batch, -1).max(dim=1)[0]
asymmetric_map_rescaled = asymmetric_map / max_abs_asym
```

**New approach** (current):
```python
# NEW: Keep raw asymmetric differences
asymmetric_map = torch.abs(positive_chunk) - torch.abs(negative_flipped)
# No rescaling - let z-normalization handle scaling
```

**Rationale for change**:
1. **Preserve natural signal strength**: Rescaling removes information about magnitude
2. **Z-normalization is sufficient**: Per-sample normalization handles scaling
3. **Avoid artificial equalization**: Different samples naturally have different directional strengths
4. **More informative gradients**: Raw differences provide clearer learning signal

**Benefits**:
- Simpler preprocessing pipeline
- Preserves relative magnitude information across samples
- Z-normalization (mean=0, std=1) provides adequate scaling
- More natural representation of directional preference

### Channel Stacking (NEW)

**Implementation**:
```python
# After computing both maps with DC prepended
symmetric_with_dc = torch.cat([dc_component, symmetric_map], dim=1)     # (batch, 61, 20)
asymmetric_with_dc = torch.cat([dc_component, asymmetric_map], dim=1)  # (batch, 61, 20)

# Stack into 2-channel tensor
processed = torch.stack([symmetric_with_dc, asymmetric_with_dc], dim=-1)  # (batch, 61, 20, 2)
```

**Dimension semantics**:
```
processed[b, f, s, 0] = Symmetric map at frequency f, scale s, sample b
processed[b, f, s, 1] = Asymmetric map at frequency f, scale s, sample b
```

**Why stack instead of concatenate**:
```python
# Option A: Concatenate along frequency (NOT USED)
concat_freq = torch.cat([symmetric_with_dc, asymmetric_with_dc], dim=1)  # (batch, 122, 20)
# Issues:
#   - Treats channels as additional frequencies (semantically wrong)
#   - Harder for Conv2d to learn cross-channel interactions
#   - Loses explicit channel structure

# Option B: Stack as separate channel dimension (USED) ✓
stack_channels = torch.stack([symmetric_with_dc, asymmetric_with_dc], dim=-1)  # (batch, 61, 20, 2)
# Benefits:
#   - Natural input for Conv2d: (batch, channels, height, width)
#   - Explicit channel separation
#   - Conv2d learns cross-channel patterns naturally
#   - Standard in computer vision (RGB images, etc.)
```

### Per-Sample Normalization (UPDATED)

**After preprocessing, normalize across all dimensions**:
```python
# STM_processed: (batch, 61, 20)
means = STM_processed.mean(dim=(1, 2), keepdim=True)  # (batch, 1, 1)
stds = STM_processed.std(dim=(1, 2), keepdim=True)    # (batch, 1, 1)
STM_processed = (STM_processed - means) / (stds + 1e-8)
```

**Effect**:
- Centers each sample to mean=0, std=1
- Applied AFTER symmetric averaging
- Ensures both channels have comparable scale after z-normalization
- Prevents any remaining magnitude imbalance

## Integration with ASM Architecture

### Data Flow (SYMMETRIC ONLY)

```
Raw STM (flattened) → Reshape to 2D (121, 20)
                   ↓
         1/f Normalization (by rate)
                   ↓
         Separate Negative/Positive Rates
                   ↓
         Flip & Align
                   ↓
         Compute Symmetric Map (averaged energy)
                   ↓
         Concatenate DC with symmetric map
                   ↓
         Single-channel tensor (61, 20)
                   ↓
         Per-sample Z-normalization
                   ↓
         (batch, 61, 20) → ASM Model
```

### Model Architecture (SYMMETRIC ONLY)

**Key changes for single-channel input with increased capacity**:
```python
ModifiedFeatureASMClassifier(
    time_steps=20,      # Scale dimension
    freq_steps=61,      # Frequency dimension (0-15 Hz)
    num_classes=6,
    dim=144,           # Increased from 128
    num_blocks=4,
    shift_range=2,
    expansion_factor=2,
    dropout=0.1
)
```

**Input Processing**:
```python
def forward(self, x, return_features=False):
    # x: (batch, 61, 20) - modified STM features
    
    # Add channel dimension: (batch, 1, 61, 20)
    x = x.unsqueeze(1)
    
    x = self.spec_augment(x)
    
    # Input projection: 1 input channel, increased first layer
    x = self.input_proj(x)  # Conv2d(1, dim//3, ...) → Conv2d(dim//3, dim, ...)
    
    # ...rest unchanged...
```

**Updated Input Projection Layer**:
```python
self.input_proj = nn.Sequential(
    nn.Conv2d(1, dim // 3, kernel_size=3, padding=1),  # 1 input channel, dim//3 = 48
    nn.BatchNorm2d(dim // 3),
    nn.GELU(),
    nn.Conv2d(dim // 3, dim, kernel_size=3, padding=1),
    nn.BatchNorm2d(dim),
    nn.GELU()
)
```

**Parameter count**: ~900K (slightly more than v3 due to dim=144)

## Training Configuration

### Same as ASM v3 (Proven Recipe)

**Loss Function**:
```python
ContrastiveFocalLoss(
    alpha=confusion_aware_weights,  # sqrt-based
    gamma=2.0,
    label_smoothing=0.01,
    contrastive_weight=0.1,
    similar_pairs=[(0, 1), (2, 3)]
)
```

**Optimizer & Scheduler**:
```python
AdamW(lr=1e-3, weight_decay=1e-4)
CosineAnnealingWarmRestarts(T_0=10, T_mult=2, eta_min=1e-6)
Warmup: 5 epochs
```

**Hyperparameters**:
```python
batch_size = 128
num_epochs = 50
gradient_clip_norm = 1.0
```

## Expected Performance Improvements

### Hypothesis 1: Better High-Frequency Discrimination

**Prediction**:
- Classes with high-rate modulations benefit from 1/f correction
- **Music classes** (2, 3): Often have rapid modulations → should improve
- **Speech classes** (0, 1): Slower modulations → less impact

**Target**:
```
Class 2 (music: vocal):     0.84 → 0.86 (+2 points)
Class 3 (music: non-vocal): 0.74 → 0.77 (+3 points)
```

### Hypothesis 2: Directional Discrimination (UPDATED)

**Prediction**:
- **Rescaled asymmetric channel** ensures equal contribution to learning
- Averaging symmetric map prevents magnitude imbalance
- 2-channel structure allows Conv2d to learn cross-channel patterns

**Expected benefits**:
```
Symmetric channel (averaged energy):
    - More stable estimates (averaging reduces noise)
    - Comparable magnitude to asymmetric channel
    - Better for classes with strong overall modulation

Asymmetric channel (rescaled direction):
    - Normalized importance across samples
    - Prevents outlier samples from dominating gradients
    - Better for classes with directional preferences

Combined:
    - Conv2d can learn complementary features from both channels
    - Cross-channel interactions captured by spatial convolutions
    - More balanced feature learning
```

**Target**:
```
Speech classes (0, 1):  Better separation via prosodic directionality
Music classes (2, 3):   Better separation via melodic energy+direction
Env classes (4, 5):     Maintain performance (less directional)
```

### Hypothesis 3: Overall Performance (UPDATED)

**Conservative Target**:
```
Test Macro F1: 0.87-0.88 (match or slightly exceed ASM v3)

Per-Class F1:
  Class 0: 0.95-0.96 (maintain)
  Class 1: 0.82-0.85 (+1-4 points over v3) ← Improved via rescaling
  Class 2: 0.85-0.87 (+1-3 points over v3) ← Improved via 2-channel
  Class 3: 0.76-0.79 (+2-5 points over v3) ← KEY IMPROVEMENT
  Class 4: 0.92-0.94 (maintain)
  Class 5: 0.91-0.93 (maintain)
```

**Optimistic Target**:
```
Test Macro F1: 0.88-0.90 (beat ASM v3 by 1-3 points)
Class 3 F1 ≥ 0.78 (break through 0.75 barrier)
All classes F1 ≥ 0.80 (uniform improvement)
```

**Why this version might perform better**:
1. **Balanced channels**: Rescaling prevents asymmetric channel from being ignored
2. **Reduced redundancy**: Single-sided spectrum (0-15 Hz) is more efficient
3. **Better Conv2d input**: 2 channels vs 1 channel with artificial concatenation
4. **Stable gradients**: Normalized ranges prevent gradient issues

## Comparison with ASM v3

### Key Differences (SYMMETRIC ONLY)

| Aspect | ASM v3 | Modified Features (Symmetric) |
|--------|--------|-------------------------------|
| **Preprocessing** | dB + normalize | dB + normalize + 1/f + symmetric |
| **Feature Dim** | (121, 20) | (61, 20) |
| **Channels** | 1 | 1 |
| **Frequency Range** | -15 to +15 Hz | 0 to +15 Hz |
| **Symmetric** | N/A | Averaged: (|P+|+|P-|)/2 |
| **Asymmetric** | N/A | Not used |
| **Feature Count** | 2,420 | 1,220 (50% reduction) |
| **Model Dim** | 128 | 144 (+12.5%) |
| **Params** | ~800K | ~900K |
| **Bio-Inspired** | No | Yes (1/f, symmetric) |
| **Expected F1** | 0.87+ | **0.87-0.89** |

**Why symmetric-only might work**:
1. **1/f normalization is key**: Spectral balancing may be main driver
2. **Simpler is better**: Reduced complexity, easier optimization
3. **Averaging reduces noise**: More stable energy estimates
4. **Directional info redundant**: Energy sufficient for classification
5. **Model compensation**: Increased capacity offsets information loss

### Complementary Approaches

**ASM v3 Strategy**: "Better learning algorithm"
- Softer class weighting
- Contrastive loss for separation
- Minimal label smoothing

**Modified Features Strategy**: "Better input representation"
- 1/f normalization for spectral balance
- Symmetric/asymmetric decomposition for directionality
- Biologically-inspired preprocessing

**Combined Power**:
- If both work independently → potentially 0.88-0.89 F1
- If synergistic → could reach 0.90 F1
- If redundant → only incremental gain

## Ablation Study Design

### Experiments to Run

**1. Baseline Comparison**:
```bash
# ASM v3 (no modified features)
python STMasm_enhanced3.py 0

# Modified Features (this script)
python STMasm_feateng.py 0
```

**2. Feature Component Ablation**:
```python
# Experiment A: Only 1/f normalization
def preprocess_stm_features_1f_only(stm_2d):
    # Apply 1/f, but don't decompose
    frequency_vector = torch.linspace(-15.0, 15.0, 121)
    abs_freq = torch.abs(frequency_vector).view(1, 121, 1)
    return stm_2d * abs_freq

# Experiment B: Only symmetric/asymmetric (no 1/f)
def preprocess_stm_features_decomp_only(stm_2d):
    # Decompose without 1/f normalization
    # ...decomposition logic...
    
# Experiment C: Both (full pipeline)
# → Main script
```

**3. Component Weight Analysis**:
```python
# Try different concatenation orders
# [Asym | DC | Sym] vs [Sym | DC | Asym]
# Weight channels differently
# Zero out one channel to measure contribution
```

### Expected Ablation Results

**If 1/f normalization is key**:
```
Baseline:         0.8566 (v1)
Only 1/f:         0.87-0.88
Only decomp:      0.86-0.87
Both:             0.88-0.89
```

**If decomposition is key**:
```
Baseline:         0.8566
Only 1/f:         0.86-0.87
Only decomp:      0.87-0.88
Both:             0.88-0.89
```

**If both are redundant**:
```
Baseline:         0.8566
Only 1/f:         0.86-0.87
Only decomp:      0.86-0.87
Both:             0.86-0.87 (no synergy)
```

## Implementation Details

### Frequency Vector Generation

**Critical for Correctness**:
```python
# STM parameters from STM06_STMpreproc.py
xmin, xmax = 190, 310  # Indices in original 500-bin array
xrange = [-15, 15]     # Hz range
x_ds_factor = 1        # No downsampling on time axis

# After cropping: 121 bins
n_time = (xmax - xmin + 1) // x_ds_factor  # = 121

# Frequency vector
rate_min = -15.0
rate_max = 15.0
frequency_vector = torch.linspace(rate_min, rate_max, n_time)

# Resolution check
rate_resolution = (rate_max - rate_min) / (n_time - 1)
# = 30 / 120 = 0.25 Hz ✓ Correct
```

**Index Mapping (UPDATED)**:
```python
Original (121 bins):
    Index    Frequency (Hz)
    0        -15.00
    1        -14.75
    ...
    59       -0.25
    60       0.00          ← DC component
    61       +0.25
    ...
    120      +15.00

After Processing (61 bins, 0-15 Hz):
    Index    Frequency (Hz)
    0        0.00           ← DC
    1        0.25
    2        0.50
    ...
    60       15.00
```

### Asymmetric Map Rescaling (REMOVED)

**Previous approach** (now removed):
```python
# OLD: Per-sample rescaling to [-1, 1]
max_abs_asym = torch.abs(asymmetric_map).reshape(batch, -1).max(dim=1)[0]
asymmetric_map_rescaled = asymmetric_map / max_abs_asym
```

**New approach** (current):
```python
# NEW: Keep raw asymmetric differences
asymmetric_map = torch.abs(positive_chunk) - torch.abs(negative_flipped)
# No rescaling - let z-normalization handle scaling
```

**Rationale for change**:
1. **Preserve natural signal strength**: Rescaling removes information about magnitude
2. **Z-normalization is sufficient**: Per-sample normalization handles scaling
3. **Avoid artificial equalization**: Different samples naturally have different directional strengths
4. **More informative gradients**: Raw differences provide clearer learning signal

**Benefits**:
- Simpler preprocessing pipeline
- Preserves relative magnitude information across samples
- Z-normalization (mean=0, std=1) provides adequate scaling
- More natural representation of directional preference

### Channel Stacking (NEW)

**Implementation**:
```python
# After computing both maps with DC prepended
symmetric_with_dc = torch.cat([dc_component, symmetric_map], dim=1)     # (batch, 61, 20)
asymmetric_with_dc = torch.cat([dc_component, asymmetric_map], dim=1)  # (batch, 61, 20)

# Stack into 2-channel tensor
processed = torch.stack([symmetric_with_dc, asymmetric_with_dc], dim=-1)  # (batch, 61, 20, 2)
```

**Dimension semantics**:
```
processed[b, f, s, 0] = Symmetric map at frequency f, scale s, sample b
processed[b, f, s, 1] = Asymmetric map at frequency f, scale s, sample b
```

**Why stack instead of concatenate**:
```python
# Option A: Concatenate along frequency (NOT USED)
concat_freq = torch.cat([symmetric_with_dc, asymmetric_with_dc], dim=1)  # (batch, 122, 20)
# Issues:
#   - Treats channels as additional frequencies (semantically wrong)
#   - Harder for Conv2d to learn cross-channel interactions
#   - Loses explicit channel structure

# Option B: Stack as separate channel dimension (USED) ✓
stack_channels = torch.stack([symmetric_with_dc, asymmetric_with_dc], dim=-1)  # (batch, 61, 20, 2)
# Benefits:
#   - Natural input for Conv2d: (batch, channels, height, width)
#   - Explicit channel separation
#   - Conv2d learns cross-channel patterns naturally
#   - Standard in computer vision (RGB images, etc.)
```

### Per-Sample Normalization (UPDATED)

**After preprocessing, normalize across all dimensions**:
```python
# STM_processed: (batch, 61, 20)
means = STM_processed.mean(dim=(1, 2), keepdim=True)  # (batch, 1, 1)
stds = STM_processed.std(dim=(1, 2), keepdim=True)    # (batch, 1, 1)
STM_processed = (STM_processed - means) / (stds + 1e-8)
```

**Effect**:
- Centers each sample to mean=0, std=1
- Applied AFTER symmetric averaging
- Ensures both channels have comparable scale after z-normalization
- Prevents any remaining magnitude imbalance

## Integration with ASM Architecture

### Data Flow (SYMMETRIC ONLY)

```
Raw STM (flattened) → Reshape to 2D (121, 20)
                   ↓
         1/f Normalization (by rate)
                   ↓
         Separate Negative/Positive Rates
                   ↓
         Flip & Align
                   ↓
         Compute Symmetric Map (averaged energy)
                   ↓
         Concatenate DC with symmetric map
                   ↓
         Single-channel tensor (61, 20)
                   ↓
         Per-sample Z-normalization
                   ↓
         (batch, 61, 20) → ASM Model
```

### Model Architecture (SYMMETRIC ONLY)

**Key changes for single-channel input with increased capacity**:
```python
ModifiedFeatureASMClassifier(
    time_steps=20,      # Scale dimension
    freq_steps=61,      # Frequency dimension (0-15 Hz)
    num_classes=6,
    dim=144,           # Increased from 128
    num_blocks=4,
    shift_range=2,
    expansion_factor=2,
    dropout=0.1
)
```

**Input Processing**:
```python
def forward(self, x, return_features=False):
    # x: (batch, 61, 20) - modified STM features
    
    # Add channel dimension: (batch, 1, 61, 20)
    x = x.unsqueeze(1)
    
    x = self.spec_augment(x)
    
    # Input projection: 1 input channel, increased first layer
    x = self.input_proj(x)  # Conv2d(1, dim//3, ...) → Conv2d(dim//3, dim, ...)
    
    # ...rest unchanged...
```

**Updated Input Projection Layer**:
```python
self.input_proj = nn.Sequential(
    nn.Conv2d(1, dim // 3, kernel_size=3, padding=1),  # 1 input channel, dim//3 = 48
    nn.BatchNorm2d(dim // 3),
    nn.GELU(),
    nn.Conv2d(dim // 3, dim, kernel_size=3, padding=1),
    nn.BatchNorm2d(dim),
    nn.GELU()
)
```

**Parameter count**: ~900K (slightly more than v3 due to dim=144)

## Training Configuration

### Same as ASM v3 (Proven Recipe)

**Loss Function**:
```python
ContrastiveFocalLoss(
    alpha=confusion_aware_weights,  # sqrt-based
    gamma=2.0,
    label_smoothing=0.01,
    contrastive_weight=0.1,
    similar_pairs=[(0, 1), (2, 3)]
)
```

**Optimizer & Scheduler**:
```python
AdamW(lr=1e-3, weight_decay=1e-4)
CosineAnnealingWarmRestarts(T_0=10, T_mult=2, eta_min=1e-6)
Warmup: 5 epochs
```

**Hyperparameters**:
```python
batch_size = 128
num_epochs = 50
gradient_clip_norm = 1.0
```

## Expected Performance Improvements

### Hypothesis 1: Better High-Frequency Discrimination

**Prediction**:
- Classes with high-rate modulations benefit from 1/f correction
- **Music classes** (2, 3): Often have rapid modulations → should improve
- **Speech classes** (0, 1): Slower modulations → less impact

**Target**:
```
Class 2 (music: vocal):     0.84 → 0.86 (+2 points)
Class 3 (music: non-vocal): 0.74 → 0.77 (+3 points)
```

### Hypothesis 2: Directional Discrimination (UPDATED)

**Prediction**:
- **Rescaled asymmetric channel** ensures equal contribution to learning
- Averaging symmetric map prevents magnitude imbalance
- 2-channel structure allows Conv2d to learn cross-channel patterns

**Expected benefits**:
```
Symmetric channel (averaged energy):
    - More stable estimates (averaging reduces noise)
    - Comparable magnitude to asymmetric channel
    - Better for classes with strong overall modulation

Asymmetric channel (rescaled direction):
    - Normalized importance across samples
    - Prevents outlier samples from dominating gradients
    - Better for classes with directional preferences

Combined:
    - Conv2d can learn complementary features from both channels
    - Cross-channel interactions captured by spatial convolutions
    - More balanced feature learning
```

**Target**:
```
Speech classes (0, 1):  Better separation via prosodic directionality
Music classes (2, 3):   Better separation via melodic energy+direction
Env classes (4, 5):     Maintain performance (less directional)
```

### Hypothesis 3: Overall Performance (UPDATED)

**Conservative Target**:
```
Test Macro F1: 0.87-0.88 (match or slightly exceed ASM v3)

Per-Class F1:
  Class 0: 0.95-0.96 (maintain)
  Class 1: 0.82-0.85 (+1-4 points over v3) ← Improved via rescaling
  Class 2: 0.85-0.87 (+1-3 points over v3) ← Improved via 2-channel
  Class 3: 0.76-0.79 (+2-5 points over v3) ← KEY IMPROVEMENT
  Class 4: 0.92-0.94 (maintain)
  Class 5: 0.91-0.93 (maintain)
```

**Optimistic Target**:
```
Test Macro F1: 0.88-0.90 (beat ASM v3 by 1-3 points)
Class 3 F1 ≥ 0.78 (break through 0.75 barrier)
All classes F1 ≥ 0.80 (uniform improvement)
```

**Why this version might perform better**:
1. **Balanced channels**: Rescaling prevents asymmetric channel from being ignored
2. **Reduced redundancy**: Single-sided spectrum (0-15 Hz) is more efficient
3. **Better Conv2d input**: 2 channels vs 1 channel with artificial concatenation
4. **Stable gradients**: Normalized ranges prevent gradient issues

## Comparison with ASM v3

### Key Differences (SYMMETRIC ONLY)

| Aspect | ASM v3 | Modified Features (Symmetric) |
|--------|--------|-------------------------------|
| **Preprocessing** | dB + normalize | dB + normalize + 1/f + symmetric |
| **Feature Dim** | (121, 20) | (61, 20) |
| **Channels** | 1 | 1 |
| **Frequency Range** | -15 to +15 Hz | 0 to +15 Hz |
| **Symmetric** | N/A | Averaged: (|P+|+|P-|)/2 |
| **Asymmetric** | N/A | Not used |
| **Feature Count** | 2,420 | 1,220 (50% reduction) |
| **Model Dim** | 128 | 144 (+12.5%) |
| **Params** | ~800K | ~900K |
| **Bio-Inspired** | No | Yes (1/f, symmetric) |
| **Expected F1** | 0.87+ | **0.87-0.89** |

**Why symmetric-only might work**:
1. **1/f normalization is key**: Spectral balancing may be main driver
2. **Simpler is better**: Reduced complexity, easier optimization
3. **Averaging reduces noise**: More stable energy estimates
4. **Directional info redundant**: Energy sufficient for classification
5. **Model compensation**: Increased capacity offsets information loss

### Complementary Approaches

**ASM v3 Strategy**: "Better learning algorithm"
- Softer class weighting
- Contrastive loss for separation
- Minimal label smoothing

**Modified Features Strategy**: "Better input representation"
- 1/f normalization for spectral balance
- Symmetric/asymmetric decomposition for directionality
- Biologically-inspired preprocessing

**Combined Power**:
- If both work independently → potentially 0.88-0.89 F1
- If synergistic → could reach 0.90 F1
- If redundant → only incremental gain

## Ablation Study Design

### Experiments to Run

**1. Baseline Comparison**:
```bash
# ASM v3 (no modified features)
python STMasm_enhanced3.py 0

# Modified Features (this script)
python STMasm_feateng.py 0
```

**2. Feature Component Ablation**:
```python
# Experiment A: Only 1/f normalization
def preprocess_stm_features_1f_only(stm_2d):
    # Apply 1/f, but don't decompose
    frequency_vector = torch.linspace(-15.0, 15.0, 121)
    abs_freq = torch.abs(frequency_vector).view(1, 121, 1)
    return stm_2d * abs_freq

# Experiment B: Only symmetric/asymmetric (no 1/f)
def preprocess_stm_features_decomp_only(stm_2d):
    # Decompose without 1/f normalization
    # ...decomposition logic...
    
# Experiment C: Both (full pipeline)
# → Main script
```

**3. Component Weight Analysis**:
```python
# Try different concatenation orders
# [Asym | DC | Sym] vs [Sym | DC | Asym]
# Weight channels differently
# Zero out one channel to measure contribution
```

### Expected Ablation Results

**If 1/f normalization is key**:
```
Baseline:         0.8566 (v1)
Only 1/f:         0.87-0.88
Only decomp:      0.86-0.87
Both:             0.88-0.89
```

**If decomposition is key**:
```
Baseline:         0.8566
Only 1/f:         0.86-0.87
Only decomp:      0.87-0.88
Both:             0.88-0.89
```

**If both are redundant**:
```
Baseline:         0.8566
Only 1/f:         0.86-0.87
Only decomp:      0.86-0.87
Both:             0.86-0.87 (no synergy)
```

## Implementation Details

### Frequency Vector Generation

**Critical for Correctness**:
```python
# STM parameters from STM06_STMpreproc.py
xmin, xmax = 190, 310  # Indices in original 500-bin array
xrange = [-15, 15]     # Hz range
x_ds_factor = 1        # No downsampling on time axis

# After cropping: 121 bins
n_time = (xmax - xmin + 1) // x_ds_factor  # = 121

# Frequency vector
rate_min = -15.0
rate_max = 15.0
frequency_vector = torch.linspace(rate_min, rate_max, n_time)

# Resolution check
rate_resolution = (rate_max - rate_min) / (n_time - 1)
# = 30 / 120 = 0.25 Hz ✓ Correct
```

**Index Mapping (UPDATED)**:
```python
Original (121 bins):
    Index    Frequency (Hz)
    0        -15.00
    1        -14.75
    ...
    59       -0.25
    60       0.00          ← DC component
    61       +0.25
    ...
    120      +15.00

After Processing (61 bins, 0-15 Hz):
    Index    Frequency (Hz)
    0        0.00           ← DC
    1        0.25
    2        0.50
    ...
    60       15.00
```

### Asymmetric Map Rescaling (REMOVED)

**Previous approach** (now removed):
```python
# OLD: Per-sample rescaling to [-1, 1]
max_abs_asym = torch.abs(asymmetric_map).reshape(batch, -1).max(dim=1)[0]
asymmetric_map_rescaled = asymmetric_map / max_abs_asym
```

**New approach** (current):
```python
# NEW: Keep raw asymmetric differences
asymmetric_map = torch.abs(positive_chunk) - torch.abs(negative_flipped)
# No rescaling - let z-normalization handle scaling
```

**Rationale for change**:
1. **Preserve natural signal strength**: Rescaling removes information about magnitude
2. **Z-normalization is sufficient**: Per-sample normalization handles scaling
3. **Avoid artificial equalization**: Different samples naturally have different directional strengths
4. **More informative gradients**: Raw differences provide clearer learning signal

**Benefits**:
- Simpler preprocessing pipeline
- Preserves relative magnitude information across samples
- Z-normalization (mean=0, std=1) provides adequate scaling
- More natural representation of directional preference

### Channel Stacking (NEW)

**Implementation**:
```python
# After computing both maps with DC prepended
symmetric_with_dc = torch.cat([dc_component, symmetric_map], dim=1)     # (batch, 61, 20)
asymmetric_with_dc = torch.cat([dc_component, asymmetric_map], dim=1)  # (batch, 61, 20)

# Stack into 2-channel tensor
processed = torch.stack([symmetric_with_dc, asymmetric_with_dc], dim=-1)  # (batch, 61, 20, 2)
```

**Dimension semantics**:
```
processed[b, f, s, 0] = Symmetric map at frequency f, scale s, sample b
processed[b, f, s, 1] = Asymmetric map at frequency f, scale s, sample b
```

**Why stack instead of concatenate**:
```python
# Option A: Concatenate along frequency (NOT USED)
concat_freq = torch.cat([symmetric_with_dc, asymmetric_with_dc], dim=1)  # (batch, 122, 20)
# Issues:
#   - Treats channels as additional frequencies (semantically wrong)
#   - Harder for Conv2d to learn cross-channel interactions
#   - Loses explicit channel structure

# Option B: Stack as separate channel dimension (USED) ✓
stack_channels = torch.stack([symmetric_with_dc, asymmetric_with_dc], dim=-1)  # (batch, 61, 20, 2)
# Benefits:
#   - Natural input for Conv2d: (batch, channels, height, width)
#   - Explicit channel separation
#   - Conv2d learns cross-channel patterns naturally
#   - Standard in computer vision (RGB images, etc.)
```

### Per-Sample Normalization (UPDATED)

**After preprocessing, normalize across all dimensions**:
```python
# STM_processed: (batch, 61, 20)
means = STM_processed.mean(dim=(1, 2), keepdim=True)  # (batch, 1, 1)
stds = STM_processed.std(dim=(1, 2), keepdim=True)    # (batch, 1, 1)
STM_processed = (STM_processed - means) / (stds + 1e-8)
```

**Effect**:
- Centers each sample to mean=0, std=1
- Applied AFTER symmetric averaging
- Ensures both channels have comparable scale after z-normalization
- Prevents any remaining magnitude imbalance

## Integration with ASM Architecture

### Data Flow (SYMMETRIC ONLY)

```
Raw STM (flattened) → Reshape to 2D (121, 20)
                   ↓
         1/f Normalization (by rate)
                   ↓
         Separate Negative/Positive Rates
                   ↓
         Flip & Align
                   ↓
         Compute Symmetric Map (averaged energy)
                   ↓
         Concatenate DC with symmetric map
                   ↓
         Single-channel tensor (61, 20)
                   ↓
         Per-sample Z-normalization
                   ↓
         (batch, 61, 20) → ASM Model
```

### Model Architecture (SYMMETRIC ONLY)

**Key changes for single-channel input with increased capacity**:
```python
ModifiedFeatureASMClassifier(
    time_steps=20,      # Scale dimension
    freq_steps=61,      # Frequency dimension (0-15 Hz)
    num_classes=6,
    dim=144,           # Increased from 128
    num_blocks=4,
    shift_range=2,
    expansion_factor=2,
    dropout=0.1
)
```

**Input Processing**:
```python
def forward(self, x, return_features=False):
    # x: (batch, 61, 20) - modified STM features
    
    # Add channel dimension: (batch, 1, 61, 20)
    x = x.unsqueeze(1)
    
    x = self.spec_augment(x)
    
    # Input projection: 1 input channel, increased first layer
    x = self.input_proj(x)  # Conv2d(1, dim//3, ...) → Conv2d(dim//3, dim, ...)
    
    # ...rest unchanged...
```

**Updated Input Projection Layer**:
```python
self.input_proj = nn.Sequential(
    nn.Conv2d(1, dim // 3, kernel_size=3, padding=1),  # 1 input channel, dim//3 = 48
    nn.BatchNorm2d(dim // 3),
    nn.GELU(),
    nn.Conv2d(dim // 3, dim, kernel_size=3, padding=1),
    nn.BatchNorm2d(dim),
    nn.GELU()
)
```

**Parameter count**: ~900K (slightly more than v3 due to dim=144)

## Training Configuration

### Same as ASM v3 (Proven Recipe)

**Loss Function**:
```python
ContrastiveFocalLoss(
    alpha=confusion_aware_weights,  # sqrt-based
    gamma=2.0,
    label_smoothing=0.01,
    contrastive_weight=0.1,
    similar_pairs=[(0, 1), (2, 3)]
)
```

**Optimizer & Scheduler**:
```python
AdamW(lr=1e-3, weight_decay=1e-4)
CosineAnnealingWarmRestarts(T_0=10, T_mult=2, eta_min=1e-6)
Warmup: 5 epochs
```

**Hyperparameters**:
```python
batch_size = 128
num_epochs = 50
gradient_clip_norm = 1.0
```

## Expected Performance Improvements

### Hypothesis 1: Better High-Frequency Discrimination

**Prediction**:
- Classes with high-rate modulations benefit from 1/f correction
- **Music classes** (2, 3): Often have rapid modulations → should improve
- **Speech classes** (0, 1): Slower modulations → less impact

**Target**:
```
Class 2 (music: vocal):     0.84 → 0.86 (+2 points)
Class 3 (music: non-vocal): 0.74 → 0.77 (+3 points)
```

### Hypothesis 2: Directional Discrimination (UPDATED)

**Prediction**:
- **Rescaled asymmetric channel** ensures equal contribution to learning
- Averaging symmetric map prevents magnitude imbalance
- 2-channel structure allows Conv2d to learn cross-channel patterns

**Expected benefits**:
```
Symmetric channel (averaged energy):
    - More stable estimates (averaging reduces noise)
    - Comparable magnitude to asymmetric channel
    - Better for classes with strong overall modulation

Asymmetric channel (rescaled direction):
    - Normalized importance across samples
    - Prevents outlier samples from dominating gradients
    - Better for classes with directional preferences

Combined:
    - Conv2d can learn complementary features from both channels
    - Cross-channel interactions captured by spatial convolutions
    - More balanced feature learning
```

**Target**:
```
Speech classes (0, 1):  Better separation via prosodic directionality
Music classes (2, 3):   Better separation via melodic energy+direction
Env classes (4, 5):     Maintain performance (less directional)
```

### Hypothesis 3: Overall Performance (UPDATED)

**Conservative Target**:
```
Test Macro F1: 0.87-0.88 (match or slightly exceed ASM v3)

Per-Class F1:
  Class 0: 0.95-0.96 (maintain)
  Class 1: 0.82-0.85 (+1-4 points over v3) ← Improved via rescaling
  Class 2: 0.85-0.87 (+1-3 points over v3) ← Improved via 2-channel
  Class 3: 0.76-0.79 (+2-5 points over v3) ← KEY IMPROVEMENT
  Class 4: 0.92-0.94 (maintain)
  Class 5: 0.91-0.93 (maintain)
```

**Optimistic Target**:
```
Test Macro F1: 0.88-0.90 (beat ASM v3 by 1-3 points)
Class 3 F1 ≥ 0.78 (break through 0.75 barrier)
All classes F1 ≥ 0.80 (uniform improvement)
```

**Why this version might perform better**:
1. **Balanced channels**: Rescaling prevents asymmetric channel from being ignored
2. **Reduced redundancy**: Single-sided spectrum (0-15 Hz) is more efficient
3. **Better Conv2d input**: 2 channels vs 1 channel with artificial concatenation
4. **Stable gradients**: Normalized ranges prevent gradient issues

## Comparison with ASM v3

### Key Differences (SYMMETRIC ONLY)

| Aspect | ASM v3 | Modified Features (Symmetric) |
|--------|--------|-------------------------------|
| **Preprocessing** | dB + normalize | dB + normalize + 1/f + symmetric |
| **Feature Dim** | (121, 20) | (61, 20) |
| **Channels** | 1 | 1 |
| **Frequency Range** | -15 to +15 Hz | 0 to +15 Hz |
| **Symmetric** | N/A | Averaged: (|P+|+|P-|)/2 |
| **Asymmetric** | N/A | Not used |
| **Feature Count** | 2,420 | 1,220 (50% reduction) |
| **Model Dim** | 128 | 144 (+12.5%) |
| **Params** | ~800K | ~900K |
| **Bio-Inspired** | No | Yes (1/f, symmetric) |
| **Expected F1** | 0.87+ | **0.87-0.89** |

**Why symmetric-only might work**:
1. **1/f normalization is key**: Spectral balancing may be main driver
2. **Simpler is better**: Reduced complexity, easier optimization
3. **Averaging reduces noise**: More stable energy estimates
4. **Directional info redundant**: Energy sufficient for classification
5. **Model compensation**: Increased capacity offsets information loss

### Complementary Approaches

**ASM v3 Strategy**: "Better learning algorithm"
- Softer class weighting
- Contrastive loss for separation
- Minimal label smoothing

**Modified Features Strategy**: "Better input representation"
- 1/f normalization for spectral balance
- Symmetric/asymmetric decomposition for directionality
- Biologically-inspired preprocessing

**Combined Power**:
- If both work independently → potentially 0.88-0.89 F1
- If synergistic → could reach 0.90 F1
- If redundant → only incremental gain

## Ablation Study Design

### Experiments to Run

**1. Baseline Comparison**:
```bash
# ASM v3 (no modified features)
python STMasm_enhanced3.py 0

# Modified Features (this script)
python STMasm_feateng.py 0
```

**2. Feature Component Ablation**:
```python
# Experiment A: Only 1/f normalization
def preprocess_stm_features_1f_only(stm_2d):
    # Apply 1/f, but don't decompose
    frequency_vector = torch.linspace(-15.0, 15.0, 121)
    abs_freq = torch.abs(frequency_vector).view(1, 121, 1)
    return stm_2d * abs_freq

# Experiment B: Only symmetric/asymmetric (no 1/f)
def preprocess_stm_features_decomp_only(stm_2d):
    # Decompose without 1/f normalization
    # ...decomposition logic...
    
# Experiment C: Both (full pipeline)
# → Main script
```

**3. Component Weight Analysis**:
```python
# Try different concatenation orders
# [Asym | DC | Sym] vs [Sym | DC | Asym]
# Weight channels differently
# Zero out one channel to measure contribution
```

### Expected Ablation Results

**If 1/f normalization is key**:
```
Baseline:         0.8566 (v1)
Only 1/f:         0.87-0.88
Only decomp:      0.86-0.87
Both:             0.88-0.89
```

**If decomposition is key**:
```
Baseline:         0.8566
Only 1/f:         0.86-0.87
Only decomp:      0.87-0.88
Both:             0.88-0.89
```

**If both are redundant**:
```
Baseline:         0.8566
Only 1/f:         0.86-0.87
Only decomp:      0.86-0.87
Both:             0.86-0.87 (no synergy)
```

## Implementation Details

### Frequency Vector Generation

**Critical for Correctness**:
```python
# STM parameters from STM06_STMpreproc.py
xmin, xmax = 190, 310  # Indices in original 500-bin array
xrange = [-15, 15]     # Hz range
x_ds_factor = 1        # No downsampling on time axis

# After cropping: 121 bins
n_time = (xmax - xmin + 1) // x_ds_factor  # = 121
