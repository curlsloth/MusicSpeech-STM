# ASM Modified Features Summary

## Executive Summary

**ASM Modified Features** implements custom STM preprocessing inspired by auditory neuroscience:
- **Root Hypothesis**: Raw STM features treat all frequencies equally, but auditory systems have 1/f response characteristics
- **Solution**: Apply 1/f normalization + symmetric/asymmetric decomposition to enhance directional information
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

### Step 2: Symmetric/Asymmetric Decomposition

**Conceptual Framework**:
```
Rate Axis Structure:
    Negative rates (0-59):   Up-sweeps    (-15 Hz to -0.25 Hz)
    DC component (60):       No modulation (0 Hz)
    Positive rates (61-120): Down-sweeps  (+0.25 Hz to +15 Hz)

Hypothesis:
    - Symmetric information: Total energy regardless of direction
    - Asymmetric information: Directional preference (up vs down)
```

**Mathematical Formulation**:
```
Let:
    P⁻(ω, s) = power at negative rate ω (indices 0-59)
    P⁺(ω, s) = power at positive rate ω (indices 61-120)

After flipping negative chunk to align:
    P⁻_flipped(i, s) = P⁻(59-i, s)  for i ∈ [0, 59]

Compute:
    M_sym(i, s) = |P⁺(i, s)| + |P⁻_flipped(i, s)|  (Energy)
    M_asym(i, s) = |P⁺(i, s)| - |P⁻_flipped(i, s)|  (Direction)

Output structure:
    [M_asym(0:59) | DC(60) | M_sym(61:120)]
```

**Implementation**:
```python
def preprocess_stm_features(stm_2d):
    # ...after 1/f normalization...
    
    # Separate components
    negative_chunk = stm_normalized[:, :60, :]      # (batch, 60, 20)
    positive_chunk = stm_normalized[:, 61:, :]      # (batch, 60, 20)
    dc_component = stm_normalized[:, 60:61, :]      # (batch, 1, 20)
    
    # Flip negative to align with positive
    negative_flipped = torch.flip(negative_chunk, dims=[1])
    
    # Compute maps
    symmetric_map = torch.abs(positive_chunk) + torch.abs(negative_flipped)
    asymmetric_map = torch.abs(positive_chunk) - torch.abs(negative_flipped)
    
    # Concatenate: [Asym | DC | Sym]
    processed = torch.cat([asymmetric_map, dc_component, symmetric_map], dim=1)
    # Output: (batch, 121, 20)
```

**Output Interpretation**:
```
Index 0-59:   Asymmetric map
    - Positive values: Down-sweeps dominate
    - Negative values: Up-sweeps dominate
    - Near zero: Balanced or absent

Index 60:     DC component
    - Preserved from original

Index 61-120: Symmetric map
    - Always positive
    - Encodes total energy
    - Direction-invariant
```

**Rationale**:
- **Symmetric channel**: Helps classify by overall modulation strength
- **Asymmetric channel**: Helps classify by sweep direction
- **Biological parallel**: Opponent processing in visual/auditory systems
- **Directional cues**: Speech (gradual sweeps) vs music (rapid sweeps) may differ

## Integration with ASM Architecture

### Data Flow

```
Raw STM (flattened) → Reshape to 2D
                   ↓
         1/f Normalization (by rate)
                   ↓
         Symmetric/Asymmetric Decomposition
                   ↓
         Per-sample Z-normalization
                   ↓
         (batch, 121, 20) → ASM Model
```

**Modified Data Prep Class**:
```python
class ModifiedSTMDataPrep(prepData_STM_Conformer):
    def prepare_datasets(self):
        # Load raw STM features
        STM_all, target, train_ind, val_ind, test_ind = self.load_data()
        
        # Reshape to 2D
        STM_all_2d = STM_all.reshape(-1, 121, 20)
        
        # Apply modified preprocessing
        STM_processed = preprocess_stm_features(STM_all_2d)
        
        # Normalize per sample (AFTER preprocessing)
        means = STM_processed.mean(dim=(1, 2), keepdim=True)
        stds = STM_processed.std(dim=(1, 2), keepdim=True)
        STM_processed = (STM_processed - means) / (stds + 1e-8)
        
        # Create datasets...
```

**Key Decision**: Normalize AFTER preprocessing
- Ensures modified features have similar scale to original
- Prevents asymmetric channel from dominating (can have negative values)
- Maintains compatibility with ASM architecture

### Model Architecture

**Unchanged from ASM v3**:
```python
ModifiedFeatureASMClassifier(
    time_steps=121,
    freq_steps=20,
    num_classes=6,
    dim=128,
    num_blocks=4,
    shift_range=2,
    expansion_factor=2,
    dropout=0.1
)
```

**Why no architectural changes**:
1. Feature engineering is orthogonal to model design
2. ASM v3 architecture already optimized
3. Can directly compare: preprocessing impact only
4. Maintains parameter count (~1.52M)

**Input Processing**:
```python
def forward(self, x, return_features=False):
    # x: (batch, 121, 20) - modified STM features
    # Same processing as ASM v3:
    x = self.spec_augment(x)
    x = x.unsqueeze(1)  # Add channel dimension
    x = self.input_proj(x)  # Conv2d projection
    # ...rest unchanged...
```

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

### Hypothesis 2: Directional Discrimination

**Prediction**:
- Asymmetric channel helps separate classes with different sweep preferences
- **Speech**: Prosodic contours (up/down intonation)
- **Music**: Melodic direction
- **Environment**: Less directional structure

**Target**:
```
Speech classes (0, 1):  May show better separation if intonation differs
Music classes (2, 3):   May benefit from melodic contour information
Env classes (4, 5):     Less impact (noise-like, no clear direction)
```

### Hypothesis 3: Overall Performance

**Conservative Target**:
```
Test Macro F1: 0.87-0.88 (match or slightly exceed ASM v3)

Per-Class F1:
  Class 0: 0.95-0.96 (maintain)
  Class 1: 0.82-0.84 (similar to v3)
  Class 2: 0.85-0.87 (+1-3 points over v3) ← KEY HOPE
  Class 3: 0.75-0.78 (+1-3 points over v3) ← KEY HOPE
  Class 4: 0.92-0.94 (maintain)
  Class 5: 0.91-0.93 (maintain)
```

**Optimistic Target**:
```
Test Macro F1: 0.88-0.89 (beat ASM v3 by 1-2 points)
All classes F1 ≥ 0.80 (uniform improvement)
```

## Comparison with ASM v3

### Key Differences

| Aspect | ASM v3 | Modified Features |
|--------|--------|-------------------|
| **Preprocessing** | dB + normalize | dB + normalize + 1/f + sym/asym |
| **Feature Engineering** | Minimal (from STM06) | Heavy (custom pipeline) |
| **Input Structure** | Raw rate spectrum | Decomposed channels |
| **Model Changes** | None (uses raw features) | None (same architecture) |
| **Parameters** | 1.52M | 1.52M (identical) |
| **Training** | v3 loss + weights | v3 loss + weights (identical) |
| **Philosophy** | Optimize architecture | Optimize input representation |

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

**Index Mapping**:
```python
Index    Frequency (Hz)
0        -15.00
1        -14.75
...
59       -0.25
60       0.00          ← DC component
61       +0.25
...
120      +15.00
```

### DC Component Handling

**Three Options Considered**:

**Option A: Zero out DC** (not chosen)
```python
# DC becomes 0 after multiplication by |0 Hz| = 0
# Simple but loses information
```

**Option B: Preserve original DC** (chosen)
```python
dc_index = 60
stm_normalized[:, dc_index, :] = stm_2d[:, dc_index, :]
# Keeps DC information
# Prevents artificial discontinuity
```

**Option C: Interpolate DC** (not chosen)
```python
# Average of neighbors
stm_normalized[:, 60, :] = (stm_normalized[:, 59, :] + 
                             stm_normalized[:, 61, :]) / 2
# More complex, unclear benefit
```

**Rationale for Option B**:
- DC component carries important baseline information
- Zeroing creates artificial gap in spectrum
- Preserving maintains continuity
- Minimal impact on overall processing

### Alignment in Decomposition

**Critical Step**: Flip negative chunk correctly

**Before Flip**:
```
Negative chunk (indices 0-59):
    Index 0:  -15.00 Hz
    Index 59: -0.25 Hz

Positive chunk (indices 61-120):
    Index 61:  +0.25 Hz (corresponds to -0.25 Hz)
    Index 120: +15.00 Hz (corresponds to -15.00 Hz)
```

**After Flip**:
```python
negative_flipped = torch.flip(negative_chunk, dims=[1])

# Now aligned:
negative_flipped[0] ↔ positive_chunk[0]  # Both at 0.25 Hz magnitude
negative_flipped[59] ↔ positive_chunk[59]  # Both at 15.0 Hz magnitude
```

**Verification**:
```python
# Test alignment
assert negative_chunk[:, 0, :].shape == positive_chunk[:, 0, :].shape
assert torch.allclose(
    frequency_vector[0],   # -15.0 Hz
    -frequency_vector[120] # -15.0 Hz
)
```

## Usage Instructions

### Training from Scratch

**Standard mode**:
```bash
python STMasm_feateng.py 0
```

**Downsampled mode** (reduce non-tonal speech to 100k samples):
```bash
python STMasm_feateng.py 1
```

### Resuming Training

```bash
python STMasm_feateng.py 0 --resume \
  model/STM/ASM_FeatEng/standard/ckpt/2026-01-18_14-30
```

### Monitoring Key Metrics

**Per epoch output**:
```
Epoch 25/50
============================================================
Train Loss: 0.1234
Val Loss: 0.0876, Val Macro F1: 0.8734

Per-class F1 scores:
  Class 0: 0.9456
  Class 1: 0.8234
  Class 2: 0.8567  ← Watch for improvement
  Class 3: 0.7645  ← Watch for improvement
  Class 4: 0.9312
  Class 5: 0.9145

Confusion between Similar Classes:
  Class 0→1:   850 | Class 1→0:  1100
  Class 2→3:   620 | Class 3→2:   780  ← Key metric

Current learning rate: 0.000456
✓ Saved best model with Val F1: 0.8734
```

### Testing After Training

```bash
# Automatic test evaluation at end of training
# Or manually:
python STMasm_feateng.py 0  # Will load best_model.pt
```

## Expected Training Behavior

### Convergence Pattern

**Phase 1: Warmup (Epochs 1-5)**
```
Loss decreases rapidly: 0.8 → 0.3
Val F1 increases: 0.60 → 0.78
Learning rate: Linear warmup 0 → 1e-3
```

**Phase 2: Fast Learning (Epochs 5-20)**
```
Loss continues decreasing: 0.3 → 0.15
Val F1 increases: 0.78 → 0.85
Feature adaptation to modified input
```

**Phase 3: Refinement (Epochs 20-35)**
```
Loss stabilizes: 0.15 → 0.10
Val F1: 0.85 → 0.87-0.88
Contrastive loss separates similar classes
```

**Phase 4: Fine-tuning (Epochs 35-50)**
```
Loss: 0.10 → 0.08
Val F1: 0.87-0.88 → 0.88-0.89
LR decay helps final optimization
```

### Comparison with ASM v3 Training

**Similarities**:
- Same loss curves shape
- Same learning rate schedule
- Same warmup behavior

**Potential Differences**:
- May converge slightly faster (better input representation)
- Contrastive loss may be more effective (clearer directional features)
- Final F1 may plateau higher

## Troubleshooting Guide

### Issue 1: No Improvement Over ASM v3

**Possible Causes**:
1. Preprocessing doesn't help for these classes
2. Model architecture can't exploit new features
3. Normalization removes useful signal

**Diagnostic**:
```python
# Visualize preprocessed features
import matplotlib.pyplot as plt

# Original vs processed
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(stm_original[0], aspect='auto')
axes[0].set_title('Original STM')
axes[1].imshow(stm_processed[0, :60, :], aspect='auto')
axes[1].set_title('Asymmetric Channel')
axes[2].imshow(stm_processed[0, 61:, :], aspect='auto')
axes[2].set_title('Symmetric Channel')
plt.savefig('feature_comparison.png')
```

**Solutions**:
```python
# Try softer 1/f correction
abs_freq_softened = torch.pow(abs_freq, 0.5)  # Square root
stm_normalized = stm_2d * abs_freq_softened

# Try different decomposition
# Option: Use power instead of absolute value
symmetric_map = (positive_chunk**2 + negative_flipped**2)**0.5
asymmetric_map = (positive_chunk**2 - negative_flipped**2)**0.5
```

### Issue 2: Class 2/3 Still Confused

**If modified features don't help music/environment separation**:

**Hypothesis**: These classes may not differ in rate/directionality
- Both have broadband modulations
- Both lack clear sweep direction

**Alternatives**:
```python
# Add scale axis processing
def process_scale_axis(stm_2d):
    # Apply Gabor-like filtering on scale dimension
    # Enhance specific cyc/oct ranges
    pass

# Or try spectral envelope features
def compute_envelope(stm_2d):
    # Summarize across scale axis
    envelope = stm_2d.mean(dim=-1, keepdim=True)
    return torch.cat([stm_2d, envelope], dim=-1)
```

### Issue 3: Asymmetric Channel Dominates

**If asymmetric channel has very large values**:

**Diagnostic**:
```python
# Check channel statistics
asym_mean = processed[:, :60, :].mean()
sym_mean = processed[:, 61:, :].mean()
print(f"Asymmetric mean: {asym_mean}")
print(f"Symmetric mean: {sym_mean}")

# If ratio > 5, asymmetric dominates
```

**Solution**:
```python
# Normalize channels separately before concatenation
asym_normalized = (asymmetric_map - asymmetric_map.mean()) / asymmetric_map.std()
sym_normalized = (symmetric_map - symmetric_map.mean()) / symmetric_map.std()
processed = torch.cat([asym_normalized, dc_component, sym_normalized], dim=1)
```

### Issue 4: Training Instability

**If loss spikes or NaN**:

**Possible Causes**:
- Extreme values after preprocessing
- DC preservation creates discontinuity

**Solutions**:
```python
# Clip extreme values
stm_normalized = torch.clamp(stm_normalized, min=-10, max=10)

# Or use softer activation before multiplication
abs_freq_soft = torch.tanh(abs_freq)  # Bounded [0, 1]
```

## Experimental Predictions

### Scenario A: Strong Success (Probability: 30%)

**Outcome**: Test F1 ≥ 0.88, all classes ≥ 0.80

**Interpretation**:
- 1/f correction was critical bottleneck
- Directional decomposition provides useful signal
- Feature engineering > architecture optimization

**Next Steps**:
- Publish preprocessing as standalone contribution
- Apply to other models (Conformer, Kanformer)
- Extend to other audio tasks

### Scenario B: Moderate Success (Probability: 50%)

**Outcome**: Test F1 = 0.87-0.88, selective improvement

**Interpretation**:
- Helps some classes (music), not others (speech)
- Incremental gain, not breakthrough
- Useful but not transformative

**Next Steps**:
- Combine with ASM v3 ensemble
- Try class-specific preprocessing
- Focus on classes that benefited

### Scenario C: No Improvement (Probability: 15%)

**Outcome**: Test F1 ≤ 0.87, no gain over ASM v3

**Interpretation**:
- Preprocessing removes signal
- Model can't exploit new features
- Architecture already optimal for raw features

**Next Steps**:
- Ablation to identify harmful components
- Try simpler preprocessing (1/f only)
- Stick with ASM v3 as best approach

### Scenario D: Degradation (Probability: 5%)

**Outcome**: Test F1 < 0.86, worse than ASM v3

**Interpretation**:
- Preprocessing actively harmful
- Over-engineering input
- Loss of critical information

**Next Steps**:
- Debug implementation (check for bugs)
- Verify preprocessing math
- Abandon modified features

## Theoretical Analysis

### Why 1/f Normalization Should Help

**Signal Processing Perspective**:
```
Natural sounds have 1/f spectrum:
    S(f) ∝ 1/f^α  where α ≈ 1

Result:
    - Low frequencies: High power, dominate loss function
    - High frequencies: Low power, ignored by model

After 1/f correction:
    S'(f) = S(f) × |f| ∝ 1/f × f = constant

Benefit:
    - Flat spectrum: All frequencies weighted equally
    - High-freq info preserved
    - Better gradient flow for high-freq features
```

**Machine Learning Perspective**:
```
Without correction:
    Loss gradient = Σ(error × input)
    → Dominated by high-power (low-freq) features
    → High-freq features undertrained

With correction:
    Inputs balanced across frequency
    → Equal learning opportunity for all frequencies
    → Better representation capacity
```

### Why Symmetric/Asymmetric Should Help

**Information Theory Perspective**:
```
Original representation:
    121 bins encoding signed rate values
    Positive and negative rates independent
    Information: I(rate, class)

Decomposed representation:
    60 bins symmetric (total energy)
    60 bins asymmetric (directional preference)
    1 bin DC
    Information: I(energy, class) + I(direction, class)

If:
    I(energy, class) + I(direction, class) > I(rate, class)

Then:
    Decomposition increases mutual information
    → Better discriminability
```

**Neuroscience Perspective**:
```
Biological opponent processing:
    ON/OFF cells in retina
    Excitatory/Inhibitory neurons
    Encode difference (opponent) and sum (symmetric)

Advantages:
    - Efficient coding (decorrelates signal)
    - Enhanced contrast (difference amplifies boundaries)
    - Robust to noise (sum averages out noise)

Applied to STM:
    Symmetric = common modulation (genre/class signature)
    Asymmetric = directional cues (prosody, melody)
```

### Potential Failure Modes

**1. Over-normalization**:
```
If original 1/f structure is informative:
    → Removing it hurts performance
    → Model needs low-freq dominance as cue

Example: Speech often has strong low-freq modulation
         Music has balanced spectrum
         → 1/f correction may blur this distinction
```

**2. Directional Redundancy**:
```
If positive/negative rates already learned separately:
    → Decomposition just rearranges information
    → No gain, just different representation

Model already has:
    Conv layers → Can learn directional filters
    FFT mixing → Can process symmetries
    → May not need explicit decomposition
```

**3. Increased Noise**:
```
Asymmetric map = difference operation:
    → Amplifies noise (errors add in subtraction)
    → Less stable than symmetric (errors cancel)

If SNR is critical:
    → Asymmetric channel may hurt more than help
```

## Comparison Table

| Aspect | Raw STM | ASM v3 | Modified Features |
|--------|---------|--------|-------------------|
| **Preprocessing** | dB + normalize | dB + normalize | dB + normalize + 1/f + decomp |
| **Feature Dim** | (121, 20) | (121, 20) | (121, 20) |
| **Channels** | 1 (rate spectrum) | 1 | 3 (asym, DC, sym) |
| **Frequency Balance** | 1/f natural bias | 1/f natural bias | Flattened |
| **Directionality** | Implicit | Implicit | Explicit (asym channel) |
| **Bio-Inspired** | No | No | Yes |
| **Params** | - | 1.52M | 1.52M |
| **Training** | - | Contrastive focal | Contrastive focal |
| **Expected F1** | 0.8566 (v1) | 0.87+ | **0.87-0.89** |

## Files and Locations

### Source Code
- **Implementation**: `/vast/ac8888/MusicSpeech-STM/STMasm_feateng.py`
- **Documentation**: `/vast/ac8888/MusicSpeech-STM/ASM_FEATENG_SUMMARY.md`
- **Base STM Preprocessing**: `/vast/ac8888/MusicSpeech-STM/STM06_STMpreproc.py`

### Related Files
- **ASM v3**: `STMasm_enhanced3.py` (architecture baseline)
- **Conformer**: `STMconformer_model.py` (data loader)
- **Original STM**: `STM06_STMpreproc.py` (feature extraction)

### Checkpoints
```
model/STM/ASM_FeatEng/
├── standard/
│   └── ckpt/
│       └── [timestamp]/
│           ├── best_model.pt
│           ├── latest_checkpoint.pt
│           ├── checkpoint_epoch_X.pt
│           ├── test_predictions.npy
│           ├── test_targets.npy
│           └── confusion_matrix.npy
└── downsample/
```

## Next Steps and Future Work

### Immediate (After Training)

**1. Compare with ASM v3**:
```python
# Load both models
model_v3 = torch.load('ASM_Enhanced3/.../best_model.pt')
model_modified = torch.load('ASM_FeatEng/.../best_model.pt')

# Evaluate on same test set
f1_v3 = evaluate(model_v3, test_loader)
f1_modified = evaluate(model_modified, test_loader)

# Statistical significance test
from scipy.stats import ttest_rel
t_stat, p_value = ttest_rel(preds_v3, preds_modified)
```

**2. Ablation Study**:
```bash
# Train 1/f only variant
python STMasm_feateng_ablation.py --mode 1f_only

# Train decomposition only variant  
python STMasm_feateng_ablation.py --mode decomp_only

# Compare results
```

**3. Feature Visualization**:
```python
# t-SNE of learned features
from sklearn.manifold import TSNE

# Extract features before classifier
features = model.feature_extractor(test_embeddings)
tsne = TSNE(n_components=2).fit_transform(features.cpu())

# Plot by class
plt.scatter(tsne[:, 0], tsne[:, 1], c=test_labels, alpha=0.5)
plt.savefig('feature_space_tsne.png')
```

### Medium-Term

**1. Apply to Other Models**:
```python
# Modified features with Conformer
class ModifiedConformer(ConformerClassifier):
    def prepare_data(self):
        return preprocess_stm_features(raw_stm)

# Modified features with Kanformer
class ModifiedKanformer(KANformer):
    # Same preprocessing pipeline
```

**2. Optimize Decomposition**:
```python
# Learnable decomposition weights
class LearnableDecomposition(nn.Module):
    def __init__(self):
        self.alpha = nn.Parameter(torch.tensor(1.0))  # Asym weight
        self.beta = nn.Parameter(torch.tensor(1.0))   # Sym weight
    
    def forward(self, stm):
        asym = self.alpha * asymmetric_map
        sym = self.beta * symmetric_map
        return torch.cat([asym, dc, sym], dim=1)
```

**3. Scale Axis Enhancement**:
```python
# Also decompose scale axis (cyc/oct)
def full_2d_decomposition(stm_2d):
    # Decompose both rate and scale
    rate_asym, rate_sym = decompose_rate(stm_2d)
    scale_asym, scale_sym = decompose_scale(stm_2d)
    # 4 channels: rate×scale decomposition
```

### Long-Term

**1. Biological Validation**:
- Compare with auditory cortex recordings
- Test on psychoacoustic discrimination tasks
- Validate against speech/music perception studies

**2. Generalization**:
- Apply to other audio modulation features
- Extend to visual motion energy
- Develop general opponent processing framework

**3. Publication**:
- "Biologically-Inspired STM Preprocessing for Audio Classification"
- "Opponent Coding of Temporal Modulations Improves Music/Speech Separation"
- "1/f Normalization in Deep Learning for Audio"

## Success Criteria

### Technical Success
- [ ] Test F1 ≥ 0.87 (match/beat ASM v3)
- [ ] Class 2/3 F1 ≥ 0.85/0.75 (+2 points each)
- [ ] Training stable (no NaN, smooth convergence)
- [ ] Ablation shows both components contribute

### Scientific Success
- [ ] Preprocessing principle is generalizable
- [ ] Feature visualizations show clearer separation
- [ ] Biological hypothesis supported by results

### Practical Success
- [ ] No additional compute cost (same params)
- [ ] Easy to implement (< 50 lines preprocessing)
- [ ] Works across different architectures
- [ ] Reproducible results

## Conclusion

**ASM Modified Features** represents a **feature engineering approach** to the music/speech classification problem:

**Core Innovation**:
1. **1/f normalization**: Compensates for natural spectral bias
2. **Symmetric/asymmetric decomposition**: Makes directional information explicit
3. **Biological inspiration**: Grounded in auditory neuroscience

**Expected Outcome**:
- Moderate to strong improvement over ASM v3 (0.87-0.89 F1)
- Particularly helps music classes (high-frequency modulations)
- Provides orthogonal gains to architectural improvements

**If Successful** → Feature engineering matters for STM classification  
**If Fails** → Architecture optimization (v3) already optimal for raw features

**Key Question**: Can better input representation beat better learning algorithm?

**Answer**: Will know after training completes!

---

**Last Updated**: January 2026  
**Author**: GitHub Copilot (Claude Sonnet 4.5) + User  
**Status**: Ready for training  
**Next Milestone**: Train and compare with ASM v3 results
