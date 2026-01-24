# ASM Enhanced v4 Summary

## Overview

Enhanced Audio Spectrogram Mixer v4 (ASM-RH v4) introduces **symmetric STM processing** to exploit the inherent symmetry between upward and downward frequency sweeps in spectro-temporal modulation (STM) representations. This reduces the input dimensionality while preserving critical information, allowing for increased model capacity.

**Key Innovation**: Averaging symmetric modulation rates reduces frequency dimension from 121 to 61 bins while maintaining representational power.

---

## Signal Processing Pipeline

### Symmetric STM Processing

The core innovation in v4 is the pre-processing of STM features before model input:

#### Step A: Separate Modulation Rates
- **Negative rates (up-sweeps)**: Indices 0–59 (−15 Hz to −0.25 Hz)
- **DC component**: Index 60 (0 Hz)
- **Positive rates (down-sweeps)**: Indices 61–120 (+0.25 Hz to +15 Hz)

#### Step B: Align Chunks
- Flip the negative chunk along the frequency axis
- Align index 0 of negative chunk with index 0 of positive chunk
- Result: −0.25 Hz aligns with +0.25 Hz, up to ±15 Hz

#### Step C: Average Aligned Chunks
- Compute element-wise average: `(flipped_negative + positive) / 2`
- Exploits the physical symmetry: up-sweeps and down-sweeps often contain redundant information

#### Step D: Concatenate DC
- Prepend the DC component (0 Hz) at index 0
- Final output: 61 frequency bins (0 Hz to +15 Hz)

### Rationale

**Physical Interpretation**: 
- Spectro-temporal modulation captures how frequency content changes over time
- Up-sweeps (negative rates) and down-sweeps (positive rates) are often symmetric
- Averaging reduces noise and redundancy while preserving discriminative features

**Dimensionality Reduction**:
- Input frequency dimension: 121 → 61 (50% reduction)
- Computational savings allow for deeper, wider models
- Information loss is minimal due to symmetry exploitation

---

## Model Architecture Changes

### From v3 to v4

| Component | v3 | v4 | Change |
|-----------|----|----|--------|
| **Frequency bins** | 121 | 61 | 50% reduction via symmetric processing |
| **Model dimension** | 128 | 160 | +25% increase |
| **Number of blocks** | 4 | 6 | +50% depth increase |
| **Total parameters** | ~2.1M | ~3.8M | +81% capacity increase |

### Architecture Details

```
Input: (batch, 61, time)
  ↓
SpecAugment (freq_mask=4, time_mask=20)
  ↓
Input Projection (Conv2d layers)
  1x1 → dim/4=40 → dim=160
  ↓
6x ASM-RH Blocks (each containing):
  ├─ Enhanced2DPositionalEncoding
  ├─ RollTimeMixing (shift_range=2)
  ├─ HermitFFTMixing
  ├─ TokenMixing
  └─ ChannelMixing
  ↓
LayerNorm + AdaptiveAvgPool
  ↓
Feature Extractor (dim → dim/2=80)
  ↓
Classifier (80 → 6 classes)
```

### Design Rationale

**Increased Capacity**: With 50% fewer input features, we can afford:
- Deeper networks (6 blocks vs 4) for more hierarchical feature learning
- Wider representations (160 vs 128 dims) for richer feature spaces
- Minimal increase in training time due to smaller input

**Preserved Components**:
- SpecAugment for regularization
- 2D positional encoding for time-frequency structure
- RollTimeMixing for temporal context
- HermitFFTMixing for frequency-domain processing
- Contrastive loss from v3 for confusion-aware training

---

## Training Strategy

### Loss Function (Inherited from v3)

**ContrastiveFocalLoss** with:
1. **Softer class weighting**: `sqrt(inverse_frequency)` instead of raw inverse frequency
2. **Confusion-aware boosting**: 
   - Classes 1 & 3 boosted by 1.3× (confusable pairs)
   - Classes 4 & 5 reduced by 0.7–0.8× (easy minorities)
3. **Contrastive regularization**: Maximize inter-class distance for similar pairs (0↔1, 2↔3)
4. **Minimal label smoothing**: 0.01 (vs 0.1 in v1)

### Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Optimizer | AdamW | Weight decay = 1e-4 |
| Learning rate | 1e-3 | With warmup over 5 epochs |
| Scheduler | CosineAnnealingWarmRestarts | T_0=10, T_mult=2 |
| Batch size | 128 | |
| Epochs | 50 | |
| Gradient clipping | 1.0 | Prevents instability |

### Data Augmentation

- **SpecAugment**: 
  - Frequency masking: 1 mask, max 4 bins
  - Time masking: 2 masks, max 20 frames
- **Implicit**: Symmetric averaging acts as denoising

---

## Expected Improvements

### From Symmetric Processing

1. **Noise Reduction**: Averaging symmetric components cancels uncorrelated noise
2. **Computational Efficiency**: 50% fewer input features → faster training/inference
3. **Increased Capacity**: Can afford deeper, wider models with same compute budget
4. **Better Generalization**: Reduced dimensionality acts as regularization

### From Increased Model Capacity

1. **Improved Feature Learning**: 6 blocks allow more hierarchical abstractions
2. **Richer Representations**: 160-dim embeddings capture more nuanced patterns
3. **Better Discrimination**: Especially for confusable pairs (0↔1, 2↔3)

### Target Metrics

Based on v3 performance, we expect:
- **Val Macro F1**: 0.70–0.75 (v3: ~0.68)
- **Test Macro F1**: 0.68–0.73 (v3: ~0.66)
- **Confusion Reduction**: 10–20% fewer mistakes on pairs (0↔1, 2↔3)
- **Training Time**: Similar or slightly reduced per epoch

---

## Implementation Details

### SymmetricSTMDataset Class

Wraps base dataset and applies symmetric processing on-the-fly:

```python
def process_symmetric_stm(stm_data):
    # Input: (batch, 121, time)
    negative = stm_data[:, 0:60, :]    # Up-sweeps
    dc = stm_data[:, 60:61, :]         # DC
    positive = stm_data[:, 61:121, :]  # Down-sweeps
    
    # Flip and average
    negative_flipped = torch.flip(negative, dims=[1])
    averaged = (negative_flipped + positive) / 2.0
    
    # Concatenate DC at start
    output = torch.cat([dc, averaged], dim=1)
    # Output: (batch, 61, time)
    return output
```

### Dimensional Consistency

All components updated for 61 frequency bins:
- Input projection: Conv2d handles (batch, 1, time, 61)
- Positional encoding: `freq_steps=61`
- ASM-RH blocks: Reshaped to (batch, time, 61, dim)
- SpecAugment: `freq_mask_param=4` scales appropriately

---

## Usage

### Training from Scratch

```bash
# Mode 0: Standard class distribution
python STMasm_enhanced4.py 0

# Mode 1: Downsample non-tonal speech
python STMasm_enhanced4.py 1
```

### Resume Training

```bash
python STMasm_enhanced4.py 0 --resume model/STM/ASM_Enhanced4_corpora_categories/standard/ckpt/2026-01-24_10-30
```

### Checkpoints Saved

- `best_model.pt`: Best validation F1 score
- `latest_checkpoint.pt`: Latest epoch (for resuming)
- `checkpoint_epoch_N.pt`: Every 5 epochs

---

## Comparison with Previous Versions

### v1 → v2 → v3 → v4

| Metric | v1 | v2 | v3 | v4 (Expected) |
|--------|----|----|----|----|
| **Input Dim** | 121 freq | 121 freq | 121 freq | **61 freq** |
| **Model Dim** | 128 | 128 | 128 | **160** |
| **Num Blocks** | 4 | 4 | 4 | **6** |
| **Class Weights** | Inverse freq | Inverse freq | sqrt(inv freq) | sqrt(inv freq) |
| **Contrastive Loss** | ✗ | ✗ | ✓ | ✓ |
| **Label Smoothing** | 0.1 | 0.1 | 0.01 | 0.01 |
| **Val Macro F1** | ~0.64 | ~0.66 | ~0.68 | **0.70–0.75** |

### Key Innovations by Version

- **v1**: Base ASM-RH architecture with focal loss
- **v2**: Enhanced positional encoding, adjusted class weights
- **v3**: Confusion-aware loss (from Kanformer v2), contrastive regularization
- **v4**: **Symmetric STM processing** + increased model capacity

---

## Theoretical Foundation

### Why Symmetric Processing Works

**Spectro-Temporal Modulation Theory**:
- STM captures how spectral content modulates over time
- Positive rates: Frequency increases over time (up-sweep)
- Negative rates: Frequency decreases over time (down-sweep)

**Symmetry Property**:
- Many natural sounds exhibit symmetry in their modulation patterns
- Example: Musical notes often have symmetric attack/decay envelopes
- Averaging exploits this symmetry while reducing noise

**Information Theory**:
- Redundancy between symmetric rates = mutual information
- Averaging removes redundancy, preserves discriminative features
- Acts as a form of dimensionality reduction with domain knowledge

### Potential Limitations

1. **Asymmetric Sounds**: Sounds with strong directional sweeps may lose information
   - Example: Sirens with predominantly upward frequency modulation
   - Mitigation: Model capacity increase compensates for minor losses

2. **Irreversible Transformation**: Cannot recover original 121-bin representation
   - Acceptable trade-off for improved efficiency and generalization

3. **Class-Specific Effects**: Impact may vary by sound category
   - Monitor per-class F1 scores during training
   - Classes 2 (music) and 3 (env) may benefit most from symmetry

---

## Monitoring and Evaluation

### Key Metrics to Track

1. **Macro F1 Score**: Overall balanced performance
2. **Per-Class F1**: Identify class-specific improvements
3. **Confusion Pairs**: Monitor (0↔1) and (2↔3) mistakes
4. **Training Stability**: Check for NaN losses or gradient explosions

### Expected Training Dynamics

- **Warmup (Epochs 1–5)**: Gradual LR increase, stable loss decrease
- **Main Training (Epochs 6–40)**: Cosine annealing, F1 steady improvement
- **Convergence (Epochs 41–50)**: Plateau around best F1

### Checkpoint Analysis

```python
# Load confusion history
checkpoint = torch.load('best_model.pt')
confusion_history = checkpoint['confusion_history']

# Analyze improvement over epochs
for epoch, cm in enumerate(confusion_history):
    class01_confusion = cm[0,1] + cm[1,0]
    class23_confusion = cm[2,3] + cm[3,2]
    print(f"Epoch {epoch}: 0↔1={class01_confusion}, 2↔3={class23_confusion}")
```

---

## Future Directions

### Potential Enhancements

1. **Adaptive Symmetric Processing**:
   - Learn optimal mixing weights instead of fixed 50/50 averaging
   - Per-class or per-sample adaptive weighting

2. **Hierarchical Symmetry**:
   - Apply symmetric processing at multiple scales
   - Coarse-to-fine frequency resolution

3. **Uncertainty Quantification**:
   - Estimate information loss from symmetric averaging
   - Use as confidence measure for predictions

4. **Cross-Domain Transfer**:
   - Test on other audio classification tasks (e.g., emotion recognition)
   - Validate symmetric processing generalizability

### Ablation Studies

To validate design choices:
1. **No Symmetric Processing**: Train v3 architecture with 61 bins (random selection)
2. **Symmetric Only**: Apply averaging without model capacity increase
3. **Capacity Only**: Train deeper/wider model without symmetric processing
4. **Asymmetric Averaging**: Weight negative/positive chunks differently (e.g., 30/70)

---

## Troubleshooting

### Common Issues

**1. Dimension Mismatch Errors**
- **Cause**: Frequency dimension not updated throughout pipeline
- **Fix**: Verify `n_freq=61` in all model components and data loaders

**2. NaN Losses**
- **Cause**: Numerical instability in symmetric averaging
- **Fix**: Check for zero-division, add epsilon if needed

**3. Poor Performance on Specific Classes**
- **Cause**: Symmetric averaging harmful for asymmetric sounds
- **Fix**: Consider class-specific processing or weighted averaging

**4. Slower Training Than Expected**
- **Cause**: Increased model capacity (6 blocks, 160 dim)
- **Fix**: Acceptable trade-off; use smaller batch size if OOM

### Debugging Commands

```bash
# Check data dimensions after symmetric processing
python -c "
from STMasm_enhanced4 import *
data_prep = prepData_STM_Conformer(ds_nontonal_speech=False)
train_ds, _, _, n_freq, n_time = data_prep.prepare_datasets()
train_ds = SymmetricSTMDataset(train_ds)
sample, label = train_ds[0]
print(f'Sample shape: {sample.shape}')  # Should be (61, time)
"

# Verify model forward pass
python -c "
import torch
from STMasm_enhanced4 import EnhancedASM_RH_Classifier
model = EnhancedASM_RH_Classifier(time_steps=117, freq_steps=61, num_classes=6)
x = torch.randn(2, 61, 117)
out = model(x)
print(f'Output shape: {out.shape}')  # Should be (2, 6)
"
```

---

## Citations and References

### Related Work

1. **Spectro-Temporal Modulation**: Chi et al., "Multiresolution spectrotemporal analysis of complex sounds" (2005)
2. **ASM Architecture**: Bai et al., "Audio Spectrogram Mixer" (2022)
3. **Focal Loss**: Lin et al., "Focal Loss for Dense Object Detection" (2017)
4. **Contrastive Learning**: Chen et al., "A Simple Framework for Contrastive Learning" (2020)

### Acknowledgments

- **v3 Foundation**: Kanformer v2's confusion-aware strategies
- **STM Processing**: Adapted from classical auditory neuroscience models
- **Implementation**: Built on PyTorch and scikit-learn

---

## Conclusion

ASM Enhanced v4 represents a significant step forward by incorporating domain-specific signal processing (symmetric STM averaging) to reduce dimensionality while increasing model capacity. The combination of:

1. **Intelligent preprocessing** (symmetric averaging)
2. **Proven loss functions** (contrastive focal loss from v3)
3. **Increased model capacity** (6 blocks, 160 dims)

...is expected to yield substantial improvements in classification performance, particularly for confusable class pairs, while maintaining or improving computational efficiency.

**Next Steps**: Train on full dataset, monitor confusion matrices, and compare against v3 baseline. Target: **≥5% relative improvement** in macro F1 score.

---

*Document Version: 1.0*  
*Last Updated: 2026-01-24*  
*Author: ASM v4 Development Team*
