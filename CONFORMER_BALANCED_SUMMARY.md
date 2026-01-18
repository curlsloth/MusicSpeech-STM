# Conformer Balanced Model - Training Summary

## Model Architecture

**Base**: Conformer with SpecAugment and Class Balancing

### Key Components:
1. **Input Projection**: Conv1d layer to project STM features to d_model=128
2. **Conformer Blocks**: 4 layers with multi-head attention and depthwise convolution
3. **Global Pooling**: Adaptive average pooling over time dimension
4. **Classification Head**: Two-layer MLP with ReLU and dropout

### Model Parameters:
- Input dimension: 20 (frequency bins)
- d_model: 128
- num_heads: 4
- ffn_dim: 512
- num_layers: 4
- depthwise_conv_kernel_size: 31
- dropout: 0.1
- **Total parameters**: 1,555,270

## Training Configuration

### Version 1 (Initial)
- **Class Weights**: Linear inverse frequency (sklearn's 'balanced')
- **Focal Loss Gamma**: 2.0
- **SpecAugment**: freq_mask=4, time_mask=20, n_time_masks=2
- **Label Smoothing**: 0.1
- **Learning Rate**: 1e-4 with warmup (5 epochs)
- **Weight Decay**: 1e-5
- **Batch Size**: 128
- **Epochs**: 50

**Result**: Test F1 = 0.8279 (Epoch 40)

### Version 2 (Refined - Current)
**Key Changes**:
1. **Class Weights Strategy**:
   - Square root scaling: `w_i = sqrt(N / (n_classes * n_i))`
   - Weight capping at 3.0 to prevent over-emphasis
   - Normalization to mean=1.0
   
2. **Focal Loss**:
   - Reduced gamma from 2.0 → 1.5 (less aggressive focusing)
   
3. **SpecAugment**:
   - Reduced freq_mask from 4 → 3
   - Reduced time_mask from 20 → 15
   - Reduced n_time_masks from 2 → 1

**Rationale**:
- Environmental classes (5 & 6) already perform well (F1 > 0.92) with fewer samples
- Square root scaling provides gentler balancing
- Weight capping prevents destabilizing over-emphasis on minority classes
- Reduced augmentation preserves more distinguishable features

## Class Distribution and Weights

### Training Data Distribution:
| Class | Count | Percentage | Weight |
|-------|-------|------------|--------|
| speech:non-tonal | ~500K | ~65% | 0.31 |
| speech:tonal | ~50K | ~6.5% | 3.00 |
| music:vocal | ~30K | ~4% | 5.00 |
| music:non-vocal | ~150K | ~19.5% | 1.00 |
| env:urban | ~15K | ~2% | 10.00 |
| env:wildlife | ~25K | ~3% | 6.00 |

### Weights in Version 2:
- **speech:non-tonal**: 0.31
- **speech:tonal**: 3.00
- **music:vocal**: 5.00
- **music:non-vocal**: 1.00
- **env:urban**: 10.00
- **env:wildlife**: 6.00

## Expected Performance

### Baseline (Unbalanced Conformer)
- **Test Macro F1**: 0.8636
- **Issue**: Potential bias toward majority classes (non-tonal speech)

### Expected Improvements with Balancing
- **Better minority class performance**: Tonal speech, urban sounds
- **More stable training**: Reduced variance in per-class F1
- **Maintained or improved macro F1**: Should exceed 0.8636
- **Slight drop in majority class F1**: Acceptable trade-off for better balance

## Usage

### Training from scratch
```bash
python STMconformer_balanced.py 0  # Standard mode
python STMconformer_balanced.py 1  # Downsample non-tonal speech mode
```

### Resume training
```bash
python STMconformer_balanced.py 0 --resume model/STM/Conformer_Balanced_corpora_categories/standard/ckpt/YYYY-MM-DD_HH-MM
```

## Output Files

Saved in checkpoint directory:
- `best_model.pt`: Best model by validation macro F1
- `latest_checkpoint.pt`: Most recent checkpoint (for resuming)
- `checkpoint_epoch_*.pt`: Periodic checkpoints every 5 epochs
- `test_predictions.npy`: Test set predictions
- `test_targets.npy`: Test set ground truth

## Key Differences from Base Model

| Aspect | Base Conformer | Balanced Conformer |
|--------|----------------|-------------------|
| Loss Function | Cross-Entropy | Weighted Focal Loss + Label Smoothing |
| Class Weighting | None | Automatic from training data |
| Augmentation | None | SpecAugment |
| LR Schedule | ReduceLROnPlateau | Warmup + ReduceLROnPlateau |
| Monitoring | Overall F1 only | Overall + Per-class F1 |
| Resume Support | No | Yes |

## Implementation Notes

1. **Class Weight Computation**: Done automatically in trainer initialization using scikit-learn
2. **Focal Loss Gamma**: Set to 2.0 (standard value from paper)
3. **Label Smoothing**: 0.1 (prevents overconfidence)
4. **SpecAugment**: Conservative parameters to avoid over-augmentation
5. **Warmup**: 5 epochs to stabilize early training

## References

- Base Conformer: `STMconformer_model.py` (Test F1: 0.8636)
- ASM Enhanced: `STMasm_enhanced.py` (inspiration for class balancing)
- Focal Loss: Lin et al., "Focal Loss for Dense Object Detection" (2017)
- SpecAugment: Park et al., "SpecAugment: A Simple Data Augmentation Method for ASR" (2019)
- Conformer: Gulati et al., "Conformer: Convolution-augmented Transformer" (2020)

## Contact

For questions about this model, refer to the base implementation or the ASM enhanced version for similar techniques.
