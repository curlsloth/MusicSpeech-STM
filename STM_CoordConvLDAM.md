# STM Classification with CoordConv-ResNet and LDAM Loss
## Phase 1: The Robust Baseline

### Overview

This implementation addresses the fundamental architectural mismatch identified in the Audio Classification Model Improvement document: **standard CNNs assume translation invariance, which is catastrophic for STM features where absolute position encodes semantic meaning**.

### Core Problem

For STM features (20×121 modulation spectrum):
- **Position matters**: Energy at (Low Rate, Low Scale) = "Speech"
- **Shift breaks semantics**: Same energy at (High Rate, High Scale) = "Mechanical Noise"
- **Standard convolution fails**: Shared filters cannot distinguish absolute locations

### Solution: CoordConv + LDAM

#### 1. CoordConv (Coordinate-Aware Convolution)

**From**: "An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution" (Liu et al., NeurIPS 2018)

**Mechanism**: Concatenate coordinate channels to input

```
Input channels:
  Channel 0: STM energy (20×121)
  Channel 1: x-coordinate [-1, 1] (temporal modulation bins)
  Channel 2: y-coordinate [-1, 1] (spectral modulation bins)
```

**Effect**: Filters learn position-dependent features
- A "Speech Detector" activates when: `Energy is high` AND `x ≈ 0` (low rate) AND `y is low` (low scale)
- Mathematical form: `f(STM_energy, x_coord, y_coord)`

#### 2. Architecture Modifications for STM

**Standard ResNet-18 issues**:
- 7×7 conv + stride-2 pooling destroys small spectral dimension (20 bins)
- Global Average Pooling removes spatial information we need to preserve

**Our adaptations**:

1. **Modified Stem**
   - Replace: 7×7 conv, stride=2, maxpool
   - With: 3×3 **CoordConv**, stride=1, no pooling
   - Rationale: Preserve spectral resolution (20 bins is small!)

2. **CoordConv in Every Residual Block**
   - First convolution of each BasicBlock uses CoordConv
   - Propagates coordinate information deep into network
   - Allows high-level features to remain location-aware

3. **Flatten + MLP Head (No Global Average Pooling)**
   - GAP destroys spatial information by averaging across locations
   - For STM, we NEED to know WHERE energy occurred
   - MLP head processes spatially-structured features directly

**Architecture**:
```
Input: (batch, 1, 20, 121)
├─ Stem: CoordConv(1→64, 3×3, stride=1) → (batch, 64, 20, 121)
├─ Layer1: 2× BasicBlock(64) → (batch, 64, 20, 121)
├─ Layer2: 2× BasicBlock(128), stride=2 → (batch, 128, 10, 61)
├─ Layer3: 2× BasicBlock(256), stride=2 → (batch, 256, 5, 31)
├─ Layer4: 2× BasicBlock(512), stride=2 → (batch, 512, 3, 16)
├─ Flatten → (batch, 24576)
└─ MLP: 24576 → 512 → 256 → 6
```

Total parameters: ~12M (comparable to baseline MLP)

#### 3. LDAM Loss with Deferred Reweighting (DRW)

**Problem**: Dataset is highly imbalanced (~1M samples, speech-dominated)
- Standard Cross-Entropy → model becomes "Speech Detector"
- Poor minority class recall (Environment sounds)

**LDAM Loss** (Cao et al., NeurIPS 2019):

**Principle**: Enforce larger decision margins for minority classes

**Margin formula**:
```
Δⱼ = C / (nⱼ)^(1/4)
```
where `nⱼ` = number of training samples in class j

**Intuition**: 
- Majority class (Speech): Small margin → easier to classify
- Minority class (Environment): Large margin → model must be more confident
- Result: Decision boundary pushed away from minority classes

**Modified Loss**:
```
L_LDAM = CrossEntropy(s × (z - Δ), y)
```
where:
- `z` = logits
- `Δ` = per-class margins
- `s` = scale factor (default: 30)
- Margin subtracted from true class logit

**Deferred Reweighting (DRW)**:

Two-stage training schedule:

**Stage 1 (Epochs 0-80%)**:
- Train with LDAM loss, NO class weights
- Goal: Learn good feature representations from abundant majority data
- Model sees raw distribution

**Stage 2 (Epochs 80%-100%)**:
- Train with LDAM loss + inverse frequency weights
- Goal: Fine-tune decision boundaries to favor minority classes
- Weights: `wⱼ = 1 / nⱼ` (normalized)

**Why DRW works**:
1. Early training: Feature extractor benefits from majority class data
2. Late training: Classifier boundary refined with class balance
3. Avoids underfitting majority classes while boosting minorities

### Key Implementation Details

#### Data Handling

```python
# Reshape from flattened (2420,) to 2D (20, 121)
STM_2d = STM_flat.reshape(-1, 20, 121)

# Per-sample normalization (preserves relative energy patterns)
mean = STM_2d.mean(axis=(1,2), keepdims=True)
std = STM_2d.std(axis=(1,2), keepdims=True)
STM_normalized = (STM_2d - mean) / (std + 1e-8)

# Add channel dimension: (batch, 1, 20, 121)
STM_input = STM_normalized[:, np.newaxis, :, :]
```

**Critical**: NO PCA. We preserve all 2420 dimensions as recommended in Section 10.

#### Training Configuration

```python
Optimizer: AdamW
  - Learning rate: 1e-4
  - Weight decay: 1e-4
  
Scheduler: CosineAnnealingLR (T_max=50)

Batch size: 256 (larger than Conformer due to simpler architecture)

Dropout: 0.3 (in MLP head)

Gradient clipping: max_norm=1.0

DRW transition: Epoch 40 (80% of 50 epochs)
```

#### LDAM Hyperparameters

```python
max_m: 0.5  # Maximum margin
s: 30       # Scale factor (amplifies logit differences)
exponent: 1/4  # For margin calculation (empirically optimal)
```

### Expected Performance Improvements

Based on the document analysis:

1. **CoordConv** should improve performance by:
   - Allowing model to distinguish absolute modulation rate/scale positions
   - Preventing "Speech" features at low frequencies from being confused with shifted patterns

2. **LDAM + DRW** should improve:
   - Macro-F1 score (balanced across classes)
   - Recall on minority classes (Environment, Music)
   - Without sacrificing majority class (Speech) accuracy

3. **Compared to baseline MLP**:
   - MLP learns implicit position through fully-connected layers
   - CoordConv-ResNet learns hierarchical + position-aware features
   - Expected: +3-5% Macro-F1

4. **Compared to standard ResNet**:
   - Standard ResNet struggles with position-dependence
   - CoordConv explicitly solves this
   - Expected: +2-3% Macro-F1 over standard ResNet

### File Structure

```
model/STM/CoordConvLDAM_corpora_categories/
├── standard/           # Full dataset training
│   └── ckpt/
│       └── YYYY-MM-DD_HH-MM/
│           ├── best_model.pt
│           ├── checkpoint_epoch_10.pt
│           ├── checkpoint_epoch_20.pt
│           ├── ...
│           ├── test_predictions.npy
│           └── test_targets.npy
└── downsample/         # Downsampled non-tonal speech
    └── ckpt/
        └── ...
```

### Usage

```bash
# Standard training (full dataset)
python STM_CoordConvLDAM.py 0

# Downsampled non-tonal speech (100k samples)
python STM_CoordConvLDAM.py 1
```

### Monitoring Training

The model prints:
- Class distribution (to verify imbalance)
- Loss curves (train/val)
- F1 scores (macro-averaged)
- DRW activation message at epoch 40
- Best model checkpoints

**Key indicators**:
- Train loss should decrease steadily in Stage 1
- Val F1 should increase, potentially plateau
- At DRW transition: brief loss spike (normal), then rapid F1 improvement
- Final test report: Check per-class recall (minorities should improve)

### Theoretical Justification

From the document (Section 5.3 & 7.1):

> "To surpass the MLP baseline on a 1-million-sample imbalanced dataset, one must adopt architectures that respect **absolute coordinate information** and **global context**... CoordConv fixes translation invariance. ResNet provides deep feature extraction. LDAM fixes the imbalance bias."

This implementation embodies all three principles:
1. ✅ **Absolute coordinates**: CoordConv in every block
2. ✅ **Deep extraction**: ResNet-18 (8 layers of processing)
3. ✅ **Imbalance correction**: LDAM + DRW

### Next Steps

If this baseline performs well:
- Phase 2: Vision Mamba (global context, linear complexity)
- Phase 3: FT-Transformer (tabular learning, feature interactions)

If performance plateaus:
- Add SpecAugment-style augmentations
- Experiment with MixUp / Remix (balanced mixup)
- Try SupCon pretraining (Section 7.3)
- Tune max_m and s hyperparameters in LDAM

### References

1. Liu et al., "An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution", NeurIPS 2018
2. Cao et al., "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss", NeurIPS 2019
3. Audio Classification Model Improvement document (Sections 2, 5, 7, 9)
