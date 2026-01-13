# Summary: Conformer Implementation for STM Audio Classification

## Files Created

### 1. Main Training Script
**File**: `STM08gpu_Conformer_STM_corpus.py`

A complete PyTorch implementation that replaces the MLP architecture with Conformer:

**Key Features**:
- Uses `torchaudio.models.Conformer` (built-in PyTorch implementation)
- Handles data loading from existing `.npy` files
- Reshapes flattened STM data (2420,) back to 2D (20, 121)
- Per-sample normalization for stable training
- Training modes:
  - Mode 0: Standard training
  - Mode 1: Downsample non-tonal speech
- Saves checkpoints and best model based on validation F1
- Evaluates on test set automatically

**Architecture**:
```
Input (batch, 20 freq, 121 time)
  → Conv1D Input Projection (→ 128 dims)
  → 4 Conformer Blocks (attention + convolution)
  → Global Average Pooling
  → Classification Head
  → Output (batch, 6 classes)
```

**Model Complexity**: ~1.5M parameters (efficient for 900K samples)

---

### 2. Model Checking Script
**File**: `STM08gpu-Conformer-1_checkmodel.py`

Utility script for evaluating trained models:

**Features**:
- Loads model checkpoints
- Displays classification report
- Generates confusion matrix visualization
- Shows per-class accuracy
- Prints model architecture details

**Usage**:
```bash
python STM08gpu-Conformer-1_checkmodel.py model/STM/Conformer_corpora_categories/standard/ckpt/2024-01-12_10-30
```

---

### 3. Test Suite
**File**: `test_conformer_implementation.py`

Comprehensive testing before full training:

**Tests**:
- ✓ Model architecture instantiation
- ✓ Forward pass with dummy data
- ✓ Data reshaping (flattened → 2D)
- ✓ GPU availability and compatibility
- ✓ Batch processing with various sizes

**Usage**:
```bash
python test_conformer_implementation.py
```

Run this first to verify everything works!

---

### 4. HPC Batch Script
**File**: `HPC_sbatch/STM08/run_conformer.sh`

SLURM job submission script for HPC:

**Configuration**:
- Array job: modes 0-1
- Time: 24 hours
- Memory: 64GB
- CPUs: 8
- GPU: 1

**Usage**:
```bash
sbatch HPC_sbatch/STM08/run_conformer.sh
```

---

### 5. Requirements File
**File**: `conformer_requirements.txt`

PyTorch dependencies:
- torch >= 2.0.0
- torchaudio >= 2.0.0
- numpy, pandas, scikit-learn
- matplotlib, seaborn

**Installation**:
```bash
pip install -r conformer_requirements.txt
```

---

### 6. Documentation
**File**: `CONFORMER_IMPLEMENTATION_NOTES.md`

Complete documentation including:
- Architecture details
- Comparison with MLP implementation
- Usage instructions
- Troubleshooting guide
- Performance considerations

---

## Quick Start Guide

### Step 1: Install Dependencies
```bash
pip install -r conformer_requirements.txt
```

### Step 2: Test Implementation
```bash
python test_conformer_implementation.py
```

### Step 3: Run Training

**Local/Interactive**:
```bash
# Standard training
python STM08gpu_Conformer_STM_corpus.py 0

# With downsampling
python STM08gpu_Conformer_STM_corpus.py 1
```

**HPC**:
```bash
sbatch HPC_sbatch/STM08/run_conformer.sh
```

### Step 4: Check Results
```bash
python STM08gpu-Conformer-1_checkmodel.py model/STM/Conformer_corpora_categories/standard/ckpt/<timestamp>
```

---

## Key Differences from MLP Implementation

| Aspect | MLP (Original) | Conformer (New) |
|--------|----------------|-----------------|
| Framework | Keras/TensorFlow | PyTorch |
| Input Shape | (2420,) flat | (20, 121) 2D |
| Architecture | Dense layers | Conv + Transformer + Conv |
| Hyperparameter Search | Keras Tuner (40 trials) | Fixed architecture |
| Parameters | Variable (tuned) | ~1.5M |
| Batch Size | 256 | 128 |
| Data Loading | tf.data.Dataset | torch.utils.data.DataLoader |

---

## Data Flow

### Original (MLP)
```
.mat files → STM06 → flattened (2420,) → PCA (1024) → MLP → prediction
```

### New (Conformer)
```
.mat files → STM06 → flattened (2420,) → reshape (20,121) → Conformer → prediction
                                           ↓
                                    normalize per sample
```

The key insight: **We unflatten the data to restore 2D structure for Conformer!**

---

## Expected Performance

Based on the architecture and dataset:

- **Training time**: ~2-3 hours/epoch on V100 GPU
- **Memory usage**: 8-12 GB GPU memory
- **Total training**: ~2-3 days for 50 epochs
- **Expected F1**: 0.75-0.85 (comparable or better than MLP)

---

## Output Structure

```
model/STM/Conformer_corpora_categories/
├── standard/
│   └── ckpt/
│       └── 2024-01-12_10-30/
│           ├── best_model.pt          # Best model by val F1
│           ├── checkpoint_epoch_5.pt   # Periodic checkpoints
│           ├── test_predictions.npy    # Test set predictions
│           ├── test_targets.npy        # Test set labels
│           └── confusion_matrix.png    # Visualization
└── downsample/
    └── ckpt/
        └── ...
```

---

## Troubleshooting

### "CUDA out of memory"
→ Reduce batch_size in the script (try 64 or 32)

### "Module not found: torchaudio"
→ `pip install torchaudio`

### "Data shape mismatch"
→ Run `test_conformer_implementation.py` to diagnose

### Slow training
→ Ensure GPU is being used (check `torch.cuda.is_available()`)

---

## Next Steps

1. **Run test suite**: Verify implementation works
2. **Small test run**: Try with 1-2 epochs first
3. **Full training**: Run on HPC with full dataset
4. **Compare with MLP**: Use STM12_evaluate_model.ipynb
5. **Optimize**: Try different hyperparameters if needed

---

## Files Modified

**None** - All new files, no modifications to existing code!

This ensures backward compatibility with existing MLP implementation.

---

## Questions?

Refer to `CONFORMER_IMPLEMENTATION_NOTES.md` for detailed documentation.
