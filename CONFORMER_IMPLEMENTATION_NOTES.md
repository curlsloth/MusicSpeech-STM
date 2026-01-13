# Conformer Implementation for STM Audio Classification

## Overview

This implementation replaces the MLP architecture with a Conformer model for classifying audio Spectro-Temporal Modulation (STM) features. The Conformer architecture is particularly suitable for 2D image-like STM representations.

## Key Changes from MLP Implementation

### Architecture
- **MLP (STM08gpu_MLP_STM_corpus.py)**: Flattened 1D input → Dense layers
- **Conformer (STM08gpu_Conformer_STM_corpus.py)**: 2D input (freq × time) → Conv + Conformer blocks → Classification

### Framework
- **Original**: Keras/TensorFlow
- **New**: PyTorch with torchaudio's built-in Conformer

### Data Format
- **Original**: Flattened 1D array (2420 features = 20 freq × 121 time)
- **New**: 2D array reshaped to (20 freq × 121 time) for Conformer input

## Files Created

1. **STM08gpu_Conformer_STM_corpus.py**: Main training script
2. **STM08gpu-Conformer-1_checkmodel.py**: Model checking and evaluation
3. **HPC_sbatch/STM08/run_conformer.sh**: SLURM batch script for HPC
4. **conformer_requirements.txt**: PyTorch dependencies
5. **CONFORMER_IMPLEMENTATION_NOTES.md**: This file

## Model Architecture

```
Input: (batch, 20 freq, 121 time)
  ↓
Input Projection: Conv1d + ReLU + BatchNorm + Dropout
  ↓ (batch, 128, 121)
Conformer Blocks (4 layers):
  - Multi-head Self-Attention (4 heads)
  - Depthwise Separable Convolution (kernel=31)
  - Feed-Forward Network (512 hidden)
  - Group Normalization + Dropout
  ↓ (batch, 128, 121)
Global Average Pooling
  ↓ (batch, 128)
Classification Head: Linear → ReLU → Dropout → Linear
  ↓
Output: (batch, 6 classes)
```

## Model Parameters

- **Total Parameters**: ~1.5M (much more efficient than large MLPs)
- **Input dimension**: 20 (frequency bins)
- **Sequence length**: 121 (time steps)
- **Model dimension (d_model)**: 128
- **Number of attention heads**: 4
- **FFN dimension**: 512
- **Number of layers**: 4
- **Dropout**: 0.1

## Data Preprocessing

### Original (STM06_STMpreproc.py)
```python
# Flattens 2D STM to 1D
data_small_db = 10 * np.log10(data[75:115:2, 190:311:1])
flattened = data_small_db.flatten()  # Shape: (2420,)
```

### New (STM08gpu_Conformer_STM_corpus.py)
```python
# Reshapes back to 2D
STM_all_2d = STM_all.reshape(-1, 20, 121)  # (N, freq, time)
# Per-sample normalization
normalized = (STM_all_2d - means) / (stds + 1e-8)
```

## Training Configuration

- **Batch size**: 128 (smaller than MLP due to model complexity)
- **Optimizer**: AdamW with weight decay (1e-5)
- **Learning rate**: 1e-4 with ReduceLROnPlateau scheduler
- **Loss**: CrossEntropyLoss
- **Metrics**: Macro F1-score
- **Epochs**: 50
- **Gradient clipping**: Max norm 1.0

## Usage

### Install Dependencies
```bash
pip install -r conformer_requirements.txt
```

### Run Training
```bash
# Mode 0: Standard training
python STM08gpu_Conformer_STM_corpus.py 0

# Mode 1: Downsample non-tonal speech
python STM08gpu_Conformer_STM_corpus.py 1
```

### HPC Submission
```bash
sbatch HPC_sbatch/STM08/run_conformer.sh
```

### Check Model Results
```bash
python STM08gpu-Conformer-1_checkmodel.py model/STM/Conformer_corpora_categories/standard/ckpt/2024-01-12_10-30
```

## Output Directory Structure

```
model/STM/Conformer_corpora_categories/
├── standard/
│   └── ckpt/
│       └── 2024-01-12_10-30/
│           ├── best_model.pt
│           ├── checkpoint_epoch_5.pt
│           ├── checkpoint_epoch_10.pt
│           ├── ...
│           ├── test_predictions.npy
│           ├── test_targets.npy
│           └── confusion_matrix.png
└── downsample/
    └── ckpt/
        └── ...
```

## Classification Categories

0. speech: non-tonal
1. speech: tonal
2. music: vocal
3. music: non-vocal
4. env: urban
5. env: wildlife

## Advantages of Conformer over MLP

1. **2D Structure Preservation**: Maintains the spatial relationship in STM data
2. **Local and Global Context**: Combines convolution (local) and attention (global)
3. **Parameter Efficiency**: More effective feature extraction with fewer parameters
4. **State-of-the-art**: Proven architecture for sequence modeling tasks
5. **Interpretability**: Attention weights can provide insights into important features

## Memory and Performance Considerations

- **Dataset**: ~900K samples
- **Model size**: ~6 MB
- **Training time**: ~2-3 hours per epoch on V100 GPU
- **Memory usage**: ~8-12 GB GPU memory with batch_size=128
- **Inference**: ~100 samples/second on GPU

## Future Improvements

1. **Data Augmentation**: SpecAugment, time/frequency masking
2. **Model Variants**: Try different d_model, num_layers configurations
3. **Ensemble**: Combine multiple Conformer models
4. **Transfer Learning**: Pre-train on larger audio datasets
5. **Mixed Precision**: Use torch.cuda.amp for faster training

## References

- Gulati et al. (2020). "Conformer: Convolution-augmented Transformer for Speech Recognition"
- PyTorch Audio Documentation: https://pytorch.org/audio/stable/index.html
- Original MLP implementation: STM08gpu_MLP_STM_corpus.py

## Troubleshooting

### CUDA Out of Memory
- Reduce batch_size (try 64 or 32)
- Reduce d_model or num_layers
- Enable gradient checkpointing

### Slow Training
- Increase num_workers in DataLoader
- Use mixed precision training (torch.cuda.amp)
- Reduce data preprocessing overhead

### Poor Performance
- Check data normalization
- Adjust learning rate
- Try different model configurations
- Verify data loading (check shapes)

## Contact

For questions or issues, please refer to the main README.md or contact the project maintainer.
