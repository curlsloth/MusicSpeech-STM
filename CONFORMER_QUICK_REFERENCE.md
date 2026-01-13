# Conformer Implementation - Quick Reference

## 📁 Files Created (7 total)

1. **STM08gpu_Conformer_STM_corpus.py** - Main training script
2. **STM08gpu-Conformer-1_checkmodel.py** - Model evaluation tool  
3. **test_conformer_implementation.py** - Pre-training test suite
4. **compare_mlp_conformer.py** - MLP vs Conformer comparison
5. **HPC_sbatch/STM08/run_conformer.sh** - SLURM submission script
6. **conformer_requirements.txt** - PyTorch dependencies
7. **CONFORMER_IMPLEMENTATION_NOTES.md** - Full documentation

## 🚀 Quick Start (3 commands)

```bash
# 1. Install
pip install -r conformer_requirements.txt

# 2. Test
python test_conformer_implementation.py

# 3. Train
python STM08gpu_Conformer_STM_corpus.py 0
```

## 📊 Key Information

| Property | Value |
|----------|-------|
| **Framework** | PyTorch + torchaudio |
| **Architecture** | Conformer (Conv + Transformer) |
| **Input Shape** | (batch, 20 freq, 121 time) |
| **Parameters** | ~1.5M |
| **Batch Size** | 128 |
| **Classes** | 6 audio categories |
| **Dataset** | ~900K samples |

## 🎯 Training Modes

```bash
python STM08gpu_Conformer_STM_corpus.py 0  # Standard
python STM08gpu_Conformer_STM_corpus.py 1  # Downsample non-tonal speech
```

## 🔍 Check Results

```bash
python STM08gpu-Conformer-1_checkmodel.py model/STM/Conformer_corpora_categories/standard/ckpt/<timestamp>
```

## ⚖️ Compare with MLP

```bash
python compare_mlp_conformer.py <mlp_dir> <conformer_dir>
```

## 🖥️ HPC Submission

```bash
sbatch HPC_sbatch/STM08/run_conformer.sh
```

## 📈 Expected Output

```
model/STM/Conformer_corpora_categories/
└── standard/ckpt/2024-01-12_10-30/
    ├── best_model.pt          ← Best model
    ├── checkpoint_epoch_*.pt  ← Periodic saves
    ├── test_predictions.npy   ← Test predictions
    ├── test_targets.npy       ← Test labels
    └── confusion_matrix.png   ← Visualization
```

## 🎓 Architecture

```
Input (20, 121) → Conv1D → 4x Conformer → Pool → Dense → Output (6 classes)
```

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| CUDA OOM | Reduce batch_size to 64 or 32 |
| Slow | Check GPU available: `nvidia-smi` |
| Import error | `pip install torch torchaudio` |
| Shape error | Run test suite first |

## 📚 Documentation

- Full docs: `CONFORMER_IMPLEMENTATION_NOTES.md`
- Summary: `CONFORMER_SUMMARY.md`
- This card: `CONFORMER_QUICK_REFERENCE.md`

## ✨ Key Advantage

**2D Structure**: Unlike MLP which uses flattened data, Conformer preserves the 2D spatial structure of STM features, enabling better feature extraction through convolution + attention.

## 🎯 Classes

0. speech: non-tonal
1. speech: tonal  
2. music: vocal
3. music: non-vocal
4. env: urban
5. env: wildlife

---
Last updated: January 12, 2026
