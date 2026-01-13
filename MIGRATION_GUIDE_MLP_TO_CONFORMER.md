# Migration Guide: MLP → Conformer

This guide helps you transition from the MLP implementation to the Conformer implementation.

## Side-by-Side Comparison

### Import Statements

**MLP (Keras/TensorFlow)**
```python
import keras
from keras import layers
import keras_tuner as kt
import tensorflow as tf
from prepData import prepData_STM as prepData
```

**Conformer (PyTorch)**
```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchaudio.models import Conformer
from STM08gpu_Conformer_STM_corpus import prepData_STM_Conformer
```

---

### Data Loading

**MLP**
```python
train_dataset, val_dataset, test_dataset, n_feat, n_target = prepData(n_pca=1024)
# Data is flattened and PCA reduced: (N, 1024)
# Returns tf.data.Dataset objects
```

**Conformer**
```python
data_prep = prepData_STM_Conformer(ds_nontonal_speech=False)
train_dataset, val_dataset, test_dataset, n_freq, n_time = data_prep.prepare_datasets()
# Data is reshaped to 2D: (N, 20, 121)
# Returns torch.utils.data.TensorDataset objects

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
```

---

### Model Definition

**MLP**
```python
model = keras.Sequential()
model.add(keras.Input(shape=(n_feat,)))
for i in range(num_layers):
    model.add(layers.Dense(units=units, activation="relu"))
    model.add(layers.Dropout(rate=dropout))
model.add(layers.Dense(n_target, activation="softmax"))
```

**Conformer**
```python
model = ConformerClassifier(
    input_dim=20,          # frequency bins
    num_classes=6,
    d_model=128,
    num_heads=4,
    ffn_dim=512,
    num_layers=4,
    dropout=0.1
)
```

---

### Training Loop

**MLP (Keras - automatic)**
```python
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=lr),
    loss="categorical_focal_crossentropy",
    metrics=[ROC_AUC, macroF1]
)

model.fit(
    train_dataset,
    epochs=2,
    validation_data=val_dataset,
    callbacks=[...]
)
```

**Conformer (PyTorch - manual)**
```python
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    model.train()
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
```

---

### Saving/Loading Models

**MLP**
```python
# Save
model.save("model.keras")

# Load
model = keras.models.load_model("model.keras")
```

**Conformer**
```python
# Save
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, 'checkpoint.pt')

# Load
checkpoint = torch.load('checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])
```

---

### Evaluation

**MLP**
```python
# Automatic with Keras
results = model.evaluate(test_dataset)
predictions = model.predict(test_dataset)
```

**Conformer**
```python
# Manual loop
model.eval()
all_preds = []
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        preds = torch.argmax(output, dim=1)
        all_preds.extend(preds.cpu().numpy())
```

---

## Key Conceptual Differences

### 1. Data Format

| Aspect | MLP | Conformer |
|--------|-----|-----------|
| Input shape | `(batch, 2420)` flat | `(batch, 20, 121)` 2D |
| Processing | PCA to 1024 dims | Per-sample normalization |
| Structure | No spatial info | Preserves freq-time structure |

### 2. Framework Philosophy

| Aspect | Keras/TensorFlow | PyTorch |
|--------|------------------|---------|
| Style | High-level, declarative | Low-level, imperative |
| Training loop | Automatic (`model.fit`) | Manual (`for` loop) |
| Dynamic graphs | No | Yes |
| Debugging | Harder | Easier |

### 3. Hyperparameter Tuning

**MLP**: Uses Keras Tuner for automatic search
```python
tuner = kt.BayesianOptimization(
    hypermodel=hm,
    objective="val_auc",
    max_trials=40,
)
tuner.search(train_dataset, epochs=2)
```

**Conformer**: Fixed architecture (manual tuning if needed)
```python
# Modify hyperparameters directly in code
model = ConformerClassifier(
    d_model=128,      # Change this
    num_layers=4,     # Or this
    num_heads=4,      # Or this
)
```

---

## Command Equivalence

### MLP Training
```bash
python STM08gpu_MLP_STM_corpus.py 0  # PCA + Dropout + ROC-AUC
python STM08gpu_MLP_STM_corpus.py 2  # PCA + Dropout + MacroF1
python STM08gpu_MLP_STM_corpus.py 4  # Downsample + ROC-AUC
```

### Conformer Training
```bash
python STM08gpu_Conformer_STM_corpus.py 0  # Standard
python STM08gpu_Conformer_STM_corpus.py 1  # Downsample
```

---

## Output Comparison

### MLP Output Structure
```
model/STM/MLP_corpora_categories/PCA/Dropout/macroF1/
└── MLP_2024-04-26_21-50/
    ├── best_model0.keras
    ├── best_model1.keras
    ├── best_model2.keras
    └── trial_*/
```

### Conformer Output Structure
```
model/STM/Conformer_corpora_categories/standard/
└── ckpt/2024-01-12_10-30/
    ├── best_model.pt
    ├── checkpoint_epoch_5.pt
    ├── test_predictions.npy
    └── confusion_matrix.png
```

---

## Performance Expectations

| Metric | MLP (tuned) | Conformer (fixed) |
|--------|-------------|-------------------|
| Macro F1 | 0.75-0.85 | 0.75-0.85 (comparable) |
| Parameters | Variable | ~1.5M |
| Training time | ~1-2 hrs/epoch | ~2-3 hrs/epoch |
| GPU memory | 6-8 GB | 8-12 GB |

---

## Common Pitfalls

### ❌ Using flattened data with Conformer
```python
# WRONG - This is for MLP
data = np.load('STM.npy')  # Shape: (N, 2420)
```

```python
# RIGHT - Reshape for Conformer
data = np.load('STM.npy').reshape(-1, 20, 121)  # Shape: (N, 20, 121)
```

### ❌ Mixing TensorFlow and PyTorch
```python
# WRONG - Don't mix frameworks
import tensorflow as tf
import torch
data = tf.data.Dataset.from_tensor_slices(...)  # TF
model = torch.nn.Module(...)  # PyTorch - Won't work!
```

### ❌ Forgetting to call .to(device)
```python
# WRONG - Data on CPU, model on GPU
model = model.to('cuda')
data = torch.tensor(...)  # On CPU
output = model(data)  # ERROR!

# RIGHT
data = data.to('cuda')
output = model(data)  # Works!
```

---

## Advantages of Each Approach

### MLP Advantages
✅ Simpler code (Keras handles everything)  
✅ Automatic hyperparameter tuning  
✅ Established baseline  
✅ Lighter memory footprint  

### Conformer Advantages  
✅ Preserves 2D structure of STM  
✅ State-of-the-art architecture  
✅ More interpretable (attention)  
✅ Better for sequential data  
✅ PyTorch ecosystem (more flexible)  

---

## When to Use Which?

**Use MLP if**:
- You want quick experiments
- You need automatic hyperparameter tuning
- You're comfortable with Keras
- You have limited GPU memory

**Use Conformer if**:
- You want state-of-the-art architecture
- You value 2D structure preservation
- You prefer PyTorch
- You have sufficient GPU resources
- You want interpretability (attention weights)

---

## Gradual Migration Strategy

1. **Phase 1**: Run test suite
   ```bash
   python test_conformer_implementation.py
   ```

2. **Phase 2**: Train Conformer on small subset (1 epoch)
   ```bash
   # Modify script to use 1 epoch for testing
   python STM08gpu_Conformer_STM_corpus.py 0
   ```

3. **Phase 3**: Full training
   ```bash
   sbatch HPC_sbatch/STM08/run_conformer.sh
   ```

4. **Phase 4**: Compare results
   ```bash
   python compare_mlp_conformer.py <mlp_dir> <conformer_dir>
   ```

5. **Phase 5**: Choose best model for production

---

## Getting Help

- **Implementation details**: `CONFORMER_IMPLEMENTATION_NOTES.md`
- **Quick reference**: `CONFORMER_QUICK_REFERENCE.md`
- **Testing**: `test_conformer_implementation.py`
- **Comparison**: `compare_mlp_conformer.py`

---

**Remember**: Both implementations are valid! Choose based on your specific needs and constraints.
