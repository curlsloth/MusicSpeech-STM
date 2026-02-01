# STM CoordConvLDAM V5 and V6 - Quick Reference

## Version Comparison

| Version | Test Macro F1 | Strategy | Key Features |
|---------|---------------|----------|--------------|
| **V2** | 0.8623 | Baseline | CoordConv + LDAM + DRW |
| **V4** | 0.8631 (+0.0008) | Architecture | + Attention (CA/SE) + Multi-scale |
| **V5** | Target: 0.88-0.89 | **Hybrid** | V4 architecture + V3 training dynamics |
| **V6** | Target: 0.87-0.88 | **Regularization** | V4 architecture + DropBlock + CutMix |

---

## V5: Hybrid Approach (Architecture + Training Dynamics)

### Philosophy
**Combine the best of V3 and V4:**
- V3's balanced sampling + focal loss (training dynamics)
- V4's attention mechanisms + multi-scale fusion (architecture)
- **Hypothesis**: Synergy between better features AND better training

### Key Features
1. **From V4 (Architecture)**:
   - CoordinateAttention (CA) in layers 1-2
   - Squeeze-and-Excitation (SE) in layers 3-4
   - Multi-scale fusion (layer3 + layer4)
   - ~13.5M parameters

2. **From V3 (Training Dynamics)**:
   - Class-balanced batch sampler (252 samples, 42 per class)
   - Focal Loss component (0.3 weight)
   - Remix mixup (class-balanced augmentation)
   - Adaptive LDAM margins (0.3 → 0.5 → 0.7)

3. **Training Configuration**:
   - Batch size: **252** (divisible by 6 classes)
   - Max epochs: **120**
   - DRW starts: **Epoch 60**
   - Hybrid loss: **0.7 × LDAM + 0.3 × Focal**
   - LR scheduler: ReduceLROnPlateau
   - Early stopping patience: 25

### Expected Improvements
- **music:non-vocal recall**: 0.71 (V4) → 0.75-0.78 (+4-7%)
- **speech:tonal recall**: 0.71 (V4) → 0.76-0.78 (+5-7%)
- **Test Macro F1**: 0.8631 (V4) → 0.88-0.89 (+1.7-2.7%)
- **Training stability**: 100-120 epochs (vs V4: 28 epochs early stop)

### Why It Should Work
- **Balanced batches** ensure attention sees all classes equally
- **Focal loss** emphasizes hard examples (loss level)
- **Attention** emphasizes discriminative features (feature level)
- **Complementary mechanisms** at different levels

### Usage
```bash
# Standard training (full dataset)
python STM_CoordConvLDAM5.py 0

# Downsampled non-tonal speech
python STM_CoordConvLDAM5.py 1
```

### Files Created
- `STM_CoordConvLDAM5.py` - Implementation (1040 lines)
- `STM_CoordConvLDAM5.md` - Detailed documentation

---

## V6: Advanced Regularization (DropBlock + CutMix + Stochastic Depth)

### Philosophy
**Address V4's overfitting problem with stronger regularization:**
- V4 stopped early (epoch 28) due to overfitting
- Keep V4's architecture (attention proven useful)
- Add advanced regularization techniques
- **Hypothesis**: Better regularization allows longer training

### Key Features
1. **From V4 (Architecture)**:
   - Same CoordConv + CA/SE attention + multi-scale fusion
   - ~13.5M parameters (+ small overhead for DropBlock)

2. **NEW Regularization Techniques**:
   - **DropBlock**: Spatial-aware dropout (blocks instead of pixels)
     - Drop prob: 0.0 → 0.15 (linear ramp epochs 20-60)
     - Block sizes: 5×5 (layers 2-3), 3×3 (layer 4)
   
   - **CutMix**: Region-based augmentation (replaces Mixup)
     - Cut random boxes from shuffled samples
     - Preserves local STM structure
     - 80% probability per batch
   
   - **Stochastic Depth**: Random block skipping
     - Layer-wise probabilities: 0.0 (layer1) → 0.2 (layer4)
     - Implicit ensemble effect
   
   - **Stronger weight decay**: 2e-4 → **5e-4**
   
   - **Cosine Annealing LR**: Replaces ReduceLROnPlateau
     - Smooth decay: 1e-4 → 1e-6 over 150 epochs

3. **Training Configuration**:
   - Batch size: **256** (standard random sampling)
   - Max epochs: **150** (longer for cosine schedule)
   - DRW starts: **Epoch 75** (50% of 150)
   - LR scheduler: CosineAnnealingLR
   - Early stopping patience: 30

### Expected Improvements
- **music:non-vocal recall**: 0.71 (V4) → 0.74-0.77 (+3-6%)
- **Test Macro F1**: 0.8631 (V4) → 0.87-0.88 (+0.7-1.7%)
- **Training stability**: 100-130 epochs (vs V4: 28 epochs)
- **Overfitting prevention**: Strong regularization allows full capacity utilization

### Why It Should Work
- **DropBlock** prevents spatial overfitting (can't memorize locations)
- **CutMix** creates realistic augmented samples (preserves STM structure)
- **Stochastic Depth** prevents layer co-adaptation
- **Cosine annealing** allows gentle fine-tuning at end

### Usage
```bash
# Standard training (full dataset)
python STM_CoordConvLDAM6.py 0

# Downsampled non-tonal speech
python STM_CoordConvLDAM6.py 1
```

### Files Created
- `STM_CoordConvLDAM6.py` - Implementation (1061 lines)
- `STM_CoordConvLDAM6.md` - Detailed documentation

---

## V5 vs V6: Which to Use?

### Use V5 if:
- ✅ Class imbalance is the main problem
- ✅ Want to maximize minority class performance
- ✅ Willing to use custom batch sampler
- ✅ Prefer interpretable training (explicit class balancing)

### Use V6 if:
- ✅ Overfitting is the main problem (like V4)
- ✅ Want longer, more stable training
- ✅ Prefer standard random sampling
- ✅ Value regularization over sampling tricks

### Likely Outcomes
- **V5 > V6** if: Class imbalance dominates (music:non-vocal needs more exposure)
- **V6 > V5** if: Model capacity is the issue (V4 stopped too early)
- **V5 ≈ V6** if: Both strategies address different aspects of same problem

### Recommended Workflow
1. **Run V5 first** (combines proven strategies from V3 + V4)
2. **Run V6 in parallel** (addresses V4's early stopping)
3. **Compare results**:
   - If V5 >> V6: Class imbalance is key → use balanced sampling
   - If V6 >> V5: Regularization is key → use advanced techniques
   - If V5 ≈ V6: **Ensemble V5 + V6** for +1-2% boost

---

## Expected Results Summary

| Metric | V2 (Baseline) | V4 (Actual) | V5 (Target) | V6 (Target) |
|--------|---------------|-------------|-------------|-------------|
| **Test Macro F1** | 0.8623 | 0.8631 | **0.88-0.89** | **0.87-0.88** |
| **music:non-vocal recall** | 0.66 | 0.71 | **0.75-0.78** | **0.74-0.77** |
| **speech:tonal recall** | 0.72 | 0.71 | **0.76-0.78** | **0.73-0.76** |
| **Training epochs** | ~100 | 28 (early stop) | 100-120 | 100-130 |
| **Parameters** | 12M | 13.5M | 13.5M | 13.5M |
| **Training complexity** | Medium | Medium | **High** | Medium |
| **Batch sampler** | Random | Random | **Balanced** | Random |

### Success Criteria for V5
- ✅ Test Macro F1 ≥ 0.88
- ✅ music:non-vocal recall ≥ 0.75
- ✅ Training stable for 80+ epochs
- ✅ Balanced batches prevent majority class dominance

### Success Criteria for V6
- ✅ Test Macro F1 ≥ 0.87
- ✅ music:non-vocal recall ≥ 0.74
- ✅ Training for 100+ epochs without overfitting
- ✅ DropBlock + CutMix prevent spatial overfitting

---

## Implementation Verification

### Data I/O (Both V5 and V6)
- ✅ Paths use `/vast-ac8888/MusicSpeech-STM/` (verified working in V4)
- ✅ Same CSV metadata loading
- ✅ Same fold-based splitting (folds 0-7=train, 8=val, 9=test)
- ✅ Same 2D reshaping (1, 20, 121)

### Critical Fixes Applied
- ✅ `self.in_channels = 64` initialized BEFORE `_make_layer()` calls
- ✅ Bias initialization checks `if m.bias is not None`
- ✅ No `verbose` parameter in ReduceLROnPlateau (V5)
- ✅ All imports present and correct

### Syntax Check
- ✅ V5: No errors found
- ✅ V6: No errors found

---

## Next Steps

1. **Submit both jobs**:
   ```bash
   # V5 (hybrid approach)
   sbatch HPC_sbatch/STM_CoordConvLDAM5.sbatch
   
   # V6 (advanced regularization)
   sbatch HPC_sbatch/STM_CoordConvLDAM6.sbatch
   ```

2. **Monitor training**:
   - V5: Check balanced class distribution in batches
   - V6: Track DropBlock schedule and CutMix effectiveness

3. **Compare results**:
   - After both complete, analyze which strategy worked better
   - Use best model(s) as baseline for Phase 2 (Vision Mamba)

4. **Potential ensemble**:
   - If both V5 and V6 succeed, ensemble them for +1-2% gain

---

## Troubleshooting

### If V5 fails:
- Check batch sampler is working (log class distribution)
- Verify focal loss component is active (log both losses)
- May need to adjust focal weight (try 0.2 or 0.4)

### If V6 fails:
- DropBlock may be too strong (reduce to 0.10)
- CutMix may be too aggressive (reduce alpha to 0.7)
- Try V6-lite: Only DropBlock + CutMix (no stochastic depth)

### If both fail:
- V4's improvement was already near ceiling
- Consider ensemble V2 + V4 + V5 + V6
- Or move to Phase 2 (Vision Mamba) for architecture change

---

**Files created**:
1. `STM_CoordConvLDAM5.py` - Hybrid model implementation
2. `STM_CoordConvLDAM5.md` - V5 documentation
3. `STM_CoordConvLDAM6.py` - Advanced regularization implementation
4. `STM_CoordConvLDAM6.md` - V6 documentation
5. `STM_CoordConvLDAM_V5_V6_SUMMARY.md` - This summary

**Ready to train!** 🚀
