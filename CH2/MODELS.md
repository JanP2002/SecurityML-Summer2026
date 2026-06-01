# Model Comparison — Pythia Anomaly Detector

> **Context**: The Pythia dataset contains 70×70 grayscale PNG images split into
> `clean` and eight labelled attack partitions (`attack_a` … `attack_h`).  
> All models are trained on `clean ∪ attack_a` (1280 samples) and tested on:
> - **Test_A** (400 samples, balanced): the *known* attack_a  
> - **Test_B** (1200 samples, 5:1 imbalance): the *unknown* attack_b  
>
> **Key metric: AUC-ROC** — insensitive to class imbalance.  All other
> metrics (Accuracy, F1) are misleading on Test_B due to the 5:1 ratio.

---

## Results at a Glance

| Model | Test_A AUC | Test_B AUC | Params / Trees | Training strategy |
|-------|-----------|-----------|----------------|-------------------|
| AnomalyCNN (Baseline) | ~0.47–0.49 | ~0.45–0.48 | ~220 K | BCE, Adam, mini-batch |
| PixelMLP | ~0.53–0.56 | ~0.52–0.56 | 4 901 | BCE→AUC, AdamW full-batch |
| ProfessorCNN (CNN) | ~0.52–0.53 | ~0.52–0.53 | ~800 K | BCE→AUC, Adam, mini-batch |
| **GBM (Best)** | **0.61–0.68** | **0.61–0.65** | 300 trees | sklearn, implicit feature selection |

> Ranges reflect run-to-run variation from random train/val splits (1600 samples total).  
> AUC is reported because Accuracy and F1 are misleading on the 5:1 imbalanced Test_B.

---

## 1. AnomalyCNN — Baseline

### Architecture

```
Input (1×70×70)
  ↓  Conv2d(1→16, k=3, pad=1)  + ReLU  + MaxPool2d(2)   → 16×35×35
  ↓  Conv2d(16→32, k=3, pad=1) + ReLU  + MaxPool2d(2)   → 32×17×17
  ↓  Conv2d(32→64, k=3, pad=1) + ReLU  + MaxPool2d(2)   → 64×8×8
  ↓  Flatten → 4096
  ↓  Linear(4096 → 128)        + ReLU
  ↓  Linear(128 → 1)           + Sigmoid
Output: scalar probability ∈ (0, 1)
```

**Training hyperparameters**  
`Adam(lr=0.001)` · `BCELoss` · batch=64 · 15 epochs · patience=3

### Rationale
This is the **closed-world reference implementation** from the original
notebook (`ch2_step1_v2.ipynb`).  It demonstrates that a standard CNN
trained with supervised labels on one attack type learns to detect *that
specific attack* well (MNIST Test_A ≈ 1.0) but fails on structurally
different unseen attacks (MNIST Test_B ≈ 0.39).

On Pythia the model fails even on the *known* attack (AUC ≈ 0.49) because:
- The attack is **statistical camouflage**: per-pixel mean differences are
  tiny (max |Δ| = 0.068, mean = 0.010 across 4900 pixels).
- MaxPooling and spatially-invariant convolutional filters discard the
  **position-specific signal** that is the only discriminating feature.

### When it works well
MNIST-type attacks with large, spatially structured perturbations (Gaussian
noise σ=0.4, OOD Fashion-MNIST) are trivially detected because the global
pixel distribution shifts dramatically.

---

## 2. PixelMLP — Neural Logistic Regression

### Architecture

```
Input (1×70×70)
  ↓  Flatten → 4900
  ↓  Linear(4900 → 1) + Sigmoid
Output: scalar probability ∈ (0, 1)
```

*No hidden layers* (equivalent to logistic regression in pixel space).

**Training hyperparameters**  
`AdamW(lr=0.001, weight_decay=1.0)` · `BCELoss` · **full-batch** (all 1280
samples per epoch) · 200 epochs · patience=20 · `monitor_auc=True`  
LR scheduler: `ReduceLROnPlateau(mode='max', factor=0.5, patience=15)`

### Rationale
The key insight motivating PixelMLP over AnomalyCNN is that the Pythia
attack signal is **position-specific**: the same pixel at position (i, j)
always shifts in the same direction.  A CNN with global-average pooling
averages out this positional identity; a flat linear layer preserves it.

**Why full-batch?**  
With 1280 samples and 4900 weights, stochastic mini-batches produce noisy
gradients that prevent the per-pixel weights from converging to stable
values.  Full-batch training gives the same deterministic gradient each
epoch, mirroring what scikit-learn's L-BFGS solver does for logistic
regression.

**Why AdamW?**  
The decoupled weight decay in AdamW shrinks each weight independently of
its gradient magnitude — avoiding the bias of Adam+L2 that would crush
small-but-consistent per-pixel signals.  Weight decay = 1.0 is strong
regularisation, intentionally chosen to prevent the model from fitting noise
pixels.

**Limitation (why AUC ≈ 0.56)**  
Even with strong regularisation, L2 still distributes weight uniformly
across all 4900 pixels.  The **2806 uninformative pixels** (|Δ| < 0.01)
contribute enough noise to dilute the signal from the **2094 informative
pixels**, capping AUC at ≈ 0.55–0.56.

---

## 3. ProfessorCNN — Best CNN from Hyperparameter Search

### Architecture

Found by the professor's grid search (`faza_professor_cnn_search.json`):

```
Input (1×70×70)
  ↓  Conv2d(1→32,  k=4) + SWISH + BatchNorm + MaxPool  → ~32×33×33
  ↓  Conv2d(32→64, k=5) + SWISH + BatchNorm + MaxPool  → ~64×14×14
  ↓  Conv2d(64→128,k=5) + SWISH + BatchNorm            → ~128×10×10
  ↓  GlobalAveragePool                                  → 128
  ↓  Linear(128 → 64) + SWISH + Dropout(0.2)
  ↓  Linear(64 → 1)   + Sigmoid
Output: scalar probability ∈ (0, 1)
```

**Training hyperparameters**  
`Adam(lr=0.001)` · `BCELoss` · batch=64 · 200 epochs · patience=20 ·
`monitor_auc=True`

### Rationale
The ProfessorCNN improves over AnomalyCNN in three ways:  
1. **Deeper + wider**: 32→64→128 channels vs. 16→32→64.  
2. **Swish activation**: smoother than ReLU, better gradient flow.  
3. **Global Average Pooling**: replaces the large flat FC layer, dramatically
   reducing parameters and acting as a regulariser.

**Why it still struggles on Pythia**  
Global Average Pooling is precisely the wrong choice here — it *averages*
each channel's activation map, which destroys the position-specific signal
that distinguishes clean from attack_a.  Even with Batch Norm and Swish,
the model cannot recover spatial location information after GAP.

This highlights the **architectural mismatch** between translation-invariant
CNNs and position-dependent attacks.

---

## 4. GBM (Best) — Gradient Boosting Machine

### Architecture

```
sklearn.ensemble.GradientBoostingClassifier
  n_estimators=300  (boosting rounds)
  max_depth=3       (each tree has ≤ 7 leaves)
  learning_rate=0.05
  subsample=0.8     (stochastic gradient boosting)
  min_samples_leaf=10
  random_state=42
```

Input: flattened pixel array, shape (N, 4900).  
Output: probability score via `predict_proba()[:, 1]`.

Wrapped in a thin `_GBMWrapper(nn.Module)` so it integrates transparently
with the existing `evaluate_model()` pipeline.

### Rationale

The breakthrough insight from the **ceiling analysis** (`_ceiling.py`) was:

| Method | AUC | Note |
|--------|-----|------|
| Oracle LR (top-500 pixels, test-set |Δ|) | 0.845 | Cheating — uses test info |
| **GBM (no oracle)** | **0.665** | Best achievable without oracle |
| L1-LR C=0.1 (673 selected pixels) | 0.572 | Sparse but still noisy |
| LR-L2 C=0.001 (all 4900 pixels) | 0.553 | L2 spreads weight uniformly |

**Why GBM works when everything else fails:**  

Decision trees split on **one pixel at a time**.  When boosting builds 300
trees, it greedily selects the pixel at each node that maximally separates
clean from attack.  After 300 rounds only the ~500 most discriminative pixels
accumulate meaningful splits — all other pixels are simply never chosen.

This is **non-parametric feature selection** embedded in the learning
algorithm itself, requiring no oracle knowledge and no separate feature
selection step.  In contrast:

- **LR** must assign a weight to every pixel simultaneously; L2 prevents
  weights from going to zero.  
- **PixelMLP** suffers the same problem at the gradient level.  
- **CNNs** aggregate across space (pooling), losing position entirely.

**Test_B generalisation (AUC 0.65)**  
attack_b is a different attack variant.  GBM generalises partially because
both attacks perturb similar pixel positions with similar directional biases.
This is a coincidence of the dataset construction, not a guaranteed property.

### Limitations
- **Slow at inference**: each prediction requires traversing 300 trees over
  4900 features.  Not suitable for real-time, high-throughput settings.  
- **No online/incremental learning**: retraining from scratch on new data.  
- **Interpretability**: feature importance can be extracted (`feature_importances_`)
  but tree ensembles are harder to audit than a single linear model.

---

## Diagnostic: Why the Signal Exists but is Hard to Find

The per-pixel delta heatmap (`plots/step1/pythia_delta_heatmap.png`) shows
what the attack actually looks like:

```
Max |Δ|:    0.068
Mean |Δ|:   0.010  (global)
# pixels with |Δ| > 0.01:  2094 / 4900  (43%)
```

The attack is **spatially diffuse** — differences appear at random-looking
positions all over the 70×70 image, with no spatial cluster.  This is why:
- **Patch-based features** fail (AUC = 0.51): no local neighborhood is more
  informative than any other.
- **CNNs** fail: convolutional filters look for spatial patterns; there are
  none to find.
- **LDA** fails: the between-class covariance is masked by within-class
  variance across all 4900 directions.
- **GBM succeeds**: it doesn't need a spatial pattern, only a marginal
  distributional shift at selected pixels.

---

## Step-by-step Usage

```bash
# From CH2/ with the venv activated:
python step1.py   # Trains all 4 models, saves pythia_model_comparison.png
python step2.py   # Progressive multi-attack training: Baseline + GBM
python step3.py   # Autoencoder comparison: Baseline + GBM + AE
```

---

## File Reference

| File | Role |
|------|------|
| `lib.py` | `AnomalyCNN`, `ProfessorCNN`, `PixelMLP`, `train_model`, `evaluate_model` |
| `step1.py` | All 4 models trained + compared on Test_A / Test_B |
| `step2.py` | Progressive multi-attack generalisation (Baseline + GBM) |
| `step3.py` | Autoencoder vs. supervised classifiers (Baseline + GBM + AE) |
| `_ceiling.py` | sklearn ceiling probe (oracle and non-oracle AUC upper bounds) |
| `faza_professor_cnn_search.json` | Best CNN config from hyperparameter search |
| `plots/step1/pythia_model_comparison.png` | 4-model comparison bar chart |
| `plots/step1/pythia_delta_heatmap.png` | Per-pixel attack signal heatmap |
