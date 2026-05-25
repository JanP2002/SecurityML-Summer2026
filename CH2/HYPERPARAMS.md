# Hyperparameter Reference — Chapter 2

> **Status: `faza_professor_cnn_search.json` does NOT exist.**
> All steps are currently using the **fallback default config** for `ProfessorCNN`.
> Run `python step_professor_search.py` to discover the best configuration and
> populate the JSON; after that every step will load it automatically.

---

## 1. How ProfessorCNN config is selected

```
step_professor_search.py  →  faza_professor_cnn_search.json  →  get_professor_cnn_best()
                                                                       ↑
                                                         called by step1 / step2 / step3
```

`get_professor_cnn_best(input_size=70)` in `lib.py` reads `faza_professor_cnn_search.json`
and instantiates the model with the `best_config` block.  If the file is missing it falls
back to the **default config** and prints a warning.

---

## 2. Current default (fallback) ProfessorCNN config

These values are used by step1, step2, and step3 **until the search is run**.

| Hyperparameter | Default value | Candidates searched |
|---|---|---|
| `kernel_size` | `[5, 5]` | `(4,5)`, `(5,5)` |
| `activation` | `gelu` | `leaky_relu`, `swish (SiLU)`, `gelu` |
| `dropout` | `0.2` | `0.1`, `0.2`, `0.3` |
| `dense_head` | `[128, 64]` | `[128,64]`, `[256,128]`, `[64,32]` |
| `pooling` | `global_average` | `global_average`, `flatten` |

Defined in `lib.py`:

```python
_PROFESSOR_CNN_DEFAULT_CONFIG = {
    "kernel_size": [5, 5],
    "activation":  "gelu",
    "dropout":     0.2,
    "dense_head":  [128, 64],
    "pooling":     "global_average",
}
```

---

## 3. Training hyperparameters (all steps)

These are **fixed** and identical across step1, step2, step3, and the search script
so all models are compared on an equal footing.

| Parameter | Value | Location |
|---|---|---|
| `BATCH_SIZE` | 64 | every step |
| `NUM_EPOCHS` (classifier) | 15 | step1, step2, step3, search |
| `PATIENCE` (early stopping) | 3 | step1, step2, step3, search |
| Optimiser | Adam | all |
| Learning rate | 0.001 | all |
| Loss function | `BCELoss` | all classifiers (both output sigmoid probability) |
| `RANDOM_SEED` | 42 | search script |

Autoencoder-specific (step3 only):

| Parameter | Value |
|---|---|
| `AE_NUM_EPOCHS` | 30 |
| `AE_PATIENCE` | 5 |
| `LATENT_DIM` | 32 |
| `AE_VAL_RATIO` | 0.15 |
| `THRESHOLD_PERCENTILE` | 95 |
| AE loss | `MSELoss` |
| AE optimiser | Adam lr=0.001 |

---

## 4. ProfessorCNN architecture details

```
Input: (B, 1, H, H)   H = 70 for Pythia

Conv block × 3:
  Conv2d(ch_in → ch_out, kernel_size, padding='same')
  → Activation  (leaky_relu | swish | gelu)
  → Dropout2d(p)
  → MaxPool2d(2, 2)

  channels:  1 → 32 → 64 → 128   (fixed, not searched)

Pooling layer (searched):
  global_average  →  output shape (B, 128)
  flatten         →  output shape (B, 128 × h × h)  where h = H // 8

Dense head × 2:
  Linear(in → d_i) → Activation → Dropout(p)
  d_i from dense_head list, e.g. [128, 64]

Output:
  Linear(d_last → 1) → Sigmoid
  ŷ ∈ (0, 1)  — probability of anomaly
```

Spatial size at each stage (H=70, three MaxPool(2,2)):

| After stage | Spatial size |
|---|---|
| Input | 70 × 70 |
| Block 1 | 35 × 35 |
| Block 2 | 17 × 17 (floor) |
| Block 3 | 8 × 8 |

Flat size with `flatten` pooling: 128 × 8 × 8 = **8 192**
Flat size with `global_average` pooling: **128** (always, regardless of H)

---

## 5. AnomalyCNN architecture (baseline — unchanged)

Used in all steps as the primary Pythia baseline and sole MNIST model.

```
Input: (B, 1, H, H)

Block 1:  Conv2d(1  → 16, 3×3, pad=1) → ReLU → MaxPool2d(2,2)
Block 2:  Conv2d(16 → 32, 3×3, pad=1) → ReLU → MaxPool2d(2,2)
Block 3:  Conv2d(32 → 64, 3×3, pad=1) → ReLU → MaxPool2d(2,2)

Flatten → Linear(flat → 128) → ReLU → Linear(128 → 1) → Sigmoid
```

| H | flat size |
|---|---|
| 28 (MNIST) | 64 × 3 × 3 = 576 |
| 70 (Pythia) | 64 × 8 × 8 = 4 096 |

Fixed hyperparameters: kernel=3, activation=ReLU, no dropout, no BN.

---

## 6. Search design — `step_professor_search.py`

A **sequential staged grid search** (15 experiments total).
Each stage fixes the winners of all previous stages.

### Stage 1 — kernel × activation  (6 experiments)

| # | kernel_size | activation | dropout | dense_head | pooling |
|---|---|---|---|---|---|
| 1 | (4, 5) | leaky_relu | 0.2 | [128, 64] | global_average |
| 2 | (4, 5) | swish | 0.2 | [128, 64] | global_average |
| 3 | (4, 5) | gelu | 0.2 | [128, 64] | global_average |
| 4 | (5, 5) | leaky_relu | 0.2 | [128, 64] | global_average |
| 5 | (5, 5) | swish | 0.2 | [128, 64] | global_average |
| 6 | (5, 5) | gelu | 0.2 | [128, 64] | global_average |

### Stage 2 — dropout  (3 experiments)

Fixed: best `kernel_size` + `activation` from Stage 1.

| # | dropout | dense_head | pooling |
|---|---|---|---|
| 7 | 0.1 | [128, 64] | global_average |
| 8 | 0.2 | [128, 64] | global_average |
| 9 | 0.3 | [128, 64] | global_average |

### Stage 3 — dense_head × pooling  (6 experiments)

Fixed: best `kernel_size` + `activation` + `dropout` from Stages 1–2.

| # | dense_head | pooling |
|---|---|---|
| 10 | [128, 64] | global_average |
| 11 | [128, 64] | flatten |
| 12 | [256, 128] | global_average |
| 13 | [256, 128] | flatten |
| 14 | [64, 32] | global_average |
| 15 | [64, 32] | flatten |

**Selection metric:** validation AUC-ROC (threshold-free, robust to class imbalance).

---

## 7. Search output JSON schema — `faza_professor_cnn_search.json`

```json
{
  "timestamp": "YYYY-MM-DD HH:MM:SS",
  "experiment": "ProfessorCNN staged hyperparameter search (Pythia)",
  "search_config": {
    "dataset": "Pythia (70x70 px)",
    "num_experiments": 15,
    "epochs": 15,
    "patience": 3,
    "batch_size": 64,
    "random_seed": 42,
    "selection_metric": "val_AUC_ROC"
  },
  "best_config": {
    "kernel_size": [...],
    "activation": "...",
    "dropout": ...,
    "dense_head": [...],
    "pooling": "..."
  },
  "best_val_metrics":    { "Accuracy": ..., "Precision": ..., "Recall": ..., "F1_Score": ..., "AUC_ROC": ... },
  "best_test_a_metrics": { ... },
  "best_test_b_metrics": { ... },
  "pythia_baseline": {
    "model": "AnomalyCNN",
    "val_metrics": { ... },
    "test_a_metrics": { ... },
    "test_b_metrics": { ... }
  },
  "all_experiments": [
    {
      "stage": 1,
      "config": { ... },
      "val_metrics": { ... },
      "test_a_metrics": { ... },
      "test_b_metrics": { ... }
    },
    ...
  ]
}
```

---

## 8. Data splits

The same deterministic 80/20 split is used in every script so results are
comparable. `split_train_test(ds, train_ratio=0.8)` uses a fixed seed internally.

| Split | Contents | Used for |
|---|---|---|
| `train_ds` | clean_train (80%) ∪ attack_a_train (80%) | model training |
| `val_ds` | remaining 20% of combined pool | early stopping + config selection |
| `test_a_ds` | clean_test (20%) ∪ attack_a_test (20%) | **known** attack evaluation |
| `test_b_ds` | clean_test (20%) ∪ full attack_b | **unknown** attack evaluation |

For Pythia progressive training (step2) the attack_b through attack_h test sets
are constructed fresh per round — only attacks up to round `n` are seen in training.

---

## 9. Logging — `step_professor_search.py`

When the search runs it emits two outputs simultaneously:

| Output | Level | Location | Contents |
|---|---|---|---|
| Console (stdout) | INFO | terminal | Stage banners, per-experiment one-line result, winners, final summary |
| Log file | DEBUG | `professor_search.log` | Everything above + training wall times per experiment |

Log format:
```
2026-05-25 14:01:23  INFO     ★ Stage 1 WINNER: kernel=(5,5)  activation=gelu  val AUC=0.9821  val F1=0.9745
2026-05-25 14:01:23  DEBUG      Training finished in 38.4 s
```

---

## 10. Running order recommendation

```bash
# 1. Run the search FIRST to produce the best config JSON
python step_professor_search.py        # writes faza_professor_cnn_search.json
                                       # writes professor_search.log

# 2. Then run the three experiment steps
python step1.py                        # uses get_professor_cnn_best() → reads JSON
python step2.py
python step3.py
```

If you run step1/2/3 **before** the search, `get_professor_cnn_best()` prints:

```
[ProfessorCNN] Search results not found at '...faza_professor_cnn_search.json'.
  Run 'python step_professor_search.py' first to select the best variant.
  Proceeding with default config.
```

and uses `kernel=[5,5], activation=gelu, dropout=0.2, head=[128,64], pool=global_average`.
