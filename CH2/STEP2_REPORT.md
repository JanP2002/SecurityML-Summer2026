# Step 2 — Generalisation via Multi-Attack Training
### Anomaly Detection in Image Data: Does Attack Diversity Help?

**Author:** SecurityML Summer 2026 — Chapter 2  
**Generated from:** `step2.py`, `faza2_wyniki_generalizacji.json`, `plots/step2/`  
**Run timestamp:** 2026-05-13 22:16:56

---

## 1. Research Question

> *Does training a supervised anomaly detector on a more diverse set of attack types improve its ability to detect a completely **unseen** attack?*

More precisely: if we progressively add attack types A₁, A₂, …, Aₙ to training while always testing on Aₙ₊₁ (held out), do later rounds (larger n) produce better generalisation?

---

## 2. Experimental Setup

### 2.1 Datasets

| Dataset | Resolution | Samples | Role |
|---|---|---|---|
| MNIST | 28 × 28 px | 60 000 train / 10 000 test | Clean class (label = 0) |
| Fashion-MNIST | 28 × 28 px | 60 000 train / 10 000 test | OOD attack source only |
| Pythia | 70 × 70 px | ~800 per partition | Hidden labelled dataset |

**Class balance.** Every training set maintains a strict 1:1 clean-to-attack ratio by randomly sub-sampling attack images. When n attack types are combined, each contributes `total_clean_samples // n` images, so the total attack budget stays fixed as n grows. This isolates the effect of *diversity* from the effect of *data volume*.

**MNIST speed-up.** The full 60 000 clean MNIST training images would make the order-sensitivity experiment (7 orderings × 5 rounds each) impractically slow. Clean training is capped at **10 000 samples**, keeping the 1:1 ratio and reducing per-round compute by ~6×.

### 2.2 Model: AnomalyCNN

A compact convolutional network with three 3×3 conv blocks followed by a two-layer classifier head:

```
Input (1, H, H)
  Conv2d(1→16, k=3, pad=1) → ReLU → MaxPool2d(2,2)
  Conv2d(16→32, k=3, pad=1) → ReLU → MaxPool2d(2,2)
  Conv2d(32→64, k=3, pad=1) → ReLU → MaxPool2d(2,2)
  Flatten
  Linear(flatten_size → 128) → ReLU
  Linear(128 → 1) → Sigmoid
Output: probability ∈ [0, 1]   (≥ 0.5 → attack)
```

**Training:** BCELoss, Adam (lr=0.001), 15 epochs max, early stopping on validation loss with patience=3.  
**Evaluation metrics:** Accuracy, Precision, Recall, F1-Score, AUC-ROC.

### 2.3 Progressive Training Protocol

For n = 1, 2, …, K−1 (K = number of attack types):

1. **Training set:** clean ∪ balanced_sample(A₁, …, Aₙ)  — attacks seen so far
2. **Validation (20% held-out):** from the same combined set
3. **Test set:** clean_test ∪ Aₙ₊₁ — **the next attack, never seen during training**
4. Train a **fresh** AnomalyCNN from scratch for each round
5. Evaluate and record all five metrics

---

## 3. Contamination Methods (MNIST, 6 Attack Types)

All six attacks are applied to the raw MNIST tensor (shape N×1×28×28, values in [0,1]).

---

### A1 — Gaussian Noise (`σ = 0.4`)

$$\tilde{x}_{i,j} = \text{clip}\left(x_{i,j} + \varepsilon_{i,j},\; 0,\; 1\right), \quad \varepsilon_{i,j} \sim \mathcal{N}(0,\, 0.4^2)$$

Each pixel receives an independent additive perturbation drawn from a zero-mean Gaussian with standard deviation 0.4. At this strength roughly 32% of pixels would be clipped, making digits visually recognisable but heavily degraded. The corruption is *spatially incoherent* — each pixel is independently noisy.

**Visual effect:** Digits are still faintly visible under a heavy "static" texture. Background pixels are no longer pure black.

---

### A2 — Salt & Pepper Noise (`p = 0.15`)

$$\tilde{x}_{i,j} = \begin{cases} 0 & u_{i,j} < 0.075 \\ 1 & u_{i,j} > 0.925 \\ x_{i,j} & \text{otherwise} \end{cases} \quad u_{i,j} \sim U(0,1)$$

With probability 0.075 a pixel is forced to 0 (pepper/black) and with probability 0.075 it is forced to 1 (salt/white). The remaining ~85% of pixels are unchanged. Unlike Gaussian noise the corruption is *binary and sparse*.

**Visual effect:** Scattered bright white and pure black dots over the image.  The digit outline is mostly preserved.

---

### A3 — Geometric Distortion (`max_displacement = 5 px`)

A smooth random displacement field is generated at low resolution and bilinearly upsampled to 28×28. The resulting vector field is added to the identity sampling grid and applied via `F.grid_sample` with reflection padding.

**Key property:** no pixel values are changed — the image is only *warped*. All intensity statistics (mean, std, histogram) are identical to the clean image; only the *spatial arrangement* differs.

**Visual effect:** Digits appear twisted, stretched, or bent in organic-looking ways. A "5" may look like a warped squiggle; a "4" may have its crossbar misaligned.

> ⚠️ **This attack is particularly hard for the model.** A detector trained purely on corrupted-pixel attacks (Gaussian, S&P) has learned to look for *noise signatures*. Geometric distortion has no noise — the model must learn a different discriminating feature.

---

### A4 — Blended Attack (`α = 0.30`, random pattern)

$$\tilde{x} = (1 - \alpha)\cdot x + \alpha \cdot p, \quad \alpha = 0.30$$

where *p* is a fixed random pattern tensor (shape 1×1×28×28) drawn once at the start of the experiment and shared between training and test images of the same blended type. The pattern is a uniform-random image in [0,1].

**Visual effect:** A semi-transparent random noise overlay. The digit is clearly visible but the background is no longer black — it has a faint random texture. The blending is additive/linear so pixel histograms shift compared to clean.

---

### A5 — Backdoor Trigger (`5×5 white square, bottom-right corner`)

A solid white 5×5 square is stamped at pixel coordinates (23:28, 23:28) of each image. All other pixels are unchanged. This is a *localised, deterministic* trigger — the attack occupies only 25 pixels out of 784 (3.2% of the image area).

**Visual effect:** Digit appears completely clean except for a small white square in the bottom-right corner, which is easy to miss at a glance.

> ⚠️ **This is the hardest attack for the model.** The "anomaly" is extremely small and localised. The model must learn to attend to a specific 5×5 region rather than any global property. Models trained on global noise patterns are not equipped to detect this trigger.

---

### A6 — OOD Replacement (Fashion-MNIST source)

The clean MNIST images are *replaced entirely* by Fashion-MNIST images (clothing items: shoes, bags, T-shirts, dresses). No pixel transform is applied — the attacked "image" is simply a Fashion-MNIST item.

**Visual effect:** Completely different content — clothing shapes rather than digits. The domain shift is maximal: texture, content, and spatial statistics all differ from MNIST digits.

> **This is the easiest attack to detect.** By the time the model is tested on A6, it has been trained on five diverse attack types. Fashion-MNIST images are so statistically distinct from MNIST digits that even a simple model achieves near-perfect detection.

---

## 4. Pythia Attack Partitions (8 Types: attack_a … attack_h)

The Pythia dataset is an opaque 70×70 px dataset. The clean partition and the eight attack partitions (attack_a through attack_h) are pre-labelled in separate folders. The nature of each attack is unknown — the labels come from the folder structure only.

**Observation from the sample images:** Both clean and attack Pythia images appear as dense, near-random-noise textures. Unlike MNIST there is no obvious visual structure. This makes the detection task fundamentally harder: the distinguishing features (if any) are subtle statistical differences rather than visible artefacts.

---

## 5. Step-by-Step Pipeline

```
1. Load MNIST clean (train + test)
2. Load Fashion-MNIST (train + test) — OOD source only
3. Generate 6 MNIST attack datasets (train split + test split) using
   the contamination functions in attacks/contamination.py
4. Visualise 5 samples from each partition → plots/step2/mnist_*.png
5. Sub-sample MNIST clean_train to 10 000 images for speed
6. Run progressive training (n = 1…5) on MNIST:
     for each n:
       build balanced combined dataset (clean + A1…An)
       split 80/20 train/val
       build test = clean_test + A_{n+1}
       train fresh AnomalyCNN (BCELoss, Adam, patience=3)
       evaluate → record metrics
7. Load Pythia clean + 8 attack partitions from pythia/ folder
8. Split each 80/20 into train/test parts
9. Visualise 5 samples from each Pythia partition
10. Run progressive training (n = 1…7) on Pythia
11. Print summary tables to console
12. Generate plots:
      step2_generalization.png  — all 5 metrics vs n, both datasets
      per_metric_curves.png     — 2×3 grid, each metric separately
      metric_heatmap.png        — heatmap: metric × round
13. Order-sensitivity analysis (7 orderings × 5 MNIST rounds,
                                   7 orderings × 7 Pythia rounds):
      orderings: original, reversed, perm_seed0 … perm_seed4
      generate: order_sensitivity_mnist.png, order_sensitivity_pythia.png
14. Save all results to faza2_wyniki_generalizacji.json
```

---

## 6. Results

### 6.1 MNIST Progressive Training

| n | Training attacks | Test attack (unseen) | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|---|---|
| 1 | A1_gaussian | **A2_salt_pepper** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |
| 2 | A1+A2 | **A3_geometric** | 0.500 | 1.000 | **0.000** | **0.000** | 0.704 |
| 3 | A1+A2+A3 | **A4_blended** | **0.995** | 0.990 | **1.000** | **0.995** | **1.000** |
| 4 | A1+A2+A3+A4 | **A5_backdoor** | 0.533 | 0.912 | **0.073** | **0.135** | 0.818 |
| 5 | A1+A2+A3+A4+A5 | **A6_ood** | **0.990** | 0.981 | **1.000** | **0.990** | **1.000** |

**Pattern: alternating success and failure.** Rounds n=1, 3, 5 succeed; rounds n=2, 4 fail.

#### Round n=1 — ✅ Perfect detection of Salt & Pepper
After training only on Gaussian noise, the model detects S&P noise perfectly (F1=1.000, AUC=1.000). Both attacks share the same statistical signature: independent per-pixel corruption. The model has learned a global "this image is noisy" feature that transfers immediately.

#### Round n=2 — ❌ Complete failure on Geometric Distortion
Accuracy = 0.500 (chance). Recall = 0.000: **every single geometric attack image is classified as clean.** Yet Precision = 1.000 (when it does predict "attack" it is never wrong), meaning the model outputs nearly all predictions as "clean". AUC-ROC = 0.704 shows the model has *some* ranking ability but cannot threshold it into actual detections.

**Why?** Gaussian and S&P noise corrupt pixel values; the model learns to detect noisy pixel distributions. Geometric distortion preserves all pixel values — only their positions change. The model has no experience with "structurally wrong" images and defaults to classifying them as clean.

#### Round n=3 — ✅ Strong detection of Blended Attack
F1=0.995, AUC=1.000. After seeing geometric distortion in training, the model generalises well to the blended attack (a fixed overlay). Both attacks can be distinguished by the modified background statistics. Adding geometric to training likely forced the model to learn more general structural features.

#### Round n=4 — ❌ Near-failure on Backdoor Trigger
Accuracy = 0.533, Recall = 0.073: only 7.3% of backdoor images are caught. The model trained on four diverse attacks still cannot detect a 5×5 white square in the corner. Precision = 0.912 means when it does flag something, it is usually right — but it almost never flags anything. AUC-ROC = 0.818 suggests potential with a lower threshold, but at the standard 0.5 threshold it practically never fires.

**Why?** The backdoor is *localised*. All global statistics (mean, variance, histogram) are nearly identical to clean images. The model has learned global discriminative features (noisy backgrounds, warped strokes, blended overlays) and has no inductive bias to examine specific 5×5 corner patches.

#### Round n=5 — ✅ Excellent detection of OOD (Fashion-MNIST)
F1=0.990, AUC≈1.000. This is the easiest test: Fashion-MNIST images (T-shirts, shoes, bags) are statistically completely different from handwritten digits. Even a model that only partially understands "anomaly" will see these as obviously non-digit images.

---

### 6.2 Pythia Progressive Training

| n | Test attack | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|---|
| 1 | attack_b | 0.500 | 0.000 | 0.000 | 0.000 | **0.432** |
| 2 | attack_c | 0.500 | 0.500 | 1.000 | 0.667 | 0.493 |
| 3 | attack_d | 0.500 | 0.500 | 1.000 | 0.667 | 0.459 |
| 4 | attack_e | 0.500 | 0.000 | 0.000 | 0.000 | 0.488 |
| 5 | attack_f | 0.500 | 0.000 | 0.000 | 0.000 | 0.506 |
| 6 | attack_g | 0.500 | 0.500 | 1.000 | 0.667 | 0.479 |
| 7 | attack_h | 0.500 | 0.500 | 1.000 | 0.667 | 0.540 |

**Pythia is unsolvable by this approach.** Every round produces accuracy = 0.500 — exactly chance level. AUC-ROC oscillates between 0.43 and 0.54, consistent with a random classifier. The model never learns to distinguish Pythia attacks from Pythia clean images.

**What the Precision/Recall pattern means:**
- When Precision = 0.000 and Recall = 0.000: the model predicts everything as clean (no attacks flagged at all).
- When Precision = 0.500 and Recall = 1.000: the model predicts everything as attack (all samples flagged). With a perfectly balanced 1:1 test set, this produces 50% accuracy and P=0.5, R=1.0 mechanically.

Neither mode reflects genuine detection — the model is simply defaulting to one class.

**Why does Pythia fail where MNIST succeeds?**
As visible in the sample images, both the Pythia clean and Pythia attack partitions look like random noise textures at 70×70 resolution. There is no interpretable visual difference. The distinguishing signal — if any — is encoded in subtle statistical properties that a three-block CNN with ~64 features at the final layer does not have sufficient capacity or inductive bias to capture. The problem likely requires unsupervised anomaly detection methods (autoencoders, normalising flows, one-class SVM on deep features) rather than a supervised binary classifier.

---

## 7. Plot-by-Plot Analysis

### Plot 1 — `step2_generalization.png`: All Metrics vs. n

**Left panel (MNIST):** The characteristic W-shaped pattern is clearly visible in Recall, F1, and Accuracy. At n=2 (test=geometric) and n=4 (test=backdoor), all metrics drop to near-zero. Precision stays near 1.0 throughout — when the model does predict "attack" it is almost always right, but at failure rounds it rarely predicts "attack" at all. AUC-ROC dips to 0.70 at n=2 and 0.82 at n=4, confirming partial discrimination even when threshold-based metrics collapse. The recovery at n=3 and n=5 is strong and consistent across all metrics.

**Right panel (Pythia):** Accuracy is a flat horizontal line at 0.500 for all 7 rounds. AUC-ROC hovers between 0.43–0.54 (a narrow band around chance). F1 oscillates between 0 and 0.667 in an alternating pattern — the two F1=0.667 rounds are rounds where the model predicts everything as attack (recall=1.0), not rounds where it genuinely detects anything.

---

### Plot 2 — `per_metric_curves.png`: 2×3 Per-Metric Grid

This view isolates each metric to make cross-dataset comparison clearer.

- **Accuracy:** MNIST shows clear valleys; Pythia is flat at 0.5.
- **Precision:** MNIST stays near 1.0 — the model is cautious and rarely makes false-positive attack predictions. Pythia oscillates between 0.0 and 0.5.
- **Recall:** MNIST has deep valleys at n=2 and n=4 (the attacks the model completely misses). Pythia bounces between 0.0 and 1.0.
- **F1-Score:** Mirrors Recall shape for MNIST. Pythia shows the same 0/0.667 oscillation.
- **AUC-ROC:** The most informative metric. MNIST AUC stays above 0.70 even at failure rounds, meaning the model's *ranked* predictions have value even when its threshold predictions fail. Pythia AUC remains flatly near 0.5 — there is no usable ranking signal at all.

---

### Plot 3 — `metric_heatmap.png`: Metric × Round Heatmap

The heatmap view confirms the pattern at a glance with colour intensity.

**MNIST (left):** Column n=2 (A3_geometric) and column n=4 (A5_backdoor) are clearly the lightest (worst), particularly in Recall and F1-Score. The AUC-ROC row remains darker (better) than the other rows even at failure columns, confirming AUC is the most robust metric here.

**Pythia (right):** Uniformly medium-orange across all cells — no column stands out as better or worse. Recall and F1-Score rows alternate between dark (1.0) and light (0.0) regardless of n, reflecting the predict-all-attack / predict-all-clean alternation rather than genuine detection.

---

### Plot 4 — `order_sensitivity_mnist.png`

Seven orderings tested: original, reversed, and 5 random permutations (perm_seed0–perm_seed4). Each ordering is a full independent progressive training run.

**Key observation: the W-shape is ordering-specific.** Whether a dip occurs at a given n depends on which attack happens to be the held-out test attack at that round — which changes with each permutation. For example, when A5_backdoor is held out at n=2 (early in some permutations), performance collapses at n=2 for that permutation. When A5_backdoor is held out at n=5, it collapses at n=5 instead.

**The ±1 std band is very wide** for Accuracy, Recall, and F1 — spanning from 0 to 1 at intermediate rounds. This means the per-round performance is highly variable across orderings.

**AUC-ROC is the most order-stable metric.** The std band is narrower for AUC-ROC, particularly at later rounds (n=4, 5), where all orderings converge toward high values. By the time 5 attack types have been seen, the model tends to produce good ranking regardless of the curriculum order.

**Interpretation for your professor:** The curriculum order matters a great deal for threshold-based metrics (accuracy, F1) but much less for ranking quality (AUC-ROC). This is because the *which attack is held out* determines whether a model has seen "similar enough" attacks — and that depends entirely on ordering.

---

### Plot 5 — `order_sensitivity_pythia.png`

**Accuracy panel:** A completely flat line at 0.5 for every ordering. The std band has zero width. This is the definitive proof that Pythia is unsolvable by this method regardless of curriculum order.

**AUC-ROC panel:** All orderings cluster tightly in a narrow [0.44, 0.56] band at every round. No ordering produces a meaningful ranking signal.

**Precision/Recall/F1:** Wild oscillations, but these reflect the model defaulting to one class (predict-all-clean or predict-all-attack) rather than any real detection. The high std band for Recall reflects that some permutations hit rounds where the default class is "attack" and others hit rounds where it is "clean."

---

## 8. Summary of Findings

### 8.1 Does Attack Diversity Help? (MNIST)

**Partially yes, but attack type matters more than diversity.**

| Finding | Evidence |
|---|---|
| Diversity helps when attacks share feature-space properties | A1→A2 transfer (both are per-pixel noise) is perfect |
| Diversity does not bridge the geometric/textural gap | Training on 4 diverse attacks still fails on the 5×5 backdoor |
| The recovery effect at n=3 and n=5 is real | Adding geometric to training (n=3) enables blended attack detection |
| Diversity + enough rounds converges | At n=5, trained on 5 types, OOD detection is near-perfect |
| AUC-ROC degrades less than F1 at failure rounds | The model retains ranking signal even when threshold fails |

**The hypothesis is supported for "semantic cluster"-like diversity** but not for attacks that require completely different discriminative features (localised triggers, pure spatial transformations).

### 8.2 Does Attack Diversity Help? (Pythia)

**No.** Progressive training from 1 to 7 attack types produces no improvement. All seven rounds remain at chance-level performance. The Pythia partition differences are too subtle for a supervised CNN trained on 70×70 noise-like images to capture.

**What would work for Pythia:**
- Autoencoder-based anomaly scoring (reconstruction error)
- One-class classification (trained only on clean)
- Statistical tests on deep feature distributions
- Unsupervised density estimation (normalising flows, VAE)

### 8.3 Does Curriculum Order Matter?

| Dataset | Does order matter? |
|---|---|
| MNIST (threshold metrics) | **Yes — strongly.** Which attack is held out at each round drives almost all of the variance. |
| MNIST (AUC-ROC) | **Mildly.** All orderings converge at n=5; early rounds show moderate sensitivity. |
| Pythia (all metrics) | **No.** All orderings produce the same chance-level results. |

---

## 9. Limitations and Future Work

1. **Backdoor attacks require local attention.** A CNN without an explicit attention mechanism or data augmentation targeting small patches is not suited to detect 5×5 triggers. Future work: patch-level anomaly scoring, attention maps, or training with trigger-aware augmentation.

2. **Pythia requires unsupervised methods.** The supervised binary classifier paradigm is inappropriate when the clean-vs-attack boundary is not learnable from the pixel statistics the CNN extracts. Step 3 should explore autoencoder reconstruction loss or density-based scoring.

3. **10 000 MNIST samples is a practical speedup but may understate diversity effects.** With the full 60 000 training samples per class the model may have sufficient data to learn more general features, potentially improving backdoor detection.

4. **The AnomalyCNN is intentionally simple.** A deeper or pre-trained backbone (ResNet, ViT) would likely generalise better across attack types for MNIST, but is out of scope for this controlled study.

5. **Pythia clean images appear as noise.** This suggests the dataset may encode information in non-pixel-intensity channels (e.g., frequency domain, spectral statistics, metadata). A CNN operating on raw pixels is architecturally blind to these signals.

---

## 10. File Index

| File | Description |
|---|---|
| `step2.py` | Main experiment script |
| `attacks/contamination.py` | All six contamination functions |
| `lib.py` | AnomalyCNN, train_model, evaluate_model, data loaders |
| `faza2_wyniki_generalizacji.json` | Full numerical results (all rounds, all orderings) |
| `plots/step2/step2_generalization.png` | All 5 metrics vs n, MNIST + Pythia side-by-side |
| `plots/step2/per_metric_curves.png` | 2×3 grid: each metric separately, MNIST vs Pythia |
| `plots/step2/metric_heatmap.png` | Heatmap: metric × round, both datasets |
| `plots/step2/order_sensitivity_mnist.png` | 7 orderings × 5 metrics, MNIST |
| `plots/step2/order_sensitivity_pythia.png` | 7 orderings × 5 metrics, Pythia |
| `plots/step2/mnist_clean.png` | 5 sample clean MNIST images |
| `plots/step2/mnist_A1_gaussian.png` … `mnist_A6_ood.png` | 5 samples per MNIST attack type |
| `plots/step2/pythia_clean.png` | 5 sample clean Pythia images |
| `plots/step2/pythia_attack_a.png` … `pythia_attack_h.png` | 5 samples per Pythia attack |


TODO: RETRY the cummulative thing on the same attach ifferent instance, trigger, in a different corrner, geometric distortion in a different direction, blended attack with a different pattern, etc. to see if the same pattern of results holds.