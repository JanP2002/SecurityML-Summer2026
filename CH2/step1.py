"""
step1.py — Anomaly Detection: Supervised Classification Baseline (Step 1)
=========================================================================

Implements the closed-world supervised binary anomaly detector from
the original ch2_step1_v2.ipynb, refactored to use the shared libraries:

    lib.py                — data loaders, AnomalyCNN, training, evaluation
    attacks/contamination — all contamination method implementations

Pipeline
--------
1.  Load MNIST clean (label 0)  +  Fashion-MNIST (for OOD attack).
2.  Generate attack_a: Gaussian noise σ=0.4  (KNOWN attack, label 1).
3.  Generate attack_b: OOD Fashion-MNIST      (UNKNOWN attack, label 1).
4.  Split into 80/20 train/val from [clean ∪ attack_a], plus Test_A and Test_B.
5.  Load the hidden Pythia dataset (70×70 px grayscale PNGs) directly from
    pre-existing partition folders: ``clean``, ``attack_a``, ``attack_b``.
    Split each partition 80/20 and assemble train/val/Test_A/Test_B sets.
6.  Train AnomalyCNN for each dataset with early stopping.
7.  Evaluate on Test_A (known attack) → expected near-perfect metrics.
8.  Evaluate on Test_B (unknown attack) → expected significant degradation.
9.  Save all metrics to ``faza1_wyniki_eksperymentu.json``.

Usage
-----
    # From the CH2/ directory:
    python step1.py

Key finding
-----------
High Test_A performance confirms the model learned attack_a.
Degraded Test_B recall demonstrates the closed-world assumption failure:
the classifier learned a *noise detector*, not a *normality model*.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

# Ensure CH2/ is on sys.path so lib and attacks can be imported when
# running this script from any working directory.
sys.path.insert(0, str(Path(__file__).parent.resolve()))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader, random_split
from datetime import datetime

from lib import (
    prepare_clean_data,
    load_pythia_data,
    check_pythia_available,
    split_train_test,
    make_dataloader,
    visualize_samples,
    AnomalyCNN,
    ProfessorCNN,
    PixelMLP,
    get_professor_cnn_best,
    train_model,
    evaluate_model,
    parse_results,
    save_results,
    compute_pos_weight,
)
from attacks.contamination import make_gaussian_attack, make_ood_attack

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RANDOM_SEED = 2137


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Comparison plot helper  (Pythia baseline vs ProfessorCNNBest)
# ---------------------------------------------------------------------------

def _plot_pythia_model_comparison(
    baseline_a: dict,
    baseline_b: dict,
    pixel_a: dict,
    pixel_b: dict,
    cnn_a: dict,
    cnn_b: dict,
    gbm_a: dict,
    gbm_b: dict,
    save_path,
) -> None:
    """Bar chart comparing all four Pythia classifiers on Test_A / Test_B.

    Models shown (left to right within each metric group):
      1. AnomalyCNN (Baseline)  — steelblue
      2. PixelMLP               — mediumpurple
      3. ProfessorCNN (CNN)     — forestgreen
      4. GBM (Best)             — darkorange (bold)
    """
    metrics = ["Accuracy", "Precision", "Recall", "F1_Score", "AUC_ROC"]
    mlabels = ["Acc", "Prec", "Rec", "F1", "AUC"]
    x = np.arange(len(metrics))
    width = 0.16   # 4 × 0.16 = 0.64 total; 0.36 breathing room between groups

    model_specs = [
        ("AnomalyCNN (Baseline)", "steelblue",   "#1a4f6e", 1.0),
        ("PixelMLP",             "mediumpurple", "#4a1a8c", 1.0),
        ("ProfessorCNN (CNN)",   "forestgreen",  "#1a5c1a", 1.0),
        ("GBM (Best)",           "darkorange",   "#8B4000", 1.8),
    ]
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
    auc_idx = metrics.index("AUC_ROC")

    imbalance_notes = [None, "\u26a0 5:1 class imbalance \u2014 AUC is the reliable metric"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "Pythia \u2014 Por\u00f3wnanie modeli: AnomalyCNN | PixelMLP | ProfessorCNN | GBM",
        fontsize=12, fontweight="bold",
    )

    for ax, data_list, split_label, subtitle, imb_note in zip(
        axes,
        [[baseline_a, pixel_a, cnn_a, gbm_a],
         [baseline_b, pixel_b, cnn_b, gbm_b]],
        ["Test_A", "Test_B"],
        ["KNOWN attack_a", "UNKNOWN attack_b"],
        imbalance_notes,
    ):
        for (label, color, edge_color, lw), offset, data in zip(
            model_specs, offsets, data_list
        ):
            vals = [data.get(m) or 0.0 for m in metrics]
            bars = ax.bar(
                x + offset, vals, width,
                label=label, color=color, alpha=0.85,
                edgecolor=edge_color, linewidth=lw,
            )
            # Bold green outline on AUC bar to draw attention to the key metric
            list(bars)[auc_idx].set_edgecolor("green")
            list(bars)[auc_idx].set_linewidth(2.5)

        title_str = f"{split_label}  ({subtitle})"
        ax.set_title(title_str, fontsize=11, fontweight="bold")
        if imb_note:
            ax.set_xlabel(imb_note, fontsize=8, color="firebrick")
        ax.set_xticks(x)
        ax.set_xticklabels(mlabels, fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
        ax.set_ylabel("Metric value", fontsize=9)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    import pathlib; pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] Saved \u2192 {save_path}")



BATCH_SIZE = 64
NUM_EPOCHS = 15
PATIENCE   = 3

# Pythia-specific training budget for ProfessorCNN.
# Pythia has subtle inter-class differences; the model needs more epochs
# and a more lenient patience than the MNIST baseline.
PYTHIA_PROF_NUM_EPOCHS = 200
PYTHIA_PROF_PATIENCE   = 20


def main() -> None:
    """Execute the full Step 1 supervised baseline experiment.

    Sections (matching original notebook cell numbers)
    --------------------------------------------------
    Cell 1  — MNIST clean + attack_a
    Cell 2  — Fashion-MNIST clean (source for OOD attack)
    Cell 3  — Generate attack_b (OOD)
    Cell 4  — MNIST dataset splits
    Cell 5  — Pythia data loading
    Cell 6  — Pythia dataset splits
    Cell 7  — Model initialisation
    Cell 8  — Training (both models)
    Cell 9  — Evaluation on Test_A (known attack)
    Cell 10 — Evaluation on Test_B (unknown attack)
    Cell 11 — Result persistence
    """

    PLOTS = Path("plots") / "step1"
    set_global_seed(RANDOM_SEED)

    # -----------------------------------------------------------------------
    # Cell 1 — MNIST clean + attack_a (Gaussian noise)
    # -----------------------------------------------------------------------
    print("--- KOMÓRKA 1: ZBIÓR MNIST ---")
    clean_train_mnist, clean_test_mnist, raw_train_mnist, raw_test_mnist = (
        prepare_clean_data(dataset_name="mnist")
    )

    # Known attack: additive Gaussian noise, σ=0.4 (label = 1)
    attack_a_train_mnist = make_gaussian_attack(raw_train_mnist, std=0.4)
    attack_a_test_mnist = make_gaussian_attack(raw_test_mnist, std=0.4)

    visualize_samples(clean_train_mnist,    save_path=PLOTS / "mnist_clean.png",    title_prefix="MNIST Clean, ")
    visualize_samples(attack_a_train_mnist, save_path=PLOTS / "mnist_attack_a.png", title_prefix="MNIST Attack_A, ")
    print("-" * 40 + "\n")

    # -----------------------------------------------------------------------
    # Cell 2 — Fashion-MNIST (source for the unknown OOD attack)
    # -----------------------------------------------------------------------
    print("--- KOMÓRKA 2: ZBIÓR FASHION-MNIST ---")
    _, _, raw_train_fmnist, raw_test_fmnist = prepare_clean_data(
        dataset_name="fashion_mnist"
    )
    print("-" * 40 + "\n")

    # -----------------------------------------------------------------------
    # Cell 3 — Generate attack_b (OOD: Fashion-MNIST substitution)
    # -----------------------------------------------------------------------
    print("--- KOMÓRKA 3: GENERACJA ATTACK_B (OOD) ---")
    # Fashion-MNIST images share the 28×28 grayscale format with MNIST but
    # depict clothing — structurally very different from handwritten digits.
    attack_b_train = make_ood_attack(raw_train_fmnist)
    attack_b_test = make_ood_attack(raw_test_fmnist)

    visualize_samples(attack_b_train, save_path=PLOTS / "mnist_attack_b_ood.png", title_prefix="Attack_B (OOD), ")

    print("\n3 prepared datasets:")
    print("  1. Clean (MNIST)            → label 0")
    print("  2. Attack_A (Gaussian noise) → label 1  [used in training]")
    print("  3. Attack_B (OOD F-MNIST)    → label 1  [reserved for unknown-attack test]")

    # -----------------------------------------------------------------------
    # Cell 4 — MNIST dataset splits
    # -----------------------------------------------------------------------
    print("\n--- KOMÓRKA 4: PODZIAŁ NA ZBIORY ---")

    # Perfectly balanced: MNIST has exactly 60 000 samples per split,
    # so clean_train (60k) + attack_a_train (60k) = 50/50 prior.
    combined_train_val = ConcatDataset([clean_train_mnist, attack_a_train_mnist])

    train_size = int(0.8 * len(combined_train_val))   # 96 000
    val_size = len(combined_train_val) - train_size    # 24 000
    train_dataset, val_dataset = random_split(combined_train_val, [train_size, val_size])

    print(f"Train:  {len(train_dataset):>7} samples (clean ∪ attack_a, 80%)")
    print(f"Val:    {len(val_dataset):>7} samples (clean ∪ attack_a, 20%)")

    # Test_A uses dedicated test-split images never seen during training
    test_a_dataset = ConcatDataset([clean_test_mnist, attack_a_test_mnist])
    print(f"Test_A: {len(test_a_dataset):>7} samples (clean + known attack_a)")

    # Test_B replaces the known attack with OOD Fashion-MNIST
    test_b_dataset = ConcatDataset([clean_test_mnist, attack_b_test])
    print(f"Test_B: {len(test_b_dataset):>7} samples (clean + unknown attack_b OOD)\n")

    # -----------------------------------------------------------------------
    # Cell 5 — Pythia hidden dataset
    # -----------------------------------------------------------------------
    print("--- KOMÓRKA 5: FAZA 2 - ZBIÓR UKRYTY PYTHIA ---")
    PYTHIA_DIR = Path("pythia")
    check_pythia_available(PYTHIA_DIR)
    pythia_clean = load_pythia_data(data_dir=PYTHIA_DIR, partition="clean")
    pythia_attack_a = load_pythia_data(data_dir=PYTHIA_DIR, partition="attack_a")
    pythia_attack_b = load_pythia_data(data_dir=PYTHIA_DIR, partition="attack_b")

    print(f"\n  Pythia clean:    {len(pythia_clean)} samples  (label 0)")
    print(f"  Pythia attack_a: {len(pythia_attack_a)} samples (label 1)")
    print(f"  Pythia attack_b: {len(pythia_attack_b)} samples (label 1)\n")

    visualize_samples(pythia_clean,    save_path=PLOTS / "pythia_clean.png",    title_prefix="Pythia Clean, ")
    visualize_samples(pythia_attack_a, save_path=PLOTS / "pythia_attack_a.png", title_prefix="Pythia Attack_A, ")
    visualize_samples(pythia_attack_b, save_path=PLOTS / "pythia_attack_b.png", title_prefix="Pythia Attack_B, ")

    # -----------------------------------------------------------------------
    # Cell 6 — Pythia splits
    # -----------------------------------------------------------------------
    print("--- KOMÓRKA 6: FAZA 2 - PODZIAŁ (PYTHIA) ---")
    # images are already normalised to [0,1] by transforms.ToTensor() in load_pythia_data

    pythia_clean_train_base, pythia_clean_test = split_train_test(pythia_clean)
    pythia_atk_a_train_base, pythia_atk_a_test = split_train_test(pythia_attack_a)

    # NOTE (audit finding): pythia_attack_b is NOT split — the full 1 000-sample
    # partition is used as the test set, creating a 5:1 class imbalance in
    # pythia_test_b_dataset (200 clean vs. 1000 attack_b).  This matches the
    # original notebook exactly; interpret precision/recall accordingly.
    pythia_atk_b_test = pythia_attack_b

    pythia_combined_tv = ConcatDataset([pythia_clean_train_base, pythia_atk_a_train_base])
    p_train_size = int(0.8 * len(pythia_combined_tv))
    p_val_size = len(pythia_combined_tv) - p_train_size
    pythia_train_dataset, pythia_val_dataset = random_split(
        pythia_combined_tv, [p_train_size, p_val_size]
    )

    pythia_test_a_dataset = ConcatDataset([pythia_clean_test, pythia_atk_a_test])
    pythia_test_b_dataset = ConcatDataset([pythia_clean_test, pythia_atk_b_test])

    print(f"  Pythia Train:  {len(pythia_train_dataset)} samples")
    print(f"  Pythia Val:    {len(pythia_val_dataset)} samples")
    print(f"  Pythia Test_A: {len(pythia_test_a_dataset)} samples")
    print(f"  Pythia Test_B: {len(pythia_test_b_dataset)} samples  ⚠ 5:1 imbalance\n")

    # Compute class weight for ProfessorCNN weighted BCE training
    print("  Obliczanie wag klas dla ProfessorCNN (Pythia)...")
    pythia_pos_weight = compute_pos_weight(pythia_train_dataset)

    # -----------------------------------------------------------------------
    # Cell 7 — Model initialisation (identical to notebook)
    # -----------------------------------------------------------------------
    print("--- KOMÓRKA 7: FAZA 3 - ARCHITEKTURA MODELU (CNN) ---")

    print("Inicjalizacja modelu dla MNIST (wejście 28x28)...")
    model_mnist  = AnomalyCNN(input_size=28)
    print(model_mnist)

    print("\nInicjalizacja modelu dla Pythia (wejście 70x70)...")
    model_pythia = AnomalyCNN(input_size=70)

    # BCELoss — model outputs probabilities (sigmoid in forward), matches notebook
    criterion        = nn.BCELoss()
    optimizer_mnist  = optim.Adam(model_mnist.parameters(),  lr=0.001)
    optimizer_pythia = optim.Adam(model_pythia.parameters(), lr=0.001)

    print("\nKonfiguracja modelu zakończona sukcesem!")
    print(" - Architektura: 3x Conv2d + MaxPooling -> Flatten -> 2x Dense (Sigmoid)")
    print(" - Funkcja straty: Binary Crossentropy (BCELoss)")
    print(" - Optymalizator: Adam (Learning Rate = 0.001)")

    # -----------------------------------------------------------------------
    # Cell 8 — Training (identical to notebook)
    # -----------------------------------------------------------------------
    print("\n--- KOMÓRKA 8: FAZA 4 - TRENOWANIE MODELU (BASELINE) ---")

    print("Tworzenie DataLoaderów dla danych referencyjnych (MNIST)...")
    train_loader_mnist  = make_dataloader(train_dataset,        BATCH_SIZE, shuffle=True)
    val_loader_mnist    = make_dataloader(val_dataset,          BATCH_SIZE)

    print("Tworzenie DataLoaderów dla danych ukrytych (Pythia)...")
    train_loader_pythia = make_dataloader(pythia_train_dataset, BATCH_SIZE, shuffle=True)
    val_loader_pythia   = make_dataloader(pythia_val_dataset,   BATCH_SIZE)

    print("\n>>> ROZPOCZĘCIE TRENINGU: ZBIÓR MNIST (Clean + Attack_A) <<<")
    model_mnist = train_model(
        model=model_mnist,
        train_loader=train_loader_mnist,
        val_loader=val_loader_mnist,
        criterion=criterion,
        optimizer=optimizer_mnist,
        num_epochs=NUM_EPOCHS,
        patience=PATIENCE,
    )

    print("\n>>> ROZPOCZĘCIE TRENINGU: ZBIÓR PYTHIA (Clean + Attack_A) <<<")
    model_pythia = train_model(
        model=model_pythia,
        train_loader=train_loader_pythia,
        val_loader=val_loader_pythia,
        criterion=criterion,
        optimizer=optimizer_pythia,
        num_epochs=NUM_EPOCHS,
        patience=PATIENCE,
    )

    # -----------------------------------------------------------------------
    # Cell 8b — Train PixelMLP (Pythia)
    # -----------------------------------------------------------------------
    # PixelMLP = Flatten → Linear(4900,1) → Sigmoid  (pure linear-in-pixel-space).
    # Equivalent to logistic regression but trained with AdamW + full-batch so
    # the decoupled weight decay doesn't collapse informative pixel weights.
    # Uses monitor_auc=True because the AUC-ROC loss is a better training signal
    # than BCE when the per-pixel SNR is very low.
    print("\n>>> TRENING: PixelMLP (Pythia) <<<")
    _pixel_m = PixelMLP(input_size=70, hidden=[], activation="gelu", dropout=0.0)
    _pixel_opt = optim.AdamW(_pixel_m.parameters(), lr=0.001, weight_decay=1.0)
    _pixel_full = make_dataloader(
        pythia_train_dataset, batch_size=len(pythia_train_dataset), shuffle=False
    )
    _pixel_sched = optim.lr_scheduler.ReduceLROnPlateau(
        _pixel_opt, mode="max", factor=0.5, patience=15, min_lr=1e-6
    )
    _pixel_m = train_model(
        model=_pixel_m,
        train_loader=_pixel_full,
        val_loader=val_loader_pythia,
        criterion=criterion,
        optimizer=_pixel_opt,
        num_epochs=PYTHIA_PROF_NUM_EPOCHS,
        patience=PYTHIA_PROF_PATIENCE,
        scheduler=_pixel_sched,
        monitor_auc=True,
    )
    pixel_model_pythia: nn.Module = _pixel_m

    # -----------------------------------------------------------------------
    # Cell 8c — Train ProfessorCNN (Pythia) — best CNN from hyperparameter search
    # -----------------------------------------------------------------------
    # ProfessorCNN uses the best architecture found by the professor's grid
    # search (faza_professor_cnn_search.json).  It is a proper convolutional
    # network (3 conv layers + dense head) trained with Adam on mini-batches.
    # Limitation for Pythia: CNNs are translation-equivariant, but the Pythia
    # attack signal is positional (different pixels at different locations),
    # so pooling smears the position information.
    print("\n>>> TRENING: ProfessorCNN (Pythia) <<<")
    _prof_cnn_m = get_professor_cnn_best(input_size=70)
    _prof_cnn_opt = optim.Adam(_prof_cnn_m.parameters(), lr=0.001)
    _prof_cnn_m = train_model(
        model=_prof_cnn_m,
        train_loader=train_loader_pythia,
        val_loader=val_loader_pythia,
        criterion=criterion,
        optimizer=_prof_cnn_opt,
        num_epochs=PYTHIA_PROF_NUM_EPOCHS,
        patience=PYTHIA_PROF_PATIENCE,
        monitor_auc=True,
    )
    prof_cnn_model_pythia: nn.Module = _prof_cnn_m

    # -----------------------------------------------------------------------
    # Cell 8d — Train GBM (Best approach — Pythia)
    # -----------------------------------------------------------------------
    # WHY GBM instead of CNN / PixelMLP?
    #
    # Diagnostic (_ceiling.py) revealed:
    #   - The attack signal exists at ~2094 / 4900 pixels (max |delta|=0.068)
    #   - LR / PixelMLP achieve AUC≈0.55 because L2 regularisation smears
    #     weight evenly across ALL pixels, so 2806 noise pixels drown the signal
    #   - Oracle LR (top-500px by |delta|) achieves AUC=0.845 → signal IS there
    #   - GBM's tree-split mechanism implicitly selects discriminative pixels
    #     from training data alone → AUC=0.665 without any oracle knowledge
    #
    # Approach:
    #   1. Collect flat training arrays from the DataLoader
    #   2. Train sklearn GradientBoostingClassifier (n=300 shallow trees)
    #   3. Wrap in a thin nn.Module so evaluate_model() works unchanged
    # -------------------------------------------------------------------
    from sklearn.ensemble import GradientBoostingClassifier as _GBC
    from sklearn.metrics import roc_auc_score as _rauc

    print("\n>>> INICJALIZACJA I TRENING: GBM (BEST) - PYTHIA <<<")
    print("  Metoda: Gradient Boosting (GBM) — implicitna selekcja cech pikselowych")

    # ── Collect flat training arrays ────────────────────────────────────
    _full_loader = make_dataloader(
        pythia_train_dataset, batch_size=len(pythia_train_dataset), shuffle=False
    )
    _X_tr_np, _y_tr_np = next(iter(_full_loader))
    _X_tr_np = _X_tr_np.numpy().reshape(len(_X_tr_np), -1)   # (1280, 4900)
    _y_tr_np = _y_tr_np.numpy().astype(np.float32)

    # ── Train GBM ───────────────────────────────────────────────────────
    _gbm = _GBC(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=10,
        random_state=42,
    )
    print("  Trenuję GBM (300 drzew, depth=3)…")
    _gbm.fit(_X_tr_np, _y_tr_np)
    print("  Trening GBM zakończony.")

    # ── Thin nn.Module wrapper so evaluate_model() works unchanged ──────
    class _GBMWrapper(nn.Module):
        """Wraps a fitted sklearn GBM as an nn.Module for evaluate_model()."""
        def __init__(self, fitted_gbm):
            super().__init__()
            self._gbm = fitted_gbm

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x_np = x.cpu().numpy().reshape(len(x), -1)
            probs = self._gbm.predict_proba(x_np)[:, 1].astype(np.float32)
            return torch.tensor(probs, device=x.device).unsqueeze(1)

        def __repr__(self) -> str:
            est = self._gbm
            return (
                f"GBMWrapper(n_estimators={est.n_estimators}, "
                f"max_depth={est.max_depth}, "
                f"lr={est.learning_rate}, subsample={est.subsample})"
            )

    prof_model_pythia: nn.Module = _GBMWrapper(_gbm)

    # ── Report val AUC ──────────────────────────────────────────────────
    _scores_v, _yt_v = [], []
    for _xi, _yi in val_loader_pythia:
        with torch.no_grad():
            _scores_v.extend(prof_model_pythia(_xi).cpu().numpy().flatten())
        _yt_v.extend(_yi.numpy().flatten())
    _val_auc = float(_rauc(_yt_v, _scores_v))
    print(f"\n  Val AUC: {_val_auc:.4f}")
    print(prof_model_pythia)

    print("\nGratulacje! Faza 4 zakończona.")
    print("Obydwa modele bazowe (Baseline) zostały wytrenowane wyłącznie z użyciem znanego ataku_a!")

    # -----------------------------------------------------------------------
    # Cell 9 — Evaluation on Test_A (known attack)
    # -----------------------------------------------------------------------
    print("\n--- KOMÓRKA 9: FAZA 5 - EWALUACJA NA ZNANYM ATAKU (TEST_A) ---")

    print("Przygotowanie DataLoaderów testowych (Test_A)...")
    test_a_loader_mnist  = make_dataloader(test_a_dataset,        BATCH_SIZE)
    test_a_loader_pythia = make_dataloader(pythia_test_a_dataset, BATCH_SIZE)

    print("\n>>> EWALUACJA MNIST (Test_A: Clean + Attack_A) <<<")
    res_mnist_a = evaluate_model(model_mnist, test_a_loader_mnist)

    print("\n>>> EWALUACJA PYTHIA (Test_A: Clean + Attack_A) <<<")
    res_pythia_a = evaluate_model(model_pythia, test_a_loader_pythia)

    print("\n>>> EWALUACJA PIXEL MLP - PYTHIA (Test_A: Clean + Attack_A) <<<")
    res_pixel_pythia_a = evaluate_model(pixel_model_pythia, test_a_loader_pythia)

    print("\n>>> EWALUACJA PROFESSOR CNN - PYTHIA (Test_A: Clean + Attack_A) <<<")
    res_prof_cnn_pythia_a = evaluate_model(prof_cnn_model_pythia, test_a_loader_pythia)

    print("\n>>> EWALUACJA GBM (BEST) - PYTHIA (Test_A: Clean + Attack_A) <<<")
    res_prof_pythia_a = evaluate_model(prof_model_pythia, test_a_loader_pythia)

    print("\n[WNIOSEK KROKU 1]: Jeśli metryki są bardzo wysokie (np. blisko 1.0/100%),")
    print("oznacza to, że nasz bazowy model skutecznie nauczył się rozpoznawać ataki,")
    print("które demonstrowaliśmy mu podczas sesji treningowej.")

    # -----------------------------------------------------------------------
    # Cell 10 — Evaluation on Test_B (unknown attack)
    # -----------------------------------------------------------------------
    print("\n--- KOMÓRKA 10: FAZA 5 - EWALUACJA NA NIEZNANYM ATAKU (TEST_B) ---")

    print("Przygotowanie DataLoaderów testowych (Test_B)...")
    test_b_loader_mnist  = make_dataloader(test_b_dataset,        BATCH_SIZE)
    test_b_loader_pythia = make_dataloader(pythia_test_b_dataset, BATCH_SIZE)

    print("\n>>> EWALUACJA MNIST (Test_B: Clean + Attack_B / OOD) <<<")
    res_mnist_b = evaluate_model(model_mnist, test_b_loader_mnist)

    print("\n>>> EWALUACJA PYTHIA (Test_B: Clean + Attack_B) <<<")
    res_pythia_b = evaluate_model(model_pythia, test_b_loader_pythia)

    print("\n>>> EWALUACJA PIXEL MLP - PYTHIA (Test_B: Clean + Attack_B) <<<")
    res_pixel_pythia_b = evaluate_model(pixel_model_pythia, test_b_loader_pythia)

    print("\n>>> EWALUACJA PROFESSOR CNN - PYTHIA (Test_B: Clean + Attack_B) <<<")
    res_prof_cnn_pythia_b = evaluate_model(prof_cnn_model_pythia, test_b_loader_pythia)

    print("\n>>> EWALUACJA GBM (BEST) - PYTHIA (Test_B: Clean + Attack_B) <<<")
    res_prof_pythia_b = evaluate_model(prof_model_pythia, test_b_loader_pythia)

    print("\n[ANALIZA KROKU 2 - DRASTYCZNY SPADEK SKUTECZNOŚCI]:")
    print("Zapewne zauważyłeś drastyczny spadek skuteczności (szczególnie Recall, F1 i AUC)")
    print("na zbiorze Test_B w porównaniu do wyników ze zbioru Test_A.")
    print("Klasyfikator nadzorowany nauczył się specyficznych cech Ataku A.")
    print("Nie zbudował jednak pojęcia 'czym jest obraz normalny'.")

    # -----------------------------------------------------------------------
    # Cell 11 — Save results
    # -----------------------------------------------------------------------
    print("\n--- KOMÓRKA 11: ZAPIS WYNIKÓW ---")

    results_summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "MNIST_Experiment": {
            "Test_A_Znany_Atak": parse_results(res_mnist_a),
            "Test_B_Nieznany_Atak": parse_results(res_mnist_b),
        },
        "PYTHIA_Experiment": {
            "AnomalyCNN_Baseline": {
                "Test_A_Znany_Atak": parse_results(res_pythia_a),
                "Test_B_Nieznany_Atak": parse_results(res_pythia_b),
            },
            "PixelMLP": {
                "Test_A_Znany_Atak": parse_results(res_pixel_pythia_a),
                "Test_B_Nieznany_Atak": parse_results(res_pixel_pythia_b),
            },
            "ProfessorCNN": {
                "Test_A_Znany_Atak": parse_results(res_prof_cnn_pythia_a),
                "Test_B_Nieznany_Atak": parse_results(res_prof_cnn_pythia_b),
            },
            "GBM_Best": {
                "Test_A_Znany_Atak": parse_results(res_prof_pythia_a),
                "Test_B_Nieznany_Atak": parse_results(res_prof_pythia_b),
            },
        },
    }

    save_results(results_summary, "faza1_wyniki_eksperymentu.json")

    # Comparison bar chart — Pythia only (all 4 models)
    _plot_pythia_model_comparison(
        baseline_a=parse_results(res_pythia_a),
        baseline_b=parse_results(res_pythia_b),
        pixel_a=parse_results(res_pixel_pythia_a),
        pixel_b=parse_results(res_pixel_pythia_b),
        cnn_a=parse_results(res_prof_cnn_pythia_a),
        cnn_b=parse_results(res_prof_cnn_pythia_b),
        gbm_a=parse_results(res_prof_pythia_a),
        gbm_b=parse_results(res_prof_pythia_b),
        save_path=PLOTS / "pythia_model_comparison.png",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
