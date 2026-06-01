"""
step2.py — Anomaly Detection: Generalisation via Multi-Attack Training (Step 2)
================================================================================

Implements the Step 2 research question:

    *Does training on more diverse attack types (A_1, …, A_n) improve
    the detector's ability to identify a completely unseen attack A_{n+1}?*

Methodology
-----------
For each dataset (MNIST and Pythia) we prepare K attack types using all
five contamination methods from ``attacks/contamination.py``.  We then
run K−1 progressive training rounds:

    Round n  (1 ≤ n < K):
        • Training data:  clean ∪ balanced_sample(A_1, …, A_n)
        • Validation:     20 % held-out from the same combined set
        • Test set:       clean_test ∪ A_{n+1}   (attack n+1 is NEVER in training)
        • Metric:         Accuracy, Precision, Recall, F1, AUC-ROC on the test set

After all rounds the metrics are plotted against n and saved to
``faza2_wyniki_generalizacji.json`` and ``step2_generalization.png``.

Class balance
-------------
To isolate the effect of attack diversity from the effect of additional
training data, all training sets are kept balanced at a 1:1 clean-to-attack
ratio.  When n attack types are present, each contributes
``total_clean_samples // n`` images to the attack class.

Contamination methods (MNIST, 6 attacks total)
----------------------------------------------
  A1 — Gaussian noise        (σ = 0.4)
  A2 — Salt & pepper         (p = 0.15)
  A3 — Geometric distortion  (max_displacement = 5 px)
  A4 — Blended attack        (α = 0.30, random noise pattern)
  A5 — Backdoor trigger      (5×5 white square, bottom-right corner)
  A6 — OOD replacement       (Fashion-MNIST, no pixel transform)

Pythia uses all 8 labelled attack partitions (attack_a … attack_h).

Usage
-----
    # From the CH2/ directory:
    python step2.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Subset, random_split

from lib import (
    AnomalyCNN,
    check_pythia_available,
    evaluate_model,
    get_professor_cnn_best,
    load_pythia_data,
    make_dataloader,
    prepare_clean_data,
    save_results,
    split_train_test,
    visualize_samples,
)
from attacks.contamination import (
    make_backdoor_attack,
    make_blended_attack,
    make_gaussian_attack,
    make_geometric_attack,
    make_ood_attack,
    make_salt_pepper_attack,
)

# ---------------------------------------------------------------------------
# Global hyperparameters — identical to notebook ch2_step1_v2.ipynb
# ---------------------------------------------------------------------------
BATCH_SIZE = 64
NUM_EPOCHS = 15
PATIENCE = 3
RANDOM_SEED = 2137
# Cap MNIST clean-class training samples for Step 2.  The full 60 000 samples
# per class make the order-sensitivity experiment
# impractically slow.  10 000 per class keeps class balance, gives strong
# generalisation signal, and reduces per-round training time ~6×.
MNIST_STEP2_SAMPLES = 10_000
NUM_PERMUTATIONS = 5  # for permutation-based feature importance (Step 3)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ===========================================================================
# HELPERS
# ===========================================================================


def balanced_attack_subset(
    attack_datasets: list,
    total_attack_samples: int,
) -> ConcatDataset:
    """Sample equal proportions from each attack type to maintain class balance.

    When training on n attack types we want the total attack-class size to
    equal the clean-class size.  This function takes
    ``total_attack_samples // n`` random samples from each dataset and
    concatenates them.

    Parameters
    ----------
    attack_datasets : list[Dataset]
        List of n attack TensorDatasets (label = 1).
    total_attack_samples : int
        Target total number of attack-class training samples.
        Typically ``len(clean_train_dataset)``.

    Returns
    -------
    ConcatDataset
        Balanced concatenation of random subsets, one per attack type.

    Notes
    -----
    If ``samples_per_type`` exceeds the length of a particular dataset,
    all available samples from that dataset are used (no over-sampling).
    """
    n = len(attack_datasets)
    samples_per_type = total_attack_samples // n
    subsets = []
    for ds in attack_datasets:
        actual = min(samples_per_type, len(ds))
        # Random permutation for unbiased subsetting
        indices = torch.randperm(len(ds))[:actual].tolist()
        subsets.append(Subset(ds, indices))
    return ConcatDataset(subsets)


def run_progressive_training(
    clean_train,
    clean_test,
    attack_train_datasets: list,
    attack_test_datasets: list,
    attack_names: list[str],
    input_size: int,
    experiment_name: str,
    model_factory=None,
    num_epochs: int | None = None,
    patience: int | None = None,
    monitor_auc: bool = False,
) -> list[dict]:
    """Execute the progressive multi-attack training experiment.

    For n = 1, 2, …, K−1 (where K = len(attack_train_datasets)):

    1. Build a **balanced** training set:
         clean_train  ∪  balanced_sample(A_1, …, A_n)
         total size ≈ 2 × len(clean_train), class ratio 1:1.
    2. Split 80/20 into train / val.
    3. Construct the test set:
         clean_test  ∪  attack_test_datasets[n]   (A_{n+1}, never seen)
    4. Train a fresh :class:`AnomalyCNN` with early stopping.
    5. Evaluate and record metrics.

    Parameters
    ----------
    clean_train : Dataset
        Clean-class training data (label = 0).
    clean_test : Dataset
        Clean-class test data (label = 0).
    attack_train_datasets : list[Dataset]
        K attack-class training datasets [A_1_train, …, A_K_train].
    attack_test_datasets : list[Dataset]
        K attack-class test datasets [A_1_test, …, A_K_test].
    attack_names : list[str]
        Human-readable name for each attack (used in logs and plots).
    input_size : int
        Spatial size H of square input images (28 for MNIST, 70 for Pythia).
    experiment_name : str
        Label used in console output (e.g. ``'MNIST'`` or ``'Pythia'``).
    model_factory : callable(input_size) -> nn.Module, optional
        Factory for the model trained in each round.  Defaults to
        ``AnomalyCNN(input_size)`` when ``None``.

    Returns
    -------
    list[dict]
        One dict per round with keys:
        ``n_training_attacks``, ``training_attacks``, ``test_attack``,
        ``Accuracy``, ``Precision``, ``Recall``, ``F1_Score``, ``AUC_ROC``.
    """
    n_attacks = len(attack_train_datasets)
    results: list[dict] = []

    for n in range(1, n_attacks):  # n = number of attack types in training set
        held_out_name = attack_names[n]
        bar = "=" * 65
        print(f"\n{bar}")
        print(f"  [{experiment_name}] Round n={n}")
        print(f"  Training on:  clean + {attack_names[:n]}")
        print(f"  Testing on:   {held_out_name}  (unseen)")
        print(f"{bar}")

        # ------------------------------------------------------------------
        # Build balanced training set
        # ------------------------------------------------------------------
        n_clean = len(clean_train)
        attack_subset = balanced_attack_subset(attack_train_datasets[:n], n_clean)
        combined_tv = ConcatDataset([clean_train, attack_subset])

        # The effective class ratio after balancing:
        #   clean: n_clean,  attack: n_clean  →  1:1 ratio
        # (slight rounding when n_clean is not divisible by n)

        tv_train_size = int(0.8 * len(combined_tv))
        tv_val_size = len(combined_tv) - tv_train_size
        train_ds, val_ds = random_split(combined_tv, [tv_train_size, tv_val_size])

        # Test set: clean_test + unseen attack A_{n+1}
        test_ds = ConcatDataset([clean_test, attack_test_datasets[n]])

        print(
            f"  Train: {len(train_ds):>6} | Val: {len(val_ds):>6} | "
            f"Test: {len(test_ds):>6}  "
            f"(clean={len(clean_test)}, attack_{n+1}={len(attack_test_datasets[n])})"
        )

        # ------------------------------------------------------------------
        # DataLoaders
        # ------------------------------------------------------------------
        train_loader = make_dataloader(train_ds, BATCH_SIZE, shuffle=True)
        val_loader   = make_dataloader(val_ds,   BATCH_SIZE)
        test_loader  = make_dataloader(test_ds,  BATCH_SIZE)

        # ------------------------------------------------------------------
        # Fresh model + optimiser for each round
        # (models trained on n attacks are independent experiments)
        # ------------------------------------------------------------------
        if model_factory is None:
            model = AnomalyCNN(input_size=input_size)
        else:
            model = model_factory(input_size)
        criterion = nn.BCELoss()  # model outputs probabilities via sigmoid
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        from lib import train_model  # local import to keep namespace clean
        model = train_model(
            model, train_loader, val_loader, criterion, optimizer,
            num_epochs=num_epochs if num_epochs is not None else NUM_EPOCHS,
            patience=patience if patience is not None else PATIENCE,
            monitor_auc=monitor_auc,
        )

        # ------------------------------------------------------------------
        # Evaluation on the unseen attack
        # ------------------------------------------------------------------
        print(f"\n  Evaluation on unseen attack '{held_out_name}':")
        acc, prec, rec, f1, auc = evaluate_model(model, test_loader)

        results.append(
            {
                "n_training_attacks": n,
                "training_attacks": attack_names[:n],
                "test_attack": held_out_name,
                "Accuracy": float(acc),
                "Precision": float(prec),
                "Recall": float(rec),
                "F1_Score": float(f1),
                "AUC_ROC": float(auc) if auc == auc else None,  # None for NaN
            }
        )

    return results


# ===========================================================================
# GBM PROGRESSIVE TRAINING  (sklearn GradientBoostingClassifier)
# ===========================================================================


def run_progressive_training_gbm(
    clean_train,
    clean_test,
    attack_train_datasets: list,
    attack_test_datasets: list,
    attack_names: list[str],
    experiment_name: str = "Pythia (GBM)",
) -> list[dict]:
    """Progressive multi-attack training using GBM instead of AnomalyCNN.

    For each round n a fresh GradientBoostingClassifier is trained on flat
    pixel arrays collected from the balanced training DataLoader.  GBM
    implicitly selects discriminative pixels via tree splits, which is the
    key advantage over LR / PixelMLP on the low-SNR Pythia dataset.

    Returns
    -------
    list[dict]
        Same schema as :func:`run_progressive_training`.
    """
    from sklearn.ensemble import GradientBoostingClassifier as _GBC

    n_attacks = len(attack_train_datasets)
    results: list[dict] = []

    for n in range(1, n_attacks):
        held_out_name = attack_names[n]
        bar = "=" * 65
        print(f"\n{bar}")
        print(f"  [{experiment_name}] Round n={n}")
        print(f"  Training on:  clean + {attack_names[:n]}")
        print(f"  Testing on:   {held_out_name}  (unseen)")
        print(f"{bar}")

        # Build balanced training set (same logic as run_progressive_training)
        n_clean = len(clean_train)
        attack_subset = balanced_attack_subset(attack_train_datasets[:n], n_clean)
        combined_tv = ConcatDataset([clean_train, attack_subset])

        tv_train_size = int(0.8 * len(combined_tv))
        tv_val_size = len(combined_tv) - tv_train_size
        train_ds, _ = random_split(combined_tv, [tv_train_size, tv_val_size])

        test_ds = ConcatDataset([clean_test, attack_test_datasets[n]])
        print(
            f"  Train: {len(train_ds):>6} | Test: {len(test_ds):>6}  "
            f"(clean={len(clean_test)}, attack_{n+1}={len(attack_test_datasets[n])})"
        )

        # Collect flat numpy arrays (GBM needs X: (N, 4900), y: (N,))
        full_loader = make_dataloader(train_ds, batch_size=len(train_ds), shuffle=False)
        _Xb, _yb = next(iter(full_loader))
        X_tr = _Xb.numpy().reshape(len(_Xb), -1)
        y_tr = _yb.numpy().astype(np.float32)

        gbm = _GBC(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=10, random_state=42,
        )
        print(f"  Training GBM (300 trees, depth=3)…")
        gbm.fit(X_tr, y_tr)

        # Thin nn.Module wrapper for evaluate_model compatibility
        class _GBMWrap(nn.Module):
            def __init__(self, g):
                super().__init__()
                self._g = g
            def forward(self, x):
                xn = x.cpu().numpy().reshape(len(x), -1)
                p = self._g.predict_proba(xn)[:, 1].astype(np.float32)
                return torch.tensor(p, device=x.device).unsqueeze(1)

        model = _GBMWrap(gbm)
        test_loader = make_dataloader(test_ds, BATCH_SIZE)

        print(f"\n  Evaluation on unseen attack '{held_out_name}':")
        acc, prec, rec, f1, auc = evaluate_model(model, test_loader)

        results.append(
            {
                "n_training_attacks": n,
                "training_attacks": attack_names[:n],
                "test_attack": held_out_name,
                "Accuracy": float(acc),
                "Precision": float(prec),
                "Recall": float(rec),
                "F1_Score": float(f1),
                "AUC_ROC": float(auc) if auc == auc else None,
            }
        )

    return results


# ===========================================================================
# ATTACK SUBSET COMPARISON
# ===========================================================================

# Default predefined subsets for Pythia (attack_a … attack_h).
# Organised to answer two questions:
#   (A) Does *how many* attack types matter?  (sequential growth 1→4)
#   (B) Does *which* attack types matter?     (same-size diverse vs sequential)
PYTHIA_ATTACK_SUBSETS: list[tuple[str, list[str]]] = [
    # ── Size 1 ────────────────────────────────────────────────────────────
    ("1 — A only",          ["attack_a"]),
    # ── Size 2 ────────────────────────────────────────────────────────────
    ("2 — A+B (seq)",       ["attack_a", "attack_b"]),
    # ── Size 3 ────────────────────────────────────────────────────────────
    ("3 — A+B+C (seq)",     ["attack_a", "attack_b", "attack_c"]),
    ("3 — A+B+E (diverse)", ["attack_a", "attack_b", "attack_e"]),   # professor example
    ("3 — A+D+G (spread)",  ["attack_a", "attack_d", "attack_g"]),
    # ── Size 4 ────────────────────────────────────────────────────────────
    ("4 — A+B+C+D (seq)",   ["attack_a", "attack_b", "attack_c", "attack_d"]),
    ("4 — A+B+E+G (div)",   ["attack_a", "attack_b", "attack_e", "attack_g"]),
    ("4 — A+C+E+G (alt)",   ["attack_a", "attack_c", "attack_e", "attack_g"]),
    # ── Size 6 ────────────────────────────────────────────────────────────
    ("6 — A-F (majority)",  ["attack_a", "attack_b", "attack_c",
                              "attack_d", "attack_e", "attack_f"]),
]


def run_subset_comparison(
    clean_train,
    clean_test,
    attack_train_datasets: list,
    attack_test_datasets: list,
    attack_names: list[str],
    input_size: int,
    subsets: list[tuple[str, list[str]]] | None = None,
    model_factory=None,
    num_epochs: int | None = None,
    patience: int | None = None,
    monitor_auc: bool = False,
) -> list[dict]:
    """Train a fresh model per attack subset and evaluate on all unseen attacks.

    For each (label, train_attack_names) entry in *subsets*:
      1. Build a balanced training set: clean_train ∪ balanced_sample(chosen attacks).
      2. 80/20 train/val split and train a fresh model with early stopping.
      3. Evaluate on clean_test ∪ A_i for every attack *not* in the subset.
      4. Record per-attack metrics and aggregate means.

    Parameters
    ----------
    subsets : list of (label, attack_names_to_train_on)
        Defaults to :data:`PYTHIA_ATTACK_SUBSETS`.

    Returns
    -------
    list[dict]
        One dict per subset, keys:
        ``subset_label``, ``train_attacks``, ``unseen_attacks``,
        ``per_attack_metrics``, ``mean_Accuracy``, ``mean_F1_Score``,
        ``mean_AUC_ROC``.
    """
    from lib import train_model  # local import to keep namespace clean

    if subsets is None:
        subsets = PYTHIA_ATTACK_SUBSETS

    _epochs = num_epochs if num_epochs is not None else NUM_EPOCHS
    _patience = patience if patience is not None else PATIENCE

    results: list[dict] = []

    for subset_label, train_attack_names in subsets:
        bar = "=" * 65
        print(f"\n{bar}")
        print(f"  [Subset] {subset_label}")
        print(f"  Training on: {train_attack_names}")
        print(f"{bar}")

        # Resolve attack indices (skip names not in this dataset)
        train_idx = [
            attack_names.index(n) for n in train_attack_names if n in attack_names
        ]
        if not train_idx:
            print(f"  WARNING: no valid attacks found — skipping '{subset_label}'")
            continue
        unseen_idx = [i for i in range(len(attack_names)) if i not in train_idx]
        unseen_names = [attack_names[i] for i in unseen_idx]

        print(f"  Unseen (test only): {unseen_names}")

        # ── Build balanced training set ──────────────────────────────────
        n_clean = len(clean_train)
        attack_subset_ds = balanced_attack_subset(
            [attack_train_datasets[i] for i in train_idx], n_clean
        )
        combined_tv = ConcatDataset([clean_train, attack_subset_ds])

        tv_train_size = int(0.8 * len(combined_tv))
        tv_val_size = len(combined_tv) - tv_train_size
        train_ds, val_ds = random_split(combined_tv, [tv_train_size, tv_val_size])

        print(
            f"  Train: {len(train_ds):>6} | Val: {len(val_ds):>6}"
        )

        train_loader = make_dataloader(train_ds, BATCH_SIZE, shuffle=True)
        val_loader   = make_dataloader(val_ds,   BATCH_SIZE)

        # ── Train fresh model ─────────────────────────────────────────────
        if model_factory is None:
            model = AnomalyCNN(input_size=input_size)
        else:
            model = model_factory(input_size)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        model = train_model(
            model, train_loader, val_loader, criterion, optimizer,
            num_epochs=_epochs, patience=_patience, monitor_auc=monitor_auc,
        )

        # ── Evaluate on every unseen attack ───────────────────────────────
        per_attack: dict[str, dict] = {}
        for i in unseen_idx:
            name = attack_names[i]
            test_ds = ConcatDataset([clean_test, attack_test_datasets[i]])
            test_loader = make_dataloader(test_ds, BATCH_SIZE)
            acc, prec, rec, f1, auc = evaluate_model(model, test_loader)
            per_attack[name] = {
                "Accuracy":  float(acc),
                "Precision": float(prec),
                "Recall":    float(rec),
                "F1_Score":  float(f1),
                "AUC_ROC":   float(auc) if auc == auc else None,
            }
            print(
                f"    {name:12s}  Acc={acc:.3f}  F1={f1:.3f}  AUC={auc:.3f}"
            )

        # ── Aggregate means (ignore None AUC) ────────────────────────────
        def _mean(key: str) -> float:
            vals = [v[key] for v in per_attack.values() if v[key] is not None]
            return float(np.mean(vals)) if vals else float("nan")

        results.append(
            {
                "subset_label":    subset_label,
                "train_attacks":   [attack_names[i] for i in train_idx],
                "unseen_attacks":  unseen_names,
                "per_attack_metrics": per_attack,
                "mean_Accuracy":  _mean("Accuracy"),
                "mean_Precision": _mean("Precision"),
                "mean_Recall":    _mean("Recall"),
                "mean_F1_Score":  _mean("F1_Score"),
                "mean_AUC_ROC":   _mean("AUC_ROC"),
            }
        )

    return results


def plot_subset_comparison(
    results: list[dict],
    plots_dir: Path | str,
    title: str = "Pythia — Attack Subset Selection Comparison",
    filename: str = "step2_subset_comparison.png",
) -> None:
    """Two-panel figure comparing attack subset strategies.

    Top panel: grouped bar chart of mean AUC-ROC and mean F1-Score on
    unseen attacks per subset, with individual-attack scatter.

    Bottom panel: heatmap of per-attack AUC-ROC across all subsets,
    making it easy to spot which unseen attack is hardest regardless of
    the training set.

    Parameters
    ----------
    results : list[dict]
        Output of :func:`run_subset_comparison`.
    plots_dir : Path or str
        Output directory.
    title : str
        Figure super-title.
    filename : str
        Output filename inside *plots_dir*.
    """
    plots_dir = Path(plots_dir)

    labels     = [r["subset_label"]  for r in results]
    mean_aucs  = [r["mean_AUC_ROC"]  for r in results]
    mean_f1s   = [r["mean_F1_Score"] for r in results]

    # Collect all unseen attack names that appear in any result
    all_unseen: list[str] = []
    for r in results:
        for k in r["unseen_attacks"]:
            if k not in all_unseen:
                all_unseen.append(k)
    all_unseen.sort()

    # Build AUC matrix: rows = subsets, cols = attacks (NaN where attack was in training)
    auc_matrix = np.full((len(results), len(all_unseen)), np.nan)
    for row_i, r in enumerate(results):
        for col_j, atk in enumerate(all_unseen):
            if atk in r["per_attack_metrics"]:
                auc_matrix[row_i, col_j] = r["per_attack_metrics"][atk]["AUC_ROC"] or np.nan

    x = np.arange(len(labels))
    bar_w = 0.35

    fig = plt.figure(figsize=(max(14, len(labels) * 1.6), 13))
    gs  = fig.add_gridspec(2, 1, height_ratios=[1, 0.9], hspace=0.45)

    # ── Top: grouped bar chart ───────────────────────────────────────────
    ax_bar = fig.add_subplot(gs[0])
    bars_auc = ax_bar.bar(x - bar_w / 2, mean_aucs, bar_w,
                          color="steelblue", alpha=0.85, label="Mean AUC-ROC (unseen)")
    bars_f1  = ax_bar.bar(x + bar_w / 2, mean_f1s,  bar_w,
                          color="darkorange", alpha=0.85, label="Mean F1-Score (unseen)")

    # Individual attack scatter points
    for row_i, r in enumerate(results):
        for atk in r["unseen_attacks"]:
            m = r["per_attack_metrics"].get(atk)
            if m and m["AUC_ROC"] is not None:
                ax_bar.scatter(row_i - bar_w / 2, m["AUC_ROC"],
                               color="navy", s=22, alpha=0.6, zorder=5)
            if m and m["F1_Score"] is not None:
                ax_bar.scatter(row_i + bar_w / 2, m["F1_Score"],
                               color="saddlebrown", s=22, alpha=0.6, zorder=5)

    # Value annotations on bars
    for bar in bars_auc:
        h = bar.get_height()
        if h == h:
            ax_bar.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=8)
    for bar in bars_f1:
        h = bar.get_height()
        if h == h:
            ax_bar.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=8)

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax_bar.set_ylim(0, 1.15)
    ax_bar.axhline(0.5, color="gray", linestyle=":", linewidth=0.9)
    ax_bar.set_ylabel("Metric value", fontsize=11)
    ax_bar.set_title(
        "Mean metrics on unseen attacks  (dots = individual attacks)",
        fontsize=12, fontweight="bold",
    )
    ax_bar.legend(fontsize=10)
    ax_bar.grid(axis="y", alpha=0.3, linestyle="--")

    # ── Bottom: heatmap ───────────────────────────────────────────────────
    ax_hm = fig.add_subplot(gs[1])
    im = ax_hm.imshow(auc_matrix, aspect="auto", cmap="RdYlGn",
                      vmin=0.4, vmax=1.0, interpolation="nearest")
    plt.colorbar(im, ax=ax_hm, label="AUC-ROC", shrink=0.8)

    ax_hm.set_xticks(range(len(all_unseen)))
    ax_hm.set_xticklabels(
        [a.replace("attack_", "atk_") for a in all_unseen],
        rotation=40, ha="right", fontsize=9,
    )
    ax_hm.set_yticks(range(len(labels)))
    ax_hm.set_yticklabels(labels, fontsize=9)
    ax_hm.set_title(
        "Per-attack AUC-ROC heatmap  (grey = attack was in training set)",
        fontsize=12, fontweight="bold",
    )

    # Annotate cells
    for i in range(len(results)):
        for j in range(len(all_unseen)):
            v = auc_matrix[i, j]
            if v == v:
                ax_hm.text(j, i, f"{v:.2f}", ha="center", va="center",
                           fontsize=7.5,
                           color="black" if 0.45 < v < 0.85 else "white")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    out = plots_dir / filename
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# ===========================================================================
# PLOTTING
# ===========================================================================


def plot_generalization_results(
    mnist_results: list[dict],
    pythia_results: list[dict],
    output_path: Path | str = Path("step2_generalization.png"),
    labels: tuple[str, str] = ("MNIST (28×28)", "Pythia (70×70)"),
) -> None:
    """Plot F1-Score and AUC-ROC vs. number of training attack types.

    Two subplots side-by-side (MNIST left, Pythia right).
    Each subplot shows three metrics (Accuracy, F1, AUC-ROC) as line
    plots with markers over the rounds n = 1, 2, \u2026.

    The x-tick labels show both n and the name of the held-out test
    attack, making it easy to see which type was evaluated at each step.

    Parameters
    ----------
    mnist_results : list[dict]
        Output of ``run_progressive_training`` for MNIST.
    pythia_results : list[dict]
        Output of ``run_progressive_training`` for Pythia.
    output_path : Path | str
        File path for the saved PNG figure (default ``Path('step2_generalization.png')``).
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(
        "Step 2: Generalisation — All Metrics vs. Number of Training Attack Types",
        fontsize=13,
        fontweight="bold",
    )

    # (metric_key, colour, marker, y-annotation offset in points)
    # Offsets staggered so labels don't overlap when lines are close.
    metric_specs = [
        ("Accuracy",  "steelblue",   "o",  +14),
        ("Precision", "darkorange",  "s",  +7),
        ("Recall",    "crimson",     "^",  -12),
        ("F1_Score",  "forestgreen", "D",  +21),
        ("AUC_ROC",   "purple",      "v",  -19),
    ]

    for ax, results, title in zip(
        axes,
        [mnist_results, pythia_results],
        list(labels),
    ):
        if not results:
            ax.set_title(f"{title}\n(no results)")
            continue

        ns = [r["n_training_attacks"] for r in results]
        held_out = [r["test_attack"] for r in results]

        for metric, color, marker, y_offset in metric_specs:
            values = [r[metric] if r[metric] is not None else float("nan") for r in results]
            ax.plot(ns, values, marker=marker, label=_METRIC_LABELS[metric], color=color, linewidth=2)
            for x, y in zip(ns, values):
                if y == y:  # not NaN
                    ax.annotate(
                        f"{y:.2f}",
                        (x, y),
                        textcoords="offset points",
                        xytext=(0, y_offset),
                        ha="center",
                        fontsize=7,
                        color=color,
                    )

        ax.set_xlabel("n  (number of attack types in training)", fontsize=10)
        ax.set_ylabel("Metric value", fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(ns)
        ax.set_xticklabels(
            [f"n={n}\n→test:{ho}" for n, ho in zip(ns, held_out)],
            fontsize=7,
        )
        ax.set_ylim(-0.05, 1.10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.axhline(y=0.5, color="gray", linestyle=":", linewidth=0.8, label="chance")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [plot] Saved → {output_path}")


def print_summary_table(results: list[dict], name: str) -> None:
    """Print a compact summary table for one experiment.

    Parameters
    ----------
    results : list[dict]
        Round results from ``run_progressive_training``.
    name : str
        Dataset name (``'MNIST'`` or ``'Pythia'``).
    """
    print(f"\n{'─'*75}")
    print(f"  {name} — F1-Score and AUC-ROC vs. n")
    print(f"{'─'*75}")
    print(f"  {'n':>3}  {'test attack':<22}  {'Accuracy':>8}  {'F1':>8}  {'AUC-ROC':>8}")
    print(f"  {'─'*3}  {'─'*22}  {'─'*8}  {'─'*8}  {'─'*8}")
    for r in results:
        auc = f"{r['AUC_ROC']:.4f}" if r["AUC_ROC"] is not None else "  NaN"
        print(
            f"  {r['n_training_attacks']:>3}  {r['test_attack']:<22}  "
            f"{r['Accuracy']:>8.4f}  {r['F1_Score']:>8.4f}  {auc:>8}"
        )
    print(f"{'─'*75}")


# ===========================================================================
# ADDITIONAL PLOTS — constants and helpers
# ===========================================================================

# Ordered metric list used by every new plot function.
METRICS: list[str] = ["Accuracy", "Precision", "Recall", "F1_Score", "AUC_ROC"]

_METRIC_LABELS: dict[str, str] = {
    "Accuracy":  "Accuracy",
    "Precision": "Precision",
    "Recall":    "Recall",
    "F1_Score":  "F1-Score",
    "AUC_ROC":   "AUC-ROC",
}

_METRIC_COLORS: dict[str, str] = {
    "Accuracy":  "steelblue",
    "Precision": "darkorange",
    "Recall":    "crimson",
    "F1_Score":  "forestgreen",
    "AUC_ROC":   "purple",
}


def _mvals(results: list[dict], metric: str) -> list[float]:
    """Extract metric values from result dicts, substituting None with NaN."""
    return [r[metric] if r[metric] is not None else float("nan") for r in results]


def _ns(results: list[dict]) -> list[int]:
    """Extract the n_training_attacks column from result dicts."""
    return [r["n_training_attacks"] for r in results]


def plot_per_metric(
    mnist_results: list[dict],
    pythia_results: list[dict],
    plots_dir: Path | str,
    labels: tuple[str, str] = ("MNIST", "Pythia"),
    suffix: str = "",
) -> None:
    """2×3 grid of per-metric generalisation curves with value annotations.

    Five metric panels (Accuracy, Precision, Recall, F1, AUC-ROC) in a 2×3
    layout; both datasets per panel with exact-value labels offset left/right
    so they don't collide.  The sixth panel holds the shared legend.

    Parameters
    ----------
    mnist_results, pythia_results : list[dict]
        Per-round result dicts from :func:`run_progressive_training`.
    plots_dir : Path | str
        Output directory.  File will be ``per_metric_curves{suffix}.png``.
    labels : tuple[str, str]
        Display names for the two datasets (default: ``('MNIST', 'Pythia')``).
    suffix : str
        Optional filename suffix, e.g. ``'_variants'``.
    """
    plots_dir = Path(plots_dir)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"Step 2 — Generalisation Curves per Metric ({labels[0]} vs. {labels[1]})",
        fontsize=14, fontweight="bold",
    )
    axes_flat = axes.flatten()

    for idx, metric in enumerate(METRICS):
        ax = axes_flat[idx]
        color_m, color_p = "steelblue", "darkorange"
        if mnist_results:
            ns_m, vals_m = _ns(mnist_results), _mvals(mnist_results, metric)
            ax.plot(ns_m, vals_m, "o-", label=labels[0], color=color_m,
                    linewidth=2.5, markersize=7)
            for x, y in zip(ns_m, vals_m):
                if y == y:
                    ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                                xytext=(-14, 5), ha="center", fontsize=8.5,
                                color=color_m)
        if pythia_results:
            ns_p, vals_p = _ns(pythia_results), _mvals(pythia_results, metric)
            ax.plot(ns_p, vals_p, "s--", label=labels[1], color=color_p,
                    linewidth=2.5, markersize=7)
            for x, y in zip(ns_p, vals_p):
                if y == y:
                    ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                                xytext=(14, 5), ha="center", fontsize=8.5,
                                color=color_p)
        ax.set_title(_METRIC_LABELS[metric], fontsize=13, fontweight="bold")
        ax.set_xlabel("n (attack types in training)", fontsize=10)
        ax.set_ylabel("Metric value", fontsize=10)
        ax.set_ylim(-0.05, 1.10)
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.9)
        ax.grid(True, alpha=0.3, linestyle="--")
        ref = mnist_results or pythia_results
        if ref:
            ax.set_xticks(_ns(ref))

    # Sixth panel: shared legend at readable size
    ax_leg = axes_flat[5]
    ax_leg.axis("off")
    handles = [
        plt.Line2D([0], [0], color="steelblue",  marker="o", linewidth=2.5,
                   markersize=9, label=labels[0]),
        plt.Line2D([0], [0], color="darkorange", marker="s", linewidth=2.5,
                   markersize=9, linestyle="--", label=labels[1]),
        plt.Line2D([0], [0], color="gray", linewidth=0.9, linestyle=":",
                   label="chance (0.5)"),
    ]
    ax_leg.legend(handles=handles, loc="center", fontsize=13, frameon=False,
                  title="Dataset", title_fontsize=13)

    plt.tight_layout()
    out = plots_dir / f"per_metric_curves{suffix}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] Saved → {out}")


def plot_heatmap(
    mnist_results: list[dict],
    pythia_results: list[dict],
    plots_dir: Path | str,
    labels: tuple[str, str] = ("MNIST (28×28)", "Pythia (70×70)"),
    suffix: str = "",
) -> None:
    """Metric × round heatmap (YlOrRd colour scale, darker = higher).

    Rows are metrics, columns are progressive rounds (n=1, 2, …).
    Cell values are annotated in black/white for readability.

    Parameters
    ----------
    mnist_results, pythia_results : list[dict]
        Per-round result dicts.
    plots_dir : Path | str
        Output directory.  File will be ``metric_heatmap{suffix}.png``.
    labels : tuple[str, str]
        Display names for the two panels.
    suffix : str
        Optional filename suffix, e.g. ``'_variants'``.
    """
    plots_dir = Path(plots_dir)
    fig, axes = plt.subplots(1, 2, figsize=(20, 5))
    fig.suptitle(
        "Step 2 — Metric Heatmap (metric × round)",
        fontsize=13, fontweight="bold",
    )

    for ax, results, title in zip(
        axes, [mnist_results, pythia_results], list(labels)
    ):
        if not results:
            ax.set_title(f"{title}\n(no data)")
            ax.axis("off")
            continue

        ns = _ns(results)
        # data shape: (n_metrics, n_rounds)
        data = np.array(
            [[r[m] if r[m] is not None else float("nan") for m in METRICS]
             for r in results]
        ).T

        im = ax.imshow(data, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
        held_out = [r["test_attack"] for r in results]
        ax.set_xticks(range(len(ns)))
        ax.set_xticklabels(
            [f"n={n}\n↳{ho}" for n, ho in zip(ns, held_out)],
            fontsize=7, rotation=45, ha="right",
        )
        ax.set_yticks(range(len(METRICS)))
        ax.set_yticklabels([_METRIC_LABELS[m] for m in METRICS], fontsize=10)
        ax.set_title(title, fontsize=12)

        for i in range(len(METRICS)):
            for j in range(len(ns)):
                v = data[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=8, color="white" if v > 0.65 else "black")

        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)

    plt.tight_layout()
    out = plots_dir / f"metric_heatmap{suffix}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] Saved → {out}")





def plot_order_sensitivity(
    orderings_results: dict[str, list[dict]],
    dataset_name: str,
    plots_dir: Path | str,
) -> None:
    """2×3 grid: all five metrics across multiple attack orderings.

    Each metric panel shows every ordering as a line.  ``'original'`` is a
    thick blue line; ``'reversed'`` is red; random permutations are greyscale
    shades.  A black dashed mean and a grey ±1 std band summarise the spread —
    a narrow band means the result is robust to curriculum order.

    Parameters
    ----------
    orderings_results : dict[str, list[dict]]
        Mapping from ordering name to per-round result dicts, as returned
        by :func:`run_order_sensitivity`.
    dataset_name : str
        ``'MNIST'`` or ``'Pythia'`` — used in title and filename.
    plots_dir : Path | str
        Output directory.  File will be ``order_sensitivity_{dataset}.png``.
    """
    plots_dir = Path(plots_dir)
    perm_names = [k for k in orderings_results if k.startswith("perm_")]
    n_perms = len(perm_names)
    grey_cmap = plt.cm.Greys

    def _style(name: str) -> tuple:
        if name == "original":
            return "steelblue", 2.8, 1.0, "o-"
        if name == "reversed":
            return "crimson", 1.6, 0.75, "s--"
        p_idx = perm_names.index(name)
        frac = p_idx / max(n_perms - 1, 1) if n_perms > 1 else 0.5
        color = grey_cmap(0.35 + 0.45 * frac)
        ls = ["^:", "D-.", "P:", "X-."][p_idx % 4]
        return color, 1.2, 0.55, ls

    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    fig.suptitle(
        f"Order Sensitivity — {dataset_name}\n"
        "(Does the curriculum order of attack types affect generalisation?)",
        fontsize=14, fontweight="bold",
    )
    axes_flat = axes.flatten()

    for m_idx, metric in enumerate(METRICS):
        ax = axes_flat[m_idx]
        all_vals_by_n: dict[int, list[float]] = {}

        for ordering_name, results in orderings_results.items():
            if not results:
                continue
            color, lw, alpha, ls = _style(ordering_name)
            ns = _ns(results)
            vals = _mvals(results, metric)
            is_orig = ordering_name == "original"
            ax.plot(ns, vals, ls, color=color, linewidth=lw, alpha=alpha,
                    label=ordering_name, zorder=5 if is_orig else 2,
                    markersize=6 if is_orig else 4)
            for n, v in zip(ns, vals):
                all_vals_by_n.setdefault(n, []).append(v)

        if all_vals_by_n:
            ns_all = sorted(all_vals_by_n)
            means = [np.nanmean(all_vals_by_n[n]) for n in ns_all]
            stds  = [np.nanstd(all_vals_by_n[n])  for n in ns_all]
            ax.plot(ns_all, means, "k--", linewidth=2, label="mean", zorder=6)
            ax.fill_between(
                ns_all,
                [m - s for m, s in zip(means, stds)],
                [m + s for m, s in zip(means, stds)],
                alpha=0.12, color="gray", label="±1 std",
            )
            ax.set_xticks(ns_all)

        ax.set_title(_METRIC_LABELS[metric], fontsize=12, fontweight="bold")
        ax.set_xlabel("n (attack types in training)", fontsize=10)
        ax.set_ylabel(_METRIC_LABELS[metric], fontsize=10)
        ax.set_ylim(-0.05, 1.10)
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.9)
        ax.grid(True, alpha=0.3, linestyle="--")

    # Sixth panel: shared legend
    ax_leg = axes_flat[5]
    ax_leg.axis("off")
    handles = []
    for ordering_name in orderings_results:
        color, lw, alpha, ls = _style(ordering_name)
        marker = ls[0] if ls[0] in "osDPX^h" else None
        linestyle_str = ls[1:] if len(ls) > 1 else "-"
        handles.append(
            plt.Line2D([0], [0], color=color, linewidth=lw,
                       alpha=max(alpha, 0.6), linestyle=linestyle_str,
                       marker=marker if marker else "None",
                       markersize=7, label=ordering_name)
        )
    handles += [
        plt.Line2D([0], [0], color="black", linewidth=2, linestyle="--",
                   label="mean"),
        plt.Rectangle((0, 0), 1, 1, fc="gray", alpha=0.25, label="±1 std"),
    ]
    ax_leg.legend(handles=handles, loc="center", fontsize=10, frameon=False,
                  title="Ordering", title_fontsize=11)

    plt.tight_layout()
    out = plots_dir / f"order_sensitivity_{dataset_name.lower()}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] Saved → {out}")


def _plot_model_order_comparison(
    models_ordering_results: dict,
    dataset_name: str,
    plots_dir,
) -> None:
    """Mean ± 1 std band per model across all ordering permutations.

    Answers: which model is most robust to curriculum order?  For each model
    the mean and ±1 std of AUC-ROC and F1-Score across all orderings at each n
    are plotted, so narrower bands signal lower order-sensitivity.

    Parameters
    ----------
    models_ordering_results : dict[str, dict[str, list[dict]]]
        Mapping model_label → {ordering_name: per-round result dicts}.
    dataset_name : str
        Used in title and output filename.
    plots_dir : Path | str
        Output directory.
    """
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    _styles: dict[str, tuple] = {
        "AnomalyCNN":   ("steelblue",   "o"),
        "ProfessorCNN": ("forestgreen", "D"),
        "GBM":          ("darkorange",  "s"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"{dataset_name} — Order Sensitivity by Model  (mean ± 1 std across orderings)",
        fontsize=13, fontweight="bold",
    )

    for ax, metric, mlabel in zip(axes, ["AUC_ROC", "F1_Score"], ["AUC-ROC", "F1-Score"]):
        for model_label, orderings_results in models_ordering_results.items():
            color, marker = _styles.get(model_label, ("gray", "o"))
            per_n: dict[int, list[float]] = {}
            for res_list in orderings_results.values():
                for r in res_list:
                    n = r["n_training_attacks"]
                    v = r[metric]
                    if v is not None:
                        per_n.setdefault(n, []).append(float(v))
            if not per_n:
                continue
            ns = sorted(per_n)
            means = [float(np.nanmean(per_n[n])) for n in ns]
            stds  = [float(np.nanstd(per_n[n]))  for n in ns]
            ax.plot(ns, means, f"{marker}-", color=color, linewidth=2.5,
                    markersize=8, label=model_label)
            ax.fill_between(
                ns,
                [m - s for m, s in zip(means, stds)],
                [m + s for m, s in zip(means, stds)],
                alpha=0.18, color=color,
            )
        ax.set_title(mlabel, fontsize=12, fontweight="bold")
        ax.set_xlabel("n (attack types in training)", fontsize=10)
        ax.set_ylabel(mlabel, fontsize=10)
        ax.set_ylim(-0.05, 1.10)
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    out = plots_dir / f"order_sensitivity_model_comparison_{dataset_name.lower()}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] Saved \u2192 {out}")


# ===========================================================================
# ORDER SENSITIVITY RUNNER
# ===========================================================================


def run_order_sensitivity(
    clean_train,
    clean_test,
    attack_train_datasets: list,
    attack_test_datasets: list,
    attack_names: list[str],
    input_size: int,
    dataset_name: str,
    n_random_perms: int = 2,
    model_factory=None,
    num_epochs: int | None = None,
    patience: int | None = None,
    monitor_auc: bool = False,
) -> dict[str, list[dict]]:
    """Run progressive training under multiple attack orderings.

    Orderings tested
    ----------------
    ``'original'``
        Attacks in the order provided (matches the main experiment).
    ``'reversed'``
        Attacks in strict reverse order.
    ``'perm_seed{k}'`` for k in 0 … n_random_perms−1
        Independently seeded uniform-random permutations of the attack list.

    By comparing metrics across orderings we can answer: *does the
    curriculum (order in which new attack types are introduced) matter,
    or do the results reflect only diversity per se?*

    Parameters
    ----------
    clean_train, clean_test : Dataset
        Clean-class splits — identical for every ordering.
    attack_train_datasets, attack_test_datasets : list[Dataset]
        Attack datasets in the original order.
    attack_names : list[str]
        Human-readable name per attack type.
    input_size : int
        Square spatial dimension of input images.
    dataset_name : str
        Used for console progress logging.
    n_random_perms : int
        Number of additional random permutations to test (default 2).

    Returns
    -------
    dict[str, list[dict]]
        Mapping ordering_name → per-round result dicts (same schema as
        :func:`run_progressive_training`).
    """
    K = len(attack_names)
    orderings: dict[str, list[int]] = {
        "original": list(range(K)),
        "reversed": list(range(K - 1, -1, -1)),
    }
    for seed in range(n_random_perms):
        rng = np.random.default_rng(seed=seed)
        orderings[f"perm_seed{seed}"] = rng.permutation(K).tolist()

    all_results: dict[str, list[dict]] = {}
    for ordering_name, idx_order in orderings.items():
        ordered_names = [attack_names[i] for i in idx_order]
        print(
            f"\n  [Order sensitivity | {dataset_name} | {ordering_name}]\n"
            f"    order: {ordered_names}"
        )
        res = run_progressive_training(
            clean_train, clean_test,
            [attack_train_datasets[i] for i in idx_order],
            [attack_test_datasets[i]  for i in idx_order],
            ordered_names,
            input_size,
            experiment_name=f"{dataset_name}[{ordering_name}]",
            model_factory=model_factory,
            num_epochs=num_epochs,
            patience=patience,
            monitor_auc=monitor_auc,
        )
        all_results[ordering_name] = res

    return all_results


def run_order_sensitivity_gbm(
    clean_train,
    clean_test,
    attack_train_datasets: list,
    attack_test_datasets: list,
    attack_names: list[str],
    dataset_name: str,
    n_random_perms: int = 2,
) -> dict[str, list[dict]]:
    """Run GBM progressive training under multiple attack orderings.

    Mirrors :func:`run_order_sensitivity` but uses
    :func:`run_progressive_training_gbm`, so GBM robustness to curriculum
    order can be compared against the AnomalyCNN baseline.

    Returns
    -------
    dict[str, list[dict]]
        Same schema as :func:`run_order_sensitivity`.
    """
    K = len(attack_names)
    orderings: dict[str, list[int]] = {
        "original": list(range(K)),
        "reversed": list(range(K - 1, -1, -1)),
    }
    for seed in range(n_random_perms):
        rng = np.random.default_rng(seed=seed)
        orderings[f"perm_seed{seed}"] = rng.permutation(K).tolist()

    all_results: dict[str, list[dict]] = {}
    for ordering_name, idx_order in orderings.items():
        ordered_names = [attack_names[i] for i in idx_order]
        print(
            f"\n  [Order sensitivity | {dataset_name} GBM | {ordering_name}]\n"
            f"    order: {ordered_names}"
        )
        res = run_progressive_training_gbm(
            clean_train, clean_test,
            [attack_train_datasets[i] for i in idx_order],
            [attack_test_datasets[i]  for i in idx_order],
            ordered_names,
            experiment_name=f"{dataset_name}_GBM[{ordering_name}]",
        )
        all_results[ordering_name] = res

    return all_results


# ===========================================================================
# PYTHIA MODEL COMPARISON PLOT  (AnomalyCNN vs ProfessorCNNBest)
# ===========================================================================


def _plot_pythia_progressive_comparison(
    baseline_results: list[dict],
    plots_dir,
    gbm_results: list[dict] | None = None,
    prof_cnn_results: list[dict] | None = None,
    dataset_name: str = "Pythia",
) -> None:
    """Overlay AnomalyCNN and ProfessorCNN progressive-training curves for a dataset.

    Plots AUC-ROC and F1-Score vs. n for both models.  An optional GBM line
    is drawn if ``gbm_results`` is supplied.

    Parameters
    ----------
    baseline_results : list[dict]
        Output of ``run_progressive_training`` with AnomalyCNN.
    plots_dir : Path
        Output directory.  File: ``pythia_model_comparison_progressive.png``.
    gbm_results : list[dict] | None
        Optional output of ``run_progressive_training_gbm`` with GBM.
    prof_cnn_results : list[dict] | None
        Optional output of ``run_progressive_training`` with ProfessorCNN.
    """
    import pathlib
    plots_dir = pathlib.Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    def _ns(r):  return [d["n_training_attacks"] for d in r]
    def _mv(r, m): return [d.get(m) or 0.0 for d in r]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"{dataset_name} — Progressive Training: AnomalyCNN vs ProfessorCNN",
        fontsize=12, fontweight="bold",
    )

    for ax, metric, mlabel in zip(axes, ["AUC_ROC", "F1_Score"], ["AUC-ROC", "F1-Score"]):
        ax.plot(_ns(baseline_results), _mv(baseline_results, metric),
                "o-", color="steelblue", linewidth=1.8, label="AnomalyCNN (Baseline)")
        if prof_cnn_results:
            ax.plot(_ns(prof_cnn_results), _mv(prof_cnn_results, metric),
                    "D--", color="forestgreen", linewidth=1.8, label="ProfessorCNN")
        if gbm_results:
            ax.plot(_ns(gbm_results), _mv(gbm_results, metric),
                    "s-", color="darkorange", linewidth=2.8, label="GBM (Best)",
                    markeredgecolor="#8B4000", markeredgewidth=1.2, markersize=8)
        ax.set_xlabel("n training attack types", fontsize=10)
        ax.set_ylabel(mlabel, fontsize=10)
        ax.set_title(f"{dataset_name} \u2014 {mlabel} vs n", fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    save_path = plots_dir / f"{dataset_name.lower().replace(' ', '_')}_model_comparison_progressive.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] Saved \u2192 {save_path}")


# ===========================================================================
# MAIN PIPELINE
# ===========================================================================


def main() -> None:
    """Run the full Step 2 generalisation experiment.

    Phase 1 — MNIST  (28×28 px)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Generates 6 attack types:
      A1: Gaussian noise  A2: Salt & pepper  A3: Geometric distortion
      A4: Blended attack  A5: Backdoor trigger  A6: OOD (Fashion-MNIST)

    Then runs 5 progressive training rounds (n = 1 … 5), training on the
    first n attack types and testing on A_{n+1}.

    Phase 2 — Pythia  (70×70 px, hidden dataset)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Loads all 8 labelled partitions (attack_a … attack_h) and runs
    7 progressive rounds (n = 1 … 7).
    """

    set_global_seed(RANDOM_SEED)

    PLOTS_DIR = Path("plots") / "step2"

    # =======================================================================
    # PHASE 1: MNIST
    # =======================================================================
    print("\n" + "=" * 65)
    print("  STEP 2 — MNIST GENERALISATION EXPERIMENT")
    print("=" * 65)

    clean_train_mnist, clean_test_mnist, raw_train_mnist, raw_test_mnist = (
        prepare_clean_data("mnist")
    )
    # Fashion-MNIST only needed as OOD source; we do not use its labels
    _, _, raw_train_fmnist, raw_test_fmnist = prepare_clean_data("fashion_mnist")

    # ------------------------------------------------------------------
    # Generate all 6 MNIST attack types (train + test splits)
    # ------------------------------------------------------------------
    print("\nGenerating 6 attack types for MNIST...")

    # A shared blended-attack pattern is generated once so that the same
    # pattern is used consistently for both training and test images.
    blend_pattern = torch.rand(1, 1, 28, 28, dtype=torch.float32)

    attack_configs = [
        ("A1_gaussian",    make_gaussian_attack,    {"std": 0.4}),
        ("A2_salt_pepper", make_salt_pepper_attack, {"prob": 0.15}),
        ("A3_geometric",   make_geometric_attack,   {"max_displacement": 5.0}),
        ("A4_blended",     make_blended_attack,     {"alpha": 0.30, "pattern": blend_pattern}),
        ("A5_backdoor",    make_backdoor_attack,    {"trigger_size": 5, "position": "bottom_right"}),
    ]

    mnist_attack_names: list[str] = []
    mnist_attack_train: list = []
    mnist_attack_test: list = []

    for name, factory_fn, kwargs in attack_configs:
        mnist_attack_train.append(factory_fn(raw_train_mnist, **kwargs))
        mnist_attack_test.append(factory_fn(raw_test_mnist, **kwargs))
        mnist_attack_names.append(name)

    # A6: OOD (Fashion-MNIST) — different source images, no per-pixel transform
    print("  Generating A6_ood (Fashion-MNIST OOD)...")
    mnist_attack_train.append(make_ood_attack(raw_train_fmnist))
    mnist_attack_test.append(make_ood_attack(raw_test_fmnist))
    mnist_attack_names.append("A6_ood")

    print(f"\nReady: {len(mnist_attack_names)} attack types: {mnist_attack_names}")

    # ------------------------------------------------------------------
    # Visualise MNIST clean + all 6 attack types
    # ------------------------------------------------------------------
    print("\nGenerating sample visualisations for MNIST partitions...")
    visualize_samples(clean_train_mnist, save_path=PLOTS_DIR / "mnist_clean.png",    title_prefix="MNIST Clean, ")
    for name, ds in zip(mnist_attack_names, mnist_attack_train):
        visualize_samples(ds, save_path=PLOTS_DIR / f"mnist_{name}.png", title_prefix=f"MNIST {name}, ")

    # ------------------------------------------------------------------
    # Subsample MNIST clean_train to keep step2 experiments fast
    # ------------------------------------------------------------------
    if len(clean_train_mnist) > MNIST_STEP2_SAMPLES:
        sub_idx = torch.randperm(len(clean_train_mnist))[:MNIST_STEP2_SAMPLES].tolist()
        clean_train_mnist_sub = Subset(clean_train_mnist, sub_idx)
    else:
        clean_train_mnist_sub = clean_train_mnist

    # ------------------------------------------------------------------
    # Progressive training — MNIST
    # ------------------------------------------------------------------
    mnist_results = run_progressive_training(
        clean_train=clean_train_mnist_sub,
        clean_test=clean_test_mnist,
        attack_train_datasets=mnist_attack_train,
        attack_test_datasets=mnist_attack_test,
        attack_names=mnist_attack_names,
        input_size=28,
        experiment_name="MNIST",
    )

    # =======================================================================
    # PHASE 2: PYTHIA
    # =======================================================================
    print("\n" + "=" * 65)
    print("  STEP 2 — PYTHIA GENERALISATION EXPERIMENT")
    print("=" * 65)

    # Load all available Pythia partitions
    PYTHIA_DIR = Path("pythia")
    check_pythia_available(PYTHIA_DIR)
    pythia_clean = load_pythia_data(PYTHIA_DIR, "clean")
    pythia_clean_train_base, pythia_clean_test = split_train_test(pythia_clean)

    pythia_partitions = [f"attack_{c}" for c in "abcdefgh"]  # attack_a … attack_h

    pythia_attack_names: list[str] = []
    pythia_attack_train: list = []
    pythia_attack_test: list = []

    for partition in pythia_partitions:
        print(f"  Loading Pythia partition '{partition}'...")
        ds = load_pythia_data(PYTHIA_DIR, partition)
        train_base, test_part = split_train_test(ds)
        pythia_attack_train.append(train_base)
        pythia_attack_test.append(test_part)
        pythia_attack_names.append(partition)

    print(f"\nReady: {len(pythia_attack_names)} Pythia attack partitions")

    # ------------------------------------------------------------------
    # Visualise all Pythia partitions (clean + attack_a … attack_h)
    # ------------------------------------------------------------------
    print("\nGenerating sample visualisations for all Pythia partitions...")
    visualize_samples(pythia_clean, save_path=PLOTS_DIR / "pythia_clean.png", title_prefix="Pythia Clean, ")
    for partition, ds_train in zip(pythia_partitions, pythia_attack_train):
        visualize_samples(
            ds_train,
            save_path=PLOTS_DIR / f"pythia_{partition}.png",
            title_prefix=f"Pythia {partition}, ",
        )

    # ------------------------------------------------------------------
    # Progressive training — Pythia
    # ------------------------------------------------------------------
    pythia_results = run_progressive_training(
        clean_train=pythia_clean_train_base,
        clean_test=pythia_clean_test,
        attack_train_datasets=pythia_attack_train,
        attack_test_datasets=pythia_attack_test,
        attack_names=pythia_attack_names,
        input_size=70,
        experiment_name="Pythia",
    )

    # =======================================================================
    # RESULTS — summary tables, all plots, order sensitivity, JSON
    # =======================================================================

    print_summary_table(mnist_results, "MNIST")
    print_summary_table(pythia_results, "Pythia")

    # --- Standard generalisation plots ---
    print("\nGenerating standard plots...")
    plot_generalization_results(
        mnist_results,
        pythia_results,
        output_path=PLOTS_DIR / "step2_generalization.png",
    )
    plot_per_metric(mnist_results, pythia_results, PLOTS_DIR)
    plot_heatmap(mnist_results, pythia_results, PLOTS_DIR)

    # --- Pythia: AnomalyCNN progressive comparison (baseline only) ---
    print("\nGenerating Pythia model comparison plot...")
    _plot_pythia_progressive_comparison(pythia_results, PLOTS_DIR)

    # --- Order sensitivity analysis ---
    print("\n" + "=" * 65)
    print("  ORDER SENSITIVITY ANALYSIS")
    print("  (Does the curriculum — order attacks are introduced — matter?)")
    print("=" * 65)

    mnist_order_results = run_order_sensitivity(
        clean_train_mnist_sub, clean_test_mnist,
        mnist_attack_train, mnist_attack_test, mnist_attack_names,
        input_size=28, dataset_name="MNIST", n_random_perms=NUM_PERMUTATIONS,
    )
    pythia_order_results = run_order_sensitivity(
        pythia_clean_train_base, pythia_clean_test,
        pythia_attack_train, pythia_attack_test, pythia_attack_names,
        input_size=70, dataset_name="Pythia", n_random_perms=NUM_PERMUTATIONS,
    )

    print("\nGenerating order-sensitivity plots...")
    plot_order_sensitivity(mnist_order_results,  "MNIST",  PLOTS_DIR)
    plot_order_sensitivity(pythia_order_results, "Pythia", PLOTS_DIR)

    # =======================================================================
    # PHASE 3: MNIST VARIANTS — same attack families, different parameters
    # =======================================================================
    # Research question: does the alternating success/failure pattern hold
    # when we use differently parameterised instances of the same attack types?
    # Variants:
    #   BV1 Gaussian      σ=0.2   (lighter than original σ=0.4)
    #   BV2 Salt & pepper p=0.30  (denser than original p=0.15)
    #   BV3 Geometric     max_displacement=8px (stronger warp)
    #   BV4 Blended       α=0.50, new random pattern (heavier blend)
    #   BV5 Backdoor      trigger at top-left corner (not bottom-right)
    #   BV6 OOD           Fashion-MNIST (same source — no alternate available)
    print("\n" + "=" * 65)
    print("  PHASE 3 — MNIST VARIANTS (same types, different parameters)")
    print("=" * 65)

    blend_pattern_v = torch.rand(1, 1, 28, 28, dtype=torch.float32)

    variant_configs = [
        ("BV1_gaussian",    make_gaussian_attack,    {"std": 0.2}),
        ("BV2_salt_pepper", make_salt_pepper_attack, {"prob": 0.30}),
        ("BV3_geometric",   make_geometric_attack,   {"max_displacement": 8.0}),
        ("BV4_blended",     make_blended_attack,     {"alpha": 0.50, "pattern": blend_pattern_v}),
        ("BV5_backdoor",    make_backdoor_attack,    {"trigger_size": 5, "position": "top_left"}),
    ]

    variant_attack_names: list[str] = []
    variant_attack_train: list = []
    variant_attack_test: list = []

    print("\nGenerating 6 variant attack types for MNIST...")
    for name, factory_fn, kwargs in variant_configs:
        variant_attack_train.append(factory_fn(raw_train_mnist, **kwargs))
        variant_attack_test.append(factory_fn(raw_test_mnist, **kwargs))
        variant_attack_names.append(name)

    # BV6: OOD — same Fashion-MNIST source
    print("  Generating BV6_ood (Fashion-MNIST OOD)...")
    variant_attack_train.append(make_ood_attack(raw_train_fmnist))
    variant_attack_test.append(make_ood_attack(raw_test_fmnist))
    variant_attack_names.append("BV6_ood")

    print(f"\nReady: {len(variant_attack_names)} variant attack types: {variant_attack_names}")

    # Sub-sample clean train to same MNIST_STEP2_SAMPLES cap
    if len(clean_train_mnist) > MNIST_STEP2_SAMPLES:
        sub_idx_v = torch.randperm(len(clean_train_mnist))[:MNIST_STEP2_SAMPLES].tolist()
        clean_train_mnist_sub_v = Subset(clean_train_mnist, sub_idx_v)
    else:
        clean_train_mnist_sub_v = clean_train_mnist

    # Visualise variant attack samples
    print("\nGenerating sample visualisations for MNIST variant partitions...")
    for name, ds in zip(variant_attack_names, variant_attack_train):
        visualize_samples(ds, save_path=PLOTS_DIR / f"mnist_{name}.png",
                          title_prefix=f"MNIST {name}, ")

    # Progressive training — variants
    variants_results = run_progressive_training(
        clean_train=clean_train_mnist_sub_v,
        clean_test=clean_test_mnist,
        attack_train_datasets=variant_attack_train,
        attack_test_datasets=variant_attack_test,
        attack_names=variant_attack_names,
        input_size=28,
        experiment_name="MNIST Variants",
    )

    print_summary_table(variants_results, "MNIST Variants")

    # Plots: compare original MNIST vs. variants side-by-side
    print("\nGenerating variant comparison plots...")
    plot_generalization_results(
        mnist_results,
        variants_results,
        output_path=PLOTS_DIR / "step2_generalization_variants.png",
        labels=("Original MNIST", "MNIST Variants"),
    )
    plot_per_metric(
        mnist_results, variants_results, PLOTS_DIR,
        labels=("Original", "Variants"),
        suffix="_variants",
    )
    plot_heatmap(
        mnist_results, variants_results, PLOTS_DIR,
        labels=("Original MNIST", "MNIST Variants"),
        suffix="_variants",
    )

    # Order sensitivity — variants
    print("\n" + "=" * 65)
    print("  ORDER SENSITIVITY — MNIST VARIANTS")
    print("=" * 65)

    variants_order_results = run_order_sensitivity(
        clean_train_mnist_sub_v, clean_test_mnist,
        variant_attack_train, variant_attack_test, variant_attack_names,
        input_size=28, dataset_name="MNIST_variants", n_random_perms=NUM_PERMUTATIONS,
    )

    print("\nGenerating variant order-sensitivity plot...")
    plot_order_sensitivity(variants_order_results, "MNIST_variants", PLOTS_DIR)

    # --- Save all results ---
    output = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "experiment": "Step 2 — Generalisation via Multi-Attack Training",
        "MNIST_progressive_results": mnist_results,
        "Pythia_progressive_results": pythia_results,
        "MNIST_order_sensitivity": mnist_order_results,
        "Pythia_order_sensitivity": pythia_order_results,
        "MNIST_variants_progressive_results": variants_results,
        "MNIST_variants_order_sensitivity": variants_order_results,
    }
    save_results(output, "faza2_wyniki_generalizacji.json")

    print(
        "\n[Interpretation] If F1 and AUC-ROC increase with n, training on more "
        "diverse attacks improves generalisation to unseen attack types, "
        "validating the hypothesis.  A flat or noisy trend suggests that "
        "diversity alone is insufficient and unsupervised/one-class methods "
        "should be explored in subsequent steps.\n"
        "\n[Order Sensitivity] Inspect the order_sensitivity_* plots: "
        "a narrow ±1 std band means the result is robust to curriculum order; "
        "a wide band flags ordering as a confound to control in future experiments."
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
