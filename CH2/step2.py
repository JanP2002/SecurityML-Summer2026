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
    load_pythia_data,
    prepare_clean_data,
    save_results,
    split_train_test,
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
# Global hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE = 64
NUM_EPOCHS = 15
PATIENCE = 3


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
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        # ------------------------------------------------------------------
        # Fresh model + optimiser for each round
        # (models trained on n attacks are independent experiments)
        # ------------------------------------------------------------------
        model = AnomalyCNN(input_size=input_size)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        from lib import train_model  # local import to keep namespace clean
        model = train_model(
            model, train_loader, val_loader, criterion, optimizer,
            num_epochs=NUM_EPOCHS, patience=PATIENCE,
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
# PLOTTING
# ===========================================================================


def plot_generalization_results(
    mnist_results: list[dict],
    pythia_results: list[dict],
    output_path: Path | str = Path("step2_generalization.png"),
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
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(
        "Step 2: Generalisation — Metric vs. Number of Training Attack Types",
        fontsize=13,
        fontweight="bold",
    )

    metric_specs = [
        ("Accuracy", "steelblue", "o"),
        ("F1_Score", "darkorange", "s"),
        ("AUC_ROC", "green", "^"),
    ]

    for ax, results, title in zip(
        axes,
        [mnist_results, pythia_results],
        ["MNIST (28×28)", "Pythia (70×70)"],
    ):
        if not results:
            ax.set_title(f"{title}\n(no results)")
            continue

        ns = [r["n_training_attacks"] for r in results]
        held_out = [r["test_attack"] for r in results]

        for metric, color, marker in metric_specs:
            values = [r[metric] if r[metric] is not None else float("nan") for r in results]
            ax.plot(ns, values, marker=marker, label=metric, color=color, linewidth=2)
            # Annotate each point with its value
            for x, y in zip(ns, values):
                if y == y:  # not NaN
                    ax.annotate(
                        f"{y:.2f}",
                        (x, y),
                        textcoords="offset points",
                        xytext=(0, 6),
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
) -> None:
    """One subplot per metric showing MNIST vs. Pythia generalisation curves.

    Five panels (Accuracy, Precision, Recall, F1, AUC-ROC) laid out in a
    single wide figure.  Both datasets are shown on each panel so that the
    reader can compare them for each metric independently.

    Parameters
    ----------
    mnist_results, pythia_results : list[dict]
        Per-round result dicts from :func:`run_progressive_training`.
    plots_dir : Path | str
        Output directory.  File will be ``per_metric_curves.png``.
    """
    plots_dir = Path(plots_dir)
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    fig.suptitle(
        "Step 2 — Per-Metric Generalisation Curves (MNIST vs. Pythia)",
        fontsize=13, fontweight="bold",
    )
    any_res = mnist_results or pythia_results
    all_ns = sorted(set(_ns(any_res))) if any_res else []

    for ax, metric in zip(axes, METRICS):
        if mnist_results:
            ax.plot(_ns(mnist_results), _mvals(mnist_results, metric),
                    "o-", label="MNIST", color="steelblue", linewidth=2)
        if pythia_results:
            ax.plot(_ns(pythia_results), _mvals(pythia_results, metric),
                    "s--", label="Pythia", color="darkorange", linewidth=2)
        ax.set_title(_METRIC_LABELS[metric], fontsize=11)
        ax.set_xlabel("n (training attack types)", fontsize=9)
        ax.set_ylim(-0.05, 1.10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
        if all_ns:
            ax.set_xticks(all_ns)
    axes[0].set_ylabel("Metric value", fontsize=9)

    plt.tight_layout()
    out = plots_dir / "per_metric_curves.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] Saved → {out}")


def plot_heatmap(
    mnist_results: list[dict],
    pythia_results: list[dict],
    plots_dir: Path | str,
) -> None:
    """Metric × round heatmap (YlOrRd colour scale, darker = higher).

    Rows are metrics, columns are progressive rounds (n=1, 2, …).
    Cell values are annotated in black/white for readability.

    Parameters
    ----------
    mnist_results, pythia_results : list[dict]
        Per-round result dicts.
    plots_dir : Path | str
        Output directory.  File will be ``metric_heatmap.png``.
    """
    plots_dir = Path(plots_dir)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle(
        "Step 2 — Metric Heatmap (metric × round)",
        fontsize=13, fontweight="bold",
    )

    for ax, results, title in zip(
        axes, [mnist_results, pythia_results], ["MNIST (28×28)", "Pythia (70×70)"]
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
        ax.set_xticks(range(len(ns)))
        ax.set_xticklabels([f"n={n}" for n in ns], fontsize=8, rotation=30)
        ax.set_yticks(range(len(METRICS)))
        ax.set_yticklabels([_METRIC_LABELS[m] for m in METRICS], fontsize=9)
        ax.set_title(title, fontsize=11)

        for i in range(len(METRICS)):
            for j in range(len(ns)):
                v = data[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=7, color="white" if v > 0.65 else "black")

        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)

    plt.tight_layout()
    out = plots_dir / "metric_heatmap.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] Saved → {out}")


def plot_metric_deltas(
    mnist_results: list[dict],
    pythia_results: list[dict],
    plots_dir: Path | str,
) -> None:
    """Bar chart of per-round metric improvement Δ = metric[n] − metric[n−1].

    Positive bars (blue/orange) indicate improvement; negative bars are shown
    in light-coral to flag regressions.  MNIST and Pythia bars are side-by-side
    at each round n.

    Parameters
    ----------
    mnist_results, pythia_results : list[dict]
        Per-round result dicts (need at least 2 rounds to compute deltas).
    plots_dir : Path | str
        Output directory.  File will be ``metric_deltas.png``.
    """
    plots_dir = Path(plots_dir)
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    fig.suptitle(
        "Step 2 — Per-Round Metric Improvement Δ (metric[n] − metric[n−1])",
        fontsize=12, fontweight="bold",
    )
    width = 0.35

    def _deltas(results: list[dict], metric: str) -> tuple[list[int], list[float]]:
        vals = _mvals(results, metric)
        ns = _ns(results)
        return ns[1:], [vals[i] - vals[i - 1] for i in range(1, len(vals))]

    for ax, metric in zip(axes, METRICS):
        color = _METRIC_COLORS[metric]
        if mnist_results and len(mnist_results) > 1:
            ns_m, d_m = _deltas(mnist_results, metric)
            ax.bar(
                [n - width / 2 for n in ns_m], d_m, width=width, label="MNIST",
                color=[color if v >= 0 else "lightcoral" for v in d_m], alpha=0.85,
            )
        if pythia_results and len(pythia_results) > 1:
            ns_p, d_p = _deltas(pythia_results, metric)
            ax.bar(
                [n + width / 2 for n in ns_p], d_p, width=width, label="Pythia",
                color=["#ff7f0e" if v >= 0 else "lightcoral" for v in d_p], alpha=0.65,
            )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(_METRIC_LABELS[metric], fontsize=10)
        ax.set_xlabel("n", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")

    axes[0].set_ylabel("Δ metric", fontsize=9)
    plt.tight_layout()
    out = plots_dir / "metric_deltas.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] Saved → {out}")


def plot_radar(
    results: list[dict],
    dataset_name: str,
    plots_dir: Path | str,
) -> None:
    """Radar (spider) chart: one coloured polygon per progressive round.

    Each polygon represents the full five-metric profile at a given value of n.
    Rounds progress from dark-purple (n=1) to yellow (n=K-1) via the viridis
    colour map so that generalisation improvement is immediately visible.

    Parameters
    ----------
    results : list[dict]
        Per-round result dicts.
    dataset_name : str
        ``'MNIST'`` or ``'Pythia'`` — used in the title and filename.
    plots_dir : Path | str
        Output directory.  File will be ``radar_{dataset_name}.png``.
    """
    if not results:
        return
    plots_dir = Path(plots_dir)
    categories = [_METRIC_LABELS[m] for m in METRICS]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    cmap = plt.cm.viridis
    colors = [cmap(i / max(len(results) - 1, 1)) for i in range(len(results))]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})

    for r, color in zip(results, colors):
        vals = [r[m] if r[m] is not None else 0.0 for m in METRICS]
        vals += vals[:1]
        ax.plot(angles, vals, "o-", linewidth=1.5, color=color,
                label=f"n={r['n_training_attacks']} → test:{r['test_attack']}")
        ax.fill(angles, vals, alpha=0.07, color=color)

    ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7, color="grey")
    ax.set_title(
        f"{dataset_name} — Metric Profile per Round",
        fontsize=12, fontweight="bold", pad=20,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = plots_dir / f"radar_{dataset_name.lower()}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] Saved → {out}")


def plot_order_sensitivity(
    orderings_results: dict[str, list[dict]],
    metric: str,
    dataset_name: str,
    plots_dir: Path | str,
) -> None:
    """Line plot comparing a metric across multiple attack orderings.

    The ``'original'`` ordering is drawn as a thick solid line; alternative
    orderings are drawn as thinner semi-transparent lines.  A black dashed
    mean line and a grey ±1 std band are overlaid to summarise the spread.

    A narrow std band means the result is robust to ordering; a wide band
    flags ordering as a confound that should be controlled in future work.

    Parameters
    ----------
    orderings_results : dict[str, list[dict]]
        Mapping from ordering name to per-round result dicts, as returned
        by :func:`run_order_sensitivity`.
    metric : str
        Key from the result dicts, e.g. ``'F1_Score'``.
    dataset_name : str
        Used in title and filename.
    plots_dir : Path | str
        Output directory.  File will be
        ``order_sensitivity_{dataset}_{metric}.png``.
    """
    plots_dir = Path(plots_dir)
    fig, ax = plt.subplots(figsize=(9, 5))

    all_vals_by_n: dict[int, list[float]] = {}
    styles = ["o-", "s--", "^:", "D-."]

    for idx, (ordering_name, results) in enumerate(orderings_results.items()):
        if not results:
            continue
        ns = _ns(results)
        vals = _mvals(results, metric)
        is_orig = ordering_name == "original"
        ax.plot(ns, vals, styles[idx % len(styles)],
                linewidth=2.5 if is_orig else 1.2,
                alpha=1.0 if is_orig else 0.6,
                label=ordering_name,
                zorder=5 if is_orig else 2)
        for n, v in zip(ns, vals):
            all_vals_by_n.setdefault(n, []).append(v)

    ns_all = sorted(all_vals_by_n)
    means = [np.nanmean(all_vals_by_n[n]) for n in ns_all]
    stds  = [np.nanstd(all_vals_by_n[n])  for n in ns_all]

    ax.plot(ns_all, means, "k--", linewidth=2, label="mean (all orderings)", zorder=6)
    ax.fill_between(
        ns_all,
        [m - s for m, s in zip(means, stds)],
        [m + s for m, s in zip(means, stds)],
        alpha=0.15, color="gray", label="±1 std",
    )

    ax.set_title(
        f"Order Sensitivity — {_METRIC_LABELS[metric]} — {dataset_name}",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlabel("n (training attack types)", fontsize=10)
    ax.set_ylabel(_METRIC_LABELS[metric], fontsize=10)
    ax.set_ylim(-0.05, 1.10)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    metric_slug = metric.lower().replace("_", "")
    out = plots_dir / f"order_sensitivity_{dataset_name.lower()}_{metric_slug}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] Saved → {out}")


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
        )
        all_results[ordering_name] = res

    return all_results


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
    # Progressive training — MNIST
    # ------------------------------------------------------------------
    mnist_results = run_progressive_training(
        clean_train=clean_train_mnist,
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
    PLOTS_DIR = Path("plots") / "step2"

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
    plot_metric_deltas(mnist_results, pythia_results, PLOTS_DIR)
    plot_radar(mnist_results, "MNIST", PLOTS_DIR)
    plot_radar(pythia_results, "Pythia", PLOTS_DIR)

    # --- Order sensitivity analysis ---
    print("\n" + "=" * 65)
    print("  ORDER SENSITIVITY ANALYSIS")
    print("  (Does the curriculum — order attacks are introduced — matter?)")
    print("=" * 65)

    mnist_order_results = run_order_sensitivity(
        clean_train_mnist, clean_test_mnist,
        mnist_attack_train, mnist_attack_test, mnist_attack_names,
        input_size=28, dataset_name="MNIST", n_random_perms=2,
    )
    pythia_order_results = run_order_sensitivity(
        pythia_clean_train_base, pythia_clean_test,
        pythia_attack_train, pythia_attack_test, pythia_attack_names,
        input_size=70, dataset_name="Pythia", n_random_perms=2,
    )

    print("\nGenerating order-sensitivity plots...")
    for metric in ["F1_Score", "AUC_ROC", "Recall"]:
        plot_order_sensitivity(mnist_order_results,   metric, "MNIST",  PLOTS_DIR)
        plot_order_sensitivity(pythia_order_results,  metric, "Pythia", PLOTS_DIR)

    # --- Save all results ---
    output = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "experiment": "Step 2 — Generalisation via Multi-Attack Training",
        "MNIST_progressive_results": mnist_results,
        "Pythia_progressive_results": pythia_results,
        "MNIST_order_sensitivity": mnist_order_results,
        "Pythia_order_sensitivity": pythia_order_results,
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
