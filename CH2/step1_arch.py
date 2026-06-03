"""
step1_arch.py -- Architecture Ablation: FCN_GAP vs ProfessorCNN on Pythia
==========================================================================

Tests whether removing spatial MaxPooling in favour of stride-2 convolutions +
Global Average Pooling (FCN_GAP) matches or beats ProfessorCNN on the Pythia
dataset, where the attack signal is a subtle *positional* perturbation.

Hypothesis
----------
ProfessorCNN is the current best CNN for Pythia.  However, it still uses
MaxPool2d which discards absolute position information.  FCN_GAP replaces
every MaxPool with stride-2 convolutions (learned downsampling) and uses
Global Average Pooling so that every spatial position contributes equally
to the final representation.

Dataset
-------
* Pythia  (70x70 grayscale PNGs):
    - clean      (label 0) -- normal traffic screenshots
    - attack_a   (label 1) -- KNOWN positional attack used during training
    - attack_b   (label 1) -- UNKNOWN attack held out for open-world test

Usage
-----
    # From the CH2/ directory:
    python step1_arch.py

Outputs
-------
    faza1_wyniki_arch_experiment.json   -- AUC/metric comparison table
    plots/step1/arch_comparison.png     -- bar chart FCN_GAP vs ProfessorCNN
"""

from __future__ import annotations

import random
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, random_split

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lib import (
    load_pythia_data,
    check_pythia_available,
    split_train_test,
    make_dataloader,
    FCN_GAP,
    get_professor_cnn_best,
    train_model,
    evaluate_model,
    parse_results,
    save_results,
)


# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
RANDOM_SEED     = 2137
BATCH_SIZE      = 64
PYTHIA_EPOCHS   = 200
PYTHIA_PATIENCE = 20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def _compare_plot(results: dict, save_path: Path) -> None:
    """Bar chart: per-metric comparison FCN_GAP vs ProfessorCNN on Pythia."""
    metrics   = ["Accuracy", "Precision", "Recall", "F1_Score", "AUC_ROC"]
    mlabels   = ["Acc", "Prec", "Rec", "F1", "AUC"]
    splits    = [("Test_A", "Test_A\n(known attack)"),
                 ("Test_B", "Test_B\n(unknown attack)")]

    gap_vals  = {s: [results["FCN_GAP"][s][m]     for m in metrics] for s, _ in splits}
    prof_vals = {s: [results["ProfessorCNN"][s][m] for m in metrics] for s, _ in splits}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    fig.suptitle(
        "Pythia — Architecture Ablation: FCN_GAP vs ProfessorCNN\n"
        "(Recommendation 1: stride-2 conv + Global Average Pooling vs MaxPool CNN)",
        fontsize=11, fontweight="bold",
    )

    x = np.arange(len(metrics))
    w = 0.35

    for ax, (split_key, split_label) in zip(axes, splits):
        b1 = ax.bar(x - w / 2, gap_vals[split_key],  w,
                    label="FCN_GAP (stride-2 + GAP)", color="darkorange",
                    alpha=0.85, edgecolor="#8B4000")
        b2 = ax.bar(x + w / 2, prof_vals[split_key], w,
                    label="ProfessorCNN (MaxPool)", color="forestgreen",
                    alpha=0.85, edgecolor="#1a5c1a")

        for bars in (b1, b2):
            for bar in bars:
                h = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.012, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=7.5, fontweight="bold",
                )

        ax.set_title(split_label, fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(mlabels, fontsize=9)
        ax.set_ylim(0, 1.22)
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.9)
        ax.set_ylabel("Metric value", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")

        # Highlight AUC bar with green outline (key metric)
        for bars in (b1, b2):
            auc_bar = list(bars)[mlabels.index("AUC")]
            auc_bar.set_edgecolor("black")
            auc_bar.set_linewidth(2.0)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] Saved -> {save_path}")


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main() -> None:
    set_seed(RANDOM_SEED)
    PLOTS = Path("plots") / "step1"

    # ------------------------------------------------------------------
    # 1. Load Pythia data
    # ------------------------------------------------------------------
    print("=" * 60)
    print("LOADING PYTHIA DATA")
    print("=" * 60)

    PYTHIA_DIR = Path("pythia")
    check_pythia_available(PYTHIA_DIR)
    py_clean = load_pythia_data(PYTHIA_DIR, "clean")
    py_atk_a = load_pythia_data(PYTHIA_DIR, "attack_a")
    py_atk_b = load_pythia_data(PYTHIA_DIR, "attack_b")

    py_clean_tr, py_clean_te = split_train_test(py_clean)
    py_atk_a_tr, py_atk_a_te = split_train_test(py_atk_a)
    py_comb  = ConcatDataset([py_clean_tr, py_atk_a_tr])
    py_n_tr  = int(0.8 * len(py_comb))
    py_train, py_val = random_split(py_comb, [py_n_tr, len(py_comb) - py_n_tr])
    py_test_a = ConcatDataset([py_clean_te, py_atk_a_te])
    py_test_b = ConcatDataset([py_clean_te, py_atk_b])

    loader_tr  = make_dataloader(py_train,  BATCH_SIZE, shuffle=True)
    loader_val = make_dataloader(py_val,    BATCH_SIZE)
    loader_ta  = make_dataloader(py_test_a, BATCH_SIZE)
    loader_tb  = make_dataloader(py_test_b, BATCH_SIZE)

    print(f"  train={len(py_train):,}  val={len(py_val):,}  "
          f"testA={len(py_test_a):,}  testB={len(py_test_b):,}")

    criterion = nn.BCELoss()
    results: dict = {}

    # ------------------------------------------------------------------
    # 2. FCN_GAP (stride-2 + GlobalAvgPool, no MaxPool)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[Pythia]  FCN_GAP  (stride-2 convs + Global Average Pooling)")
    print("=" * 60)
    set_seed(RANDOM_SEED)
    m_gap = FCN_GAP(input_size=70)
    print(m_gap)
    opt = optim.Adam(m_gap.parameters(), lr=0.001)
    m_gap = train_model(m_gap, loader_tr, loader_val, criterion, opt,
                        num_epochs=PYTHIA_EPOCHS, patience=PYTHIA_PATIENCE,
                        monitor_auc=True)
    print("  -- Test_A (known attack) --")
    r_gap_a = evaluate_model(m_gap, loader_ta)
    print("  -- Test_B (unknown attack) --")
    r_gap_b = evaluate_model(m_gap, loader_tb)
    results["FCN_GAP"] = {
        "Test_A": parse_results(r_gap_a),
        "Test_B": parse_results(r_gap_b),
    }

    # ------------------------------------------------------------------
    # 3. ProfessorCNN (best grid-searched config, still uses MaxPool)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[Pythia]  ProfessorCNN  (best grid-searched config, MaxPool)")
    print("=" * 60)
    set_seed(RANDOM_SEED)
    m_prof = get_professor_cnn_best(input_size=70)
    print(m_prof)
    opt = optim.Adam(m_prof.parameters(), lr=0.001)
    m_prof = train_model(m_prof, loader_tr, loader_val, criterion, opt,
                         num_epochs=PYTHIA_EPOCHS, patience=PYTHIA_PATIENCE,
                         monitor_auc=True)
    print("  -- Test_A (known attack) --")
    r_prof_a = evaluate_model(m_prof, loader_ta)
    print("  -- Test_B (unknown attack) --")
    r_prof_b = evaluate_model(m_prof, loader_tb)
    results["ProfessorCNN"] = {
        "Test_A": parse_results(r_prof_a),
        "Test_B": parse_results(r_prof_b),
    }

    # ------------------------------------------------------------------
    # 4. Save results + plot
    # ------------------------------------------------------------------
    results["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results["config"] = {
        "RANDOM_SEED":      RANDOM_SEED,
        "PYTHIA_EPOCHS":    PYTHIA_EPOCHS,
        "PYTHIA_PATIENCE":  PYTHIA_PATIENCE,
        "early_stopping":   "AUC-based (monitor_auc=True)",
        "dataset":          "Pythia 70x70",
    }
    save_results(results, "faza1_wyniki_arch_experiment.json")
    _compare_plot(results, PLOTS / "arch_comparison.png")

    # ------------------------------------------------------------------
    # 5. Summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY  (Pythia AUC-ROC)")
    print("=" * 60)
    print(f"{'Split':<28} {'FCN_GAP':>10} {'ProfCNN':>10} {'Delta (GAP-Prof)':>18}")
    print("-" * 68)
    for split_key, label in (("Test_A", "known attack"), ("Test_B", "unknown attack")):
        gap_auc  = results["FCN_GAP"][split_key]["AUC_ROC"]
        prof_auc = results["ProfessorCNN"][split_key]["AUC_ROC"]
        delta    = gap_auc - prof_auc
        sign     = "+" if delta >= 0 else ""
        marker   = " <-- KEY" if split_key == "Test_B" else ""
        print(f"Pythia {label:<22} {gap_auc:>10.4f} {prof_auc:>10.4f} "
              f"{sign}{delta:>16.4f}{marker}")

    print("\nDone.")


if __name__ == "__main__":
    main()
