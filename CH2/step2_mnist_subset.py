"""
step2_mnist_subset.py  — MNIST Attack Subset Selection Experiment
=================================================================

Companion script to step2_professor.py.  Answers the same two questions
that Prof. Tałataj posed for Pythia, but now applied to the MNIST dataset:

  (A) Does *how many* attack types in training matter?
      Sequential growth: n=1 → n=2 → n=3 → n=4 → n=5

  (B) Does *which* attack types matter?
      Three size-3 subsets with different diversity levels are compared at
      the same budget.

Nine subsets are evaluated using AnomalyCNN (the baseline architecture),
with 5 seeded repeats per subset (BASE_SEED=2137).  Results are saved to
``faza2_wyniki_mnist_subsets.json`` and plotted as
``plots/step2/step2_subset_comparison_mnist_cnn.png``.

Usage
-----
    # From the CH2/ directory:
    python step2_mnist_subset.py
"""

from __future__ import annotations

import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, Subset, random_split

sys.path.insert(0, str(Path(__file__).parent.resolve()))

from step2 import (
    AnomalyCNN,
    BATCH_SIZE,
    make_backdoor_attack,
    make_blended_attack,
    make_gaussian_attack,
    make_geometric_attack,
    make_ood_attack,
    make_salt_pepper_attack,
    balanced_attack_subset,
    plot_subset_comparison,
    prepare_clean_data,
)
from lib import (
    evaluate_model,
    make_dataloader,
    save_results,
    train_model,
)

# ---------------------------------------------------------------------------
# Hyper-parameters (matching step2_professor.py for consistency)
# ---------------------------------------------------------------------------
BASE_SEED   = 2137
RUN_REPEATS = 5
NUM_EPOCHS  = 15
PATIENCE    = 5
MNIST_SAMPLES = 10_000   # same cap as main step2 experiment


# ---------------------------------------------------------------------------
# MNIST attack subsets — 9 strategies, same structure as PYTHIA_ATTACK_SUBSETS
# ---------------------------------------------------------------------------
# Attacks:
#   A1_gaussian   — pixel-level, easy to detect (Gaussian noise σ=0.4)
#   A2_salt_pepper— pixel-level, easy to detect (S&P p=0.15)
#   A3_geometric  — structural, harder (smooth warp)
#   A4_blended    — pixel-level overlay (α=0.30)
#   A5_backdoor   — localised trigger (5×5 corner patch), hardest
#   A6_ood        — OOD replacement (Fashion-MNIST), trivially easy
#
# Questions answered:
#   (A) n=1 → 2 → 3 → 4 → 5 sequential subsets  (how many?)
#   (B) three size-3 subsets of different diversity (which ones?)

MNIST_ATTACK_SUBSETS: list[tuple[str, list[str]]] = [
    # ── Size 1 ────────────────────────────────────────────────────────────
    ("1 — A1 only",                ["A1_gaussian"]),
    # ── Size 2 ────────────────────────────────────────────────────────────
    ("2 — A1+A2 (seq)",            ["A1_gaussian", "A2_salt_pepper"]),
    # ── Size 3 ────────────────────────────────────────────────────────────
    ("3 — A1+A2+A3 (seq)",         ["A1_gaussian", "A2_salt_pepper", "A3_geometric"]),
    ("3 — A1+A2+A5 (diverse)",     ["A1_gaussian", "A2_salt_pepper", "A5_backdoor"]),
    ("3 — A1+A3+A5 (spread)",      ["A1_gaussian", "A3_geometric",   "A5_backdoor"]),
    # ── Size 4 ────────────────────────────────────────────────────────────
    ("4 — A1+A2+A3+A4 (seq)",      ["A1_gaussian", "A2_salt_pepper",
                                    "A3_geometric", "A4_blended"]),
    ("4 — A1+A2+A5+A6 (div)",      ["A1_gaussian", "A2_salt_pepper",
                                    "A5_backdoor",  "A6_ood"]),
    ("4 — A1+A3+A5+A6 (alt)",      ["A1_gaussian", "A3_geometric",
                                    "A5_backdoor",  "A6_ood"]),
    # ── Size 5 ────────────────────────────────────────────────────────────
    ("5 — A1–A5 (majority)",       ["A1_gaussian", "A2_salt_pepper",
                                    "A3_geometric", "A4_blended", "A5_backdoor"]),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    return float(np.mean(values)), float(np.std(values, ddof=0))


def run_subset_comparison_repeated(
    clean_train,
    clean_test,
    attack_train_datasets: list,
    attack_test_datasets: list,
    attack_names: list[str],
    subsets: list[tuple[str, list[str]]],
    n_repeats: int = RUN_REPEATS,
    base_seed: int = BASE_SEED,
) -> list[dict]:
    """Run run_subset_comparison with multiple seeds and aggregate mean ± std.

    Each repeat uses a fresh global seed derived from base_seed + repeat_idx
    and trains a fresh AnomalyCNN per subset per repeat.

    Returns
    -------
    list[dict]
        One dict per subset with keys:
        ``subset_label``, ``train_attacks``, ``unseen_attacks``,
        ``mean_AUC_ROC``, ``std_AUC_ROC``, ``mean_F1_Score``, ``std_F1_Score``,
        ``mean_Accuracy``, ``std_Accuracy``, ``per_repeat``.
    """
    # Accumulate per-repeat results keyed by subset_label
    accumulator: dict[str, dict[str, list]] = {}

    for repeat in range(n_repeats):
        seed = base_seed + repeat
        set_global_seed(seed)
        print(f"\n{'#'*65}")
        print(f"  REPEAT {repeat + 1}/{n_repeats}  (seed={seed})")
        print(f"{'#'*65}")

        for subset_label, train_attack_names in subsets:
            print(f"\n  [Subset] {subset_label}")

            # Resolve indices
            train_idx = [
                attack_names.index(n) for n in train_attack_names
                if n in attack_names
            ]
            if not train_idx:
                print(f"  WARNING: no valid attacks — skipping")
                continue
            unseen_idx = [i for i in range(len(attack_names)) if i not in train_idx]
            unseen_names = [attack_names[i] for i in unseen_idx]

            # Build balanced training set (use all attack samples — no down-sampling
            # to clean size, to match the professor's approach in step2_professor.py)
            all_attack_ds = ConcatDataset(
                [attack_train_datasets[i] for i in train_idx]
            )
            combined_tv = ConcatDataset([clean_train, all_attack_ds])

            tv_train_size = int(0.8 * len(combined_tv))
            tv_val_size   = len(combined_tv) - tv_train_size
            train_ds, val_ds = random_split(
                combined_tv, [tv_train_size, tv_val_size],
                generator=torch.Generator().manual_seed(seed),
            )

            train_loader = make_dataloader(train_ds, BATCH_SIZE, shuffle=True)
            val_loader   = make_dataloader(val_ds,   BATCH_SIZE)

            # Fresh model
            model     = AnomalyCNN(input_size=28)
            criterion = nn.BCELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            model = train_model(
                model, train_loader, val_loader, criterion, optimizer,
                num_epochs=NUM_EPOCHS, patience=PATIENCE,
            )

            # Evaluate on each unseen attack
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
                    f"    {name:18s}  Acc={acc:.3f}  F1={f1:.3f}  AUC={auc:.3f}"
                )

            if subset_label not in accumulator:
                accumulator[subset_label] = {
                    "train_attacks":  [attack_names[i] for i in train_idx],
                    "unseen_attacks": unseen_names,
                    "per_repeat":     [],
                    "_per_attack_metrics_all": {n: {"AUC_ROC": [], "F1_Score": [], "Accuracy": []}
                                                for n in unseen_names},
                }
            accumulator[subset_label]["per_repeat"].append(
                {"seed": seed, "per_attack": per_attack}
            )
            for name, metrics in per_attack.items():
                for key in ("AUC_ROC", "F1_Score", "Accuracy"):
                    v = metrics[key]
                    if v is not None:
                        accumulator[subset_label]["_per_attack_metrics_all"][name][key].append(v)

    # Build final aggregated results
    results: list[dict] = []
    for subset_label, train_attack_names in subsets:
        if subset_label not in accumulator:
            continue
        acc_data = accumulator[subset_label]

        # Mean across repeats and attacks for top-level scalars
        all_aucs: list[float] = []
        all_f1s:  list[float] = []
        all_accs: list[float] = []
        per_attack_mean: dict[str, dict] = {}

        for name in acc_data["unseen_attacks"]:
            aucs = acc_data["_per_attack_metrics_all"][name]["AUC_ROC"]
            f1s  = acc_data["_per_attack_metrics_all"][name]["F1_Score"]
            accs = acc_data["_per_attack_metrics_all"][name]["Accuracy"]
            m_auc, s_auc = _mean_std(aucs)
            m_f1,  s_f1  = _mean_std(f1s)
            m_acc, s_acc = _mean_std(accs)
            per_attack_mean[name] = {
                "mean_AUC_ROC": m_auc, "std_AUC_ROC": s_auc,
                "mean_F1_Score": m_f1, "std_F1_Score": s_f1,
                "mean_Accuracy": m_acc, "std_Accuracy": s_acc,
                # For plot_subset_comparison compatibility:
                "AUC_ROC":  m_auc,
                "F1_Score": m_f1,
                "Accuracy": m_acc,
            }
            all_aucs.extend(aucs)
            all_f1s.extend(f1s)
            all_accs.extend(accs)

        m_auc_all, s_auc_all = _mean_std(all_aucs)
        m_f1_all,  s_f1_all  = _mean_std(all_f1s)
        m_acc_all, s_acc_all = _mean_std(all_accs)

        results.append({
            "subset_label":       subset_label,
            "train_attacks":      acc_data["train_attacks"],
            "unseen_attacks":     acc_data["unseen_attacks"],
            "per_attack_metrics": per_attack_mean,
            "mean_AUC_ROC":       m_auc_all,
            "std_AUC_ROC":        s_auc_all,
            "mean_F1_Score":      m_f1_all,
            "std_F1_Score":       s_f1_all,
            "mean_Accuracy":      m_acc_all,
            "std_Accuracy":       s_acc_all,
            "per_repeat":         acc_data["per_repeat"],
        })

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    set_global_seed(BASE_SEED)

    PLOTS_DIR = Path("plots") / "step2"
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print("  STEP 2 — MNIST ATTACK SUBSET SELECTION EXPERIMENT")
    print(f"  {RUN_REPEATS} seeded repeats × {len(MNIST_ATTACK_SUBSETS)} subsets")
    print("=" * 65)

    # ── Load MNIST and FashionMNIST ──────────────────────────────────────
    clean_train_mnist, clean_test_mnist, raw_train_mnist, raw_test_mnist = (
        prepare_clean_data("mnist")
    )
    _, _, raw_train_fmnist, raw_test_fmnist = prepare_clean_data("fashion_mnist")

    # ── Sub-sample clean training set (same cap as main step2 experiment) ─
    if len(clean_train_mnist) > MNIST_SAMPLES:
        sub_idx = torch.randperm(len(clean_train_mnist),
                                 generator=torch.Generator().manual_seed(BASE_SEED))[:MNIST_SAMPLES].tolist()
        clean_train_mnist = Subset(clean_train_mnist, sub_idx)

    # ── Generate 6 attack types ──────────────────────────────────────────
    print("\nGenerating 6 MNIST attack types...")
    blend_pattern = torch.rand(
        1, 1, 28, 28, dtype=torch.float32,
        generator=torch.Generator().manual_seed(BASE_SEED),
    )

    attack_configs = [
        ("A1_gaussian",    make_gaussian_attack,    {"std": 0.4}),
        ("A2_salt_pepper", make_salt_pepper_attack, {"prob": 0.15}),
        ("A3_geometric",   make_geometric_attack,   {"max_displacement": 5.0}),
        ("A4_blended",     make_blended_attack,     {"alpha": 0.30, "pattern": blend_pattern}),
        ("A5_backdoor",    make_backdoor_attack,    {"trigger_size": 5, "position": "bottom_right"}),
    ]

    attack_names:  list[str] = []
    attack_train:  list      = []
    attack_test:   list      = []

    for name, factory_fn, kwargs in attack_configs:
        attack_train.append(factory_fn(raw_train_mnist, **kwargs))
        attack_test.append(factory_fn(raw_test_mnist, **kwargs))
        attack_names.append(name)

    attack_train.append(make_ood_attack(raw_train_fmnist))
    attack_test.append(make_ood_attack(raw_test_fmnist))
    attack_names.append("A6_ood")

    print(f"  Attack types: {attack_names}")

    # ── Run subset comparison with repeated seeds ────────────────────────
    print(f"\nRunning subset comparison ({RUN_REPEATS} repeats each)…")
    results = run_subset_comparison_repeated(
        clean_train=clean_train_mnist,
        clean_test=clean_test_mnist,
        attack_train_datasets=attack_train,
        attack_test_datasets=attack_test,
        attack_names=attack_names,
        subsets=MNIST_ATTACK_SUBSETS,
        n_repeats=RUN_REPEATS,
        base_seed=BASE_SEED,
    )

    # ── Save results JSON ────────────────────────────────────────────────
    output = {
        "experiment":  "MNIST attack subset selection",
        "timestamp":   datetime.now().isoformat(),
        "base_seed":   BASE_SEED,
        "n_repeats":   RUN_REPEATS,
        "num_epochs":  NUM_EPOCHS,
        "patience":    PATIENCE,
        "mnist_samples": MNIST_SAMPLES,
        "subsets":     results,
    }
    json_path = Path("faza2_wyniki_mnist_subsets.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)
    print(f"\n  Results saved → {json_path}")

    # ── Generate plot ────────────────────────────────────────────────────
    print("\nGenerating subset comparison plot…")
    plot_subset_comparison(
        results,
        plots_dir=PLOTS_DIR,
        title="MNIST — Attack Subset Selection Comparison  (AnomalyCNN, 5 repeats)",
        filename="step2_subset_comparison_mnist_cnn.png",
    )

    # ── Print summary table ──────────────────────────────────────────────
    print(f"\n{'─'*75}")
    print("  MNIST Subset Comparison — Summary (mean ± std across repeats & attacks)")
    print(f"{'─'*75}")
    print(f"  {'Subset':<32}  {'AUC-ROC':>12}  {'F1-Score':>12}  {'Accuracy':>10}")
    print(f"  {'─'*32}  {'─'*12}  {'─'*12}  {'─'*10}")
    for r in results:
        auc_str = f"{r['mean_AUC_ROC']:.3f}±{r['std_AUC_ROC']:.3f}"
        f1_str  = f"{r['mean_F1_Score']:.3f}±{r['std_F1_Score']:.3f}"
        acc_str = f"{r['mean_Accuracy']:.3f}"
        print(f"  {r['subset_label']:<32}  {auc_str:>12}  {f1_str:>12}  {acc_str:>10}")
    print(f"{'─'*75}")

    print("\nDone.")


if __name__ == "__main__":
    main()
