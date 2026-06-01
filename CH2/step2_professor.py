"""
step2_professor.py - Pythia ProfessorCNN reproducibility suite
===============================================================

This script is the professor-only companion to step2.py. It focuses on
Pythia and implements four improvements requested for repeatability and
fairness:

1. Fixed random seeds for every run.
2. Repeated runs with mean and standard deviation reporting.
3. Larger training sets by using all available attack samples in the
   selected training subset instead of balancing down to the clean size.
4. A direct step1-style ordered pair protocol: train on attack_A, test on
   attack_B, and vice versa for every ordered pair.

The goal is to answer a simpler question than the progressive curriculum:
which attack combinations actually help ProfessorCNN on Pythia?

Usage
-----
    # From the CH2/ directory:
    python step2_professor.py
"""

from __future__ import annotations

import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, random_split

sys.path.insert(0, str(Path(__file__).parent.resolve()))

from step2 import PYTHIA_ATTACK_SUBSETS, plot_subset_comparison
from lib import (
    check_pythia_available,
    evaluate_model,
    get_professor_cnn_best,
    load_pythia_data,
    make_dataloader,
    save_results,
    split_train_test,
    train_model,
    visualize_samples,
)


BASE_SEED = 2137
RUN_REPEATS = 5
TRAIN_NUM_EPOCHS = 200
TRAIN_PATIENCE = 20
BATCH_SIZE = 64


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return float(np.mean(values)), float(np.std(values, ddof=0))


def _train_professor_cnn(
    clean_train,
    attack_train_datasets: list,
    train_indices: list[int],
    input_size: int,
    seed: int,
    num_epochs: int = TRAIN_NUM_EPOCHS,
    patience: int = TRAIN_PATIENCE,
) -> nn.Module:
    """Train ProfessorCNN on clean data plus all samples from selected attacks."""
    set_global_seed(seed)

    train_parts = [clean_train] + [attack_train_datasets[i] for i in train_indices]
    combined = ConcatDataset(train_parts)

    generator = torch.Generator().manual_seed(seed)
    train_size = int(0.8 * len(combined))
    val_size = len(combined) - train_size
    train_ds, val_ds = random_split(combined, [train_size, val_size], generator=generator)

    train_loader = make_dataloader(train_ds, BATCH_SIZE, shuffle=True)
    val_loader = make_dataloader(val_ds, BATCH_SIZE)

    model = get_professor_cnn_best(input_size=input_size)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        num_epochs=num_epochs,
        patience=patience,
        monitor_auc=True,
    )
    return model


def _evaluate_on_attack_set(
    model: nn.Module,
    clean_test,
    attack_test_datasets: list,
    attack_names: list[str],
    test_indices: list[int],
) -> dict[str, dict[str, float | None]]:
    per_attack: dict[str, dict[str, float | None]] = {}
    for idx in test_indices:
        test_ds = ConcatDataset([clean_test, attack_test_datasets[idx]])
        test_loader = make_dataloader(test_ds, BATCH_SIZE)
        acc, prec, rec, f1, auc = evaluate_model(model, test_loader)
        per_attack[attack_names[idx]] = {
            "Accuracy": float(acc),
            "Precision": float(prec),
            "Recall": float(rec),
            "F1_Score": float(f1),
            "AUC_ROC": float(auc) if auc == auc else None,
        }
    return per_attack


def _aggregate_run_metrics(run_metrics: list[dict[str, float | None]]) -> dict[str, float | None]:
    metric_names = ["Accuracy", "Precision", "Recall", "F1_Score", "AUC_ROC"]
    aggregated: dict[str, float | None] = {}
    for metric_name in metric_names:
        values = [m[metric_name] for m in run_metrics if m[metric_name] is not None]
        mean_value, _ = _mean_std([float(v) for v in values])
        aggregated[metric_name] = mean_value
    return aggregated


def run_seeded_subset_comparison(
    clean_train,
    clean_test,
    attack_train_datasets: list,
    attack_test_datasets: list,
    attack_names: list[str],
    input_size: int,
    subsets: list[tuple[str, list[str]]],
    repeats: int = RUN_REPEATS,
    base_seed: int = BASE_SEED,
) -> list[dict]:
    """Compare predefined attack subsets using repeated seeded runs.

    Unlike the original balanced curriculum experiment, this version uses
    all available samples from the selected attacks. That makes the training
    set larger and removes the downsampling effect that was hiding possible
    gains from ProfessorCNN.
    """
    results: list[dict] = []

    for subset_index, (subset_label, train_attack_names) in enumerate(subsets):
        train_indices = [attack_names.index(name) for name in train_attack_names if name in attack_names]
        if not train_indices:
            continue

        unseen_indices = [i for i in range(len(attack_names)) if i not in train_indices]
        unseen_names = [attack_names[i] for i in unseen_indices]

        run_per_attack: dict[str, list[dict[str, float | None]]] = defaultdict(list)
        run_summaries: list[dict[str, float | None]] = []

        for repeat_index in range(repeats):
            seed = base_seed + subset_index * 1000 + repeat_index
            model = _train_professor_cnn(
                clean_train=clean_train,
                attack_train_datasets=attack_train_datasets,
                train_indices=train_indices,
                input_size=input_size,
                seed=seed,
            )
            per_attack = _evaluate_on_attack_set(
                model,
                clean_test,
                attack_test_datasets,
                attack_names,
                unseen_indices,
            )

            run_summaries.append(_aggregate_run_metrics(list(per_attack.values())))
            for attack_name, metrics in per_attack.items():
                run_per_attack[attack_name].append(metrics)

        per_attack_mean: dict[str, dict[str, float | None]] = {}
        per_attack_std: dict[str, dict[str, float | None]] = {}
        for attack_name, metrics_list in run_per_attack.items():
            per_attack_mean[attack_name] = {}
            per_attack_std[attack_name] = {}
            for metric_name in ["Accuracy", "Precision", "Recall", "F1_Score", "AUC_ROC"]:
                values = [m[metric_name] for m in metrics_list if m[metric_name] is not None]
                mean_value, std_value = _mean_std([float(v) for v in values])
                per_attack_mean[attack_name][metric_name] = mean_value
                per_attack_std[attack_name][metric_name] = std_value

        summary_mean: dict[str, float | None] = {}
        summary_std: dict[str, float | None] = {}
        for metric_name in ["Accuracy", "Precision", "Recall", "F1_Score", "AUC_ROC"]:
            values = [r[metric_name] for r in run_summaries if r[metric_name] is not None]
            mean_value, std_value = _mean_std([float(v) for v in values])
            summary_mean[f"mean_{metric_name}"] = mean_value
            summary_std[f"std_{metric_name}"] = std_value

        results.append(
            {
                "subset_label": subset_label,
                "train_attacks": [attack_names[i] for i in train_indices],
                "unseen_attacks": unseen_names,
                "per_attack_metrics": per_attack_mean,
                "per_attack_metrics_std": per_attack_std,
                "repeats": repeats,
                "run_seeds": [base_seed + subset_index * 1000 + i for i in range(repeats)],
                **summary_mean,
                **summary_std,
            }
        )

    return results


def run_seeded_pairwise_protocol(
    clean_train,
    clean_test,
    attack_train_datasets: list,
    attack_test_datasets: list,
    attack_names: list[str],
    input_size: int,
    repeats: int = RUN_REPEATS,
    base_seed: int = BASE_SEED,
) -> list[dict]:
    """Run a step1-style ordered pair protocol for every attack pair.

    For each ordered pair (train_attack, test_attack), the model is trained
    on clean_train plus all samples from the train_attack partition and then
    evaluated on clean_test plus all samples from the test_attack partition.
    """
    results: list[dict] = []

    for train_idx, train_attack_name in enumerate(attack_names):
        for test_idx, test_attack_name in enumerate(attack_names):
            if train_idx == test_idx:
                continue

            run_metrics: list[dict[str, float | None]] = []
            for repeat_index in range(repeats):
                seed = base_seed + train_idx * 1000 + test_idx * 100 + repeat_index
                model = _train_professor_cnn(
                    clean_train=clean_train,
                    attack_train_datasets=attack_train_datasets,
                    train_indices=[train_idx],
                    input_size=input_size,
                    seed=seed,
                )
                per_attack = _evaluate_on_attack_set(
                    model,
                    clean_test,
                    attack_test_datasets,
                    attack_names,
                    [test_idx],
                )
                run_metrics.append(next(iter(per_attack.values())))

            summary_mean: dict[str, float | None] = {}
            summary_std: dict[str, float | None] = {}
            for metric_name in ["Accuracy", "Precision", "Recall", "F1_Score", "AUC_ROC"]:
                values = [r[metric_name] for r in run_metrics if r[metric_name] is not None]
                mean_value, std_value = _mean_std([float(v) for v in values])
                summary_mean[f"mean_{metric_name}"] = mean_value
                summary_std[f"std_{metric_name}"] = std_value

            results.append(
                {
                    "train_attack": train_attack_name,
                    "test_attack": test_attack_name,
                    "repeats": repeats,
                    "run_seeds": [base_seed + train_idx * 1000 + test_idx * 100 + i for i in range(repeats)],
                    **summary_mean,
                    **summary_std,
                }
            )

    return results


def plot_pairwise_heatmap(results: list[dict], attack_names: list[str], plots_dir: Path | str) -> None:
    """Plot a heatmap of mean AUC-ROC for the ordered pair protocol."""
    plots_dir = Path(plots_dir)
    matrix = np.full((len(attack_names), len(attack_names)), np.nan)

    for row in results:
        i = attack_names.index(row["train_attack"])
        j = attack_names.index(row["test_attack"])
        matrix[i, j] = row.get("mean_AUC_ROC") if row.get("mean_AUC_ROC") is not None else np.nan

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0.4, vmax=1.0, interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Mean AUC-ROC")
    ax.set_xticks(range(len(attack_names)))
    ax.set_yticks(range(len(attack_names)))
    ax.set_xticklabels(attack_names, rotation=45, ha="right")
    ax.set_yticklabels(attack_names)
    ax.set_title("Pythia ProfessorCNN ordered-pair protocol (train attack -> test attack)")

    for i in range(len(attack_names)):
        for j in range(len(attack_names)):
            value = matrix[i, j]
            if value == value:
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8,
                        color="black" if value < 0.85 else "white")

    out = plots_dir / "step2_professor_pairwise_auc_heatmap.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out}")


def main() -> None:
    plots_dir = Path("plots") / "step2"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("  STEP 2 PROFESSOR - PYTHIA REPRODUCIBILITY SUITE")
    print("=" * 70)
    print(f"  base_seed={BASE_SEED}, repeats={RUN_REPEATS}")

    pythia_dir = Path("pythia")
    check_pythia_available(pythia_dir)

    pythia_clean = load_pythia_data(pythia_dir, "clean")
    pythia_clean_train, pythia_clean_test = split_train_test(pythia_clean)

    attack_partitions = [f"attack_{c}" for c in "abcdefgh"]
    attack_names: list[str] = []
    attack_train: list = []
    attack_test: list = []

    for partition in attack_partitions:
        print(f"  Loading Pythia partition '{partition}'...")
        dataset = load_pythia_data(pythia_dir, partition)
        train_ds, test_ds = split_train_test(dataset)
        attack_train.append(train_ds)
        attack_test.append(test_ds)
        attack_names.append(partition)

    print(f"\nReady: {len(attack_names)} Pythia attack partitions")

    visualize_samples(pythia_clean, save_path=plots_dir / "pythia_clean.png", title_prefix="Pythia Clean, ")
    for partition, ds_train in zip(attack_partitions, attack_train):
        visualize_samples(
            ds_train,
            save_path=plots_dir / f"pythia_{partition}.png",
            title_prefix=f"Pythia {partition}, ",
        )

    print("\n" + "=" * 70)
    print("  ATTACK SUBSET COMPARISON (ProfessorCNN)")
    print("  Uses all samples from the selected attacks; repeated runs report mean/std")
    print("=" * 70)
    subset_results = run_seeded_subset_comparison(
        clean_train=pythia_clean_train,
        clean_test=pythia_clean_test,
        attack_train_datasets=attack_train,
        attack_test_datasets=attack_test,
        attack_names=attack_names,
        input_size=70,
        subsets=PYTHIA_ATTACK_SUBSETS,
        repeats=RUN_REPEATS,
        base_seed=BASE_SEED,
    )
    plot_subset_comparison(
        subset_results,
        plots_dir,
        title="Pythia - Attack Subset Selection Comparison (ProfessorCNN, seeded repeats)",
        filename="step2_subset_comparison_prof_cnn.png",
    )

    print("\n" + "=" * 70)
    print("  ORDERED PAIR PROTOCOL (step1-style)")
    print("  Train on attack_A, test on attack_B, and vice versa for every pair")
    print("=" * 70)
    pairwise_results = run_seeded_pairwise_protocol(
        clean_train=pythia_clean_train,
        clean_test=pythia_clean_test,
        attack_train_datasets=attack_train,
        attack_test_datasets=attack_test,
        attack_names=attack_names,
        input_size=70,
        repeats=RUN_REPEATS,
        base_seed=BASE_SEED + 50_000,
    )
    plot_pairwise_heatmap(pairwise_results, attack_names, plots_dir)

    output = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "experiment": "Step 2 (Professor) - Pythia ProfessorCNN reproducibility suite",
        "base_seed": BASE_SEED,
        "repeats": RUN_REPEATS,
        "Pythia_ProfessorCNN_subset_comparison": subset_results,
        "Pythia_ProfessorCNN_ordered_pair_protocol": pairwise_results,
    }
    save_results(output, "faza2_wyniki_generalizacji_professor.json")

    print("\n" + "=" * 70)
    print("  step2_professor.py COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
