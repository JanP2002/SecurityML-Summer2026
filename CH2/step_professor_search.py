"""
step_professor_search.py — Staged hyperparameter search for ProfessorCNN
=========================================================================

Runs a 15-experiment staged grid search on the **Pythia dataset only** to
find the best ProfessorCNN configuration, as suggested by the professor.

Stage 1  (6 experiments)
    Fixed:  dropout=0.2, dense_head=[128, 64], pooling='global_average'
    Grid:   kernel_size ∈ {(4,5), (5,5)} × activation ∈ {leaky_relu, swish, gelu}

Stage 2  (3 experiments)
    Fixed:  best (kernel_size, activation) from Stage 1
    Grid:   dropout ∈ {0.1, 0.2, 0.3}

Stage 3  (6 experiments)
    Fixed:  best (kernel_size, activation, dropout) from Stages 1–2
    Grid:   dense_head ∈ {[128,64], [256,128], [64,32]} × pooling ∈ {flatten, global_average}

Model selection
    Primary metric:   validation AUC-ROC   (threshold-free, robust to class imbalance)
    Secondary metric: validation F1-Score

Outputs
    faza_professor_cnn_search.json  — all experiment metrics + best_config

Usage
-----
    # From the CH2/ directory (virtual-env active):
    python step_professor_search.py
"""

from __future__ import annotations

import copy
import json
import logging
import random
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, random_split

from lib import (
    ProfessorCNN,
    check_pythia_available,
    evaluate_model,
    load_pythia_data,
    make_dataloader,
    parse_results,
    save_results,
    split_train_test,
    train_model,
)

# ---------------------------------------------------------------------------
# Hyperparameters — match step1.py for a fair comparison
# ---------------------------------------------------------------------------
BATCH_SIZE  = 64
NUM_EPOCHS  = 15
PATIENCE    = 3
RANDOM_SEED = 42

OUTPUT_JSON = Path("faza_professor_cnn_search.json")
LOG_FILE    = Path("professor_search.log")


# ===========================================================================
# LOGGING SETUP
# ===========================================================================

def _setup_logging() -> logging.Logger:
    """Configure root logger: INFO to console (with colour prefix) + DEBUG to file."""
    log_path = Path(__file__).parent / LOG_FILE
    log_path.parent.mkdir(parents=True, exist_ok=True)

    fmt_file    = logging.Formatter(
        "%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fmt_console = logging.Formatter("[%(levelname)s] %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8", mode="a")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt_file)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt_console)

    logger = logging.getLogger("professor_search")
    logger.setLevel(logging.DEBUG)
    # Avoid duplicate handlers if main() is called more than once in a session
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


def _fmt_config(cfg: dict) -> str:
    """One-line human-readable config summary."""
    return (
        f"kernel={cfg.get('kernel_size')}  act={cfg.get('activation')!s:<10}"
        f"  drop={cfg.get('dropout')}  head={cfg.get('dense_head')}  pool={cfg.get('pooling')}"
    )


def _fmt_metrics(m: dict, label: str = "") -> str:
    """One-line metric summary."""
    prefix = f"{label}: " if label else ""
    return (
        f"{prefix}Acc={m.get('Accuracy', 0):.4f}  "
        f"Prec={m.get('Precision', 0):.4f}  "
        f"Rec={m.get('Recall', 0):.4f}  "
        f"F1={m.get('F1_Score', 0):.4f}  "
        f"AUC={m.get('AUC_ROC', 0):.4f}"
    )





# ===========================================================================
# HELPERS
# ===========================================================================


def _train_and_eval(
    config: dict,
    train_ds,
    val_ds,
    test_a_ds,
    test_b_ds,
    stage_label: str,
) -> dict:
    """Train one ProfessorCNN variant and evaluate on val + Test_A + Test_B.

    Parameters
    ----------
    config : dict
        Keys: kernel_size, activation, dropout, dense_head, pooling.
    train_ds, val_ds : Dataset
        80/20 split of clean ∪ attack_a.
    test_a_ds : Dataset
        clean_test + attack_a_test  (known attack).
    test_b_ds : Dataset
        clean_test + attack_b_test  (unknown attack).
    stage_label : str
        String used in console output (e.g. "Stage 1, exp 2/6").

    Returns
    -------
    dict
        config + val_metrics + test_a_metrics + test_b_metrics.
    """
    ks = config["kernel_size"]
    if isinstance(ks, list):
        ks = tuple(ks)

    model = ProfessorCNN(
        input_size=70,
        kernel_size=ks,
        activation=config["activation"],
        dropout=config["dropout"],
        dense_head=config["dense_head"],
        pooling=config["pooling"],
    )
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    log = logging.getLogger("professor_search")
    log.info("")
    log.info("-" * 60)
    log.info("%-20s  %s", stage_label, _fmt_config(config))
    log.info("-" * 60)

    t0 = time.perf_counter()
    model = train_model(
        model,
        make_dataloader(train_ds, BATCH_SIZE, shuffle=True),
        make_dataloader(val_ds,   BATCH_SIZE),
        criterion, optimizer,
        num_epochs=NUM_EPOCHS, patience=PATIENCE,
    )
    elapsed = time.perf_counter() - t0
    log.debug("  Training finished in %.1f s", elapsed)

    log.info("  Evaluating...")
    val_metrics    = parse_results(evaluate_model(model, make_dataloader(val_ds,    BATCH_SIZE)))
    test_a_metrics = parse_results(evaluate_model(model, make_dataloader(test_a_ds, BATCH_SIZE)))
    test_b_metrics = parse_results(evaluate_model(model, make_dataloader(test_b_ds, BATCH_SIZE)))

    log.info("  %s", _fmt_metrics(val_metrics,    "Val  "))
    log.info("  %s", _fmt_metrics(test_a_metrics,  "Test_A"))
    log.info("  %s", _fmt_metrics(test_b_metrics,  "Test_B"))
    log.info("  Wall time: %.1f s", elapsed)

    return {
        "config": {
            "kernel_size": list(config["kernel_size"])
            if isinstance(config["kernel_size"], tuple) else config["kernel_size"],
            "activation":  config["activation"],
            "dropout":     config["dropout"],
            "dense_head":  config["dense_head"],
            "pooling":     config["pooling"],
        },
        "val_metrics":    val_metrics,
        "test_a_metrics": test_a_metrics,
        "test_b_metrics": test_b_metrics,
    }


def _score(result: dict) -> float:
    """Primary selection score: validation AUC-ROC."""
    auc = result["val_metrics"].get("AUC_ROC")
    return float(auc) if auc is not None and auc == auc else 0.0  # NaN → 0


def _best(results: list[dict]) -> dict:
    """Return the result dict with the highest validation AUC-ROC."""
    return max(results, key=_score)


# ===========================================================================
# MAIN SEARCH
# ===========================================================================

def main() -> None:
    log = _setup_logging()
    run_start = time.perf_counter()

    log.info("=" * 60)
    log.info("ProfessorCNN Hyperparameter Search  (%s)",
             datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    log.info("  BATCH_SIZE=%d  NUM_EPOCHS=%d  PATIENCE=%d  SEED=%d",
             BATCH_SIZE, NUM_EPOCHS, PATIENCE, RANDOM_SEED)
    log.info("  Output: %s", OUTPUT_JSON)
    log.info("  Log:    %s", LOG_FILE)
    log.info("=" * 60)

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    PYTHIA_DIR = Path("pythia")
    check_pythia_available(PYTHIA_DIR)

    log.info("\nLoading Pythia partitions for search...")
    pythia_clean    = load_pythia_data(PYTHIA_DIR, "clean")
    pythia_attack_a = load_pythia_data(PYTHIA_DIR, "attack_a")
    pythia_attack_b = load_pythia_data(PYTHIA_DIR, "attack_b")

    pythia_clean_train_base, pythia_clean_test   = split_train_test(pythia_clean)
    pythia_atk_a_train_base, pythia_atk_a_test   = split_train_test(pythia_attack_a)
    pythia_atk_b_test = pythia_attack_b          # full partition = test (matches step1)

    # Build combined train+val pool (clean + attack_a, 80/20 split inside)
    combined_tv = ConcatDataset([pythia_clean_train_base, pythia_atk_a_train_base])
    t_size = int(0.8 * len(combined_tv))
    v_size = len(combined_tv) - t_size
    train_ds, val_ds = random_split(combined_tv, [t_size, v_size])

    test_a_ds = ConcatDataset([pythia_clean_test, pythia_atk_a_test])
    test_b_ds = ConcatDataset([pythia_clean_test, pythia_atk_b_test])

    log.info("  Split sizes — Train: %d | Val: %d | Test_A: %d | Test_B: %d",
             len(train_ds), len(val_ds), len(test_a_ds), len(test_b_ds))

    all_results: list[dict] = []
    total_experiments = 15  # 6 + 3 + 6
    exp_counter = 0

    # =======================================================================
    # Stage 1: kernel_size × activation  (6 experiments)
    # Fixed: dropout=0.2, dense_head=[128,64], pooling='global_average'
    # =======================================================================
    print("\n" + "=" * 60)
    print("  STAGE 1 — kernel_size × activation  (6 experiments)")
    print("=" * 60)

    stage1_results: list[dict] = []
    stage1_grid = [
        {"kernel_size": (4, 5), "activation": "leaky_relu", "dropout": 0.2, "dense_head": [128, 64], "pooling": "global_average"},
        {"kernel_size": (4, 5), "activation": "swish",      "dropout": 0.2, "dense_head": [128, 64], "pooling": "global_average"},
        {"kernel_size": (4, 5), "activation": "gelu",       "dropout": 0.2, "dense_head": [128, 64], "pooling": "global_average"},
        {"kernel_size": (5, 5), "activation": "leaky_relu", "dropout": 0.2, "dense_head": [128, 64], "pooling": "global_average"},
        {"kernel_size": (5, 5), "activation": "swish",      "dropout": 0.2, "dense_head": [128, 64], "pooling": "global_average"},
        {"kernel_size": (5, 5), "activation": "gelu",       "dropout": 0.2, "dense_head": [128, 64], "pooling": "global_average"},
    ]

    for i, cfg in enumerate(stage1_grid, start=1):
        exp_counter += 1
        log.info("")
        log.info("STAGE 1  exp %d/%d  (overall %d/%d)",
                 i, len(stage1_grid), exp_counter, total_experiments)
        result = _train_and_eval(
            cfg, train_ds, val_ds, test_a_ds, test_b_ds,
            f"Stage 1, exp {i}/{len(stage1_grid)}"
        )
        result["stage"] = 1
        stage1_results.append(result)
        all_results.append(result)
        log.info("  RESULT  Val AUC-ROC=%.4f  F1=%.4f",
                 result["val_metrics"].get("AUC_ROC", 0),
                 result["val_metrics"].get("F1_Score", 0))

    best1 = _best(stage1_results)
    best_kernel = tuple(best1["config"]["kernel_size"]) if isinstance(best1["config"]["kernel_size"], list) else best1["config"]["kernel_size"]
    best_act    = best1["config"]["activation"]
    log.info("")
    log.info("★ Stage 1 WINNER: kernel=%s  activation=%s  val AUC=%.4f  val F1=%.4f",
             best_kernel, best_act,
             best1["val_metrics"].get("AUC_ROC", 0),
             best1["val_metrics"].get("F1_Score", 0))

    # =======================================================================
    # Stage 2: dropout  (3 experiments)
    # Fixed: best kernel + activation from Stage 1
    # =======================================================================
    print("\n" + "=" * 60)
    print("  STAGE 2 — dropout  (3 experiments)")
    print("=" * 60)

    stage2_results: list[dict] = []
    for drop in [0.1, 0.2, 0.3]:
        exp_counter += 1
        log.info("")
        log.info("STAGE 2  dropout=%.1f  (overall %d/%d)",
                 drop, exp_counter, total_experiments)
        cfg = {
            "kernel_size": best_kernel,
            "activation":  best_act,
            "dropout":     drop,
            "dense_head":  [128, 64],
            "pooling":     "global_average",
        }
        result = _train_and_eval(
            cfg, train_ds, val_ds, test_a_ds, test_b_ds,
            f"Stage 2, dropout={drop}"
        )
        result["stage"] = 2
        stage2_results.append(result)
        all_results.append(result)
        log.info("  RESULT  Val AUC-ROC=%.4f  F1=%.4f",
                 result["val_metrics"].get("AUC_ROC", 0),
                 result["val_metrics"].get("F1_Score", 0))

    best2        = _best(stage2_results)
    best_dropout = best2["config"]["dropout"]
    log.info("")
    log.info("★ Stage 2 WINNER: dropout=%.2f  val AUC=%.4f  val F1=%.4f",
             best_dropout,
             best2["val_metrics"].get("AUC_ROC", 0),
             best2["val_metrics"].get("F1_Score", 0))

    # =======================================================================
    # Stage 3: dense_head × pooling  (6 experiments)
    # Fixed: best kernel + activation + dropout from Stages 1–2
    # =======================================================================
    print("\n" + "=" * 60)
    print("  STAGE 3 — dense_head × pooling  (6 experiments)")
    print("=" * 60)

    stage3_results: list[dict] = []
    for head in [[128, 64], [256, 128], [64, 32]]:
        for pool in ["global_average", "flatten"]:
            exp_counter += 1
            log.info("")
            log.info("STAGE 3  head=%s  pool=%s  (overall %d/%d)",
                     head, pool, exp_counter, total_experiments)
            cfg = {
                "kernel_size": best_kernel,
                "activation":  best_act,
                "dropout":     best_dropout,
                "dense_head":  head,
                "pooling":     pool,
            }
            result = _train_and_eval(
                cfg, train_ds, val_ds, test_a_ds, test_b_ds,
                f"Stage 3, head={head}  pool={pool}"
            )
            result["stage"] = 3
            stage3_results.append(result)
            all_results.append(result)
            log.info("  RESULT  Val AUC-ROC=%.4f  F1=%.4f",
                     result["val_metrics"].get("AUC_ROC", 0),
                     result["val_metrics"].get("F1_Score", 0))

    # =======================================================================
    # Select overall best
    # =======================================================================
    best_overall = _best(all_results)
    best_config  = best_overall["config"]
    total_elapsed = time.perf_counter() - run_start

    log.info("")
    log.info("=" * 60)
    log.info("SEARCH COMPLETE  (%d experiments  |  total %.1f min)",
             len(all_results), total_elapsed / 60)
    log.info("=" * 60)
    log.info("★ BEST CONFIG:")
    log.info("    kernel_size : %s", best_config["kernel_size"])
    log.info("    activation  : %s", best_config["activation"])
    log.info("    dropout     : %s", best_config["dropout"])
    log.info("    dense_head  : %s", best_config["dense_head"])
    log.info("    pooling     : %s", best_config["pooling"])
    log.info("")
    log.info("★ BEST METRICS:")
    log.info("    %s", _fmt_metrics(best_overall["val_metrics"],    "Val   "))
    log.info("    %s", _fmt_metrics(best_overall["test_a_metrics"], "Test_A"))
    log.info("    %s", _fmt_metrics(best_overall["test_b_metrics"], "Test_B"))

    # Stage-winner summary table
    log.info("")
    log.info("--- Stage winners (val AUC-ROC) ---")
    log.info("  Stage 1  kernel=%s  act=%-10s  AUC=%.4f",
             best_kernel, best_act, best1["val_metrics"].get("AUC_ROC", 0))
    log.info("  Stage 2  dropout=%.2f                 AUC=%.4f",
             best_dropout, best2["val_metrics"].get("AUC_ROC", 0))
    best3 = _best(stage3_results)
    log.info("  Stage 3  head=%s  pool=%-14s  AUC=%.4f",
             best3["config"]["dense_head"], best3["config"]["pooling"],
             best3["val_metrics"].get("AUC_ROC", 0))

    # --- Also compare best ProfessorCNN against the Pythia AnomalyCNN baseline ---
    print("\nTraining AnomalyCNN baseline for comparison...")
    log.info("")
    log.info("Training AnomalyCNN baseline for comparison...")
    from lib import AnomalyCNN
    baseline = AnomalyCNN(input_size=70)
    baseline_opt = optim.Adam(baseline.parameters(), lr=0.001)
    t_base = time.perf_counter()
    baseline = train_model(
        baseline,
        make_dataloader(train_ds, BATCH_SIZE, shuffle=True),
        make_dataloader(val_ds,   BATCH_SIZE),
        nn.BCELoss(), baseline_opt,
        num_epochs=NUM_EPOCHS, patience=PATIENCE,
    )
    log.debug("  Baseline training: %.1f s", time.perf_counter() - t_base)
    print("\n  [AnomalyCNN Baseline] Val:")
    baseline_val   = parse_results(evaluate_model(baseline, make_dataloader(val_ds,   BATCH_SIZE)))
    print("\n  [AnomalyCNN Baseline] Test_A:")
    baseline_test_a = parse_results(evaluate_model(baseline, make_dataloader(test_a_ds, BATCH_SIZE)))
    print("\n  [AnomalyCNN Baseline] Test_B:")
    baseline_test_b = parse_results(evaluate_model(baseline, make_dataloader(test_b_ds, BATCH_SIZE)))

    log.info("")
    log.info("--- AnomalyCNN baseline ---")
    log.info("  %s", _fmt_metrics(baseline_val,    "Val   "))
    log.info("  %s", _fmt_metrics(baseline_test_a, "Test_A"))
    log.info("  %s", _fmt_metrics(baseline_test_b, "Test_B"))
    log.info("")
    log.info("--- Comparison: Test_B AUC-ROC ---")
    log.info("  AnomalyCNN   : %.4f", baseline_test_b.get("AUC_ROC", 0))
    log.info("  ProfessorCNN : %.4f  (%+.4f)",
             best_overall["test_b_metrics"].get("AUC_ROC", 0),
             best_overall["test_b_metrics"].get("AUC_ROC", 0) - baseline_test_b.get("AUC_ROC", 0))

    print("\n  Comparison:")
    print(f"    AnomalyCNN  Test_B AUC-ROC: {baseline_test_b['AUC_ROC']:.4f}")
    print(f"    ProfessorCNN Test_B AUC-ROC: {best_overall['test_b_metrics']['AUC_ROC']:.4f}")

    # =======================================================================
    # Save results
    # =======================================================================
    output = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "experiment": "ProfessorCNN staged hyperparameter search (Pythia)",
        "search_config": {
            "dataset": "Pythia (70x70 px)",
            "num_experiments": len(all_results),
            "epochs": NUM_EPOCHS,
            "patience": PATIENCE,
            "batch_size": BATCH_SIZE,
            "random_seed": RANDOM_SEED,
            "selection_metric": "val_AUC_ROC",
        },
        "best_config": best_config,
        "best_val_metrics": best_overall["val_metrics"],
        "best_test_a_metrics": best_overall["test_a_metrics"],
        "best_test_b_metrics": best_overall["test_b_metrics"],
        "pythia_baseline": {
            "model": "AnomalyCNN",
            "val_metrics":    baseline_val,
            "test_a_metrics": baseline_test_a,
            "test_b_metrics": baseline_test_b,
        },
        "all_experiments": all_results,
    }

    save_results(output, OUTPUT_JSON)

    log.info("")
    log.info("Results saved to: %s", OUTPUT_JSON)
    log.info("Log file:         %s", LOG_FILE)
    log.info("Total run time:   %.1f min", (time.perf_counter() - run_start) / 60)
    log.info("get_professor_cnn_best() will now load the best config automatically.")
    print(
        f"\nResults saved to: {OUTPUT_JSON}\n"
        f"Log file:         {LOG_FILE}\n"
        f"The best config will be loaded automatically by get_professor_cnn_best().\n"
        f"Now run step1.py, step2.py, and step3.py to see the full comparison."
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
