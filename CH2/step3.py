"""
step3.py — Anomaly Detection: Unsupervised Detection via Autoencoder (Step 3)
==============================================================================

Implements the Step 3 research question:

    *Does an unsupervised autoencoder — trained ONLY on clean data — detect a
    completely unseen attack better than the supervised classifier of Step 1?*

Rationale
---------
The Step 1 classifier learns to recognise one specific attack (attack_a).
It therefore fails on a structurally different unseen attack (attack_b),
because it has learned a *noise detector*, not a *model of normality*.

An autoencoder takes the opposite approach.  It is trained only to compress
and reconstruct **clean** images, so it becomes an expert on normality and a
complete ignorant about everything else.  The per-pixel reconstruction error

    Score = || x - f(g(x)) ||^2          (g = encoder, f = decoder)

is then used as the anomaly score: clean images reconstruct well (low score),
anything off the clean manifold reconstructs poorly (high score).  Because the
autoencoder never sees any attack in training, *no* attack is "known" to it —
which is exactly why it should generalise to unseen attacks.

Methodology
-----------
For each dataset (MNIST and Pythia) the script trains two detectors on the
SAME data splits, so the comparison is strictly apples-to-apples:

    1. Baseline classifier  — AnomalyCNN trained on  clean + attack_a
                              (a faithful reproduction of Step 1).
    2. Autoencoder          — ConvAutoencoder trained on  clean ONLY.
                              The anomaly threshold is the 95th percentile of
                              reconstruction error on a held-out CLEAN
                              validation split — chosen without ever looking
                              at attack data.

Both detectors are then evaluated on identical test sets:

    Test_A = clean_test + attack_a       (attack KNOWN to the classifier)
    Test_B = clean_test + attack_b       (attack UNKNOWN to both detectors)

Extended analysis
-----------------
To draw richer conclusions, both trained detectors are additionally evaluated
across the full battery of contamination types from Step 2 (MNIST: Gaussian,
salt & pepper, geometric, blended, backdoor, OOD; Pythia: attack_a … attack_h).
This is pure inference on the already-trained models — it shows *which* kinds
of anomaly each paradigm catches and which it misses.

Architecture — encoder "3+2"
----------------------------
The encoder uses the instructor-suggested 3+2 topology (3 conv blocks + 2 FC),
identical in shape to AnomalyCNN, so the encoder can be reused as a feature
extractor in Step 4.  See ConvAutoencoder in lib.py.

Outputs
-------
    faza3_wyniki_autoenkodera.json   — all numerical results
    plots/step3/*.png                — reconstructions, score distributions,
                                       ROC comparison, per-attack comparison

Usage
-----
    # From the CH2/ directory:
    python step3.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure CH2/ is on sys.path so lib and attacks import regardless of the
# working directory the script is launched from.
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, random_split

# lib is imported before pyplot so its matplotlib.use("Agg") call (non-
# interactive backend) takes effect before any figure backend is selected.
from lib import (
    AnomalyCNN,
    ConvAutoencoder,
    check_pythia_available,
    evaluate_autoencoder,
    evaluate_model,
    load_pythia_data,
    make_dataloader,
    parse_results,
    prepare_clean_data,
    reconstruction_scores,
    save_results,
    select_threshold,
    split_train_test,
    train_autoencoder,
    train_model,
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

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

# ---------------------------------------------------------------------------
# Global hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE = 64

# Baseline classifier (Step 1 reproduction) — identical to step1.py.
CLF_NUM_EPOCHS = 15
CLF_PATIENCE = 3

# Autoencoder — reconstruction training benefits from more epochs than the
# classifier, so it is given a longer budget and more patience.
AE_NUM_EPOCHS = 30
AE_PATIENCE = 5
LATENT_DIM = 32           # bottleneck size of the autoencoder

# Fraction of the clean TRAINING data held out as a clean validation split.
# Used both for early stopping and for choosing the anomaly threshold.
AE_VAL_RATIO = 0.1

# Anomaly threshold = this percentile of clean-validation reconstruction
# error.  95 -> the detector accepts a ~5 % false-positive rate on clean data.
THRESHOLD_PERCENTILE = 95.0

# Fixed seed for reproducible splits / results across runs.
RANDOM_SEED = 42


# ===========================================================================
# HELPERS
# ===========================================================================


def predict_probs(model: nn.Module, loader) -> tuple[np.ndarray, np.ndarray]:
    """Return the classifier's attack-probability outputs and ground-truth labels.

    AnomalyCNN already applies a sigmoid, so its output is the probability of
    the attack class.  This mirrors the internal logic of ``evaluate_model``
    but exposes the continuous scores, which are needed to draw ROC curves.

    Parameters
    ----------
    model : nn.Module
        A trained AnomalyCNN.
    loader : DataLoader
        Yields ``(image, label)`` batches.

    Returns
    -------
    probs : np.ndarray, shape (N,)
        Predicted probability of the attack class.
    labels : np.ndarray, shape (N,)
        Ground-truth labels (0 = clean, 1 = attack).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    probs: list = []
    labels: list = []
    with torch.no_grad():
        for inputs, lbls in loader:
            inputs = inputs.to(device)
            out = model(inputs).squeeze()
            if out.dim() == 0:                 # batch of exactly 1
                out = out.unsqueeze(0)
            probs.extend(out.cpu().numpy())
            labels.extend(lbls.numpy())

    return np.array(probs), np.array(labels)


# ===========================================================================
# EXPERIMENT RUNNERS
# ===========================================================================


def run_baseline_classifier(
    clean_train,
    clean_test,
    attack_a_train,
    attack_a_test,
    attack_b_test,
    input_size: int,
    name: str,
) -> tuple[nn.Module, dict]:
    """Train the Step 1 baseline classifier and evaluate it on Test_A / Test_B.

    A faithful reproduction of Step 1: an AnomalyCNN trained on
    ``clean + attack_a`` (80/20 train/val split), then evaluated on the known
    attack (Test_A) and the unknown attack (Test_B).  The model is retrained
    inside this script (rather than loading Step 1's saved results) so the
    comparison against the autoencoder uses identical data splits.

    Parameters
    ----------
    clean_train, clean_test : Dataset
        Clean-class splits (label = 0).
    attack_a_train, attack_a_test : Dataset
        Known-attack splits (label = 1).
    attack_b_test : Dataset
        Unknown-attack test split (label = 1).
    input_size : int
        Square spatial size of the input images (28 MNIST, 70 Pythia).
    name : str
        Dataset label used in console output.

    Returns
    -------
    model : nn.Module
        The trained classifier (reused later for the across-attacks analysis).
    results : dict
        ``{"Test_A": {...}, "Test_B": {...}}`` of parsed metric dicts.
    """
    bar = "=" * 65
    print(f"\n{bar}")
    print(f"  [{name}] BASELINE CLASSIFIER  (Step 1 reproduction)")
    print(f"  Training on:  clean + attack_a   (supervised, labels used)")
    print(f"{bar}")

    train_val = ConcatDataset([clean_train, attack_a_train])
    t_size = int(0.8 * len(train_val))
    v_size = len(train_val) - t_size
    train_ds, val_ds = random_split(train_val, [t_size, v_size])
    print(f"  Train: {len(train_ds):>7} | Val: {len(val_ds):>7}  "
          f"(clean ∪ attack_a, 80/20 split)")

    model = AnomalyCNN(input_size=input_size)
    criterion = nn.BCELoss()                   # AnomalyCNN outputs probabilities
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model = train_model(
        model,
        make_dataloader(train_ds, BATCH_SIZE, shuffle=True),
        make_dataloader(val_ds, BATCH_SIZE),
        criterion, optimizer,
        num_epochs=CLF_NUM_EPOCHS, patience=CLF_PATIENCE,
    )

    test_a = ConcatDataset([clean_test, attack_a_test])
    test_b = ConcatDataset([clean_test, attack_b_test])

    print(f"\n  [{name}] Classifier — Test_A (KNOWN attack_a):")
    res_a = evaluate_model(model, make_dataloader(test_a, BATCH_SIZE))
    print(f"\n  [{name}] Classifier — Test_B (UNKNOWN attack_b):")
    res_b = evaluate_model(model, make_dataloader(test_b, BATCH_SIZE))

    return model, {"Test_A": parse_results(res_a), "Test_B": parse_results(res_b)}


def run_autoencoder(
    clean_train,
    clean_test,
    attack_a_test,
    attack_b_test,
    input_size: int,
    name: str,
) -> tuple[nn.Module, float, dict]:
    """Train the unsupervised autoencoder and evaluate it on Test_A / Test_B.

    The autoencoder is trained on **clean data only**.  The clean training
    pool is split into an autoencoder-train part and a clean-validation part;
    the validation part drives early stopping AND supplies the anomaly
    threshold (its 95th-percentile reconstruction error).  No attack sample is
    ever used during training or threshold selection.

    Parameters
    ----------
    clean_train, clean_test : Dataset
        Clean-class splits (label = 0).
    attack_a_test, attack_b_test : Dataset
        Attack test splits (label = 1).  Used for evaluation only.
    input_size : int
        Square spatial size of the input images.
    name : str
        Dataset label used in console output.

    Returns
    -------
    model : nn.Module
        The trained autoencoder.
    threshold : float
        The reconstruction-error anomaly threshold.
    results : dict
        ``{"threshold", "threshold_percentile", "Test_A", "Test_B"}``.
    """
    bar = "=" * 65
    print(f"\n{bar}")
    print(f"  [{name}] AUTOENCODER  (Step 3 — unsupervised)")
    print(f"  Training on:  clean ONLY   (no attack labels, no attack images)")
    print(f"{bar}")

    # Split the clean training pool into AE-train and clean-validation.
    ae_train, ae_val = split_train_test(clean_train, train_ratio=1.0 - AE_VAL_RATIO)
    print(f"  AE-train: {len(ae_train):>7} | clean-val: {len(ae_val):>7}  "
          f"(clean only, {int(100*(1-AE_VAL_RATIO))}/{int(100*AE_VAL_RATIO)} split)")

    model = ConvAutoencoder(input_size=input_size, latent_dim=LATENT_DIM)
    criterion = nn.MSELoss()                   # reconstruction loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    val_loader = make_dataloader(ae_val, BATCH_SIZE)
    model = train_autoencoder(
        model,
        make_dataloader(ae_train, BATCH_SIZE, shuffle=True),
        val_loader,
        criterion, optimizer,
        num_epochs=AE_NUM_EPOCHS, patience=AE_PATIENCE,
    )

    # Threshold: 95th percentile of reconstruction error on the CLEAN
    # validation split — selected without ever inspecting attack data.
    clean_val_scores, _ = reconstruction_scores(model, val_loader)
    threshold = select_threshold(clean_val_scores, THRESHOLD_PERCENTILE)
    print(f"\n  Threshold (p{THRESHOLD_PERCENTILE:g} of clean-val error): {threshold:.6f}")

    test_a = ConcatDataset([clean_test, attack_a_test])
    test_b = ConcatDataset([clean_test, attack_b_test])

    print(f"\n  [{name}] Autoencoder — Test_A (attack_a):")
    res_a = evaluate_autoencoder(model, make_dataloader(test_a, BATCH_SIZE), threshold)
    print(f"\n  [{name}] Autoencoder — Test_B (attack_b):")
    res_b = evaluate_autoencoder(model, make_dataloader(test_b, BATCH_SIZE), threshold)

    return model, threshold, {
        "threshold": threshold,
        "threshold_percentile": THRESHOLD_PERCENTILE,
        "Test_A": parse_results(res_a),
        "Test_B": parse_results(res_b),
    }


def evaluate_across_attacks(
    classifier: nn.Module,
    autoencoder: nn.Module,
    threshold: float,
    clean_test,
    attack_tests: dict,
    name: str,
) -> dict:
    """Evaluate both trained detectors across every available attack type.

    Pure inference on the already-trained classifier and autoencoder — no
    retraining.  For each attack type a balanced test set ``clean_test +
    attack`` is built and scored with both detectors.  This reveals which
    anomaly families each paradigm catches and which it misses.

    Parameters
    ----------
    classifier : nn.Module
        The trained baseline classifier.
    autoencoder : nn.Module
        The trained autoencoder.
    threshold : float
        The autoencoder's anomaly threshold.
    clean_test : Dataset
        Clean test split (label = 0).
    attack_tests : dict[str, Dataset]
        Mapping attack-type name -> attack test dataset (label = 1).
    name : str
        Dataset label used in console output.

    Returns
    -------
    dict
        ``{attack_name: {"classifier": {...}, "autoencoder": {...}}}``.
    """
    bar = "=" * 65
    print(f"\n{bar}")
    print(f"  [{name}] EXTENDED ANALYSIS — both detectors vs. every attack type")
    print(f"{bar}")

    rows: dict = {}
    for atk_name, atk_ds in attack_tests.items():
        test_ds = ConcatDataset([clean_test, atk_ds])
        loader = make_dataloader(test_ds, BATCH_SIZE)
        print(f"\n  --- attack type: {atk_name}  "
              f"(clean={len(clean_test)}, attack={len(atk_ds)}) ---")
        print(f"  Classifier:")
        clf_res = evaluate_model(classifier, loader)
        print(f"  Autoencoder:")
        ae_res = evaluate_autoencoder(autoencoder, loader, threshold)
        rows[atk_name] = {
            "classifier": parse_results(clf_res),
            "autoencoder": parse_results(ae_res),
        }
    return rows


# ===========================================================================
# PLOTTING
# ===========================================================================


def plot_reconstructions(
    model: nn.Module,
    partitions: dict,
    save_path: Path | str,
    n_samples: int = 6,
    title: str = "",
) -> None:
    """Save an original / reconstruction / error-map grid for several partitions.

    For each partition the figure shows three sub-rows: the original image,
    the autoencoder's reconstruction, and the per-pixel squared-error heat
    map.  Clean partitions should reconstruct faithfully (cool error maps);
    attack partitions should reconstruct poorly (hot error maps) — the visual
    intuition behind the reconstruction-error anomaly score.

    Parameters
    ----------
    model : nn.Module
        A trained autoencoder.
    partitions : dict[str, Dataset]
        Mapping partition name -> dataset (e.g. clean, attack_a, attack_b).
    save_path : Path | str
        Output PNG path.
    n_samples : int
        Number of sample images per partition (default 6).
    title : str
        Figure suptitle.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    n_part = len(partitions)
    fig, axes = plt.subplots(
        3 * n_part, n_samples,
        figsize=(2.0 * n_samples, 2.0 * 3 * n_part),
    )
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")

    for p_idx, (pname, ds) in enumerate(partitions.items()):
        imgs = torch.stack([ds[i][0] for i in range(min(n_samples, len(ds)))])
        with torch.no_grad():
            recon = model(imgs.to(device)).cpu()
        err = (recon - imgs) ** 2

        row0 = 3 * p_idx
        row_labels = [f"{pname}\nORIGINAL", "RECONSTRUCTED", "SQUARED ERROR"]
        for sub in range(3):
            for j in range(n_samples):
                ax = axes[row0 + sub, j]
                if sub == 0:
                    ax.imshow(imgs[j].squeeze(), cmap="gray", vmin=0, vmax=1)
                elif sub == 1:
                    ax.imshow(recon[j].squeeze(), cmap="gray", vmin=0, vmax=1)
                else:
                    ax.imshow(err[j].squeeze(), cmap="hot")
                ax.set_xticks([])
                ax.set_yticks([])
                if j == 0:
                    ax.set_ylabel(row_labels[sub], fontsize=9, fontweight="bold")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] Saved → {save_path}")


def plot_score_distributions(
    scores_by_partition: dict,
    threshold: float,
    save_path: Path | str,
    title: str = "",
) -> None:
    """Plot histograms of reconstruction error for clean vs. attack partitions.

    Overlapping density histograms make the separation (or lack of it)
    between clean and attack reconstruction errors directly visible.  The
    dashed vertical line marks the anomaly threshold: ideally clean mass sits
    left of it and attack mass sits right of it.

    Parameters
    ----------
    scores_by_partition : dict[str, np.ndarray]
        Mapping partition name -> array of reconstruction errors.
    threshold : float
        The anomaly threshold (vertical reference line).
    save_path : Path | str
        Output PNG path.
    title : str
        Figure title.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    all_scores = np.concatenate(list(scores_by_partition.values()))
    # Clip the upper edge at the 99th percentile so a few extreme outliers
    # do not squash the informative part of the histogram.
    hi = np.percentile(all_scores, 99)
    lo = float(all_scores.min())
    bins = np.linspace(lo, hi, 60)

    for pname, scores in scores_by_partition.items():
        ax.hist(scores, bins=bins, alpha=0.55, density=True,
                label=f"{pname}  (n={len(scores)})")

    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.8,
               label=f"threshold (p{THRESHOLD_PERCENTILE:g})")
    ax.set_xlabel("reconstruction error  (mean squared error per pixel)", fontsize=10)
    ax.set_ylabel("density", fontsize=10)
    ax.set_title(title or "Reconstruction-error distributions", fontsize=12,
                 fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] Saved → {save_path}")


def plot_roc_comparison(
    roc_data: dict,
    save_path: Path | str,
    title: str = "",
) -> None:
    """Plot ROC curves comparing the classifier and the autoencoder.

    ROC / AUC is threshold-free, so it is the fairest single view for
    comparing a supervised classifier against an unsupervised autoencoder.

    Parameters
    ----------
    roc_data : dict[str, tuple]
        Mapping curve label -> ``(y_true, y_score)`` arrays.
    save_path : Path | str
        Output PNG path.
    title : str
        Figure title.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    styles = {
        "Classifier — Test_A":  ("steelblue",   "-"),
        "Classifier — Test_B":  ("steelblue",   "--"),
        "Autoencoder — Test_A": ("crimson",     "-"),
        "Autoencoder — Test_B": ("crimson",     "--"),
    }

    for label, (y_true, y_score) in roc_data.items():
        # ROC is undefined if only one class is present.
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_score)
        # Trapezoidal AUC straight from the curve.
        auc = float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") \
            else float(np.trapz(tpr, fpr))
        color, ls = styles.get(label, ("gray", "-"))
        ax.plot(fpr, tpr, color=color, linestyle=ls, linewidth=2,
                label=f"{label}  (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=1,
            label="chance (AUC=0.500)")
    ax.set_xlabel("False positive rate", fontsize=10)
    ax.set_ylabel("True positive rate", fontsize=10)
    ax.set_title(title or "ROC — classifier vs. autoencoder", fontsize=12,
                 fontweight="bold")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] Saved → {save_path}")


def plot_metric_comparison(
    clf_results: dict,
    ae_results: dict,
    save_path: Path | str,
    title: str = "",
) -> None:
    """Grouped bar chart: classifier vs. autoencoder on Test_A and Test_B.

    Two side-by-side panels (Test_A, Test_B); each panel groups the five
    metrics with one bar per detector.

    Parameters
    ----------
    clf_results : dict
        ``{"Test_A": {...}, "Test_B": {...}}`` for the classifier.
    ae_results : dict
        Same structure for the autoencoder.
    save_path : Path | str
        Output PNG path.
    title : str
        Figure suptitle.
    """
    metrics = ["Accuracy", "Precision", "Recall", "F1_Score", "AUC_ROC"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")

    for ax, split in zip(axes, ["Test_A", "Test_B"]):
        x = np.arange(len(metrics))
        width = 0.38

        def _vals(res: dict) -> list:
            return [res[split][m] if res[split][m] is not None else 0.0
                    for m in metrics]

        ax.bar(x - width / 2, _vals(clf_results), width,
               label="Classifier (Step 1)", color="steelblue")
        ax.bar(x + width / 2, _vals(ae_results), width,
               label="Autoencoder (Step 3)", color="crimson")

        subtitle = "KNOWN attack_a" if split == "Test_A" else "UNKNOWN attack_b"
        ax.set_title(f"{split}  ({subtitle})", fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, fontsize=9)
        ax.set_ylim(0, 1.10)
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.9)
        ax.set_ylabel("Metric value", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] Saved → {save_path}")


def plot_per_attack(
    per_attack: dict,
    save_path: Path | str,
    title: str = "",
) -> None:
    """Grouped bar chart: classifier vs. autoencoder across every attack type.

    Two panels (AUC-ROC and Recall) — the two most telling metrics.  AUC-ROC
    captures threshold-free discrimination; Recall captures how many real
    anomalies are actually caught at the operating threshold.

    Parameters
    ----------
    per_attack : dict
        ``{attack_name: {"classifier": {...}, "autoencoder": {...}}}``.
    save_path : Path | str
        Output PNG path.
    title : str
        Figure suptitle.
    """
    attack_names = list(per_attack.keys())
    x = np.arange(len(attack_names))
    width = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(max(12, 2 * len(attack_names)), 6))
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")

    for ax, metric, mlabel in zip(axes,
                                  ["AUC_ROC", "Recall"],
                                  ["AUC-ROC", "Recall"]):
        clf_vals = [per_attack[a]["classifier"][metric]
                    if per_attack[a]["classifier"][metric] is not None else 0.0
                    for a in attack_names]
        ae_vals = [per_attack[a]["autoencoder"][metric]
                   if per_attack[a]["autoencoder"][metric] is not None else 0.0
                   for a in attack_names]

        ax.bar(x - width / 2, clf_vals, width,
               label="Classifier (Step 1)", color="steelblue")
        ax.bar(x + width / 2, ae_vals, width,
               label="Autoencoder (Step 3)", color="crimson")

        ax.set_title(mlabel, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(attack_names, fontsize=8, rotation=30, ha="right")
        ax.set_ylim(0, 1.10)
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.9)
        ax.set_ylabel(mlabel, fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] Saved → {save_path}")


# ===========================================================================
# CONSOLE SUMMARY
# ===========================================================================


def print_comparison_table(clf_results: dict, ae_results: dict, name: str) -> None:
    """Print a side-by-side classifier-vs-autoencoder metric table.

    Parameters
    ----------
    clf_results, ae_results : dict
        ``{"Test_A": {...}, "Test_B": {...}}`` for each detector.
    name : str
        Dataset label.
    """
    metrics = ["Accuracy", "Precision", "Recall", "F1_Score", "AUC_ROC"]
    print(f"\n{'─' * 78}")
    print(f"  {name} — Classifier (Step 1)  vs.  Autoencoder (Step 3)")
    print(f"{'─' * 78}")
    for split in ["Test_A", "Test_B"]:
        tag = "KNOWN attack_a" if split == "Test_A" else "UNKNOWN attack_b"
        print(f"\n  {split}  ({tag})")
        print(f"  {'metric':<12} {'classifier':>12} {'autoencoder':>12} {'winner':>12}")
        print(f"  {'─' * 12} {'─' * 12} {'─' * 12} {'─' * 12}")
        for m in metrics:
            cv = clf_results[split][m]
            av = ae_results[split][m]
            cv = float("nan") if cv is None else cv
            av = float("nan") if av is None else av
            if cv != cv or av != av:
                winner = "—"
            elif abs(cv - av) < 1e-4:
                winner = "tie"
            else:
                winner = "classifier" if cv > av else "autoencoder"
            print(f"  {m:<12} {cv:>12.4f} {av:>12.4f} {winner:>12}")
    print(f"{'─' * 78}")


# ===========================================================================
# MAIN PIPELINE
# ===========================================================================


def main() -> None:
    """Run the full Step 3 autoencoder experiment (MNIST + Pythia).

    Phase 1 — MNIST  (28x28 px)
        Generate clean, attack_a (Gaussian noise) and attack_b (OOD
        Fashion-MNIST).  Train the baseline classifier and the autoencoder,
        compare them on Test_A / Test_B, then run the extended across-attacks
        analysis using all six contamination types.

    Phase 2 — Pythia  (70x70 px, hidden dataset)
        Same protocol on the hidden dataset, using attack_a as the known
        attack and attack_b as the unknown one, with the extended analysis
        spanning all eight labelled partitions (attack_a … attack_h).
    """
    # Reproducible splits and results.
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    PLOTS_DIR = Path("plots") / "step3"

    output = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "experiment": "Step 3 — Unsupervised Anomaly Detection via Autoencoder",
        "config": {
            "batch_size": BATCH_SIZE,
            "classifier_epochs": CLF_NUM_EPOCHS,
            "autoencoder_epochs": AE_NUM_EPOCHS,
            "latent_dim": LATENT_DIM,
            "threshold_percentile": THRESHOLD_PERCENTILE,
            "random_seed": RANDOM_SEED,
        },
    }

    # =======================================================================
    # PHASE 1 — MNIST
    # =======================================================================
    print("\n" + "=" * 65)
    print("  STEP 3 — MNIST AUTOENCODER EXPERIMENT")
    print("=" * 65)

    clean_train_mnist, clean_test_mnist, raw_train_mnist, raw_test_mnist = (
        prepare_clean_data("mnist")
    )
    # Fashion-MNIST is needed only as the OOD attack source.
    _, _, raw_train_fmnist, raw_test_fmnist = prepare_clean_data("fashion_mnist")

    # --- attack_a (Gaussian noise) and attack_b (OOD) — same as Step 1 ---
    print("\nGenerating MNIST attack_a (Gaussian) and attack_b (OOD)...")
    attack_a_train_mnist = make_gaussian_attack(raw_train_mnist, std=0.4)
    attack_a_test_mnist = make_gaussian_attack(raw_test_mnist, std=0.4)
    attack_b_test_mnist = make_ood_attack(raw_test_fmnist)

    visualize_samples(clean_train_mnist, save_path=PLOTS_DIR / "mnist_clean.png",
                      title_prefix="MNIST Clean, ")
    visualize_samples(attack_a_train_mnist,
                      save_path=PLOTS_DIR / "mnist_attack_a.png",
                      title_prefix="MNIST attack_a (Gaussian), ")
    visualize_samples(attack_b_test_mnist,
                      save_path=PLOTS_DIR / "mnist_attack_b.png",
                      title_prefix="MNIST attack_b (OOD), ")

    # --- Baseline classifier (Step 1 reproduction) ---
    mnist_clf, mnist_clf_results = run_baseline_classifier(
        clean_train_mnist, clean_test_mnist,
        attack_a_train_mnist, attack_a_test_mnist, attack_b_test_mnist,
        input_size=28, name="MNIST",
    )

    # --- Autoencoder (Step 3) ---
    mnist_ae, mnist_threshold, mnist_ae_results = run_autoencoder(
        clean_train_mnist, clean_test_mnist,
        attack_a_test_mnist, attack_b_test_mnist,
        input_size=28, name="MNIST",
    )

    print_comparison_table(mnist_clf_results, mnist_ae_results, "MNIST")

    # --- Extended across-attacks analysis (MNIST: 6 contamination types) ---
    mnist_attack_battery = {
        "gaussian":    make_gaussian_attack(raw_test_mnist, std=0.4),
        "salt_pepper": make_salt_pepper_attack(raw_test_mnist, prob=0.15),
        "geometric":   make_geometric_attack(raw_test_mnist, max_displacement=5.0),
        "blended":     make_blended_attack(raw_test_mnist, alpha=0.30),
        "backdoor":    make_backdoor_attack(raw_test_mnist, trigger_size=5,
                                            position="bottom_right"),
        "ood":         make_ood_attack(raw_test_fmnist),
    }
    mnist_per_attack = evaluate_across_attacks(
        mnist_clf, mnist_ae, mnist_threshold,
        clean_test_mnist, mnist_attack_battery, name="MNIST",
    )

    # --- MNIST plots ---
    print("\nGenerating MNIST plots...")
    plot_reconstructions(
        mnist_ae,
        {"clean": clean_test_mnist,
         "attack_a (Gaussian)": attack_a_test_mnist,
         "attack_b (OOD)": attack_b_test_mnist},
        PLOTS_DIR / "mnist_reconstructions.png",
        title="MNIST — Autoencoder reconstructions (trained on clean only)",
    )

    clean_scores_mnist, _ = reconstruction_scores(
        mnist_ae, make_dataloader(clean_test_mnist, BATCH_SIZE))
    atk_a_scores_mnist, _ = reconstruction_scores(
        mnist_ae, make_dataloader(attack_a_test_mnist, BATCH_SIZE))
    atk_b_scores_mnist, _ = reconstruction_scores(
        mnist_ae, make_dataloader(attack_b_test_mnist, BATCH_SIZE))
    plot_score_distributions(
        {"clean": clean_scores_mnist,
         "attack_a (Gaussian)": atk_a_scores_mnist,
         "attack_b (OOD)": atk_b_scores_mnist},
        mnist_threshold,
        PLOTS_DIR / "mnist_score_distributions.png",
        title="MNIST — Reconstruction error: clean vs. attacks",
    )

    # ROC data: classifier probabilities and autoencoder scores per test set.
    mnist_test_a = ConcatDataset([clean_test_mnist, attack_a_test_mnist])
    mnist_test_b = ConcatDataset([clean_test_mnist, attack_b_test_mnist])
    clf_a_p, clf_a_y = predict_probs(mnist_clf, make_dataloader(mnist_test_a, BATCH_SIZE))
    clf_b_p, clf_b_y = predict_probs(mnist_clf, make_dataloader(mnist_test_b, BATCH_SIZE))
    ae_a_s, ae_a_y = reconstruction_scores(mnist_ae, make_dataloader(mnist_test_a, BATCH_SIZE))
    ae_b_s, ae_b_y = reconstruction_scores(mnist_ae, make_dataloader(mnist_test_b, BATCH_SIZE))
    plot_roc_comparison(
        {"Classifier — Test_A":  (clf_a_y, clf_a_p),
         "Classifier — Test_B":  (clf_b_y, clf_b_p),
         "Autoencoder — Test_A": (ae_a_y, ae_a_s),
         "Autoencoder — Test_B": (ae_b_y, ae_b_s)},
        PLOTS_DIR / "mnist_roc_comparison.png",
        title="MNIST — ROC: classifier vs. autoencoder",
    )
    plot_metric_comparison(
        mnist_clf_results, mnist_ae_results,
        PLOTS_DIR / "mnist_metric_comparison.png",
        title="MNIST — Classifier vs. Autoencoder",
    )
    plot_per_attack(
        mnist_per_attack,
        PLOTS_DIR / "mnist_per_attack.png",
        title="MNIST — Both detectors across all attack types",
    )

    output["MNIST"] = {
        "classifier_baseline": mnist_clf_results,
        "autoencoder": mnist_ae_results,
        "across_attacks": mnist_per_attack,
    }

    # =======================================================================
    # PHASE 2 — PYTHIA
    # =======================================================================
    print("\n" + "=" * 65)
    print("  STEP 3 — PYTHIA AUTOENCODER EXPERIMENT")
    print("=" * 65)

    PYTHIA_DIR = Path("pythia")
    check_pythia_available(PYTHIA_DIR)

    # Clean partition: split into a training pool and a test split.
    pythia_clean = load_pythia_data(PYTHIA_DIR, "clean")
    pythia_clean_train, pythia_clean_test = split_train_test(pythia_clean)

    # Load all eight labelled attack partitions; split each 80/20.  The AE
    # never trains on attacks, so only the test split of each partition is
    # actually used here — kept balanced (~1:1) against the clean test split.
    pythia_partitions = [f"attack_{c}" for c in "abcdefgh"]
    pythia_attack_train: dict = {}
    pythia_attack_test: dict = {}
    for part in pythia_partitions:
        print(f"  Loading Pythia partition '{part}'...")
        ds = load_pythia_data(PYTHIA_DIR, part)
        tr, te = split_train_test(ds)
        pythia_attack_train[part] = tr
        pythia_attack_test[part] = te

    visualize_samples(pythia_clean, save_path=PLOTS_DIR / "pythia_clean.png",
                      title_prefix="Pythia Clean, ")
    for part in pythia_partitions:
        visualize_samples(
            load_pythia_data(PYTHIA_DIR, part),
            save_path=PLOTS_DIR / f"pythia_{part}.png",
            title_prefix=f"Pythia {part}, ",
        )

    # --- Baseline classifier (attack_a known) ---
    pythia_clf, pythia_clf_results = run_baseline_classifier(
        pythia_clean_train, pythia_clean_test,
        pythia_attack_train["attack_a"], pythia_attack_test["attack_a"],
        pythia_attack_test["attack_b"],
        input_size=70, name="Pythia",
    )

    # --- Autoencoder (clean only) ---
    pythia_ae, pythia_threshold, pythia_ae_results = run_autoencoder(
        pythia_clean_train, pythia_clean_test,
        pythia_attack_test["attack_a"], pythia_attack_test["attack_b"],
        input_size=70, name="Pythia",
    )

    print_comparison_table(pythia_clf_results, pythia_ae_results, "Pythia")

    # --- Extended across-attacks analysis (Pythia: attack_a … attack_h) ---
    pythia_per_attack = evaluate_across_attacks(
        pythia_clf, pythia_ae, pythia_threshold,
        pythia_clean_test, pythia_attack_test, name="Pythia",
    )

    # --- Pythia plots ---
    print("\nGenerating Pythia plots...")
    plot_reconstructions(
        pythia_ae,
        {"clean": pythia_clean_test,
         "attack_a": pythia_attack_test["attack_a"],
         "attack_b": pythia_attack_test["attack_b"]},
        PLOTS_DIR / "pythia_reconstructions.png",
        title="Pythia — Autoencoder reconstructions (trained on clean only)",
    )

    clean_scores_pythia, _ = reconstruction_scores(
        pythia_ae, make_dataloader(pythia_clean_test, BATCH_SIZE))
    atk_a_scores_pythia, _ = reconstruction_scores(
        pythia_ae, make_dataloader(pythia_attack_test["attack_a"], BATCH_SIZE))
    atk_b_scores_pythia, _ = reconstruction_scores(
        pythia_ae, make_dataloader(pythia_attack_test["attack_b"], BATCH_SIZE))
    plot_score_distributions(
        {"clean": clean_scores_pythia,
         "attack_a": atk_a_scores_pythia,
         "attack_b": atk_b_scores_pythia},
        pythia_threshold,
        PLOTS_DIR / "pythia_score_distributions.png",
        title="Pythia — Reconstruction error: clean vs. attacks",
    )

    pythia_test_a = ConcatDataset([pythia_clean_test, pythia_attack_test["attack_a"]])
    pythia_test_b = ConcatDataset([pythia_clean_test, pythia_attack_test["attack_b"]])
    pclf_a_p, pclf_a_y = predict_probs(pythia_clf, make_dataloader(pythia_test_a, BATCH_SIZE))
    pclf_b_p, pclf_b_y = predict_probs(pythia_clf, make_dataloader(pythia_test_b, BATCH_SIZE))
    pae_a_s, pae_a_y = reconstruction_scores(pythia_ae, make_dataloader(pythia_test_a, BATCH_SIZE))
    pae_b_s, pae_b_y = reconstruction_scores(pythia_ae, make_dataloader(pythia_test_b, BATCH_SIZE))
    plot_roc_comparison(
        {"Classifier — Test_A":  (pclf_a_y, pclf_a_p),
         "Classifier — Test_B":  (pclf_b_y, pclf_b_p),
         "Autoencoder — Test_A": (pae_a_y, pae_a_s),
         "Autoencoder — Test_B": (pae_b_y, pae_b_s)},
        PLOTS_DIR / "pythia_roc_comparison.png",
        title="Pythia — ROC: classifier vs. autoencoder",
    )
    plot_metric_comparison(
        pythia_clf_results, pythia_ae_results,
        PLOTS_DIR / "pythia_metric_comparison.png",
        title="Pythia — Classifier vs. Autoencoder",
    )
    plot_per_attack(
        pythia_per_attack,
        PLOTS_DIR / "pythia_per_attack.png",
        title="Pythia — Both detectors across all attack partitions",
    )

    output["Pythia"] = {
        "classifier_baseline": pythia_clf_results,
        "autoencoder": pythia_ae_results,
        "across_attacks": pythia_per_attack,
    }

    # =======================================================================
    # SAVE RESULTS
    # =======================================================================
    save_results(output, "faza3_wyniki_autoenkodera.json")

    print(
        "\n[Interpretacja — Krok 3]\n"
        "Hipoteza: autoenkoder (uczony wyłącznie na danych clean) powinien\n"
        "wykrywać NIEZNANY atak (Test_B) lepiej niż klasyfikator z Kroku 1,\n"
        "kosztem słabszego wyniku na ZNANYM ataku (Test_A).\n"
        "\n"
        "  • Test_A: jeśli klasyfikator > autoenkoder — to oczekiwany koszt\n"
        "    podejścia nienadzorowanego (klasyfikator zna dokładnie attack_a).\n"
        "  • Test_B: jeśli autoenkoder > klasyfikator — hipoteza potwierdzona;\n"
        "    model normalności generalizuje na atak, którego nikt nie pokazał.\n"
        "  • AUC-ROC to najuczciwsza metryka — nie zależy od progu.\n"
        "  • Wykres per_attack pokazuje, które rodzaje anomalii każde podejście\n"
        "    łapie: np. autoenkoder zwykle dobrze radzi sobie z OOD i szumem,\n"
        "    a słabiej z lokalnym backdoorem (mały trigger 5x5 wnosi znikomy\n"
        "    wkład do błędu rekonstrukcji całego obrazu).\n"
        "\n"
        "Pełne wyniki liczbowe: faza3_wyniki_autoenkodera.json\n"
        "Wykresy: plots/step3/\n"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()