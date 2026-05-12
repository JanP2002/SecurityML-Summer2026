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
5.  Mirror steps 1-4 for the hidden Pythia dataset (70×70 px grayscale).
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
    visualize_samples,
    AnomalyCNN,
    train_model,
    evaluate_model,
    parse_results,
    save_results,
)
from attacks.contamination import make_gaussian_attack, make_ood_attack

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE = 64
NUM_EPOCHS = 15
PATIENCE = 3


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

    # -----------------------------------------------------------------------
    # Cell 7 — Model initialisation
    # -----------------------------------------------------------------------
    print("--- KOMÓRKA 7: ARCHITEKTURA MODELU (CNN) ---")

    model_mnist = AnomalyCNN(input_size=28)   # flatten_size = 3×3×64 = 576
    print(model_mnist)

    model_pythia = AnomalyCNN(input_size=70)  # flatten_size = 8×8×64 = 4096

    # BCELoss is shared — it has no learnable parameters
    criterion = nn.BCELoss()
    optimizer_mnist = optim.Adam(model_mnist.parameters(), lr=0.001)
    optimizer_pythia = optim.Adam(model_pythia.parameters(), lr=0.001)

    # -----------------------------------------------------------------------
    # Cell 8 — Training
    # -----------------------------------------------------------------------
    print("\n--- KOMÓRKA 8: TRENING (BASELINE) ---")

    # DataLoaders — shuffle=True for train to prevent ordering artefacts
    train_loader_mnist = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader_mnist = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    train_loader_pythia = DataLoader(pythia_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader_pythia = DataLoader(pythia_val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("\n>>> TRENING: MNIST (Clean + Attack_A) <<<")
    model_mnist = train_model(
        model_mnist, train_loader_mnist, val_loader_mnist,
        criterion, optimizer_mnist, num_epochs=NUM_EPOCHS, patience=PATIENCE,
    )

    print("\n>>> TRENING: PYTHIA (Clean + Attack_A) <<<")
    model_pythia = train_model(
        model_pythia, train_loader_pythia, val_loader_pythia,
        criterion, optimizer_pythia, num_epochs=NUM_EPOCHS, patience=PATIENCE,
    )

    # -----------------------------------------------------------------------
    # Cell 9 — Evaluation on Test_A (known attack)
    # -----------------------------------------------------------------------
    print("\n--- KOMÓRKA 9: EWALUACJA — TEST_A (ZNANY ATAK) ---")

    test_a_loader_mnist = DataLoader(test_a_dataset, batch_size=BATCH_SIZE)
    test_a_loader_pythia = DataLoader(pythia_test_a_dataset, batch_size=BATCH_SIZE)

    print("\n>>> EWALUACJA MNIST (Test_A) <<<")
    res_mnist_a = evaluate_model(model_mnist, test_a_loader_mnist)

    print("\n>>> EWALUACJA PYTHIA (Test_A) <<<")
    res_pythia_a = evaluate_model(model_pythia, test_a_loader_pythia)

    # -----------------------------------------------------------------------
    # Cell 10 — Evaluation on Test_B (unknown attack)
    # -----------------------------------------------------------------------
    print("\n--- KOMÓRKA 10: EWALUACJA — TEST_B (NIEZNANY ATAK) ---")

    test_b_loader_mnist = DataLoader(test_b_dataset, batch_size=BATCH_SIZE)
    test_b_loader_pythia = DataLoader(pythia_test_b_dataset, batch_size=BATCH_SIZE)

    print("\n>>> EWALUACJA MNIST (Test_B: OOD) <<<")
    res_mnist_b = evaluate_model(model_mnist, test_b_loader_mnist)

    print("\n>>> EWALUACJA PYTHIA (Test_B) <<<")
    res_pythia_b = evaluate_model(model_pythia, test_b_loader_pythia)

    print(
        "\n[Closed-world assumption failure]: The supervised classifier learned "
        "attack-specific features of attack_a and cannot generalise to the "
        "structurally different attack_b — recall and AUC-ROC on Test_B "
        "will be significantly lower than on Test_A."
    )

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
            "Test_A_Znany_Atak": parse_results(res_pythia_a),
            "Test_B_Nieznany_Atak": parse_results(res_pythia_b),
        },
    }

    save_results(results_summary, "faza1_wyniki_eksperymentu.json")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
