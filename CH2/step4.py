"""
step4.py — Anomaly Detection: Hybrid System (Step 4)
==============================================================================

Implementuje architekturę hybrydową (Krok 4) polegającą na:
    1. Wytrenowaniu nienadzorowanego Autoenkodera na danych 'clean'.
    2. Zamrożeniu wag jego enkodera (Freezing).
    3. Dodaniu do zamrożonego enkodera nowej "głowy" klasyfikacyjnej
       (2 warstwy gęste + Softmax).
    4. Nadzorowanym dotrenowaniu (Fine-Tuning) wyłącznie parametrów nowej głowy
       na zbiorze treningowym składającym się z danych 'clean' oraz 'attack_a'.

Skrypt następnie analizuje zjawisko generalizacji, sprawdzając jak ta
struktura latentna (wyuczona tylko na danych poprawnych) połączona 
z klasyfikatorem (wyuczonym na jednym ataku) radzi sobie w zderzeniu
z zupełnie nieznanymi atakami (attack_b, OOD) w porównaniu do:
    - Klasyfikatora Baseline (z Kroku 1)
    - Czystego Autoenkodera (z Kroku 3)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Upewnienie się, że ścieżki do współdzielonych modułów są dostępne
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

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

# ---------------------------------------------------------------------------
# Hiperparametry
# ---------------------------------------------------------------------------
BATCH_SIZE = 64
CLF_NUM_EPOCHS = 15     # Epoki dla Baseline'u i Hybrydy
CLF_PATIENCE = 3        # Cierpliwość dla Baseline'u i Hybrydy
AE_NUM_EPOCHS = 30      # Epoki dla Autoenkodera
AE_PATIENCE = 5         # Cierpliwość dla Autoenkodera

LATENT_DIM = 32         # Wymiar warstwy latentnej
AE_VAL_RATIO = 0.1      # Proporcja walidacyjna dla Autoenkodera
THRESHOLD_PERCENTILE = 95.0 
RANDOM_SEED = 42

# ===========================================================================
# DEFINICJA ARCHITEKTURY HYBRYDOWEJ (Krok 3 z planu)
# ===========================================================================

class HybridClassifier(nn.Module):
    """
    Hybrydowy system detekcji anomalii.
    Złożony z zamrożonego ekstraktora cech (wykorzystuje metodę encode autoenkodera) 
    i trenowalnej głowy.
    """
    def __init__(self, pretrained_autoencoder: ConvAutoencoder, latent_dim: int = 32, num_classes: int = 2):
        super().__init__()
        
        # 1. Zapisujemy referencję do całego wytrenowanego autoenkodera
        self.pretrained_autoencoder = pretrained_autoencoder
        
        # 2. Zamrożenie wag całego autoenkodera (odłączenie propagacji wstecznej)
        # Dekoder też zostanie zamrożony, ale ponieważ nie używamy go w metodzie forward, 
        # nie wpływa to na proces uczenia naszej głowy klasyfikacyjnej.
        for param in self.pretrained_autoencoder.parameters():
            param.requires_grad = False
            
        # 3. Zbudowanie nowej głowy klasyfikacyjnej z dwiema warstwami gęstymi i aktywacją Softmax
        self.head = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Enkoder rzutuje obraz do wektora przestrzeni latentnej za pomocą metody encode()
        z = self.pretrained_autoencoder.encode(x)
        
        # Głowa klasyfikuje wektor latentny i zwraca prawdopodobieństwa dla 2 klas
        probs = self.head(z)
        
        # Aby zachować kompatybilność z evaluate_model() i loss = BCELoss(),
        # moduł musi zwracać prawdopodobieństwo z dopasowanym kształtem.
        # Używamy .unsqueeze(1), by zmienić kształt z [BATCH_SIZE] na [BATCH_SIZE, 1],
        # ponieważ takiego wymiaru etykiet (target) wymaga funkcja BCELoss.
        return probs[:, 1].unsqueeze(1)

# ===========================================================================
# HELPERY I FUNKCJE POMOCNICZE
# ===========================================================================

def predict_probs(model: nn.Module, loader) -> tuple[np.ndarray, np.ndarray]:
    """Zwraca ciągłe prawdopodobieństwa ataku do wyrysowania krzywych ROC."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    probs: list = []
    labels: list = []
    with torch.no_grad():
        for inputs, lbls in loader:
            inputs = inputs.to(device)
            out = model(inputs).squeeze()
            if out.dim() == 0:
                out = out.unsqueeze(0)
            probs.extend(out.cpu().numpy())
            labels.extend(lbls.numpy())

    return np.array(probs), np.array(labels)

# ===========================================================================
# EXPERYMENTY - TRENING MODULÓW
# ===========================================================================

def build_datasets_and_loaders(clean_train, attack_a_train):
    """Buduje zbiorczy DataLoader dla Baseline i Hybrydy (clean + attack_a)."""
    train_val = ConcatDataset([clean_train, attack_a_train])
    t_size = int(0.8 * len(train_val))
    v_size = len(train_val) - t_size
    train_ds, val_ds = random_split(train_val, [t_size, v_size])
    
    return make_dataloader(train_ds, BATCH_SIZE, shuffle=True), make_dataloader(val_ds, BATCH_SIZE)

def run_experiment_triad(clean_train, clean_test, attack_a_train, attack_a_test, attack_b_test, input_size: int, name: str):
    """
    Centralna funkcja wykonująca trening wszystkich 3 podejść dla bezpośredniego porównania.
    """
    bar = "=" * 65
    print(f"\n{bar}")
    print(f"  [{name}] TRENOWANIE AUTOENKODERA (Nienadzorowany - Baza dla hybrydy)")
    print(f"{bar}")
    
    # 1. AUTOENKODER
    ae_train, ae_val = split_train_test(clean_train, train_ratio=1.0 - AE_VAL_RATIO)
    ae_model = ConvAutoencoder(input_size=input_size, latent_dim=LATENT_DIM)
    ae_model = train_autoencoder(
        ae_model,
        make_dataloader(ae_train, BATCH_SIZE, shuffle=True),
        make_dataloader(ae_val, BATCH_SIZE),
        nn.MSELoss(), optim.Adam(ae_model.parameters(), lr=0.001),
        num_epochs=AE_NUM_EPOCHS, patience=AE_PATIENCE,
    )
    clean_val_scores, _ = reconstruction_scores(ae_model, make_dataloader(ae_val, BATCH_SIZE))
    ae_threshold = select_threshold(clean_val_scores, THRESHOLD_PERCENTILE)

    # Przygotowanie danych nadzorowanych (Dla Baseline'u i Hybrydy)
    sup_train_loader, sup_val_loader = build_datasets_and_loaders(clean_train, attack_a_train)
    
    print(f"\n{bar}")
    print(f"  [{name}] TRENOWANIE KLASYFIKATORA BAZOWEGO (Baseline - Krok 1)")
    print(f"{bar}")
    
    # 2. BASELINE CLASSIFIER
    baseline_model = AnomalyCNN(input_size=input_size)
    baseline_model = train_model(
        baseline_model, sup_train_loader, sup_val_loader,
        nn.BCELoss(), optim.Adam(baseline_model.parameters(), lr=0.001),
        num_epochs=CLF_NUM_EPOCHS, patience=CLF_PATIENCE,
    )
    
    print(f"\n{bar}")
    print(f"  [{name}] TRENOWANIE SYSTEMU HYBRYDOWEGO (Krok 4)")
    print(f"{bar}")
    
    # 3. HYBRID CLASSIFIER (z użyciem parametru zamrożonego AE z punktu 1)
    hybrid_model = HybridClassifier(ae_model, latent_dim=LATENT_DIM)
    
    # Ograniczenie optymalizatora WYŁĄCZNIE do parametrów nowej głowy klasyfikującej
    optimizer_hybrid = optim.Adam(hybrid_model.head.parameters(), lr=0.001)
    
    hybrid_model = train_model(
        hybrid_model, sup_train_loader, sup_val_loader,
        nn.BCELoss(), optimizer_hybrid,  # BCELoss pasuje idealnie dzięki Softmax i zwrotowi prob[:,1]
        num_epochs=CLF_NUM_EPOCHS, patience=CLF_PATIENCE,
    )
    
    return ae_model, ae_threshold, baseline_model, hybrid_model

# ===========================================================================
# PLOTTING
# ===========================================================================

def plot_roc_comparison_3way(roc_data: dict, save_path: Path | str, title: str = "") -> None:
    """Rysuje krzywe ROC dla trzech systemów naraz na wybranym zbiorze."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    styles = {
        "Baseline":    ("steelblue",   "-"),
        "Autoencoder": ("crimson",     "-"),
        "Hybrid":      ("forestgreen", "-"),
    }

    for label, (y_true, y_score) in roc_data.items():
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") else float(np.trapz(tpr, fpr))
        
        system_type = label.split(" — ")[0]
        color, ls = styles.get(system_type, ("gray", "-"))
        is_test_b = "Test_B" in label
        
        # Test_B rysujemy przerywaną linią, Test_A ciągłą
        final_ls = "--" if is_test_b else ls
        
        ax.plot(fpr, tpr, color=color, linestyle=final_ls, linewidth=2.5 if is_test_b else 1.5,
                alpha=1.0 if is_test_b else 0.5, label=f"{label} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=1, label="chance (AUC=0.500)")
    ax.set_xlabel("False positive rate", fontsize=10)
    ax.set_ylabel("True positive rate", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] Saved → {save_path}")

def plot_metric_comparison_3way(base_res: dict, ae_res: dict, hybrid_res: dict, save_path: Path | str, title: str = "") -> None:
    """Grupowany bar chart dla trzech systemów: Baseline vs AE vs Hybrid."""
    metrics = ["Accuracy", "Precision", "Recall", "F1_Score", "AUC_ROC"]
    labels = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")

    for ax, split in zip(axes, ["Test_A", "Test_B"]):
        x = np.arange(len(metrics))
        width = 0.25

        def _v(res):
            return [res[split][m] if res[split][m] is not None else 0.0 for m in metrics]

        ax.bar(x - width, _v(base_res), width, label="Baseline (Step 1)", color="steelblue")
        ax.bar(x,         _v(ae_res),   width, label="Autoencoder (Step 3)", color="crimson")
        ax.bar(x + width, _v(hybrid_res), width, label="Hybrid (Step 4)", color="forestgreen")

        subtitle = "KNOWN attack_a" if split == "Test_A" else "UNKNOWN attack_b"
        ax.set_title(f"{split} ({subtitle})", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylim(0, 1.10)
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.9)
        ax.legend(fontsize=10)
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] Saved → {save_path}")



def evaluate_models_and_get_results(base, ae, ae_thresh, hybrid, test_a_ds, test_b_ds):
    """Dokonuje ewaluacji na zbiorach A i B dla podanych 3 modeli i opakowuje w dict."""
    loader_a = make_dataloader(test_a_ds, BATCH_SIZE)
    loader_b = make_dataloader(test_b_ds, BATCH_SIZE)
    
    res_base = {
        "Test_A": parse_results(evaluate_model(base, loader_a)),
        "Test_B": parse_results(evaluate_model(base, loader_b))
    }
    res_ae = {
        "Test_A": parse_results(evaluate_autoencoder(ae, loader_a, ae_thresh)),
        "Test_B": parse_results(evaluate_autoencoder(ae, loader_b, ae_thresh))
    }
    res_hybrid = {
        "Test_A": parse_results(evaluate_model(hybrid, loader_a)),
        "Test_B": parse_results(evaluate_model(hybrid, loader_b))
    }
    return res_base, res_ae, res_hybrid

def prepare_roc_data_3way(base, ae, hybrid, test_ds, split_name):
    """Zwraca dane do wykresu ROC dla konkretnego splitu."""
    loader = make_dataloader(test_ds, BATCH_SIZE)
    base_p, base_y = predict_probs(base, loader)
    hybr_p, hybr_y = predict_probs(hybrid, loader)
    ae_s, ae_y = reconstruction_scores(ae, loader)
    
    return {
        f"Baseline — {split_name}": (base_y, base_p),
        f"Autoencoder — {split_name}": (ae_y, ae_s),
        f"Hybrid — {split_name}": (hybr_y, hybr_p),
    }

def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    PLOTS_DIR = Path("plots") / "step4"
    
    output_json = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "experiment": "Step 4 — Hybrid System (Frozen AE + Trainable Head)"
    }
    
    print("\n" + "=" * 65)
    print("  KROK 4 — SYSTEM HYBRYDOWY (FAZA 1: MNIST)")
    print("=" * 65)
    
    # KROK 1: Przygotowanie Danych (MNIST)
    cl_tr_m, cl_te_m, rw_tr_m, rw_te_m = prepare_clean_data("mnist")
    _, _, rw_tr_fm, rw_te_fm = prepare_clean_data("fashion_mnist")
    
    atk_a_tr_m = make_gaussian_attack(rw_tr_m, std=0.4)
    atk_a_te_m = make_gaussian_attack(rw_te_m, std=0.4)
    atk_b_te_m = make_ood_attack(rw_te_fm)
    
    # TRENOWANIE ZESTAWU 3 MODELI
    m_ae, m_thresh, m_base, m_hybrid = run_experiment_triad(
        cl_tr_m, cl_te_m, atk_a_tr_m, atk_a_te_m, atk_b_te_m, input_size=28, name="MNIST"
    )
    
    test_a_m = ConcatDataset([cl_te_m, atk_a_te_m])
    test_b_m = ConcatDataset([cl_te_m, atk_b_te_m])
    
    # EWALUACJA
    res_base_m, res_ae_m, res_hybr_m = evaluate_models_and_get_results(
        m_base, m_ae, m_thresh, m_hybrid, test_a_m, test_b_m
    )
    output_json["MNIST"] = {"Baseline": res_base_m, "Autoencoder": res_ae_m, "Hybrid": res_hybr_m}
    
    # PLOTY MNIST
    plot_metric_comparison_3way(res_base_m, res_ae_m, res_hybr_m, PLOTS_DIR / "mnist_metrics_3way.png", "MNIST — Porównanie systemów")
    
    roc_m_data = {**prepare_roc_data_3way(m_base, m_ae, m_hybrid, test_a_m, "Test_A"), 
                  **prepare_roc_data_3way(m_base, m_ae, m_hybrid, test_b_m, "Test_B")}
    plot_roc_comparison_3way(roc_m_data, PLOTS_DIR / "mnist_roc_3way.png", "MNIST — Krzywe ROC (Test A i Test B)")
    
    # -------------------------------------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  KROK 4 — SYSTEM HYBRYDOWY (FAZA 2: ZBIÓR UKRYTY - PYTHIA)")
    print("=" * 65)
    
    PYTHIA_DIR = Path("pythia")
    check_pythia_available(PYTHIA_DIR)
    
    pythia_clean = load_pythia_data(PYTHIA_DIR, "clean")
    cl_tr_p, cl_te_p = split_train_test(pythia_clean)
    
    pythia_atk_a = load_pythia_data(PYTHIA_DIR, "attack_a")
    atk_a_tr_p, atk_a_te_p = split_train_test(pythia_atk_a)
    
    # Zgodnie ze step1.py, cały zbiór B idzie do testów (1000 elementów)
    atk_b_te_p = load_pythia_data(PYTHIA_DIR, "attack_b")
    
    # TRENOWANIE ZESTAWU 3 MODELI
    p_ae, p_thresh, p_base, p_hybrid = run_experiment_triad(
        cl_tr_p, cl_te_p, atk_a_tr_p, atk_a_te_p, atk_b_te_p, input_size=70, name="Pythia"
    )
    
    test_a_p = ConcatDataset([cl_te_p, atk_a_te_p])
    test_b_p = ConcatDataset([cl_te_p, atk_b_te_p])
    
    res_base_p, res_ae_p, res_hybr_p = evaluate_models_and_get_results(
        p_base, p_ae, p_thresh, p_hybrid, test_a_p, test_b_p
    )
    output_json["Pythia"] = {"Baseline": res_base_p, "Autoencoder": res_ae_p, "Hybrid": res_hybr_p}
    
    plot_metric_comparison_3way(res_base_p, res_ae_p, res_hybr_p, PLOTS_DIR / "pythia_metrics_3way.png", "Pythia — Porównanie systemów")
    
    roc_p_data = {**prepare_roc_data_3way(p_base, p_ae, p_hybrid, test_a_p, "Test_A"), 
                  **prepare_roc_data_3way(p_base, p_ae, p_hybrid, test_b_p, "Test_B")}
    plot_roc_comparison_3way(roc_p_data, PLOTS_DIR / "pythia_roc_3way.png", "Pythia — Krzywe ROC (Test A i Test B)")
    
    save_results(output_json, "faza4_wyniki_hybrydy.json")
    

if __name__ == "__main__":
    main()