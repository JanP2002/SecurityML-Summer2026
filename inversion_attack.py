#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WIZUALNY atak inwersyjny na CIFAR (metryka PRYWATNOŚCI, nie użyteczności).
Regresor: embedding_grafu → kolor-per-superpiksel (obraz uśredniony po segmentach SLIC,
złożony po masce). Atakujący = silniejszy z {RidgeCV, MLP} (spójnie z recon_leak). Mierzymy
SSIM/MSE odtworzenia do oryginału (uśrednionego po superpikselach) i degradację przy feature-DP.

Oczekiwane: attr (treść/kolor węzłów) odtwarza ROZPOZNAWALNY obraz; reprezentacje strukturalne
(wl, fgsd) dają śmieci. feature-DP psuje odtworzenie z attr (struktura niewrażliwa).

Użycie:
  python inversion_attack.py --classes 0 1 8 --per-class 200 --out-dir results_inversion
"""
from __future__ import annotations
import argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from skimage.segmentation import slic
from skimage.metrics import structural_similarity as ssim_fn
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.model_selection import KFold, cross_val_predict

from graphrep.config import Config
from graphrep import data, features
from graphrep.embeddings import build_structural, _block_norm
from graphrep.features import attr_matrix


def superpixel_avg_images(images, cfg):
    """Cel ataku: każdy obraz uśredniony po segmentach SLIC (kolor-per-superpiksel złożony
    po masce). Piecewise-stały po superpikselach — to maksimum osiągalne na granularności grafu."""
    out, S = [], cfg.img_size
    for img in images:
        lab = slic(img, n_segments=cfg.n_segments, compactness=cfg.compactness,
                   start_label=0, channel_axis=-1)
        avg = np.zeros_like(img)
        for s in np.unique(lab):
            m = lab == s
            avg[m] = img[m].mean(axis=0)
        out.append(avg)
    return np.array(out)                       # (N, S, S, 3) w [0,1]


def attack(X, T, seed):
    """X→T w CV; zwraca predykcje silniejszego modelu (wyższy śr. SSIM) + (ssim, mse)."""
    Xs = StandardScaler().fit_transform(X)
    cv = KFold(n_splits=3, shuffle=True, random_state=seed)
    best = None
    for name, mdl in [("ridge", RidgeCV(alphas=(0.1, 1, 10, 100, 1000))),
                      ("mlp", MLPRegressor(hidden_layer_sizes=(128,), max_iter=200, random_state=seed))]:
        try:
            pred = cross_val_predict(mdl, Xs, T, cv=cv)
        except Exception as e:
            print("  ", name, "err", e); continue
        pred = np.clip(pred, 0, 1)
        N = T.shape[0]; S = int(round((T.shape[1] / 3) ** 0.5))
        Ti = T.reshape(N, S, S, 3); Pi = pred.reshape(N, S, S, 3)
        ss = np.mean([ssim_fn(Ti[i], Pi[i], channel_axis=-1, data_range=1.0) for i in range(N)])
        mse = float(np.mean((T - pred) ** 2))
        if best is None or ss > best[0]:
            best = (ss, mse, pred, name)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", type=int, nargs="+", default=[0, 1, 8])
    ap.add_argument("--per-class", type=int, default=200)
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 1.0, 2.0, 4.0])
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="results_inversion")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = Config(source="cifar", graph_type="slic", classes=args.classes,
                 per_class=args.per_class, data_dir=args.data_dir, seed=args.seed)
    images, _ = data.load_cifar_images(cfg)
    graphs, y, names = data.get_graphs(cfg)        # ta sama kolejność (ten sam seed/subsample)
    graphs = features.ensure_labels(graphs, cfg)
    T = superpixel_avg_images(images, cfg).reshape(len(images), -1)
    N, S = len(images), cfg.img_size
    print(f"[cifar {args.classes}] N={N} target_dim={T.shape[1]}")

    methods = ["attr", "wl", "fgsd"]
    sig = args.sigmas
    recon = {}                                      # (method, sigma) -> (ssim, mse, pred, model)

    # struktura (wl/fgsd) jest NIEWRAŻLIWA na feature-DP → liczymy raz i powielamy po σ
    struct_cache = {}
    for m in ["wl", "fgsd"]:
        Xs = build_structural(graphs, [m], cfg)
        res = attack(Xs, T, args.seed)
        struct_cache[m] = res
        print(f"  {m:5} (σ-niezależny) SSIM={res[0]:.3f} MSE={res[1]:.4f} [{res[3]}]")

    for s in sig:
        gobf = data.obfuscate(graphs, "feature", s, cfg.seed) if s > 0 else graphs
        A = normalize(_block_norm(attr_matrix(gobf), "standard_l2"))
        res = attack(A, T, args.seed)
        recon[("attr", s)] = res
        print(f"  attr  σ={s:.0f} SSIM={res[0]:.3f} MSE={res[1]:.4f} [{res[3]}]")
        for m in ["wl", "fgsd"]:
            recon[(m, s)] = struct_cache[m]

    # --- raport SSIM/MSE ---
    print("\n  SSIM (↑ = lepsze odtworzenie = gorsza prywatność) / MSE:")
    print(f"  {'metoda':6} " + " ".join(f"σ={s:.0f}".rjust(14) for s in sig))
    for m in methods:
        cells = [f"{recon[(m,s)][0]:.3f}/{recon[(m,s)][1]:.3f}".rjust(14) for s in sig]
        print(f"  {m:6} " + " ".join(cells))

    # --- siatka obrazów: wiersz=metoda, kol=[oryginał, σ...] dla jednego przykładu ---
    # wybierz przykład z 1. klasy, w miarę rozpoznawalny
    sample = int(np.where(y == 0)[0][3])
    Tg = T.reshape(N, S, S, 3)
    ncols = 1 + len(sig)
    fig, axes = plt.subplots(len(methods), ncols, figsize=(2.0 * ncols, 2.0 * len(methods)))
    for r, m in enumerate(methods):
        axes[r, 0].imshow(np.clip(Tg[sample], 0, 1)); axes[r, 0].set_ylabel(m, fontsize=11)
        axes[r, 0].set_title("oryginał (SLIC)" if r == 0 else "", fontsize=9)
        for c, s in enumerate(sig):
            pred = recon[(m, s)][2].reshape(N, S, S, 3)[sample]
            ax = axes[r, c + 1]; ax.imshow(np.clip(pred, 0, 1))
            if r == 0:
                ax.set_title(f"σ={s:.0f}", fontsize=9)
        for c in range(ncols):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
    fig.suptitle("Wizualny atak inwersyjny (CIFAR 0/1/8) — odtworzenie obrazu z embeddingu grafu\n"
                 "wiersze: metoda · kolumny: σ feature-DP · attr odtwarza obraz, struktura = śmieci",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "fig_inversion.png")
    fig.savefig(path, dpi=130); print("\nzapisano", path)


if __name__ == "__main__":
    main()
