#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D pokrętło prywatności: edge-DP (oś struktury) ⟂ feature-DP (oś treści).
Dla siatki (siła_edge × siła_feature) stosujemy OBIE obfuskacje, budujemy strumień atrybutowy
i mierzymy recon_leak∈[0,1] (odporny atak). Pokazuje, że dopiero feature-DP redukuje wyciek
TREŚCI; edge-DP nie rusza osi treści. Zapisuje heatmapę.

Użycie:
  python privacy2d.py --source tu --dataset PROTEINS_full --per-class 200 --out-dir results_priv2d
"""
from __future__ import annotations
import argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

from graphrep.config import Config
from graphrep import data, features, evaluate
from graphrep.embeddings import _block_norm
from graphrep.features import attr_matrix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="tu")
    ap.add_argument("--dataset", default="PROTEINS_full")
    ap.add_argument("--classes", type=int, nargs="+", default=None)
    ap.add_argument("--per-class", type=int, default=200)
    ap.add_argument("--edge-strengths", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0])
    ap.add_argument("--feat-strengths", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="results_priv2d")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = Config(source=args.source, dataset=args.dataset, classes=args.classes,
                 per_class=args.per_class, data_dir=args.data_dir, seed=args.seed)
    graphs, y, names = data.get_graphs(cfg)
    graphs = features.ensure_labels(graphs, cfg)
    target = evaluate.content_target(graphs)           # czysta treść (cel ataku)
    print(f"[{args.source}:{args.dataset}] N={len(graphs)} target_dim={None if target is None else target.shape[1]}")

    ES, FS = args.edge_strengths, args.feat_strengths
    leak = np.full((len(FS), len(ES)), np.nan)
    for j, es in enumerate(ES):
        ge = data.obfuscate(graphs, "edp", es, cfg.seed) if es > 0 else graphs
        for i, fs in enumerate(FS):
            gf = data.obfuscate(ge, "feature", fs, cfg.seed) if fs > 0 else ge
            A = normalize(_block_norm(attr_matrix(gf), "standard_l2"))
            rr = evaluate.reconstruction_attack(A, target, cfg)
            leak[i, j] = rr["recon_leak"]
            print(f"  edge={es:.2f} feat={fs:.2f}  recon_leak={leak[i,j]:.3f}")

    os.makedirs(args.out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    im = ax.imshow(leak, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(ES))); ax.set_xticklabels([f"{e:.2f}" for e in ES])
    ax.set_yticks(range(len(FS))); ax.set_yticklabels([f"{f:.1f}" for f in FS])
    ax.set_xlabel("edge-DP (siła; ← więcej szumu krawędzi)")
    ax.set_ylabel("feature-DP (siła ×std; ↑ więcej szumu cech)")
    ax.set_title(f"recon_leak (treść) — {args.dataset}\n2D pokrętło prywatności: struktura ⟂ treść")
    for i in range(len(FS)):
        for j in range(len(ES)):
            ax.text(j, i, f"{leak[i,j]:.2f}", ha="center", va="center",
                    color="white" if leak[i, j] < 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="recon_leak ∈ [0,1] (↓ = prywatniej)")
    fig.tight_layout()
    path = os.path.join(args.out_dir, "fig_privacy_2d.png")
    fig.savefig(path, dpi=130); print("zapisano", path)
    np.save(os.path.join(args.out_dir, "leak_grid.npy"), leak)


if __name__ == "__main__":
    main()
