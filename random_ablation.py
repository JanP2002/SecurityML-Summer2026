#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ABLACJA: grafy LOSOWE jako REPREZENTACJA (nie obfuskacja).
Ile sygnału klastrowania pochodzi z PRAWDZIWEJ struktury vs losowej? Dla najlepszego
deskryptora strukturalnego liczymy km_ARI/NMI na: (i) prawdziwym grafie, (ii) ER o dopasowanej
gęstości, (iii) dropedge p=0.5. Atrybuty/etykiety węzłów ZACHOWANE (zmieniamy tylko strukturę),
więc test izoluje wkład topologii. Oczekiwane: losowy << prawdziwy → struktura niesie sygnał.

Użycie:
  python random_ablation.py --source tu --dataset PROTEINS_full --struct fgsd
  python random_ablation.py --source tu --dataset ENZYMES --struct wl
"""
from __future__ import annotations
import argparse
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

from graphrep.config import Config
from graphrep import data, features
from graphrep.embeddings import build_structural


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="tu")
    ap.add_argument("--dataset", default="PROTEINS_full")
    ap.add_argument("--struct", nargs="+", default=["fgsd"])
    ap.add_argument("--classes", type=int, nargs="+", default=None)
    ap.add_argument("--per-class", type=int, default=None)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = Config(source=args.source, dataset=args.dataset, classes=args.classes,
                 per_class=args.per_class, data_dir=args.data_dir, seed=args.seed)
    graphs, y, names = data.get_graphs(cfg)
    graphs = features.ensure_labels(graphs, cfg)
    k = len(names)
    print(f"[{args.source}:{args.dataset}] struct={args.struct} N={len(graphs)} k={k}")

    variants = {
        "(i) prawdziwy graf": graphs,
        "(ii) ER (dopasowana gęstość)": data.obfuscate(graphs, "er", 1.0, cfg.seed),
        "(iii) dropedge p=0.5": data.obfuscate(graphs, "dropedge", 0.5, cfg.seed),
    }
    print(f"\n  {'wariant':32} {'km_ARI':>7} {'km_NMI':>7} {'silh':>7}")
    for name, gs in variants.items():
        X = build_structural(gs, args.struct, cfg)
        pred = KMeans(n_clusters=k, n_init=10, random_state=args.seed).fit_predict(X)
        ari = adjusted_rand_score(y, pred); nmi = normalized_mutual_info_score(y, pred)
        silh = silhouette_score(X, pred) if len(set(pred)) > 1 else float("nan")
        print(f"  {name:32} {ari:7.3f} {nmi:7.3f} {silh:7.3f}")


if __name__ == "__main__":
    main()
