#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RYGOR#6 — czy LABEL-FREE kryterium trafia w komplementarny punkt fuzji?

Fuzja struktura⊕attr (jak w pipeline): X(w) = [w·S | (1-w)·A], S/A jednostkowe.
Dla każdego w liczymy ARI/NMI (WYNIK, nie cel) oraz kryteria bezetykietowe:
silhouette, Calinski-Harabasz, Davies-Bouldin, oraz STABILNOŚĆ bootstrapową (śr. ARI między
klastrowaniem podpróbek a pełnym). Pytanie: czy argmax(silh/CH/stab)/argmin(DB) ≈ argmax(ARI)?

Użycie:
  python fusion_select.py --source cifar --classes 0 1 8 --per-class 500 --struct wl
  python fusion_select.py --source tu --dataset PROTEINS_full --struct fgsd
"""
from __future__ import annotations
import argparse
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             silhouette_score, calinski_harabasz_score, davies_bouldin_score)

from graphrep.config import Config
from graphrep import data, features
from graphrep.embeddings import build_structural, attr_matrix, _block_norm


def bootstrap_stability(X, k, seed, B=15, frac=0.8):
    """Średnie ARI między klastrowaniem podpróbki (80%) a etykietami pełnego zbioru na
    wspólnych punktach. Wyższe = stabilniejsza struktura klastrów (label-free)."""
    rng = np.random.default_rng(seed)
    ref = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
    scores = []
    n = X.shape[0]
    for b in range(B):
        idx = rng.choice(n, int(frac * n), replace=False)
        lab = KMeans(n_clusters=k, n_init=5, random_state=seed + b + 1).fit_predict(X[idx])
        scores.append(adjusted_rand_score(ref[idx], lab))
    return float(np.mean(scores))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="cifar")
    ap.add_argument("--dataset", default="ENZYMES")
    ap.add_argument("--struct", nargs="+", default=["wl"])
    ap.add_argument("--classes", type=int, nargs="+", default=None)
    ap.add_argument("--per-class", type=int, default=None)
    ap.add_argument("--label-rich", action="store_true")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = Config(source=args.source, dataset=args.dataset, classes=args.classes,
                 per_class=args.per_class, label_rich=args.label_rich, data_dir=args.data_dir,
                 seed=args.seed)
    graphs, y, names = data.get_graphs(cfg)
    graphs = features.ensure_labels(graphs, cfg)
    S = build_structural(graphs, args.struct, cfg)
    A = normalize(_block_norm(attr_matrix(graphs), "standard_l2"))
    k = len(names)
    print(f"[{args.source}:{args.dataset}] struct={args.struct} N={S.shape[0]} k={k}")

    rows = []
    for w in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        parts = []
        if w > 0:
            parts.append(w * S)
        if w < 1:
            parts.append((1 - w) * A)
        X = np.hstack(parts)
        pred = KMeans(n_clusters=k, n_init=10, random_state=args.seed).fit_predict(X)
        ari = adjusted_rand_score(y, pred); nmi = normalized_mutual_info_score(y, pred)
        silh = silhouette_score(X, pred) if len(set(pred)) > 1 else float("nan")
        ch = calinski_harabasz_score(X, pred) if len(set(pred)) > 1 else float("nan")
        db = davies_bouldin_score(X, pred) if len(set(pred)) > 1 else float("nan")
        stab = bootstrap_stability(X, k, args.seed)
        rows.append(dict(w=w, ARI=ari, NMI=nmi, silh=silh, CH=ch, DB=db, stab=stab))

    print(f"\n  {'w':>4} {'ARI':>6} {'NMI':>6} {'silh':>6} {'CH':>8} {'DB':>6} {'stab':>6}")
    for r in rows:
        print(f"  {r['w']:>4} {r['ARI']:6.3f} {r['NMI']:6.3f} {r['silh']:6.3f} "
              f"{r['CH']:8.1f} {r['DB']:6.2f} {r['stab']:6.3f}")

    def pick(key, hi=True):
        c = [r for r in rows if np.isfinite(r[key])]
        return (max if hi else min)(c, key=lambda r: r[key])

    best_ari = pick("ARI")
    print(f"\n  ARI-optimal w={best_ari['w']} (ARI={best_ari['ARI']:.3f})  [oracle — nieosiągalne label-free]")
    for crit, hi in [("silh", True), ("CH", True), ("DB", False), ("stab", True)]:
        p = pick(crit, hi)
        hit = "✓" if abs(p["w"] - best_ari["w"]) < 1e-9 else "✗"
        print(f"  {crit:5} → w={p['w']}  (tam ARI={p['ARI']:.3f})  {hit}")


if __name__ == "__main__":
    main()
