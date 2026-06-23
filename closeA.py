#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Domknięcie Cel A — dwa eksperymenty:
  1) JĄDRO WL: macierz jądra WL (kosinusowe na histogramach poddrzew) → spectral clustering
     (affinity='precomputed') oraz kernel-KMeans (KMeans na znormalizowanych cechach WL).
     Czysto-strukturalny kierunek, jedyny niewyczerpany. ARI/NMI vs obecny lider.
  2) ORACLE upper-bound: LDA klasowo-świadome (UŻYWA ETYKIET = wyciek!) rzutuje embedding na
     osie dyskryminacyjne, potem KMeans. Pokazuje, o ile SKACZE ARI, gdy oś klasy jest znana
     → dowód, że sygnał jest w embeddingu, tylko nie jest osią wariancji. RAPORT OSOBNY.

Użycie:
  python closeA.py --source tu --dataset PROTEINS_full --struct fgsd
  python closeA.py --source cifar --classes 0 1 8 --per-class 500 --struct wl
"""
from __future__ import annotations
import argparse
import numpy as np
from collections import Counter

from sklearn.cluster import KMeans, SpectralClustering
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import normalize
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             silhouette_score)

from graphrep.config import Config
from graphrep import data, features
from graphrep.embeddings import build_structural, _init_wl_labels


def wl_count_matrix(graphs, n_iter):
    """Rzadka macierz cech WL (histogramy etykiet poddrzew, wspólny słownik) — przed SVD.
    To feature map jądra WL-subtree; jądro liniowe = X X^T."""
    node_labels = [_init_wl_labels(G) for G in graphs]
    dicts = [Counter({f"l_{l}": c for l, c in Counter(nl.values()).items()}) for nl in node_labels]
    pat2id = {}
    for it in range(n_iter):
        new = []
        for gi, G in enumerate(graphs):
            nl = node_labels[gi]; nn = {}
            for nd in G.nodes():
                pat = nl[nd] + "|" + ",".join(sorted(nl[nb] for nb in G.neighbors(nd)))
                pid = pat2id.setdefault(pat, len(pat2id)); nn[nd] = str(pid)
                dicts[gi][f"wl{it}_{pid}"] += 1
            new.append(nn)
        node_labels = new
    return DictVectorizer(sparse=True).fit_transform(dicts)


def score(name, X, pred, y, rows):
    m = {"name": name, "ARI": adjusted_rand_score(y, pred),
         "NMI": normalized_mutual_info_score(y, pred)}
    try:
        m["silh"] = silhouette_score(X, pred) if len(set(pred)) > 1 else float("nan")
    except Exception:
        m["silh"] = float("nan")
    rows.append(m)


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
    print(f"[{args.source}:{args.dataset}] N={len(graphs)} k={k}")

    rows = []

    # --- 1) JĄDRO WL ---
    Xwl = wl_count_matrix(graphs, cfg.wl_iterations)
    Xn = normalize(Xwl)                       # cosine
    K = (Xn @ Xn.T).toarray()                 # jądro WL kosinusowe ∈ [0,1]
    K = np.clip(K, 0.0, None)
    Xn_dense = np.asarray(Xn.todense())
    # kernel-KMeans ≈ KMeans na znormalizowanych cechach WL (jądro liniowe)
    score("wl-kernel: kmeans(feat)", Xn_dense,
          KMeans(n_clusters=k, n_init=10, random_state=args.seed).fit_predict(Xn_dense), y, rows)
    # spectral na macierzy jądra (precomputed)
    try:
        sp_pred = SpectralClustering(n_clusters=k, affinity="precomputed",
                                     assign_labels="kmeans", random_state=args.seed).fit_predict(K)
        score("wl-kernel: spectral(K)", K, sp_pred, y, rows)
    except Exception as e:
        print("  spectral(K) err:", e)
    # spectral z dyskretyzacją
    try:
        sp_pred2 = SpectralClustering(n_clusters=k, affinity="precomputed",
                                      assign_labels="discretize", random_state=args.seed).fit_predict(K)
        score("wl-kernel: spectral-disc(K)", K, sp_pred2, y, rows)
    except Exception as e:
        print("  spectral-disc err:", e)

    # --- ref: obecny lider (struct + KMeans) ---
    Xs = build_structural(graphs, args.struct, cfg)
    score(f"ref: {'+'.join(args.struct)}+kmeans", Xs,
          KMeans(n_clusters=k, n_init=10, random_state=args.seed).fit_predict(Xs), y, rows)

    print("\n  --- KLASTROWANIE (honest, label-free selekcja po silh) ---")
    print(f"  {'metoda':30} {'ARI':>6} {'NMI':>6} {'silh':>6}")
    for r in sorted(rows, key=lambda r: -r["ARI"]):
        print(f"  {r['name']:30} {r['ARI']:6.3f} {r['NMI']:6.3f} {r['silh']:6.3f}")

    # --- 2) ORACLE upper-bound: LDA(X,y) → KMeans (UŻYWA ETYKIET) ---
    print("\n  --- ORACLE upper-bound (UŻYWA ETYKIET — nie do leaderboardu honest) ---")
    nc = min(k - 1, Xs.shape[1])
    Xl = LinearDiscriminantAnalysis(n_components=nc).fit_transform(Xs, y)
    pred_o = KMeans(n_clusters=k, n_init=10, random_state=args.seed).fit_predict(Xl)
    ari_o = adjusted_rand_score(y, pred_o); nmi_o = normalized_mutual_info_score(y, pred_o)
    # też na jądrze WL (LDA na gęstych cechach WL)
    from sklearn.decomposition import TruncatedSVD
    Xwl_svd = TruncatedSVD(n_components=min(100, Xn.shape[1] - 1),
                           random_state=args.seed).fit_transform(Xn)
    Xl2 = LinearDiscriminantAnalysis(n_components=nc).fit_transform(Xwl_svd, y)
    pred_o2 = KMeans(n_clusters=k, n_init=10, random_state=args.seed).fit_predict(Xl2)
    ari_o2 = adjusted_rand_score(y, pred_o2); nmi_o2 = normalized_mutual_info_score(y, pred_o2)
    ref = max(r["ARI"] for r in rows)
    print(f"  {'oracle LDA('+ '+'.join(args.struct)+')→KMeans':30} ARI={ari_o:.3f} NMI={nmi_o:.3f}")
    print(f"  {'oracle LDA(wl-kernel)→KMeans':30} ARI={ari_o2:.3f} NMI={nmi_o2:.3f}")
    print(f"  (honest best tutaj = {ref:.3f} → oracle skok ×{ari_o/max(ref,1e-9):.1f})")


if __name__ == "__main__":
    main()
