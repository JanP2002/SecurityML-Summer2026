#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Laboratorium KLASTROWANIA (Cel A) — buduje JEDEN embedding i testuje baterię algorytmów
klastrowania in-process (bez przebudowy grafów per wariant).

Hipoteza fazy: wąskim gardłem jest krok klastrowania, nie embedding (probe_acc >> szansy,
ARI niskie). Selekcja konfiguracji po kryteriach LABEL-FREE (silhouette / Calinski-Harabasz /
Davies-Bouldin); ARI/NMI raportowane jako WYNIK, nie cel strojenia.

Użycie:
  python cluster_lab.py --source tu --dataset PROTEINS_full --struct fgsd
  python cluster_lab.py --source cifar --classes 0 1 8 --per-class 500 --struct wl
  python cluster_lab.py --source tu --dataset ENZYMES --struct wl
"""
from __future__ import annotations
import argparse
import numpy as np
import networkx as nx

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             silhouette_score, calinski_harabasz_score, davies_bouldin_score)

from graphrep.config import Config
from graphrep import data, features
from graphrep.embeddings import build_structural


def corr_ratio(labels, x):
    """Współczynnik korelacji η² (eta-squared): jaki ułamek wariancji zmiennej ciągłej x
    tłumaczy przynależność do grup `labels`. 0 = klastry niezależne od x, 1 = x deterministyczne
    po klastrze. Służy do diagnozy: czy klastry rozdzielają po ROZMIARZE zamiast po klasie."""
    x = np.asarray(x, float)
    grand = x.mean()
    ss_tot = ((x - grand) ** 2).sum()
    if ss_tot == 0:
        return 0.0
    ss_between = 0.0
    for c in set(labels):
        xi = x[labels == c]
        if len(xi):
            ss_between += len(xi) * (xi.mean() - grand) ** 2
    return float(ss_between / ss_tot)


def metrics(X, pred, y):
    m = {"ARI": adjusted_rand_score(y, pred), "NMI": normalized_mutual_info_score(y, pred)}
    if len(set(pred)) >= 2:
        m["silh"] = silhouette_score(X, pred)
        m["CH"] = calinski_harabasz_score(X, pred)
        m["DB"] = davies_bouldin_score(X, pred)
    else:
        m["silh"] = m["CH"] = m["DB"] = float("nan")
    return m


def get_embedding(cfg, struct_names):
    graphs, y, names = data.get_graphs(cfg)
    graphs = features.ensure_labels(graphs, cfg)
    X = build_structural(graphs, struct_names, cfg)
    sizes = np.array([g.number_of_nodes() for g in graphs], float)
    dens = np.array([nx.density(g) if g.number_of_nodes() > 1 else 0.0 for g in graphs], float)
    return X, y, sizes, dens, names


def cluster_battery(X, y, sizes, dens, k, seed):
    rows = []

    def run(name, pred, Xeval):
        m = metrics(Xeval, pred, y)
        m["name"] = name
        m["eta_size"] = corr_ratio(pred, sizes)
        m["eta_dens"] = corr_ratio(pred, dens)
        rows.append(m)

    # 1) KMeans (baseline) — X jest już L2-znormalizowane (≈ cosine/spherical)
    run("kmeans", KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X), X)

    # 2) GMM full covariance
    try:
        gm = GaussianMixture(n_components=k, covariance_type="full", n_init=3,
                             random_state=seed).fit(X)
        run("gmm-full", gm.predict(X), X)
    except Exception as e:
        print("  gmm-full err:", e)

    # 3) GMM diag (stabilniejsze w wysokim wymiarze)
    try:
        gm = GaussianMixture(n_components=k, covariance_type="diag", n_init=3,
                             random_state=seed).fit(X)
        run("gmm-diag", gm.predict(X), X)
    except Exception as e:
        print("  gmm-diag err:", e)

    # 4) PCA-whitening -> KMeans
    nc = min(X.shape[1], X.shape[0] - 1, 50)
    Xw = PCA(n_components=nc, whiten=True, random_state=seed).fit_transform(X)
    run("pca-whiten+kmeans", KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Xw), Xw)
    run("pca-whiten+gmm-diag",
        GaussianMixture(n_components=k, covariance_type="diag", n_init=3,
                        random_state=seed).fit(Xw).predict(Xw), Xw)

    # 5) PCA-whiten -> re-normalize (spherical) -> KMeans
    Xws = normalize(Xw)
    run("pca-whiten+sph-kmeans", KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Xws), Xws)

    # 6) Agglomerative (ward na whitened)
    run("agglo-ward(whiten)", AgglomerativeClustering(n_clusters=k).fit_predict(Xw), Xw)

    # 7) UMAP -> KMeans / GMM (sweep n_neighbors, min_dist)
    try:
        import umap
        for nn in (10, 30, 50):
            for md in (0.0, 0.1):
                emb = umap.UMAP(n_neighbors=nn, min_dist=md, n_components=10,
                                random_state=seed).fit_transform(X)
                run(f"umap(nn{nn},md{md})+kmeans",
                    KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(emb), emb)
                run(f"umap(nn{nn},md{md})+gmm",
                    GaussianMixture(n_components=k, covariance_type="full", n_init=3,
                                    random_state=seed).fit(emb).predict(emb), emb)
    except Exception as e:
        print("  umap err:", e)

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="tu")
    ap.add_argument("--dataset", default="PROTEINS_full")
    ap.add_argument("--struct", nargs="+", default=["fgsd"])
    ap.add_argument("--classes", type=int, nargs="+", default=None)
    ap.add_argument("--per-class", type=int, default=None)
    ap.add_argument("--label-rich", action="store_true")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = Config(source=args.source, dataset=args.dataset, classes=args.classes,
                 per_class=args.per_class, label_rich=args.label_rich, data_dir=args.data_dir,
                 seed=args.seed)
    X, y, sizes, dens, names = get_embedding(cfg, args.struct)
    k = len(names)
    print(f"[{args.source}:{args.dataset}] struct={args.struct} | N={X.shape[0]} dim={X.shape[1]} "
          f"k={k} | |V| med={np.median(sizes):.0f}")
    # diagnoza: czy ROZMIAR/GĘSTOŚĆ korelują z PRAWDZIWĄ klasą?
    print(f"  η²(size|class_true)={corr_ratio(y, sizes):.3f}  η²(dens|class_true)={corr_ratio(y, dens):.3f}")

    rows = cluster_battery(X, y, sizes, dens, k, args.seed)
    print(f"\n  {'algorytm':26} {'ARI':>6} {'NMI':>6} {'silh':>6} {'CH':>7} {'DB':>6} "
          f"{'ηsize':>6} {'ηdens':>6}")
    for r in sorted(rows, key=lambda r: -r["ARI"]):
        print(f"  {r['name']:26} {r['ARI']:6.3f} {r['NMI']:6.3f} {r['silh']:6.3f} "
              f"{r['CH']:7.1f} {r['DB']:6.2f} {r['eta_size']:6.3f} {r['eta_dens']:6.3f}")


if __name__ == "__main__":
    main()
