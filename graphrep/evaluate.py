#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ewaluacja w trzech warstwach:
  1) NIENADZOROWANA (klastrowanie)  -> ARI/NMI/silhouette = ile klasy WYCIEKA bez etykiet
  2) NADZOROWANA (sonda)            -> acc/F1 = górna granica separowalności
  3) PRYWATNOŚĆ (atak rekonstrukcyjny) -> R²/MSE odtworzenia treści z embeddingu
                                          (im wyżej, tym reprezentacja bardziej odwracalna)
"""
from __future__ import annotations
import numpy as np
import networkx as nx

from sklearn.cluster import KMeans, HDBSCAN, SpectralClustering, AgglomerativeClustering
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict, KFold
from sklearn.metrics import (
    adjusted_rand_score, normalized_mutual_info_score, homogeneity_completeness_v_measure,
    silhouette_score, r2_score,
)
from .config import Config


def cluster(X: np.ndarray, algo: str, k: int, cfg: Config) -> np.ndarray:
    if algo == "kmeans":
        return KMeans(n_clusters=k, n_init=10, random_state=cfg.seed).fit_predict(X)
    if algo == "agglo":
        return AgglomerativeClustering(n_clusters=k).fit_predict(X)
    if algo == "spectral":
        nn = min(10, max(2, X.shape[0] - 1))
        return SpectralClustering(n_clusters=k, affinity="nearest_neighbors",
                                  n_neighbors=nn, assign_labels="kmeans",
                                  random_state=cfg.seed).fit_predict(X)
    if algo == "hdbscan":
        return HDBSCAN(min_cluster_size=max(5, X.shape[0] // (2 * k))).fit_predict(X)
    raise ValueError(f"Nieznany algorytm klastrowania: {algo}")


def unsupervised_metrics(X, pred, y) -> dict:
    res = {"n_clusters": len({c for c in set(pred) if c != -1}),
           "noise": float(np.mean(pred == -1))}
    mask = pred != -1
    if mask.sum() > 2 and len(set(pred[mask])) >= 2:
        res["silhouette"] = float(silhouette_score(X[mask], pred[mask]))
    else:
        res["silhouette"] = float("nan")
    res["ARI"] = float(adjusted_rand_score(y, pred))
    res["NMI"] = float(normalized_mutual_info_score(y, pred))
    _, _, v = homogeneity_completeness_v_measure(y, pred); res["Vmeasure"] = float(v)
    return res


def supervised_probe(X, y, cfg: Config) -> dict:
    counts = np.bincount(y)
    n_splits = int(min(cfg.cv_folds, counts[counts > 0].min()))
    if n_splits < 2:
        return {"probe_acc": float("nan"), "probe_f1": float("nan")}
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.seed)
    clf = LogisticRegression(max_iter=1000)
    Xs = StandardScaler().fit_transform(X)
    return {"probe_acc": float(cross_val_score(clf, Xs, y, cv=cv, scoring="accuracy").mean()),
            "probe_f1": float(cross_val_score(clf, Xs, y, cv=cv, scoring="f1_macro").mean())}


def content_target(graphs) -> np.ndarray:
    """'Treść' obiektu do odtworzenia: średni deskryptor węzła na graf
    (kolor/tekstura dla obrazu, atrybuty dla białka). Cel ataku inwersyjnego."""
    rows, dim = [], 0
    for G in graphs:
        xs = [np.atleast_1d(d["features"]) for _, d in G.nodes(data=True) if "features" in d]
        if xs:
            v = np.vstack(xs).astype(float).mean(0); dim = max(dim, len(v)); rows.append(v)
        else:
            rows.append(None)
    if dim == 0:
        return None
    return np.vstack([r if r is not None else np.zeros(dim) for r in rows])


def reconstruction_attack(X, target, cfg: Config) -> dict:
    """Atak inwersyjny: czy z embeddingu da się odtworzyć treść obiektu?
    Trenujemy MLP (CV) X -> target; R² ~ odwracalność (wyciek). Niżej = lepsza prywatność."""
    if target is None or X.shape[0] < 6:
        return {"recon_r2": float("nan")}
    Xs = StandardScaler().fit_transform(X)
    Ts = StandardScaler().fit_transform(target)
    mlp = MLPRegressor(hidden_layer_sizes=(64,), max_iter=300, random_state=cfg.seed)
    cv = KFold(n_splits=3, shuffle=True, random_state=cfg.seed)
    try:
        pred = cross_val_predict(mlp, Xs, Ts, cv=cv)
        return {"recon_r2": float(r2_score(Ts, pred))}
    except Exception:
        return {"recon_r2": float("nan")}
