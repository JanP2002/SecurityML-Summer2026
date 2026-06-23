#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Etykiety węzłów (seed dla WL/graph2vec) i pooling atrybutów."""
from __future__ import annotations
import numpy as np
from sklearn.cluster import KMeans
from .config import Config


def _has_labels(graphs) -> bool:
    for G in graphs:
        for _, d in G.nodes(data=True):
            return "label" in d
    return False


def ensure_labels(graphs, cfg: Config):
    """Jeśli węzły już mają 'label' (TU) — nic nie rób. W innym wypadku (obrazy)
    przypisz dyskretną etykietę globalnym KMeans na deskryptorze węzła (seed WL).
    `--label-rich`: seed po pełnym deskryptorze (kolor+tekstura), inaczej po kolorze."""
    if _has_labels(graphs):
        return graphs
    feats, owner = [], []
    for gi, G in enumerate(graphs):
        for n, d in G.nodes(data=True):
            f = np.atleast_1d(d.get("features", np.zeros(1)))
            feats.append(f if cfg.label_rich else f[:3])
            owner.append((gi, n))
    if not feats:
        return graphs
    X = np.vstack(feats)
    k = min(cfg.n_color_labels, max(2, X.shape[0]))
    km = KMeans(n_clusters=k, n_init=5, random_state=cfg.seed).fit_predict(X)
    for (gi, n), lab in zip(owner, km):
        graphs[gi].nodes[n]["label"] = int(lab)
    return graphs


def _attr_dim(graphs) -> int:
    for G in graphs:
        for _, d in G.nodes(data=True):
            if "features" in d:
                return len(np.atleast_1d(d["features"]))
    return 0


def attr_matrix(graphs) -> np.ndarray:
    """Strumień atrybutowy: mean ⊕ std deskryptorów węzła (jeden wektor na graf)."""
    dim = _attr_dim(graphs)
    if dim == 0:
        return np.zeros((len(graphs), 1))
    rows = []
    for G in graphs:
        xs = [np.atleast_1d(d["features"]) for _, d in G.nodes(data=True) if "features" in d]
        if xs:
            X = np.vstack(xs).astype(float)
            rows.append(np.concatenate([X.mean(0), X.std(0)]))
        else:
            rows.append(np.zeros(2 * dim))
    return np.vstack(rows)


def has_attributes(graphs) -> bool:
    return _attr_dim(graphs) > 0
