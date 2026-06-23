#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wspólna konfiguracja frameworka `graphrep`."""
from __future__ import annotations
from dataclasses import dataclass, field
import os


@dataclass
class Config:
    # --- dane / źródło grafów ---
    source: str = "tu"                 # tu | cifar | synth
    dataset: str = "ENZYMES"           # dla source=tu: ENZYMES | PROTEINS_full ; dla cifar/synth: dowolne
    data_dir: str = "data"
    out_dir: str = "results"
    classes: list[int] | None = None   # podzbiór klas (cifar/synth) lub None
    per_class: int | None = None       # ile próbek/grafów na klasę (None = wszystkie)
    num_samples: int | None = None      # alternatywnie: łączna liczba
    seed: int = 42

    # --- budowa grafu z obrazu (source=cifar/synth) ---
    graph_type: str = "slic"           # pixel | patch | slic
    connectivity: int = 8
    edge_quantile: float = 0.6         # adaptacyjny próg krawędzi
    tau: float | None = None           # stały próg zamiast kwantyla
    sigma_pixel: float = 15.0
    sigma_feat: float = 1.0
    patch_size: int = 5
    stride: int = 2
    knn_radius: int = 3                # okno przestrzenne dla kNN (warunek #1)
    k_neighbors: int = 4
    n_segments: int = 60
    compactness: float = 10.0
    edge_texture: bool = False         # mocniejszy warunek krawędzi: kolor ORAZ tekstura
    rich_features: bool = False        # mini-HOG na superpiksel
    n_orient_bins: int = 8
    img_size: int = 32                 # rozmiar obrazu syntetycznego

    # --- etykiety węzłów / WL ---
    n_color_labels: int = 8            # globalny KMeans kolorów -> seed WL (cifar/synth)
    label_rich: bool = False           # seed WL po pełnym deskryptorze (kolor+tekstura)
    wl_iterations: int = 3
    g2v_dim: int = 64
    doc2vec_epochs: int = 80

    # --- node2vec ---
    n2v_dim: int = 64
    n2v_walk_length: int = 12
    n2v_num_walks: int = 20
    n2v_window: int = 5
    n2v_workers: int = max(1, (os.cpu_count() or 2) - 1)

    # --- spectral ---
    spec_k: int = 12

    # --- rand_ens (reprezentacja: ensembling M losowych wariantów struktury + uśrednianie) ---
    rand_ens_m: int = 10               # liczba losowych wariantów na obiekt
    rand_ens_base: str = "topo"        # bazowy deskryptor: topo | wl | graph2vec | netlsd | graphlet | fgsd
    rand_ens_method: str = "dropedge"  # losowanie struktury: dropedge | shortcuts | er
    rand_ens_p: float = 0.5            # siła losowania (p dropedge / siła shortcuts)

    # --- obfuskacja (oś prywatność–użyteczność) ---
    obf_method: str = "rewire"         # rewire | dropedge | shortcuts | er
    obf_strengths: list[float] = field(default_factory=lambda: [0.0])

    # --- fuzja / klastrowanie ---
    weights: list[float] = field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0])
    cluster_algo: str = "kmeans"       # kmeans | spectral | agglo | hdbscan
    k: int | None = None               # liczba klastrów (None = liczba klas)

    # --- ewaluacja ---
    cv_folds: int = 5
    do_reconstruction: bool = True     # atak inwersyjny (metryka prywatności)
