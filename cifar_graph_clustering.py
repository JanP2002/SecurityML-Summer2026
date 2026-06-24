#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
CIFAR-10 jako grafy — reprezentacje grafowe obrazów (Node2Vec) — Lista 3
================================================================================
Każdy obraz -> graf -> Node2Vec (na węzłach) -> agregacja -> jeden wektor obrazu.
Porównujemy 3 sposoby budowy grafu (pixel / patch / slic) z baseline'ami
(RGB-mean, HOG) na klasyfikacji (LogReg / RandomForest / SVM, 5-fold CV) oraz
nienadzorowanej klasteryzacji (KMeans: silhouette, ARI).

GŁÓWNA ZMIANA WZGLĘDEM NOTATNIKA (kluczowa wskazówka, 2026-06-18):
  Krawędź powstaje TYLKO gdy spełnione są JEDNOCZEŚNIE dwa warunki:
    1) węzły są blisko siebie PRZESTRZENNIE (sąsiedztwo na obrazie / siatce), ORAZ
    2) węzły reprezentują ten sam obiekt / piksele o podobnym odcieniu
       (odległość koloru w LAB poniżej progu tau).
  W notatniku warunek (2) był tylko MIĘKKĄ wagą krawędzi — krawędź i tak
  powstawała, więc graf był prawie pełną siatką 32x32 dla KAŻDEGO obrazu i
  Node2Vec kodował głównie siatkę, a nie treść. Twardy próg tnie krawędzie na
  granicach obiektów -> graf rozpada się na regiony ~obiekty -> topologia grafu
  staje się zależna od klasy obrazu.

Próg tau jest dobierany ADAPTACYJNIE per obraz jako kwantyl rozkładu odległości
krawędzi (domyślnie 0.6 => zostaje ~60% najbardziej "wewnątrzobiektowych"
krawędzi). Można też podać stały tau przez --tau.

Uruchomienie:
  python cifar_graph_clustering.py --graph-type all --classes 0 1 8 --per-class 80
  python cifar_graph_clustering.py --graph-type slic --num-samples 1000   # pełne 10 klas
"""

from __future__ import annotations
import argparse, os, csv, sys, time, warnings, hashlib
from dataclasses import dataclass, field, replace
from collections import Counter

import numpy as np
import scipy.sparse as sp
import networkx as nx
from joblib import Parallel, delayed
warnings.filterwarnings("ignore")

import torch
import torchvision
import torchvision.transforms as transforms

from skimage.feature import hog
from skimage.segmentation import slic
from skimage.color import rgb2lab
from scipy.spatial.distance import cdist

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans

from node2vec import Node2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Wymuś UTF-8 na stdout/stderr (konsola Windows bywa cp1252 i krztusi się polskimi znakami).
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

CIFAR_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                 "dog", "frog", "horse", "ship", "truck"]


@dataclass
class Config:
    data_dir: str = "./data"
    out_dir: str = "results_cifar"
    # dobór danych
    classes: list[int] | None = None          # None => wszystkie 10 klas
    per_class: int | None = None               # liczba obrazów na klasę
    num_samples: int | None = None             # alternatywnie: łączna liczba próbek
    # twardy próg krawędzi (kluczowa zasada)
    tau: float | None = None                   # stały próg odległości; None => kwantyl
    edge_quantile: float = 0.6                 # zostaw krawędzie poniżej tego kwantyla
    # parametry grafów
    connectivity: int = 8
    patch_size: int = 4
    stride: int = 2
    knn_radius: int = 3                         # promień (w komórkach siatki) dla krawędzi kNN patchy
    k_neighbors: int = 4
    n_segments: int = 60
    compactness: float = 10.0
    sigma_pixel: float = 15.0                   # skala wag w LAB (0-100)
    sigma_feat: float = 1.0                     # skala wag w przestrzeni cech (patch/slic)
    # Node2Vec
    dim: int = 32
    walk_length: int = 30
    num_walks: int = 20
    p: float = 1.0
    q: float = 0.5
    window: int = 5
    # całografowe embeddingi (WL / graph2vec) — NOWE metody
    wl_iterations: int = 2
    g2v_dim: int = 64
    g2v_epochs: int = 60
    n_color_bins: int = 16          # kwantyzacja koloru węzła na dyskretne etykiety WL
    # bogatszy deskryptor węzła (mini-HOG na superpiksel) — kolejne usprawnienie
    rich_features: bool = False
    n_orient_bins: int = 6
    # mocniejszy warunek krawędzi: podobny kolor ORAZ podobna tekstura (typy *tex)
    edge_texture: bool = False
    n_spectral: int = 16            # liczba wartości własnych w sygnaturze spektralnej
    label_rich: bool = False        # seed WL po pełnym deskryptorze (kolor+tekstura), nie samym kolorze
    # ewaluacja
    cv_folds: int = 5
    n_jobs: int = -1
    seed: int = 42
    # waga strumienia strukturalnego (Node2Vec) vs atrybutowego (kolor/tekstura);
    # przemiatamy ją, żeby znaleźć optymalny mix i przeciwdziałać rozcieńczeniu
    weights: list[float] = field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0])
    plots: bool = False             # zapisz wykresy do opowiedzenia historii wyników


# ============================================================================
# DANE
# ============================================================================
def load_cifar10(cfg: Config):
    """Zwraca (images: list[HxWx3 float32 w [0,1]], labels: np.ndarray[int]).
    Etykiety są PRZEMAPOWANE na 0..(k-1) w kolejności podanej w cfg.classes."""
    transform = transforms.Compose([transforms.ToTensor()])
    ds = torchvision.datasets.CIFAR10(root=cfg.data_dir, train=True,
                                      download=True, transform=transform)
    targets = np.array(ds.targets)
    keep = cfg.classes if cfg.classes is not None else list(range(10))
    remap = {c: i for i, c in enumerate(keep)}

    rng = np.random.default_rng(cfg.seed)
    idxs = []
    if cfg.per_class is not None:
        for c in keep:
            pool = np.where(targets == c)[0]
            idxs.extend(rng.choice(pool, min(cfg.per_class, len(pool)), replace=False))
    else:
        pool = np.where(np.isin(targets, keep))[0]
        n = cfg.num_samples or len(pool)
        idxs = rng.choice(pool, min(n, len(pool)), replace=False)
    idxs = np.array(sorted(int(i) for i in idxs))

    images, labels = [], []
    for i in idxs:
        img, lab = ds[int(i)]
        images.append(img.permute(1, 2, 0).numpy().astype(np.float32))  # HxWx3, [0,1]
        labels.append(remap[int(lab)])
    return images, np.array(labels, dtype=int)


# ============================================================================
# BUDOWA GRAFÓW — twarda koniunkcja: bliskość przestrzenna ORAZ podobny kolor
# ============================================================================
def _threshold(dists: np.ndarray, cfg: Config) -> float:
    """Próg odległości krawędzi: stały (cfg.tau) albo adaptacyjny kwantyl per obraz."""
    if cfg.tau is not None:
        return cfg.tau
    if len(dists) == 0:
        return float("inf")
    return float(np.quantile(dists, cfg.edge_quantile))


def _ensure_connected(G: nx.Graph, best_neighbor: dict):
    """Każdy izolowany węzeł podłącz do jego NAJPODOBNIEJSZEGO sąsiada przestrzennego
    (warunek 1 zachowany), żeby Node2Vec miał gdzie chodzić."""
    for n in list(G.nodes()):
        if G.degree(n) == 0 and n in best_neighbor:
            nb, w = best_neighbor[n]
            G.add_edge(n, nb, weight=w)


def build_pixel_graph(image: np.ndarray, cfg: Config) -> nx.Graph:
    """Metoda A: graf pikselowy w przestrzeni LAB. Krawędź między sąsiadami
    8-spójnymi TYLKO jeśli odległość koloru LAB < próg (ten sam obiekt/odcień)."""
    lab = rgb2lab(image)
    H, W, _ = lab.shape
    # tekstura: magnituda gradientu kanału jasności L (przeżywa pooling jako
    # "średnia teksturowość" — gładkie niebo/statek vs poszarpane zwierzę)
    gy, gx = np.gradient(lab[:, :, 0])
    grad = np.sqrt(gx ** 2 + gy ** 2)
    G = nx.Graph()
    for i in range(H):
        for j in range(W):
            feat = np.array([lab[i, j, 0], lab[i, j, 1], lab[i, j, 2], grad[i, j]], float)
            G.add_node(i * W + j, pos=(i, j), features=feat)

    if cfg.connectivity == 8:
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    else:
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # 1) zbierz wszystkie sąsiedztwa przestrzenne: odległość koloru i tekstury
    cand = []                       # (u, v, dist_koloru, dist_tekstury)
    best_neighbor = {}              # u -> (v, weight) najbardziej podobny sąsiad
    for i in range(H):
        for j in range(W):
            u = i * W + j
            for di, dj in offsets:
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W:
                    v = ni * W + nj
                    d = float(np.sqrt(np.sum((lab[i, j] - lab[ni, nj]) ** 2)))
                    dt = abs(float(grad[i, j] - grad[ni, nj]))
                    if u < v:
                        cand.append((u, v, d, dt))
                    w = float(np.exp(-d * d / (2 * cfg.sigma_pixel ** 2)))
                    if u not in best_neighbor or w > best_neighbor[u][1]:
                        best_neighbor[u] = (v, w)

    # 2) twardy próg: krawędź wewnątrzobiektowa = podobny kolor (--edge-texture: ORAZ tekstura)
    tau = _threshold(np.array([d for _, _, d, _ in cand]), cfg)
    tau_t = _threshold(np.array([dt for *_, dt in cand]), cfg) if cfg.edge_texture else None
    for u, v, d, dt in cand:
        if d <= tau and (tau_t is None or dt <= tau_t):
            G.add_edge(u, v, weight=float(np.exp(-d * d / (2 * cfg.sigma_pixel ** 2))))
    _ensure_connected(G, best_neighbor)
    return G


def build_patch_graph(image: np.ndarray, cfg: Config) -> nx.Graph:
    """Metoda B: węzły = nakładające się patche (mean⊕std). Krawędzie:
    sąsiedztwo na siatce ORAZ kNN — ale kNN OGRANICZONE do okna przestrzennego
    (warunek 1), a obie rodziny przycinane progiem podobieństwa cech (warunek 2)."""
    H, W, _ = image.shape
    ps, st = cfg.patch_size, cfg.stride
    G = nx.Graph()
    feats, pos = [], []
    idx = 0
    for i in range(0, H - ps + 1, st):
        for j in range(0, W - ps + 1, st):
            patch = image[i:i + ps, j:j + ps, :]
            f = np.concatenate([patch.mean(axis=(0, 1)), patch.std(axis=(0, 1))])
            G.add_node(idx, pos=(i, j), grid=(i // st, j // st), features=f)
            feats.append(f); pos.append((i // st, j // st)); idx += 1
    feats = np.array(feats)
    gH = (H - ps) // st + 1
    gW = (W - ps) // st + 1

    cand, best_neighbor = [], {}
    # krawędzie siatkowe (4-sąsiedztwo)
    for n in range(len(feats)):
        gi, gj = pos[n]
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = gi + di, gj + dj
            if 0 <= ni < gH and 0 <= nj < gW:
                m = ni * gW + nj
                d = float(np.linalg.norm(feats[n] - feats[m]))
                if n < m:
                    cand.append((n, m, d))
                w = float(np.exp(-d * d / (2 * cfg.sigma_feat ** 2)))
                if n not in best_neighbor or w > best_neighbor[n][1]:
                    best_neighbor[n] = (m, w)

    # krawędzie kNN OGRANICZONE przestrzennie (okno o promieniu knn_radius)
    D = cdist(feats, feats, metric="euclidean")
    grid_pos = np.array(pos)
    for n in range(len(feats)):
        near = np.where((np.abs(grid_pos[:, 0] - grid_pos[n, 0]) <= cfg.knn_radius) &
                        (np.abs(grid_pos[:, 1] - grid_pos[n, 1]) <= cfg.knn_radius))[0]
        near = near[near != n]
        if len(near) == 0:
            continue
        order = near[np.argsort(D[n, near])][:cfg.k_neighbors]
        for m in order:
            if n < m:
                cand.append((n, int(m), float(D[n, m])))

    tau = _threshold(np.array([d for *_, d in cand]), cfg)
    for n, m, d in cand:
        if d <= tau:
            G.add_edge(n, m, weight=float(np.exp(-d * d / (2 * cfg.sigma_feat ** 2))))
    _ensure_connected(G, best_neighbor)
    return G


def build_slic_graph(image: np.ndarray, cfg: Config) -> nx.Graph:
    """Metoda C: węzły = superpiksele SLIC (mean⊕std). Krawędź między
    sąsiadującymi superpikselami TYLKO jeśli różnica średnich kolorów < próg."""
    labels = slic(image, n_segments=cfg.n_segments, compactness=cfg.compactness,
                  start_label=0, channel_axis=-1)
    H, W = labels.shape
    G = nx.Graph()

    boundary = {}
    for r in range(H):
        for c in range(W):
            lbl = labels[r, c]
            if c < W - 1 and labels[r, c + 1] != lbl:
                pair = tuple(sorted((int(lbl), int(labels[r, c + 1]))))
                boundary[pair] = boundary.get(pair, 0) + 1
            if r < H - 1 and labels[r + 1, c] != lbl:
                pair = tuple(sorted((int(lbl), int(labels[r + 1, c]))))
                boundary[pair] = boundary.get(pair, 0) + 1
    max_b = max(boundary.values()) if boundary else 1

    # tekstura na superpiksel (do mocniejszego warunku krawędzi, typy *tex): średnia
    # magnituda gradientu kanału L — gładkie niebo vs poszarpane zwierzę.
    grad_lbl = {}
    if cfg.edge_texture:
        Lt = rgb2lab(image)[:, :, 0]
        gyt, gxt = np.gradient(Lt)
        magt = np.sqrt(gxt ** 2 + gyt ** 2)
        for lbl in np.unique(labels):
            grad_lbl[int(lbl)] = float(magt[labels == lbl].mean())

    # --- bogatszy deskryptor węzła (--rich-features): mini-HOG na superpiksel ---
    # Histogram orientacji gradientu (ważony magnitudą) w obrębie superpiksela —
    # graf-natywny odpowiednik HOG, żeby `combo`/`hyb` rywalizowały strukturą, a nie
    # pożyczonym HOG. UWAGA: features[:3] zostaje średnim RGB (krawędzie bez zmian).
    if cfg.rich_features:
        Lc = rgb2lab(image)[:, :, 0]
        gy, gx = np.gradient(Lc)
        mag = np.sqrt(gx ** 2 + gy ** 2)
        nb = cfg.n_orient_bins
        obin = np.minimum((np.arctan2(gy, gx) % np.pi) / (np.pi / nb), nb - 1).astype(int)
        npx = float(H * W)

    for lbl in np.unique(labels):
        mask = labels == lbl
        f = np.concatenate([image[mask].mean(axis=0), image[mask].std(axis=0)])
        if cfg.rich_features:
            m, b = mag[mask], obin[mask]
            hist = np.bincount(b, weights=m, minlength=cfg.n_orient_bins)
            hist = hist / (hist.sum() + 1e-6)                      # mini-HOG (nb)
            cen = np.argwhere(mask).mean(axis=0) / np.array([H, W])  # znormalizowany środek
            f = np.concatenate([f, hist, [m.mean(), m.std()], [mask.sum() / npx], cen])
        G.add_node(int(lbl), features=f, size=int(mask.sum()),
                   pos=tuple(np.argwhere(mask).mean(axis=0)))

    cand, best_neighbor = [], {}
    for (u, v), blen in boundary.items():
        d = float(np.linalg.norm(G.nodes[u]["features"][:3] - G.nodes[v]["features"][:3]))
        dt = abs(grad_lbl.get(u, 0.0) - grad_lbl.get(v, 0.0)) if cfg.edge_texture else 0.0
        cand.append((u, v, d, blen, dt))
        w = float(np.exp(-d * d / (2 * cfg.sigma_feat ** 2)))
        for a, b in ((u, v), (v, u)):
            if a not in best_neighbor or w > best_neighbor[a][1]:
                best_neighbor[a] = (b, w)

    tau = _threshold(np.array([d for _, _, d, _, _ in cand]), cfg)
    tau_t = _threshold(np.array([dt for *_, dt in cand]), cfg) if cfg.edge_texture else None
    for u, v, d, blen, dt in cand:
        if d <= tau and (tau_t is None or dt <= tau_t):
            color_w = float(np.exp(-d * d / (2 * cfg.sigma_feat ** 2)))
            G.add_edge(u, v, weight=(blen / max_b) * color_w)
    _ensure_connected(G, best_neighbor)
    return G


GRAPH_BUILDERS = {"pixel": build_pixel_graph, "patch": build_patch_graph, "slic": build_slic_graph}


def base_gt(gt: str) -> str:
    """Bazowy typ grafu bez sufiksu '-tex' (np. 'slictex' -> 'slic')."""
    return gt[:-3] if gt.endswith("tex") else gt


def make_graph(image, gt: str, cfg: Config) -> nx.Graph:
    """Buduje graf danego typu. Sufiks 'tex' = mocniejszy warunek krawędzi
    (podobny kolor ORAZ tekstura) — jako NOWY typ obok oryginalnego, nic nie tracimy."""
    tex = gt.endswith("tex")
    c = replace(cfg, edge_texture=True) if tex else cfg
    return GRAPH_BUILDERS[base_gt(gt)](image, c)


# ============================================================================
# NODE2VEC + AGREGACJA (fuzja cech: embedding strukturalny ⊕ atrybuty węzła)
# ============================================================================
def node2vec_embeddings(G: nx.Graph, cfg: Config) -> np.ndarray:
    if G.number_of_edges() == 0:
        return np.zeros((G.number_of_nodes(), cfg.dim), float)
    n2v = Node2Vec(G, dimensions=cfg.dim, walk_length=cfg.walk_length,
                   num_walks=cfg.num_walks, p=cfg.p, q=cfg.q, workers=1,
                   quiet=True, seed=cfg.seed)
    model = n2v.fit(window=cfg.window, min_count=1, batch_words=4, seed=cfg.seed)
    return np.array([model.wv[str(n)] for n in sorted(G.nodes())])


def _pool(G: nx.Graph, M: np.ndarray, nodes, method: str, image_shape) -> np.ndarray:
    """Pooling macierzy węzeł×cecha do jednego wektora (M to embedding ALBO atrybuty)."""
    dim = M.shape[1]
    if method == "spatial_quadrants":
        H, W = image_shape[:2]
        mh, mw = H // 2, W // 2
        quads = [[], [], [], []]
        for k, n in enumerate(nodes):
            r, c = G.nodes[n]["pos"]
            qi = (0 if r < mh else 2) + (0 if c < mw else 1)
            quads[qi].append(M[k])
        return np.concatenate([np.mean(q, axis=0) if q else np.zeros(dim) for q in quads])
    if method == "weighted_mean":
        sizes = np.array([G.nodes[n].get("size", 1) for n in nodes], float)
        sizes = sizes / sizes.sum()
        return np.average(M, axis=0, weights=sizes)
    return np.mean(M, axis=0)


def aggregate_streams(G: nx.Graph, node_emb: np.ndarray, method: str, image_shape):
    """OSOBNO poolinguje strumień strukturalny (Node2Vec) i atrybutowy (cechy węzła),
    żeby później wyważyć ich wkład — surowa konkatenacja topiła kolor w 128 wymiarach
    Node2Vec (efekt rozcieńczenia, patrz analiza A/B)."""
    nodes = sorted(G.nodes())
    feats = np.array([G.nodes[n]["features"] for n in nodes])
    struct = _pool(G, node_emb, nodes, method, image_shape)
    attr = _pool(G, feats, nodes, method, image_shape)
    return struct, attr


def _pooling_for(graph_type: str) -> str:
    return {"pixel": "spatial_quadrants", "slic": "weighted_mean"}.get(base_gt(graph_type), "mean")


def process_single_image(idx, image, label, graph_type, cfg: Config):
    try:
        G = make_graph(image, graph_type, cfg)
        emb = node2vec_embeddings(G, cfg)
        struct, attr = aggregate_streams(G, emb, _pooling_for(graph_type), image.shape)
        return (idx, np.asarray(struct, np.float32), np.asarray(attr, np.float32),
                int(label), (G.number_of_nodes(), G.number_of_edges()))
    except Exception as e:
        print(f"  [!] obraz {idx} ({graph_type}): {type(e).__name__}: {e}", file=sys.stderr)
        return None


# ============================================================================
# CAŁOGRAFOWE EMBEDDINGI — NOWE METODY (WL / graph2vec) — bez uśredniania węzłów
# Pomysł: zamiast uśredniać embeddingi węzłów (co topiło strukturę), liczymy
# JEDEN wektor na cały graf wprost ze wzorców WL — tak jak w tu_graph_clustering.py,
# gdzie to pobiło Node2Vec. Kolor węzła kwantyzujemy na dyskretne etykiety WL.
# ============================================================================
def _build_one_graph(image, label, graph_type, cfg: Config):
    try:
        G = make_graph(image, graph_type, cfg)
        return G, int(label)
    except Exception as e:
        print(f"  [!] graf ({graph_type}): {type(e).__name__}: {e}", file=sys.stderr)
        return None


def build_graphs(images, labels, graph_type, cfg: Config):
    res = Parallel(n_jobs=cfg.n_jobs, verbose=0)(
        delayed(_build_one_graph)(images[i], labels[i], graph_type, cfg)
        for i in range(len(images)))
    res = [r for r in res if r is not None]
    return [r[0] for r in res], np.array([r[1] for r in res])


def assign_color_labels(graphs, cfg: Config):
    """Globalny KMeans na cechach węzłów -> dyskretna etykieta 'label' (wspólny
    słownik dla korpusu) -> seed dla WL/graph2vec. Domyślnie kwantyzacja po samym
    KOLORZE (pierwsze 3 cechy); przy `--label-rich` po CAŁYM (zestandaryzowanym)
    deskryptorze (kolor+tekstura+kształt) — bogatszy seed dla struktury."""
    def vec(G, n):
        f = np.atleast_1d(G.nodes[n]["features"]).astype(float)
        return f if cfg.label_rich else f[:3]
    per_graph = [(list(G.nodes()), np.array([vec(G, n) for n in G.nodes()])) for G in graphs]
    stacked = np.vstack([c for _, c in per_graph])
    if cfg.label_rich:                       # standaryzacja, by tekstura/kształt nie ginęły przy kolorze
        mu, sd = stacked.mean(0), stacked.std(0) + 1e-6
        stacked = (stacked - mu) / sd
        per_graph = [(nodes, (cols - mu) / sd) for nodes, cols in per_graph]
    rng = np.random.default_rng(cfg.seed)
    sample = stacked[rng.choice(len(stacked), min(20000, len(stacked)), replace=False)]
    km = KMeans(n_clusters=cfg.n_color_bins, n_init=5, random_state=cfg.seed).fit(sample)
    for (nodes, cols), G in zip(per_graph, graphs):
        for n, l in zip(nodes, km.predict(cols)):
            G.nodes[n]["label"] = int(l)


def _wl_relabel_docs(graphs, iterations):
    """Wspólny dla WL i graph2vec: iteracyjne przeetykietowanie WL.
    Zwraca (docs, dicts) — docs = listy tokenów na graf, dicts = liczniki wzorców."""
    node_labels = [{n: str(G.nodes[n].get("label", G.degree(n))) for n in G.nodes()} for G in graphs]
    docs = [list(nl.values()) for nl in node_labels]
    dicts = [Counter({f"l_{l}": c for l, c in Counter(nl.values()).items()}) for nl in node_labels]
    pat2id: dict[str, int] = {}
    for it in range(iterations):
        new = []
        for gi, G in enumerate(graphs):
            nl = node_labels[gi]; nn = {}
            for nd in G.nodes():
                pat = nl[nd] + "|" + ",".join(sorted(nl[nb] for nb in G.neighbors(nd)))
                pid = pat2id.setdefault(pat, len(pat2id)); nn[nd] = str(pid)
                dicts[gi][f"wl{it}_{pid}"] += 1
            new.append(nn); docs[gi].extend(nn.values())
        node_labels = new
    return docs, dicts


def embed_wl(graphs, cfg: Config) -> np.ndarray:
    """WL feature map (liczby wzorców) -> TruncatedSVD do gęstej reprezentacji."""
    _, dicts = _wl_relabel_docs(graphs, cfg.wl_iterations)
    M = DictVectorizer(sparse=True).fit_transform(dicts)
    k = min(cfg.g2v_dim, M.shape[1] - 1, M.shape[0] - 1)
    return TruncatedSVD(n_components=max(2, k), random_state=cfg.seed).fit_transform(M)


def embed_graph2vec(graphs, cfg: Config) -> np.ndarray:
    """graph2vec: te same wzorce WL, ale UCZONE gęsto przez Doc2Vec."""
    docs, _ = _wl_relabel_docs(graphs, cfg.wl_iterations)
    tagged = [TaggedDocument(words=(d if d else ["empty"]), tags=[str(i)])
              for i, d in enumerate(docs)]
    model = Doc2Vec(tagged, vector_size=cfg.g2v_dim, dm=0, min_count=1,
                    epochs=cfg.g2v_epochs, workers=1, seed=cfg.seed)
    return np.vstack([model.dv[str(i)] for i in range(len(graphs))])


def attr_pool_matrix(graphs, graph_type, cfg: Config) -> np.ndarray:
    """Pooling atrybutów węzła (kolor+tekstura) na graf — strumień atrybutowy."""
    method = _pooling_for(graph_type)
    rows = []
    for G in graphs:
        nodes = sorted(G.nodes())
        feats = np.array([G.nodes[n]["features"] for n in nodes])
        rows.append(_pool(G, feats, nodes, method, image_shape=(32, 32)))
    return np.array(rows)


def graph_topo_features(G: nx.Graph) -> np.ndarray:
    """Czysto STRUKTURALNE cechy grafu (graf-natywne, bez koloru). Najważniejsze:
    liczba i rozkład rozmiarów KOMPONENTÓW po przecięciu krawędzi — to bezpośredni
    skutek twardego warunku na krawędź (obraz rozpada się na regiony ~obiekty)."""
    n, m = G.number_of_nodes(), G.number_of_edges()
    if n == 0:
        return np.zeros(13, float)
    degs = np.array([d for _, d in G.degree()], float)
    sizes = np.array([len(c) for c in nx.connected_components(G)], float)
    try:
        asr = nx.degree_assortativity_coefficient(G); asr = 0.0 if np.isnan(asr) else asr
    except Exception:
        asr = 0.0
    return np.array([
        n, m, m / n, nx.density(G) if n > 1 else 0.0,
        len(sizes), sizes.mean(), sizes.std(), sizes.max(), sizes.max() / n,  # fragmentacja
        degs.mean(), degs.std(),
        nx.average_clustering(G) if n > 2 else 0.0, asr,
    ], float)


def topo_matrix(graphs) -> np.ndarray:
    return np.vstack([graph_topo_features(G) for G in graphs])


def spectral_features(G: nx.Graph, k: int) -> np.ndarray:
    """Sygnatura spektralna: k najmniejszych wartości własnych znormalizowanego
    Laplasjanu. Graf-natywny, klasyczny deskryptor kształtu/łączności grafu
    (liczba bliskich zeru wartości ≈ liczba komponentów — komplementarne do `topo`)."""
    n = G.number_of_nodes()
    if n < 2:
        return np.zeros(k)
    L = nx.normalized_laplacian_matrix(G)
    if n <= 400:
        vals = np.linalg.eigvalsh(L.toarray())
    else:
        try:
            from scipy.sparse.linalg import eigsh
            vals = eigsh(L.asfptype(), k=min(k, n - 2), which="SM", return_eigenvectors=False)
        except Exception:
            vals = np.linalg.eigvalsh(L.toarray())
    vals = np.sort(np.real(vals))[:k]
    if len(vals) < k:
        vals = np.pad(vals, (0, k - len(vals)), constant_values=(vals[-1] if len(vals) else 0.0))
    return vals


def spectral_matrix(graphs, k) -> np.ndarray:
    return np.vstack([spectral_features(G, k) for G in graphs])


# ============================================================================
# BASELINE'Y
# ============================================================================
def baseline_rgb(image):
    return image.mean(axis=(0, 1))


def baseline_hog(image):
    return hog(image, orientations=8, pixels_per_cell=(8, 8),
               cells_per_block=(2, 2), channel_axis=-1, feature_vector=True)


# ============================================================================
# EWALUACJA
# ============================================================================
def _prep(X, scale):
    X = np.nan_to_num(np.asarray(X, float))
    return StandardScaler().fit_transform(X) if scale else X


def fuse(Xs, Xa, w_struct):
    """Wyważona fuzja dwóch strumieni: każdy blok osobno standaryzowany i
    L2-normalizowany (każdy wnosi wektor jednostkowy, niezależnie od liczby
    wymiarów), potem skalowany wagą. w=0 -> tylko atrybuty, w=1 -> tylko struktura."""
    Zs = normalize(_prep(Xs, True))
    Za = normalize(_prep(Xa, True))
    return np.hstack([w_struct * Zs, (1.0 - w_struct) * Za])


def evaluate_classifiers(X, y, cfg: Config, scale=True) -> dict:
    X = _prep(X, scale)
    counts = np.bincount(y)
    folds = int(min(cfg.cv_folds, counts[counts > 0].min()))
    if folds < 2:
        return {}
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=cfg.seed)
    models = {
        "LogReg": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=cfg.seed),
        "SVM": SVC(kernel="rbf", random_state=cfg.seed),
    }
    out = {}
    for name, clf in models.items():
        s = cross_validate(clf, X, y, cv=cv, scoring=("accuracy", "f1_macro"))
        out[name] = (float(s["test_accuracy"].mean()), float(s["test_f1_macro"].mean()))
    return out


def evaluate_unsupervised(X, y, n_classes, cfg: Config, scale=True) -> dict:
    X = _prep(X, scale)
    km = KMeans(n_clusters=n_classes, n_init=10, random_state=cfg.seed).fit_predict(X)
    res = {"ARI": float(adjusted_rand_score(y, km)),
           "NMI": float(normalized_mutual_info_score(y, km))}
    if len(np.unique(km)) > 1:
        res["silhouette"] = float(silhouette_score(X, km))
    return res


# ============================================================================
# PIPELINE
# ============================================================================
def _cache_key(graph_type, cfg: Config, n, extra=()) -> str:
    parts = [graph_type, cfg.classes, cfg.per_class, cfg.num_samples, cfg.edge_quantile,
             cfg.tau, cfg.dim, cfg.walk_length, cfg.num_walks, cfg.p, cfg.q, cfg.n_segments,
             cfg.patch_size, cfg.stride, cfg.k_neighbors, cfg.knn_radius, cfg.seed, n, *extra]
    if cfg.rich_features:   # tylko gdy włączone -> nie unieważnia istniejących cache'y "plain"
        parts += ["rich", cfg.n_orient_bins]
    if cfg.label_rich:
        parts += ["lblrich"]
    h = hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:10]
    return f"{graph_type}_{h}"


def build_features(images, labels, graph_type, cfg: Config):
    """Zwraca (Xstruct, Xattr, y). Embeddingi są cache'owane na dysku — Node2Vec
    liczymy RAZ, a przemiatanie wag/skalowania jest potem natychmiastowe."""
    cdir = os.path.join(cfg.out_dir, "cache"); os.makedirs(cdir, exist_ok=True)
    key = _cache_key(graph_type, cfg, len(images))
    paths = {s: os.path.join(cdir, f"{key}_{s}.npy") for s in ("struct", "attr", "y")}
    if all(os.path.exists(p) for p in paths.values()):
        print(f"  [{graph_type}] wczytano z cache ({key})")
        return np.load(paths["struct"]), np.load(paths["attr"]), np.load(paths["y"])

    t0 = time.time()
    res = Parallel(n_jobs=cfg.n_jobs, verbose=0)(
        delayed(process_single_image)(i, images[i], labels[i], graph_type, cfg)
        for i in range(len(images)))
    res = [r for r in res if r is not None]
    Xs = np.array([r[1] for r in res]); Xa = np.array([r[2] for r in res])
    y = np.array([r[3] for r in res]); sizes = np.array([r[4] for r in res])
    print(f"  [{graph_type}] {len(res)} obrazów, |V|~{int(sizes[:,0].mean())} "
          f"|E|~{int(sizes[:,1].mean())}, {time.time()-t0:.1f}s")
    np.save(paths["struct"], Xs); np.save(paths["attr"], Xa); np.save(paths["y"], y)
    return Xs, Xa, y


def build_graph_reps(images, labels, graph_type, cfg: Config):
    """Całografowe reprezentacje (NOWE): zwraca (Xwl, Xg2v, Xattr, Xtopo, y). Cache na dysku."""
    cdir = os.path.join(cfg.out_dir, "cache"); os.makedirs(cdir, exist_ok=True)
    key = _cache_key("rep_" + graph_type, cfg, len(images),
                     extra=(cfg.wl_iterations, cfg.g2v_dim, cfg.g2v_epochs, cfg.n_color_bins))
    cols = ("wl", "g2v", "attr", "topo", "spec", "y")
    paths = {s: os.path.join(cdir, f"{key}_{s}.npy") for s in cols}
    if all(os.path.exists(p) for p in paths.values()):
        print(f"  [rep-{graph_type}] wczytano z cache ({key})")
        return tuple(np.load(paths[s]) for s in cols)

    t0 = time.time()
    graphs, y = build_graphs(images, labels, graph_type, cfg)
    assign_color_labels(graphs, cfg)
    Xwl = embed_wl(graphs, cfg)
    Xg2v = embed_graph2vec(graphs, cfg)
    Xattr = attr_pool_matrix(graphs, graph_type, cfg)
    Xtopo = topo_matrix(graphs)
    Xspec = spectral_matrix(graphs, cfg.n_spectral)
    print(f"  [rep-{graph_type}] {len(graphs)} grafów, wl{Xwl.shape[1]}d / g2v{Xg2v.shape[1]}d / "
          f"attr{Xattr.shape[1]}d / topo{Xtopo.shape[1]}d / spec{Xspec.shape[1]}d, {time.time()-t0:.1f}s")
    for s, arr in zip(cols, (Xwl, Xg2v, Xattr, Xtopo, Xspec, y)):
        np.save(paths[s], arr)
    return Xwl, Xg2v, Xattr, Xtopo, Xspec, y


# ============================================================================
# WYKRESY — opowiadamy historię wyników
# ============================================================================
def _best_by_method(rows, key="acc"):
    """Dla każdej metody bierze konfigurację o najlepszej wartości `key`."""
    best = {}
    for r in rows:
        m = r["method"]
        if m not in best or r[key] > best[m][key]:
            best[m] = r
    return best


def _is_graph_method(name):
    return not name.startswith("baseline")


def plot_method_comparison(rows, n_classes, path):
    best = _best_by_method(rows, "acc")
    items = sorted(best.items(), key=lambda kv: kv[1]["acc"])
    names = [k for k, _ in items]
    accs = [v["acc"] for _, v in items]
    colors = ["#4C9F70" if _is_graph_method(n) else "#999999" for n in names]
    plt.figure(figsize=(8, max(4, 0.4 * len(names))))
    plt.barh(names, accs, color=colors)
    for i, a in enumerate(accs):
        plt.text(a + 0.005, i, f"{a:.2f}", va="center", fontsize=8)
    plt.axvline(1 / n_classes, ls="--", c="#cc4444", lw=1, label=f"losowo ({1/n_classes:.2f})")
    plt.axvline(0.7, ls=":", c="#3366cc", lw=1, label="próg 0.7")
    plt.xlabel("najlepsza dokładność (5-fold CV)")
    plt.title("Porównanie metod — najlepsza dokładność")
    plt.legend(fontsize=8, loc="lower right"); plt.tight_layout()
    plt.savefig(path, dpi=130); plt.close()


def plot_weight_sweep(rows, path):
    from collections import defaultdict
    data = defaultdict(dict)   # method -> w -> {'acc','ARI'}
    for r in rows:
        if r["w"] != r["w"]:   # NaN => metoda bez przemiatania wagi
            continue
        d = data[r["method"]].get(r["w"])
        if d is None or r["acc"] > d["acc"]:
            data[r["method"]][r["w"]] = {"acc": r["acc"], "ARI": r["ARI"]}
    if not data:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for m, wd in sorted(data.items()):
        ws = sorted(wd)
        ax1.plot(ws, [wd[w]["acc"] for w in ws], "o-", label=m)
        ax2.plot(ws, [wd[w]["ARI"] for w in ws], "o-", label=m)
    ax1.set(xlabel="waga struktury  w  (0=tylko atrybuty, 1=tylko struktura)",
            ylabel="dokładność", title="Dokładność vs waga struktury")
    ax2.set(xlabel="waga struktury  w", ylabel="ARI", title="Klasteryzacja (ARI) vs waga struktury")
    ax1.legend(fontsize=8); ax2.legend(fontsize=8)
    ax1.grid(alpha=.3); ax2.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


def plot_acc_vs_ari(rows, n_classes, path):
    acc_best = _best_by_method(rows, "acc")
    ari_best = _best_by_method(rows, "ARI")
    plt.figure(figsize=(7.5, 6))
    for m in acc_best:
        x = acc_best[m]["acc"]; y = ari_best[m]["ARI"]
        graph = _is_graph_method(m)
        plt.scatter(x, y, s=70, c="#4C9F70" if graph else "#999999",
                    marker="o" if graph else "s", edgecolors="k", zorder=3)
        plt.annotate(m, (x, y), fontsize=7, xytext=(4, 4), textcoords="offset points")
    plt.axvline(1 / n_classes, ls="--", c="#cc4444", lw=1)
    plt.axvline(0.7, ls=":", c="#3366cc", lw=1)
    plt.xlabel("najlepsza dokładność  (separowalność, nadzorowana)")
    plt.ylabel("najlepsze ARI  (klasteryzowalność, NIEnadzorowana)")
    plt.title("Separowalność vs klasteryzowalność\n(graf = ●, baseline = ■)")
    plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


def plot_progression(rows, gt, hog_acc, n_classes, path):
    """Historia usprawnień: najlepsza dokładność po kolei, jak dokładaliśmy poprawki.
    Pokazuje DOKŁADNIE co dało każde ulepszenie (oś = kolejne wersje metody)."""
    steps = [
        (f"n2vflat-{gt}", "0. naiwna fuzja\n(rozcieńczenie)"),
        (f"n2v-{gt}", "1. rozdz. strumienie\n+ waga"),
        (f"g2v-{gt}", "2. graph2vec\n(całografowy)"),
        (f"combo-{gt}", "3. combo\n(g2v+kolor)"),
        (f"combo+r-{gt}", "4. combo+mini-HOG\n(bogaty węzeł)"),
        (f"gnat+r-{gt}", "5. graf-natywne\n(g2v+topo+węzeł)"),
        (f"hyb-{gt}", "6. hybryda\nHOG+g2v"),
        (f"hyb+r-{gt}", "7. hybryda+r\nHOG+g2v+r"),
    ]
    best = _best_by_method(rows, "acc")
    xs, ys, labs = [], [], []
    for m, lab in steps:
        if m in best:
            xs.append(len(xs)); ys.append(best[m]["acc"]); labs.append(lab)
    if not xs:
        return
    plt.figure(figsize=(max(8, 1.5 * len(xs)), 5))
    plt.plot(xs, ys, "o-", color="#4C9F70", lw=2, ms=8, zorder=3)
    for x, yv in zip(xs, ys):
        plt.text(x, yv + 0.012, f"{yv:.3f}", ha="center", fontsize=9)
    if hog_acc is not None:
        plt.axhline(hog_acc, ls="--", c="#cc6600", lw=1.5, label=f"baseline HOG ({hog_acc:.3f})")
    plt.axhline(1 / n_classes, ls=":", c="#cc4444", lw=1, label=f"losowo ({1/n_classes:.2f})")
    plt.axhline(0.7, ls=":", c="#3366cc", lw=1, label="próg 0.7")
    plt.xticks(xs, labs, fontsize=8)
    plt.ylabel("najlepsza dokładność (5-fold CV)")
    plt.title(f"Historia usprawnień — graf {gt}")
    plt.legend(fontsize=8, loc="lower right"); plt.grid(alpha=.3, axis="y")
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


def plot_morphospace(embs: dict, y, class_names, path):
    n = len(embs)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.6), squeeze=False)
    for ax, (name, X) in zip(axes[0], embs.items()):
        Z = PCA(n_components=2, random_state=0).fit_transform(
            StandardScaler().fit_transform(np.nan_to_num(np.asarray(X, float))))
        for lab in sorted(set(y)):
            pts = Z[y == lab]
            ax.scatter(pts[:, 0], pts[:, 1], s=12, alpha=.6, label=class_names[lab])
        ax.set(title=name, xlabel="PC1", ylabel="PC2")
    axes[0][-1].legend(fontsize=7, markerscale=1.5)
    plt.suptitle("Morfoprzestrzeń (PCA 2D, kolor = prawdziwa klasa)")
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


def run(cfg: Config, graph_types):
    classes = cfg.classes if cfg.classes is not None else list(range(10))
    names = [CIFAR_CLASSES[c] for c in classes]
    print(f"Klasy ({len(classes)}): {names}")
    images, labels = load_cifar10(cfg)
    print(f"Obrazów: {len(images)} | liczności: {np.bincount(labels).tolist()}\n")

    n_classes = len(classes)
    rows = []
    morpho = {}   # do morfoprzestrzeni: najlepszy embedding grafowy per typ grafu
    Xhog_all = np.array([baseline_hog(im) for im in images])   # cechy wyglądu (do baseline i hybrydy)
    Xrgb_all = np.array([baseline_rgb(im) for im in images])

    def add(method, w, X, yy, scale):
        clf = evaluate_classifiers(X, yy, cfg, scale=scale)
        uns = evaluate_unsupervised(X, yy, n_classes, cfg, scale=scale)
        for mname, (acc, f1) in clf.items():
            rows.append({"method": method, "w": w, "clf": mname, "acc": acc, "f1": f1,
                         "ARI": uns.get("ARI", float("nan")), "silhouette": uns.get("silhouette", float("nan"))})

    cfg_plain = replace(cfg, rich_features=False)
    for gt in graph_types:
        # --- metoda 0 (BASELINE GRAFOWY): naiwna płaska fuzja Node2Vec+kolor ---
        # standaryzowana RAZEM => 128 wym. struktury topi kolor (efekt rozcieńczenia).
        # Zostawiamy ją na stałe jako punkt odniesienia "przed poprawką".
        Xs, Xa, y = build_features(images, labels, gt, cfg_plain)
        add(f"n2vflat-{gt}", float("nan"), np.hstack([Xs, Xa]), y, scale=True)

        # --- metoda 1: Node2Vec + rozdzielone strumienie + przemiatanie wagi ---
        for w in cfg.weights:
            add(f"n2v-{gt}", w, fuse(Xs, Xa, w), y, scale=False)

        # --- metody 2-6 (całografowe WL / graph2vec + topo + spec + combo + gnat + hybryda) ---
        Xwl, Xg2v, Xattr, Xtopo, Xspec, y = build_graph_reps(images, labels, gt, cfg_plain)
        add(f"wl-{gt}", float("nan"), Xwl, y, scale=True)
        add(f"g2v-{gt}", float("nan"), Xg2v, y, scale=True)
        add(f"topo-{gt}", float("nan"), Xtopo, y, scale=True)   # czysto strukturalne (fragmentacja)
        add(f"spec-{gt}", float("nan"), Xspec, y, scale=True)   # sygnatura spektralna (Laplasjan)
        for w in cfg.weights:
            add(f"combo-{gt}", w, fuse(Xg2v, Xattr, w), y, scale=False)
        # gnat = GRAF-NATYWNE: struktura (graph2vec ⊕ topo) ⊕ atrybuty węzła. ZERO HOG.
        Xstruct = np.hstack([Xg2v, Xtopo])
        for w in cfg.weights:
            add(f"gnat-{gt}", w, fuse(Xstruct, Xattr, w), y, scale=False)
        # gspec = gnat + sygnatura spektralna w bloku struktury (też ZERO HOG)
        Xstruct_s = np.hstack([Xg2v, Xtopo, Xspec])
        for w in cfg.weights:
            add(f"gspec-{gt}", w, fuse(Xstruct_s, Xattr, w), y, scale=False)
        morpho[gt] = (fuse(Xg2v, Xattr, 0.5), y)   # combo w=0.5 do wizualizacji
        if len(Xhog_all) == len(y):
            for w in cfg.weights:
                add(f"hyb-{gt}", w, fuse(Xg2v, Xhog_all, w), y, scale=False)

        # --- metoda 7 (NOWE USPRAWNIENIE): bogatszy deskryptor węzła (mini-HOG) ---
        # combo+r / gnat+r / gspec+r liczone na grafie z deskryptorem orientacji gradientu
        # na superpiksel — czy struktura grafowa rywalizuje z HOG BEZ pożyczania HOG.
        if cfg.rich_features:
            _, Xg2v_r, Xattr_r, Xtopo_r, Xspec_r, y = build_graph_reps(images, labels, gt, replace(cfg, rich_features=True))
            Xstruct_r = np.hstack([Xg2v_r, Xtopo_r])
            Xstruct_sr = np.hstack([Xg2v_r, Xtopo_r, Xspec_r])
            for w in cfg.weights:
                add(f"combo+r-{gt}", w, fuse(Xg2v_r, Xattr_r, w), y, scale=False)
                add(f"gnat+r-{gt}", w, fuse(Xstruct_r, Xattr_r, w), y, scale=False)
                add(f"gspec+r-{gt}", w, fuse(Xstruct_sr, Xattr_r, w), y, scale=False)
            morpho[gt] = (fuse(Xstruct_sr, Xattr_r, 0.5), y)
            if len(Xhog_all) == len(y):
                for w in cfg.weights:
                    add(f"hyb+r-{gt}", w, fuse(Xg2v_r, Xhog_all, w), y, scale=False)

    # baseline'y (skalowane standardowo)
    for bname, Xb in (("baseline-rgb", Xrgb_all), ("baseline-hog", Xhog_all)):
        add(bname, float("nan"), Xb, labels, scale=True)

    # tabela
    print(f"\n{'='*80}\nWYNIKI (poziom losowy acc = {1/n_classes:.3f}; w = waga strumienia strukturalnego)\n{'='*80}")
    head = f"{'method':<14}{'w':>5}{'clf':<14}{'acc':>8}{'f1':>8}{'ARI':>8}{'silh':>8}"
    print(head); print("-" * len(head))
    for r in sorted(rows, key=lambda r: -r["acc"]):
        flag = "  <-- >0.7" if r["acc"] >= 0.7 else ""
        wv = "  -" if r["w"] != r["w"] else f"{r['w']:>5.2f}"
        print(f"{r['method']:<14}{wv:>5}{r['clf']:<14}{r['acc']:>8.3f}{r['f1']:>8.3f}"
              f"{r['ARI']:>8.3f}{r['silhouette']:>8.3f}{flag}")

    os.makedirs(cfg.out_dir, exist_ok=True)
    csv_path = os.path.join(cfg.out_dir, "cifar_porownanie.csv")
    with open(csv_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["method", "w", "clf", "acc", "f1", "ARI", "silhouette"])
        wr.writeheader(); wr.writerows(rows)
    print(f"\nZapisano: {csv_path}")

    if cfg.plots:
        tag = "_".join(map(str, classes)) if n_classes <= 4 else f"{n_classes}klas"
        hog_acc = max((r["acc"] for r in rows if r["method"] == "baseline-hog"), default=None)
        plot_method_comparison(rows, n_classes, os.path.join(cfg.out_dir, f"fig_porownanie_{tag}.png"))
        plot_weight_sweep(rows, os.path.join(cfg.out_dir, f"fig_waga_{tag}.png"))
        plot_acc_vs_ari(rows, n_classes, os.path.join(cfg.out_dir, f"fig_acc_vs_ari_{tag}.png"))
        for gt in graph_types:
            plot_progression(rows, gt, hog_acc, n_classes, os.path.join(cfg.out_dir, f"fig_postep_{gt}_{tag}.png"))
        for gt, (Xc, yc) in morpho.items():
            plot_morphospace({f"combo-{gt} (struktura+kolor)": Xc, "HOG (wygląd)": Xhog_all},
                             yc, names, os.path.join(cfg.out_dir, f"fig_morfo_{gt}_{tag}.png"))
        print(f"Zapisano wykresy: {cfg.out_dir}/fig_*.png")
    return rows


def main():
    ap = argparse.ArgumentParser(description="CIFAR-10 jako grafy (Node2Vec) — Lista 3.")
    ap.add_argument("--graph-type", nargs="+",
                    choices=["pixel", "patch", "slic", "pixeltex", "slictex", "all"], default=["all"],
                    help="jeden lub kilka typów grafu; sufiks 'tex' = mocniejszy warunek krawędzi (kolor ORAZ tekstura)")
    ap.add_argument("--classes", type=int, nargs="+", default=None,
                    help="podzbiór klas 0-9 (domyślnie wszystkie 10)")
    ap.add_argument("--per-class", type=int, default=None, help="liczba obrazów na klasę")
    ap.add_argument("--num-samples", type=int, default=None, help="łączna liczba próbek")
    ap.add_argument("--tau", type=float, default=None, help="stały próg odległości krawędzi")
    ap.add_argument("--edge-quantile", type=float, default=0.6,
                    help="kwantyl odległości — zostaw krawędzie poniżej (domyślnie 0.6)")
    ap.add_argument("--weights", type=float, nargs="+", default=None,
                    help="wagi strumienia Node2Vec do przemiecenia (0=tylko atrybuty, 1=tylko struktura)")
    ap.add_argument("--plots", action="store_true", help="zapisz wykresy (fig_*.png) do out-dir")
    ap.add_argument("--rich-features", action="store_true",
                    help="dolicz bogatszy deskryptor węzła (mini-HOG) -> metody combo+r / hyb+r")
    ap.add_argument("--wl-iter", type=int, default=None, help="liczba iteracji WL (graph2vec)")
    ap.add_argument("--g2v-dim", type=int, default=None, help="wymiar embeddingu graph2vec")
    ap.add_argument("--n-orient-bins", type=int, default=None, help="kubełki histogramu orientacji (mini-HOG)")
    ap.add_argument("--label-rich", action="store_true",
                    help="seed WL/graph2vec po pełnym deskryptorze węzła (kolor+tekstura), nie samym kolorze")
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--out-dir", default="results_cifar")
    args = ap.parse_args()

    cfg = Config(data_dir=args.data_dir, out_dir=args.out_dir, classes=args.classes,
                 per_class=args.per_class, num_samples=args.num_samples,
                 tau=args.tau, edge_quantile=args.edge_quantile, plots=args.plots,
                 rich_features=args.rich_features, label_rich=args.label_rich)
    if args.weights is not None:
        cfg.weights = args.weights
    if args.wl_iter is not None:
        cfg.wl_iterations = args.wl_iter
    if args.g2v_dim is not None:
        cfg.g2v_dim = args.g2v_dim
    if args.n_orient_bins is not None:
        cfg.n_orient_bins = args.n_orient_bins
    if cfg.per_class is None and cfg.num_samples is None:
        cfg.per_class = 80  # rozsądny domyślny rozmiar na klasę
    gts = []
    for g in args.graph_type:
        gts.extend(["pixel", "patch", "slic"] if g == "all" else [g])
    seen = set(); gts = [g for g in gts if not (g in seen or seen.add(g))]  # unikalne, kolejność
    run(cfg, gts)


if __name__ == "__main__":
    main()
