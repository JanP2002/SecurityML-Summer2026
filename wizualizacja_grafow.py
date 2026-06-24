#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Wizualizacje grafów — Lista 3 (CIFAR-10 + Białka/PROTEINS_full)
================================================================================
Szeroki przegląd wizualizacji, z NACISKIEM na nowy notatnik Jana
(`cifar_rand_graphs2_description.ipynb`): GRAF LOSOWY oparty na superpikselach SLIC z
probabilistycznym wstawianiem krawędzi (DropEdge) oraz skrótami small-world,
budowany jako ZESPÓŁ (ensemble) M losowych grafów na obraz.

Wszystkie buildery są wiernymi kopiami z notatnika Jana (oznaczone w komentarzach),
żeby wizualizacja pokazywała DOKŁADNIE jego konstrukcję. Nic w notatniku nie ruszamy.

Pytanie o bliskość przestrzenną, na które wprost odpowiada fig_05: w wersji probabilistycznej
bliskość przestrzenną gwarantuje sam zbiór kandydatów — krawędź rozważamy WYŁĄCZNIE
między superpikselami fizycznie graniczącymi (graf sąsiedztwa regionów, RAG);
losowość (rand < color_w) działa już TYLKO na tych sąsiadach i odrzuca pary o
niepodobnym kolorze, nigdy nie tworzy połączeń między odległymi regionami.

Uruchomienie:
  python wizualizacja_grafow.py                       # CIFAR (klasy 0 1 8) + Białka
  python wizualizacja_grafow.py --skip-proteins       # tylko CIFAR / Jan
  python wizualizacja_grafow.py --classes 0 1 8 9 --per-class 8 --out results_wiz
"""
from __future__ import annotations
import argparse, os, sys, urllib.request, warnings
warnings.filterwarnings("ignore")
from dataclasses import dataclass, field

import numpy as np
import networkx as nx

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, to_rgba
from matplotlib.lines import Line2D

from skimage.segmentation import slic, mark_boundaries

import torch  # noqa: F401  (wymagane przez torchvision)
import torchvision
import torchvision.transforms as transforms

# Polskie znaki na konsoli Windows (cp1252) — jak w pozostałych skryptach repo.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

CIFAR_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                 "dog", "frog", "horse", "ship", "truck"]
_GRAPHRNN = "https://raw.githubusercontent.com/snap-stanford/GraphRNN/master/dataset"


@dataclass
class Config:
    """Podzbiór konfiguracji z notatnika Jana, istotny dla wizualizacji."""
    data_dir: str = "./data"
    classes: list[int] = field(default_factory=lambda: [0, 1, 8])
    per_class: int = 8
    # parametry grafu losowego (Jan)
    probabilistic: bool = True          # DropEdge: rand < color_w zamiast twardego progu
    small_world_p: float = 0.05         # odsetek węzłów -> liczba dalekich skrótów
    ensemble_m: int = 6                 # M losowych grafów na obraz (wizualizujemy w fig_02)
    er_p: float = 0.2                   # gęstość czysto losowego grafu Erdősa-Rényiego
    tau: float | None = None
    edge_quantile: float = 0.6
    n_segments: int = 60
    compactness: float = 10.0
    sigma_pixel: float = 15.0
    sigma_feat: float = 1.0
    seed: int = 42


# ============================================================================
# DANE
# ============================================================================
def load_cifar10(cfg: Config):
    """Wierna kopia z notatnika Jana: wybór klas/próbek, obrazy w [0,1]."""
    transform = transforms.Compose([transforms.ToTensor()])
    ds = torchvision.datasets.CIFAR10(root=cfg.data_dir, train=True,
                                      download=True, transform=transform)
    targets = np.array(ds.targets)
    keep = cfg.classes if cfg.classes is not None else list(range(10))
    remap = {c: i for i, c in enumerate(keep)}
    rng = np.random.default_rng(cfg.seed)
    idxs = []
    for c in keep:
        pool = np.where(targets == c)[0]
        idxs.extend(rng.choice(pool, min(cfg.per_class, len(pool)), replace=False))
    idxs = np.array(sorted(int(i) for i in idxs))
    images, labels = [], []
    for i in idxs:
        img, lab = ds[int(i)]
        images.append(img.permute(1, 2, 0).numpy().astype(np.float32))
        labels.append(int(lab))            # zostawiamy ORYGINALNE id klasy (dla podpisów)
    return images, np.array(labels, dtype=int)


def download_tu(name: str, root: str) -> str:
    """Pobiera surowe pliki TU z mirrora GraphRNN (jeśli ich nie ma). Kopia z tu_graph_clustering.py."""
    dst = os.path.join(root, name); os.makedirs(dst, exist_ok=True)
    for f in ["A", "graph_indicator", "graph_labels", "node_labels", "node_attributes"]:
        path = os.path.join(dst, f"{name}_{f}.txt")
        if os.path.exists(path):
            continue
        url = f"{_GRAPHRNN}/{name}/{name}_{f}.txt"
        try:
            urllib.request.urlretrieve(url, path)
            print(f"  pobrano {name}_{f}.txt")
        except Exception as e:
            if f in ("node_labels", "node_attributes"):
                print(f"  (opcjonalny {name}_{f}.txt niedostępny: {e})")
            else:
                raise RuntimeError(f"Nie udało się pobrać {url}: {e}")
    return dst


def load_tu_dataset(name: str, root: str):
    """Zwraca (graphs, labels). Kopia z tu_graph_clustering.py."""
    d = os.path.join(root, name)
    if not os.path.exists(os.path.join(d, f"{name}_A.txt")):
        d = download_tu(name, root)
    p = lambda s: os.path.join(d, f"{name}_{s}.txt")
    A = np.loadtxt(p("A"), delimiter=",", dtype=int)
    indicator = np.loadtxt(p("graph_indicator"), dtype=int)
    glabels = np.loadtxt(p("graph_labels"), dtype=int)
    nlabels = np.loadtxt(p("node_labels"), dtype=int) if os.path.exists(p("node_labels")) else None
    gids = np.unique(indicator)
    G = {g: nx.Graph() for g in gids}
    for i in range(indicator.shape[0]):
        g = int(indicator[i]); attrs = {}
        if nlabels is not None:
            attrs["label"] = int(nlabels[i])
        G[g].add_node(i + 1, **attrs)
    for u, v in A:
        G[int(indicator[u - 1])].add_edge(int(u), int(v))
    graphs = [G[g] for g in gids]
    labels = [int(glabels[g - 1]) for g in gids]
    return graphs, labels


# ============================================================================
# GRAF LOSOWY SLIC (wierna rekonstrukcja z cifar_rand_graphs2_description.ipynb)
# Rozbity na etapy, żeby z JEDNEJ segmentacji narysować wszystkie warianty.
# ============================================================================
def _threshold(dists: np.ndarray, cfg: Config) -> float:
    if cfg.tau is not None:
        return cfg.tau
    if len(dists) == 0:
        return float("inf")
    return float(np.quantile(dists, cfg.edge_quantile))


def slic_prep(image: np.ndarray, cfg: Config):
    """SLIC -> węzły (mean⊕std, rozmiar, środek) + kandydaci RAG (sąsiedztwo z granic).
    To wspólny rdzeń: zwraca wszystko, czego potrzeba do dowolnego trybu krawędzi."""
    labels = slic(image, n_segments=cfg.n_segments, compactness=cfg.compactness,
                  start_label=0, channel_axis=-1)
    H, W = labels.shape
    # długości wspólnych granic = miara sąsiedztwa przestrzennego (RAG)
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

    nodes = {}
    for lbl in np.unique(labels):
        mask = labels == lbl
        f = np.concatenate([image[mask].mean(axis=0), image[mask].std(axis=0)])
        nodes[int(lbl)] = dict(features=f, size=int(mask.sum()),
                               pos=tuple(np.argwhere(mask).mean(axis=0)))

    cand, best_neighbor = [], {}
    for (u, v), blen in boundary.items():
        d = float(np.linalg.norm(nodes[u]["features"][:3] - nodes[v]["features"][:3]))
        cand.append((u, v, d, blen))
        w = float(np.exp(-d * d / (2 * cfg.sigma_feat ** 2)))
        for a, b in ((u, v), (v, u)):
            if a not in best_neighbor or w > best_neighbor[a][1]:
                best_neighbor[a] = (b, w)
    return labels, nodes, cand, max_b, best_neighbor


def _new_graph(nodes: dict) -> nx.Graph:
    G = nx.Graph()
    for n, a in nodes.items():
        G.add_node(n, **a)
    return G


def apply_small_world(G: nx.Graph, cfg: Config):
    """Kopia z notatnika Jana. Zwraca listę DODANYCH skrótów (do rysowania na czerwono)."""
    if cfg.small_world_p <= 0.0 or G.number_of_nodes() < 2:
        return []
    nodes = list(G.nodes())
    n_shortcuts = int(len(nodes) * cfg.small_world_p)
    sigma = cfg.sigma_pixel if len(nodes) > 400 else cfg.sigma_feat
    added = []
    for _ in range(n_shortcuts):
        u, v = np.random.choice(nodes, 2, replace=False)
        if not G.has_edge(u, v):
            fu = G.nodes[u]["features"][:3]; fv = G.nodes[v]["features"][:3]
            d = float(np.linalg.norm(fu - fv))
            prob = float(np.exp(-d * d / (2 * sigma ** 2)))
            if np.random.rand() < prob:
                G.add_edge(u, v, weight=prob); added.append((int(u), int(v)))
    return added


def build_from_prep(prep, cfg: Config, mode="prob", small_world=True, seed=None):
    """Z gotowego rdzenia SLIC zbuduj graf w danym trybie krawędzi.
    mode: 'rag' (wszystkie sąsiedztwa), 'det' (twardy próg d<=tau), 'prob' (rand<color_w).
    Zwraca (G, shortcuts, kept_flags) — kept_flags: dla każdego kandydata czy krawędź powstała."""
    labels, nodes, cand, max_b, best_neighbor = prep
    if seed is not None:
        np.random.seed(seed)
    G = _new_graph(nodes)
    tau = _threshold(np.array([d for _, _, d, _ in cand]), cfg)
    kept = []
    for u, v, d, blen in cand:
        color_w = float(np.exp(-d * d / (2 * cfg.sigma_feat ** 2)))
        edge_w = (blen / max_b) * color_w
        if mode == "rag":
            G.add_edge(u, v, weight=edge_w); k = True
        elif mode == "det":
            k = d <= tau
            if k:
                G.add_edge(u, v, weight=edge_w)
        else:  # prob
            k = np.random.rand() < color_w
            if k:
                G.add_edge(u, v, weight=edge_w)
        kept.append(k)
    # _ensure_connected (Jan): izolowany węzeł -> najbardziej podobny sąsiad
    for n in list(G.nodes()):
        if G.degree(n) == 0 and n in best_neighbor:
            nb, w = best_neighbor[n]; G.add_edge(n, nb, weight=w)
    shortcuts = apply_small_world(G, cfg) if small_world else []
    return G, shortcuts, kept


# ============================================================================
# POMOCNICZE RYSOWANIE
# ============================================================================
def _posxy(G: nx.Graph) -> dict:
    """pos=(row,col) -> (x=col, y=row); zgodne z imshow (origin='upper')."""
    return {n: (G.nodes[n]["pos"][1], G.nodes[n]["pos"][0]) for n in G}


def draw_overlay(ax, image, G, node_color="#ffd000", edge_color="#39ff14",
                 shortcuts=None, sizes=True, title=None, ns=None):
    """Rysuje graf na obrazie. Krawędzie przez LineCollection (szybko nawet dla
    grafu pikselowego ~1000 węzłów); dla gęstych grafów auto-zmniejsza węzły/linie."""
    ax.imshow(image, interpolation="nearest")
    pos = _posxy(G)
    sc = set(tuple(sorted(s)) for s in (shortcuts or []))
    n = G.number_of_nodes()
    dense = n > 300
    base = to_rgba(edge_color)
    segs, cols, lws = [], [], []
    for u, v, dd in G.edges(data=True):
        if tuple(sorted((u, v))) in sc:
            continue
        w = float(dd.get("weight", 0.5))
        a = min(0.97, 0.55 + 0.4 * w) * (0.6 if dense else 1.0)
        lw = (0.25 + 0.6 * w) if dense else (1.0 + 1.8 * w)
        segs.append([pos[u], pos[v]]); cols.append((base[0], base[1], base[2], a)); lws.append(lw)
    if segs:
        ax.add_collection(LineCollection(segs, colors=cols, linewidths=lws, zorder=2))
    if sc:                                   # skróty small-world na wierzchu, na czerwono
        ax.add_collection(LineCollection([[pos[u], pos[v]] for u, v in sc],
                          colors="#ff2d2d", linewidths=1.8, alpha=0.95, zorder=4))
    if ns is None:
        ns = 3 if dense else (9 if n > 120 else None)
    xs = [pos[k][0] for k in G]; ys = [pos[k][1] for k in G]
    if ns is None and sizes:
        s = [np.clip(G.nodes[k].get("size", 15) * 1.1, 8, 95) for k in G]
    else:
        s = ns if ns is not None else 20
    ax.scatter(xs, ys, s=s, c=node_color, edgecolors="k",
               linewidths=0.2 if dense else 0.3, zorder=3)
    ax.set_xlim(-0.5, image.shape[1] - 0.5); ax.set_ylim(image.shape[0] - 0.5, -0.5)
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)


def paint_labels(labels, valmap, cmap="viridis", vmin=None, vmax=None):
    """Maluje regiony superpikseli kolorem wg wartości na węzeł (komponent / stopień)."""
    uniq = np.unique(labels)
    vals = np.array([valmap.get(int(l), 0) for l in uniq], float)
    vmin = vals.min() if vmin is None else vmin
    vmax = max(vals.max(), vmin + 1e-9) if vmax is None else vmax
    mapper = plt.get_cmap(cmap)
    norm = (vals - vmin) / (vmax - vmin + 1e-9)
    out = np.zeros((*labels.shape, 3))
    for i, l in enumerate(uniq):
        out[labels == int(l)] = mapper(float(norm[i]))[:3]
    return out


def _sharp_boundaries(image, labels, scale=16, color=(1, 1, 0)):
    """Ostre granice superpikseli: powiększamy obraz i mapę etykiet (najbliższy sąsiad),
    żeby linie granic były CIENKIE względem powiększonego obrazu (czytelny podział ~Voronoi)."""
    big_img = np.kron(image, np.ones((scale, scale, 1)))
    big_lab = np.kron(labels, np.ones((scale, scale))).astype(int)
    return mark_boundaries(big_img, big_lab, color=color)


def _slic_mean_image(image, labels):
    """Każdy superpiksel zamalowany swoim średnim kolorem — pokazuje, co 'widzi' węzeł."""
    out = np.zeros_like(image)
    for lbl in np.unique(labels):
        m = labels == lbl
        out[m] = image[m].mean(axis=0)
    return out


def savefig(fig, out, name):
    path = os.path.join(out, name)
    fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  zapisano {path}")
    return path


def class_name(cid: int) -> str:
    return CIFAR_CLASSES[cid] if 0 <= cid < len(CIFAR_CLASSES) else str(cid)


def pick_one_per_class(images, labels):
    """Zwraca [(image, cid)] — pierwszy obraz każdej obecnej klasy (po id rosnąco)."""
    out = []
    for cid in sorted(set(int(x) for x in labels)):
        i = int(np.where(labels == cid)[0][0])
        out.append((images[i], cid))
    return out


# ============================================================================
# FIGURY — CIFAR / graf losowy Jana
# ============================================================================
def fig01_pipeline(images, labels, cfg, out):
    """Pełny potok: obraz -> superpiksele -> RAG (wszystkie sąsiedztwa) -> graf prob."""
    rows = pick_one_per_class(images, labels)
    n = len(rows)
    fig, ax = plt.subplots(n, 4, figsize=(11, 2.8 * n))
    if n == 1:
        ax = ax[None, :]
    for r, (img, cid) in enumerate(rows):
        prep = slic_prep(img, cfg)
        labels_s = prep[0]
        ax[r, 0].imshow(img); ax[r, 0].set_ylabel(class_name(cid), fontsize=10)
        ax[r, 0].set_xticks([]); ax[r, 0].set_yticks([])
        ax[r, 1].imshow(_sharp_boundaries(img, labels_s))
        ax[r, 1].set_xticks([]); ax[r, 1].set_yticks([])
        Grag, _, _ = build_from_prep(prep, cfg, mode="rag", small_world=False)
        draw_overlay(ax[r, 2], img, Grag, edge_color="#bbbbbb")
        Gp, sc, _ = build_from_prep(prep, cfg, mode="prob", small_world=True, seed=cfg.seed + r)
        draw_overlay(ax[r, 3], img, Gp, shortcuts=sc)
        if r == 0:
            for c, t in enumerate(["obraz", "superpiksele SLIC",
                                   "RAG: wszystkie sąsiedztwa", "graf LOSOWY (prob + small-world)"]):
                ax[0, c].set_title(t, fontsize=10)
    fig.suptitle("Potok grafu losowego Jana: obraz -> SLIC -> RAG -> próbkowanie krawędzi (DropEdge)\n"
                 "szare = wszystkie krawędzie sąsiedztwa; kolor = krawędzie wylosowane; czerwone = skróty small-world",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return savefig(fig, out, "fig_01_pipeline.png")


def fig02_ensemble(images, labels, cfg, out):
    """Klucz: graf jest LOSOWY — M niezależnych losowań tego samego obrazu różni się.
    Stąd ensemble (uśrednianie cech topologicznych po M grafach)."""
    img, cid = pick_one_per_class(images, labels)[0]
    prep = slic_prep(img, cfg)
    m = cfg.ensemble_m
    cols = 3; rows = int(np.ceil(m / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.4 * rows))
    ax = np.array(ax).reshape(rows, cols)
    for k in range(rows * cols):
        a = ax[k // cols, k % cols]
        if k >= m:
            a.axis("off"); continue
        G, sc, _ = build_from_prep(prep, cfg, mode="prob", small_world=True, seed=1000 + k)
        draw_overlay(a, img, G, shortcuts=sc,
                     title=f"losowanie #{k+1}: |E|={G.number_of_edges()}")
    fig.suptitle(f"Ensemble M={m} losowych grafów dla JEDNEGO obrazu ({class_name(cid)})\n"
                 "każde losowanie daje inny zbiór krawędzi -> cechy uśredniamy po zespole",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return savefig(fig, out, "fig_02_ensemble_losowosc.png")


def fig03_det_vs_prob(images, labels, cfg, out):
    img, cid = pick_one_per_class(images, labels)[0]
    prep = slic_prep(img, cfg)
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.8))
    ax[0].imshow(img); ax[0].set_title(f"obraz ({class_name(cid)})", fontsize=10)
    ax[0].set_xticks([]); ax[0].set_yticks([])
    Gd, _, _ = build_from_prep(prep, cfg, mode="det", small_world=False)
    draw_overlay(ax[1], img, Gd, edge_color="#39ff14",
                 title=f"DETERMINISTYCZNY  d<=tau  |E|={Gd.number_of_edges()}")
    Gp, sc, _ = build_from_prep(prep, cfg, mode="prob", small_world=False, seed=cfg.seed)
    draw_overlay(ax[2], img, Gp, edge_color="#ff2ec4",
                 title=f"PROBABILISTYCZNY  rand<color_w  |E|={Gp.number_of_edges()}")
    fig.suptitle("Twardy próg vs DropEdge: w wersji losowej krawędź sąsiada zostaje z prawd. = podobieństwo koloru",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return savefig(fig, out, "fig_03_det_vs_prob.png")


def fig04_small_world(images, labels, cfg, out):
    img, cid = pick_one_per_class(images, labels)[0]
    prep = slic_prep(img, cfg)
    fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.9))
    G0, _, _ = build_from_prep(prep, cfg, mode="prob", small_world=False, seed=cfg.seed)
    draw_overlay(ax[0], img, G0, title=f"bez skrótów  |E|={G0.number_of_edges()}")
    G1, sc, _ = build_from_prep(prep, cfg, mode="prob", small_world=True, seed=cfg.seed)
    draw_overlay(ax[1], img, G1, shortcuts=sc,
                 title=f"small-world: +{len(sc)} dalekich skrótów (czerwone)")
    fig.suptitle(f"Skróty small-world (p={cfg.small_world_p}) dokładają dalekie krawędzie\n"
                 f"obraz: {class_name(cid)}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    return savefig(fig, out, "fig_04_small_world.png")


def fig05_prawdopodobienstwo(images, labels, cfg, out):
    """Wprost odpowiada na pytanie o bliskość przestrzenną: jak ją gwarantujemy.
    Lewy panel: mechanizm losowania color_w=exp(-d^2/2sigma^2). Prawy: kandydaci to
    WYŁĄCZNIE sąsiedzi (RAG); losowość tylko odrzuca pary o niepodobnym kolorze."""
    img, cid = pick_one_per_class(images, labels)[0]
    prep = slic_prep(img, cfg)
    labels_s, nodes, cand, max_b, _ = prep
    d_arr = np.array([d for _, _, d, _ in cand])
    cw = np.exp(-d_arr ** 2 / (2 * cfg.sigma_feat ** 2))
    Gp, _, kept = build_from_prep(prep, cfg, mode="prob", small_world=False, seed=cfg.seed)
    kept = np.array(kept, bool)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    xs = np.linspace(0, max(d_arr.max(), 1e-3), 200)
    ax[0].plot(xs, np.exp(-xs ** 2 / (2 * cfg.sigma_feat ** 2)),
               "k--", lw=1.3, label=r"$P=\exp(-d^2/2\sigma^2)$")
    ax[0].scatter(d_arr[kept], cw[kept], c="#2a9d8f", s=28, label="krawędź wylosowana", zorder=3)
    ax[0].scatter(d_arr[~kept], cw[~kept], c="#e63946", s=28, label="krawędź odrzucona", zorder=3)
    ax[0].set_xlabel("odległość koloru d (sąsiednie superpiksele)")
    ax[0].set_ylabel("prawdopodobieństwo połączenia color_w")
    ax[0].set_title("Mechanizm DropEdge: P zależy od podobieństwa koloru", fontsize=10)
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)

    pos = _posxy(_new_graph(nodes))
    ax[1].imshow(img, interpolation="nearest")
    for (u, v, d, blen), k in zip(cand, kept):     # WSZYSCY kandydaci to sąsiedzi RAG
        col, al, lw = ("#2a9d8f", 0.95, 1.6) if k else ("#e63946", 0.35, 0.8)
        ls = "-" if k else ":"
        ax[1].plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                   color=col, alpha=al, lw=lw, ls=ls, zorder=2)
    ax[1].scatter([p[0] for p in pos.values()], [p[1] for p in pos.values()],
                  s=18, c="#ffd000", edgecolors="k", linewidths=0.3, zorder=3)
    ax[1].set_xticks([]); ax[1].set_yticks([])
    ax[1].set_title("Kandydaci = TYLKO sąsiedzi (RAG); zielone zostają, czerwone odrzucone", fontsize=10)
    fig.suptitle("Bliskość przestrzenną gwarantuje bramka sąsiedztwa (RAG), NIE losowanie — "
                 "losowość działa na kolorze", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return savefig(fig, out, "fig_05_prawdopodobienstwo_krawedzi.png")


def fig06_komponenty(images, labels, cfg, out):
    """Skutek przecinania krawędzi: obraz rozpada się na komponenty ~obiekty."""
    img, cid = pick_one_per_class(images, labels)[0]
    prep = slic_prep(img, cfg)
    labels_s = prep[0]
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.8))
    ax[0].imshow(img); ax[0].set_title(f"obraz ({class_name(cid)})", fontsize=10)
    ax[0].set_xticks([]); ax[0].set_yticks([])
    for j, (mode, name) in enumerate([("det", "deterministyczny"), ("prob", "probabilistyczny")]):
        G, _, _ = build_from_prep(prep, cfg, mode=mode, small_world=False, seed=cfg.seed)
        comp = {}
        for ci, nodeset in enumerate(nx.connected_components(G)):
            for nn in nodeset:
                comp[int(nn)] = ci
        ncomp = max(comp.values()) + 1 if comp else 1
        painted = paint_labels(labels_s, comp, cmap="tab20", vmin=0, vmax=max(ncomp - 1, 1))
        ax[j + 1].imshow(painted)
        ax[j + 1].set_title(f"{name}: {ncomp} komponentów", fontsize=10)
        ax[j + 1].set_xticks([]); ax[j + 1].set_yticks([])
    fig.suptitle("Komponenty spójne po przecięciu krawędzi (każdy kolor = osobny region)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return savefig(fig, out, "fig_06_komponenty.png")


def fig07_galeria_klasy(images, labels, cfg, out):
    """Topologia grafu losowego zależna od klasy — po 2 przykłady na klasę."""
    cids = sorted(set(int(x) for x in labels))
    per = 2
    fig, ax = plt.subplots(len(cids), per, figsize=(3.4 * per, 3.2 * len(cids)))
    ax = np.array(ax).reshape(len(cids), per)
    for r, cid in enumerate(cids):
        idx = np.where(labels == cid)[0][:per]
        for c in range(per):
            a = ax[r, c]
            if c >= len(idx):
                a.axis("off"); continue
            img = images[int(idx[c])]
            prep = slic_prep(img, cfg)
            G, sc, _ = build_from_prep(prep, cfg, mode="prob", small_world=True, seed=cfg.seed + c)
            draw_overlay(a, img, G, shortcuts=sc)
            if c == 0:
                a.set_ylabel(class_name(cid), fontsize=11)
    fig.suptitle("Galeria klas: graf losowy (prob + small-world) dla różnych obiektów", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return savefig(fig, out, "fig_07_galeria_klasy.png")


def fig08_n_segments(images, labels, cfg, out):
    from dataclasses import replace
    img, cid = pick_one_per_class(images, labels)[0]
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.8))
    for j, ns in enumerate([30, 60, 120]):
        prep = slic_prep(img, replace(cfg, n_segments=ns))
        G, sc, _ = build_from_prep(prep, replace(cfg, n_segments=ns),
                                   mode="prob", small_world=True, seed=cfg.seed)
        draw_overlay(ax[j], img, G, shortcuts=sc,
                     title=f"n_segments={ns}  |V|={G.number_of_nodes()}")
    fig.suptitle(f"Ziarnistość SLIC: więcej superpikseli = bogatszy graf ({class_name(cid)})", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return savefig(fig, out, "fig_08_n_segments.png")


def fig09_sigma(images, labels, cfg, out):
    """W trybie probabilistycznym sigma_feat steruje gęstością (przez color_w)."""
    from dataclasses import replace
    img, cid = pick_one_per_class(images, labels)[0]
    prep = slic_prep(img, cfg)                       # segmentacja ta sama
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.8))
    for j, sg in enumerate([0.5, 1.0, 2.0]):
        c = replace(cfg, sigma_feat=sg)
        G, sc, _ = build_from_prep(prep, c, mode="prob", small_world=True, seed=cfg.seed)
        draw_overlay(ax[j], img, G, shortcuts=sc,
                     title=f"sigma_feat={sg}  |E|={G.number_of_edges()}")
    fig.suptitle(f"Większa sigma_feat -> wyższe color_w -> gęstszy graf losowy ({class_name(cid)})", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return savefig(fig, out, "fig_09_sigma.png")


def fig10_er_baseline(images, labels, cfg, out):
    """Czysto losowy Erdős–Rényi (er_p) na tych samych węzłach = baseline 'szumu topologicznego'."""
    img, cid = pick_one_per_class(images, labels)[0]
    prep = slic_prep(img, cfg)
    labels_s, nodes, *_ = prep
    Gp, sc, _ = build_from_prep(prep, cfg, mode="prob", small_world=True, seed=cfg.seed)
    order = list(nodes.keys())
    Ger = nx.erdos_renyi_graph(n=len(order), p=cfg.er_p, seed=cfg.seed)
    Ger = nx.relabel_nodes(Ger, {i: order[i] for i in range(len(order))})
    for n in Ger:
        Ger.nodes[n].update(nodes[n])
    fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.9))
    draw_overlay(ax[0], img, Gp, shortcuts=sc,
                 title=f"STRUKTURALNY (prob SLIC)  |E|={Gp.number_of_edges()}")
    draw_overlay(ax[1], img, Ger, edge_color="#9b5de5", sizes=True,
                 title=f"LOSOWY Erdős–Rényi p={cfg.er_p}  |E|={Ger.number_of_edges()}")
    fig.suptitle(f"Graf oparty na obrazie vs czysto losowy ER (baseline fuzji) — {class_name(cid)}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    return savefig(fig, out, "fig_10_ER_baseline.png")


def fig11_wagi(images, labels, cfg, out):
    """Waga krawędzi = (długość granicy / max) * podobieństwo koloru."""
    img, cid = pick_one_per_class(images, labels)[0]
    prep = slic_prep(img, cfg)
    G, _, _ = build_from_prep(prep, cfg, mode="prob", small_world=False, seed=cfg.seed)
    pos = _posxy(G)
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.imshow(img, interpolation="nearest")
    cmap = plt.get_cmap("plasma")
    for u, v, dd in G.edges(data=True):
        w = float(dd.get("weight", 0.5))
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color=cmap(np.clip(w, 0, 1)), alpha=0.9, lw=0.6 + 2.4 * w, zorder=2)
    ax.scatter([p[0] for p in pos.values()], [p[1] for p in pos.values()],
               s=18, c="w", edgecolors="k", linewidths=0.4, zorder=3)
    ax.set_xticks([]); ax.set_yticks([])
    sm = ScalarMappable(norm=Normalize(0, 1), cmap=cmap)
    fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label="waga krawędzi")
    fig.suptitle(f"Wagi krawędzi: (granica/max) × podobieństwo koloru — {class_name(cid)}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return savefig(fig, out, "fig_11_wagi_krawedzi.png")


def fig12_stopnie(images, labels, cfg, out):
    """Superpiksele pomalowane stopniem węzła — strukturalna 'mapa ciepła'."""
    img, cid = pick_one_per_class(images, labels)[0]
    prep = slic_prep(img, cfg)
    labels_s = prep[0]
    G, _, _ = build_from_prep(prep, cfg, mode="prob", small_world=True, seed=cfg.seed)
    deg = dict(G.degree())
    painted = paint_labels(labels_s, deg, cmap="inferno")
    fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.9))
    ax[0].imshow(img); ax[0].set_title(f"obraz ({class_name(cid)})", fontsize=10)
    ax[0].set_xticks([]); ax[0].set_yticks([])
    ax[1].imshow(painted); ax[1].set_title("stopień węzła (jaśniej = wyższy)", fontsize=10)
    ax[1].set_xticks([]); ax[1].set_yticks([])
    sm = ScalarMappable(norm=Normalize(min(deg.values()), max(deg.values())), cmap="inferno")
    fig.colorbar(sm, ax=ax[1], fraction=0.046, pad=0.04, label="stopień")
    fig.suptitle("Mapa stopni węzłów w grafie losowym", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    return savefig(fig, out, "fig_12_stopnie.png")


def fig13_random_walk(images, labels, cfg, out):
    """Spacer losowy Node2Vec = 'zdanie' węzłów: kolejność odwiedzin koduje strukturę."""
    img, cid = pick_one_per_class(images, labels)[0]
    prep = slic_prep(img, cfg)
    G, sc, _ = build_from_prep(prep, cfg, mode="prob", small_world=True, seed=cfg.seed)
    np.random.seed(cfg.seed)
    start = max(G.degree, key=lambda x: x[1])[0]
    walk, cur = [start], start
    for _ in range(18):
        nbrs = list(G.neighbors(cur))
        if not nbrs:
            break
        cur = nbrs[np.random.randint(len(nbrs))]; walk.append(cur)
    pos = _posxy(G)
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    draw_overlay(ax, img, G, edge_color="#bdbdbd", node_color="#ffd000")
    for a, b in zip(walk[:-1], walk[1:]):
        ax.annotate("", xy=pos[b], xytext=pos[a],
                    arrowprops=dict(arrowstyle="-|>", color="#ff2d2d", lw=2.0, alpha=0.9),
                    zorder=6)
    ax.scatter([pos[walk[0]][0]], [pos[walk[0]][1]], s=160, c="#2a9d8f",
               edgecolors="k", zorder=7, label="start")
    ax.scatter([pos[walk[-1]][0]], [pos[walk[-1]][1]], s=160, c="#e63946",
               edgecolors="k", zorder=7, label="koniec")
    ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(f"Spacer losowy (Node2Vec) po grafie — sekwencja {len(walk)} węzłów ({class_name(cid)})\n"
                 "takie 'zdania' trafiają do Word2Vec, ucząc osadzeń węzłów", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return savefig(fig, out, "fig_13_random_walk.png")


def fig14_adjacency(images, labels, cfg, out):
    """Macierze sąsiedztwa: jak rzednie graf przy RAG -> twardy próg -> losowanie."""
    img, cid = pick_one_per_class(images, labels)[0]
    prep = slic_prep(img, cfg)
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.8))
    for j, (mode, name) in enumerate([("rag", "RAG (wszystkie sąsiedztwa)"),
                                      ("det", "deterministyczny (d<=tau)"),
                                      ("prob", "probabilistyczny (DropEdge)")]):
        G, _, _ = build_from_prep(prep, cfg, mode=mode, small_world=False, seed=cfg.seed)
        order = sorted(G.nodes())
        A = nx.to_numpy_array(G, nodelist=order)
        ax[j].imshow(A, cmap="Greys", interpolation="nearest")
        ax[j].set_title(f"{name}\n|E|={G.number_of_edges()}", fontsize=9)
        ax[j].set_xlabel("węzeł"); ax[j].set_xticks([]); ax[j].set_yticks([])
    fig.suptitle(f"Macierze sąsiedztwa grafu SLIC — rzednięcie krawędzi ({class_name(cid)})", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    return savefig(fig, out, "fig_14_adjacency.png")


def fig18_superpiksel(images, labels, cfg, out):
    """Wyjaśnienie 'czym jest superpiksel': piksele -> podział na regiony (~Voronoi)
    -> co zachowuje węzeł (średni kolor) -> graf z węzłami w centroidach."""
    img, cid = pick_one_per_class(images, labels)[0]
    prep = slic_prep(img, cfg)
    labels_s = prep[0]
    G, sc, _ = build_from_prep(prep, cfg, mode="prob", small_world=True, seed=cfg.seed)
    nseg = len(np.unique(labels_s))
    fig, ax = plt.subplots(1, 4, figsize=(14, 3.9))
    ax[0].imshow(np.kron(img, np.ones((16, 16, 1))))
    ax[0].set_title("1. obraz (1024 piksele)", fontsize=10)
    ax[1].imshow(_sharp_boundaries(img, labels_s))
    ax[1].set_title(f"2. {nseg} superpikseli (ostre granice ~Voronoi)", fontsize=10)
    ax[2].imshow(_sharp_boundaries(_slic_mean_image(img, labels_s), labels_s, color=(0, 0, 0)))
    ax[2].set_title("3. superpiksel = jego średni kolor", fontsize=10)
    draw_overlay(ax[3], img, G, shortcuts=sc)
    ax[3].set_title("4. graf: żółty węzeł = 1 superpiksel", fontsize=10)
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"Czym jest superpiksel: piksele -> podział na regiony (~Voronoi) -> węzły grafu "
                 f"w środkach ciężkości ({class_name(cid)})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return savefig(fig, out, "fig_18_superpiksel.png")


def _paint_pixel_components(G, H, W):
    """Maluje każdy komponent spójny grafu pikselowego innym kolorem (tab20)."""
    comp = {}
    for ci, nodeset in enumerate(nx.connected_components(G)):
        for nn in nodeset:
            comp[int(nn)] = ci
    ncomp = max(comp.values()) + 1 if comp else 1
    cmap = plt.get_cmap("tab20")
    img = np.zeros((H, W, 3))
    for nn, ci in comp.items():
        i, j = divmod(nn, W)
        img[i, j] = cmap(ci % 20)[:3]
    return img, ncomp


def fig15_galeria_10klas(cfg, out):
    """Graf losowy SLIC dla WSZYSTKICH 10 klas — przegląd różnorodności kształtów."""
    from dataclasses import replace
    images, labels = load_cifar10(replace(cfg, classes=list(range(10)), per_class=2))
    order = np.argsort(labels, kind="stable")[:20]
    fig, ax = plt.subplots(4, 5, figsize=(13, 10.5)); axf = ax.ravel()
    for k, i in enumerate(order):
        img = images[int(i)]
        prep = slic_prep(img, cfg)
        G, sc, _ = build_from_prep(prep, cfg, mode="prob", small_world=True, seed=cfg.seed + k)
        draw_overlay(axf[k], img, G, shortcuts=sc, title=class_name(int(labels[int(i)])))
    for k in range(len(order), 20):
        axf[k].axis("off")
    fig.suptitle("Galeria 10 klas CIFAR-10: graf losowy SLIC (prob + small-world)\n"
                 "różne obiekty -> różne kształty grafów", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return savefig(fig, out, "fig_15_galeria_10klas.png")


def fig16_ciekawe_ksztalty(cfg, out):
    """Ranking obrazów po liczbie komponentów (twardy próg q=0.5): od rozdrobnionych
    do zwartych — pokazuje najciekawsze topologicznie kształty."""
    from dataclasses import replace
    images, labels = load_cifar10(replace(cfg, classes=list(range(10)), per_class=7, seed=cfg.seed + 1))
    cf = replace(cfg, edge_quantile=0.5)
    info, preps = [], []
    for i, img in enumerate(images):
        prep = slic_prep(img, cf)
        Gd, _, _ = build_from_prep(prep, cf, mode="det", small_world=False)
        info.append((i, nx.number_connected_components(Gd))); preps.append(prep)
    info.sort(key=lambda t: t[1])
    least, most = info[:6], info[-6:][::-1]
    fig, ax = plt.subplots(2, 6, figsize=(14, 5.4))
    for c, (i, nc) in enumerate(most):
        G, sc, _ = build_from_prep(preps[i], cf, mode="prob", small_world=True, seed=cfg.seed)
        draw_overlay(ax[0, c], images[i], G, shortcuts=sc, title=f"{class_name(int(labels[i]))}  #k={nc}")
    for c, (i, nc) in enumerate(least):
        G, sc, _ = build_from_prep(preps[i], cf, mode="prob", small_world=True, seed=cfg.seed)
        draw_overlay(ax[1, c], images[i], G, shortcuts=sc, title=f"{class_name(int(labels[i]))}  #k={nc}")
    ax[0, 0].set_ylabel("najbardziej\nrozdrobnione", fontsize=10)
    ax[1, 0].set_ylabel("najbardziej\nzwarte", fontsize=10)
    fig.suptitle("Ciekawe kształty grafów: ranking po liczbie komponentów (twardy próg q=0.5)\n"
                 "rozdrobnienie na regiony to wprost skutek twardego warunku na krawędź", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.9], h_pad=2.6)
    return savefig(fig, out, "fig_16_ciekawe_ksztalty.png")


def fig17_pixel_obiekty(cfg, out):
    """Graf PIKSELOWY: obiekt oddziela się od tła."""
    from dataclasses import replace
    from cifar_graph_clustering import Config as CC, make_graph
    images, labels = load_cifar10(replace(cfg, classes=[0, 1, 7, 8], per_class=2, seed=cfg.seed + 2))
    rows = pick_one_per_class(images, labels)[:4]
    cc = CC(edge_quantile=0.5, connectivity=8)
    fig, ax = plt.subplots(len(rows), 3, figsize=(9, 3.0 * len(rows)))
    ax = np.array(ax).reshape(len(rows), 3)
    for r, (img, cid) in enumerate(rows):
        G = make_graph(img, "pixel", cc)
        ax[r, 0].imshow(img); ax[r, 0].set_ylabel(class_name(cid), fontsize=10)
        ax[r, 0].set_xticks([]); ax[r, 0].set_yticks([])
        draw_overlay(ax[r, 1], img, G, edge_color="#39ff14")
        painted, ncomp = _paint_pixel_components(G, img.shape[0], img.shape[1])
        ax[r, 2].imshow(painted); ax[r, 2].set_title(f"{ncomp} komponentów", fontsize=9)
        ax[r, 2].set_xticks([]); ax[r, 2].set_yticks([])
        if r == 0:
            ax[0, 0].set_title("obraz", fontsize=10); ax[0, 1].set_title("graf pikselowy", fontsize=10)
    fig.suptitle("Graf PIKSELOWY: twardy próg LAB tnie krawędzie na granicach\n"
                 "-> obiekt oddziela się od tła w osobne komponenty", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return savefig(fig, out, "fig_17_pixel_obiekty.png")


def fig23_wszystkie_metody(cfg, out):
    """Przegląd WSZYSTKICH metod budowy grafu na różnych klasach obrazów.
    Kolumny = metoda (kolor podpisu = rodzina), wiersze = klasa. Jednoznacznie
    pokazuje, która metoda jest która i jak różni się topologia."""
    from dataclasses import replace
    # różnorodne klasy: samolot / auto / ptak / kot / pies / statek
    div = replace(cfg, classes=[0, 1, 2, 3, 5, 8], per_class=2, seed=cfg.seed)
    images, labels = load_cifar10(div)
    rows = pick_one_per_class(images, labels)

    # nasze metody deterministyczne (cifar_graph_clustering.py) — import leniwy
    from cifar_graph_clustering import Config as CC, make_graph
    cc = CC(n_segments=cfg.n_segments, edge_quantile=cfg.edge_quantile, compactness=cfg.compactness)

    def det(gt):
        return lambda img: (make_graph(img, gt, cc), [])

    def jan(mode, sw):
        def f(img):
            prep = slic_prep(img, cfg)
            G, scs, _ = build_from_prep(prep, cfg, mode=mode, small_world=sw, seed=cfg.seed)
            return G, scs
        return f

    def er(img):
        _, nodes, *_ = slic_prep(img, cfg)
        order = list(nodes.keys())
        G = nx.erdos_renyi_graph(len(order), cfg.er_p, seed=cfg.seed)
        G = nx.relabel_nodes(G, {i: order[i] for i in range(len(order))})
        for nn in G:
            G.nodes[nn].update(nodes[nn])
        return G, []

    methods = [
        ("pixel",     "nasz .py",     det("pixel")),
        ("pixel+tex", "nasz .py",     det("pixeltex")),
        ("patch",     "nasz .py",     det("patch")),
        ("slic",      "nasz .py",     det("slic")),
        ("slic+tex",  "nasz .py",     det("slictex")),
        ("slic-prob", "Jan (losowy)", jan("prob", False)),
        ("slic+SW",   "Jan (losowy)", jan("prob", True)),
        ("ER",        "baseline",     er),
    ]
    fam_color = {"nasz .py": "#1f6feb", "Jan (losowy)": "#d62728", "baseline": "#555555"}

    R, Cn = len(rows), len(methods)
    fig, ax = plt.subplots(R, Cn, figsize=(1.75 * Cn, 1.95 * R))
    ax = np.array(ax).reshape(R, Cn)
    for r, (img, cid) in enumerate(rows):
        for c, (lab, fam, fn) in enumerate(methods):
            a = ax[r, c]
            try:
                G, scs = fn(img)
                ec = "#9b5de5" if fam == "baseline" else "#39ff14"
                draw_overlay(a, img, G, shortcuts=scs, edge_color=ec)
            except Exception as e:
                a.imshow(img); a.set_xticks([]); a.set_yticks([])
                print(f"    [fig23 {lab}/{class_name(cid)}: {e}]")
            if r == 0:
                a.set_title(lab, fontsize=9, color=fam_color[fam], fontweight="bold")
            if c == 0:
                a.set_ylabel(class_name(cid), fontsize=10)
    handles = [Line2D([0], [0], color=fam_color[k], lw=4, label=k) for k in fam_color]
    handles.append(Line2D([0], [0], color="#ff2d2d", lw=2, label="skrót small-world"))
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9, frameon=False)
    fig.suptitle("Wszystkie metody budowy grafu na różnych klasach obrazów\n"
                 "kolumny = metoda (kolor podpisu = rodzina), wiersze = klasa obrazu",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    return savefig(fig, out, "fig_23_wszystkie_metody.png")


# ============================================================================
# FIGURY — BIAŁKA (PROTEINS_full)
# ============================================================================
# Kolor etykiety węzła = typ elementu struktury wtórnej (SSE): 0 / 1 / 2.
SSE_COLORS = {0: "#e63946", 1: "#457b9d", 2: "#f4a261"}
SSE_NAMES = {0: "typ 0 (np. helisa)", 1: "typ 1 (np. arkusz)", 2: "typ 2 (np. pętla)"}


def _node_colors(G):
    return [SSE_COLORS.get(int(G.nodes[n].get("label", 0)), "#999999") for n in G]


def _protein_layout(G, seed=1):
    """Kamada–Kawai dla małych grafów (ładniejszy, 'organiczny'), spring dla dużych."""
    if 1 < G.number_of_nodes() <= 80:
        try:
            return nx.kamada_kawai_layout(G)
        except Exception:
            pass
    return nx.spring_layout(G, seed=seed, k=0.45)


def _pstats(G):
    n, m = G.number_of_nodes(), G.number_of_edges()
    deg = 2 * m / n if n else 0.0
    return n, m, deg, nx.average_clustering(G), nx.number_connected_components(G)


def _sse_handles():
    return [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markeredgecolor="k",
                   markersize=9, label=SSE_NAMES[k]) for k, c in SSE_COLORS.items()]


def _draw_protein(ax, G, title, size_by_degree=True):
    pos = _protein_layout(G)
    deg = dict(G.degree())
    ns = [55 + 26 * deg[n] for n in G] if size_by_degree else 90
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, width=0.7, edge_color="#555555")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=ns, node_color=_node_colors(G),
                           linewidths=0.4, edgecolors="k")
    ax.set_title(title, fontsize=9); ax.axis("off")


def fig_proteins(out, root="data"):
    print("Białka: wczytuję PROTEINS_full ...")
    graphs, labels = load_tu_dataset("PROTEINS_full", root)
    labels = np.array(labels)
    sizes = np.array([g.number_of_nodes() for g in graphs])
    classes = sorted(set(int(x) for x in labels))[:2]
    cls_name = {cl: f"klasa {cl}" for cl in classes}
    paths = []

    # ---- fig_20: galeria (kamada-kawai, kolor=SSE, rozmiar~stopień, legenda) ----
    sel = [i for i in np.argsort(sizes) if 14 <= sizes[i] <= 40][:9]
    fig, ax = plt.subplots(3, 3, figsize=(11, 10)); axf = ax.ravel()
    for k, i in enumerate(sel):
        n, _, dg, _, _ = _pstats(graphs[i])
        _draw_protein(axf[k], graphs[i], f"#{i}  {cls_name[int(labels[i])]}  |V|={n}  ⟨deg⟩={dg:.1f}")
    fig.legend(handles=_sse_handles(), loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.suptitle("PROTEINS_full: przykładowe grafy białek\n"
                 "kolor węzła = typ struktury wtórnej (SSE), rozmiar ~ stopień", fontsize=12)
    fig.tight_layout(rect=[0, 0.04, 1, 0.94]); paths.append(savefig(fig, out, "fig_20_bialka_galeria.png"))

    # ---- fig_21: klasa vs klasa, ze statystykami strukturalnymi ----
    fig, ax = plt.subplots(2, 3, figsize=(11, 7.6))
    for r, cl in enumerate(classes):
        pool = [i for i in np.argsort(sizes) if labels[i] == cl and 16 <= sizes[i] <= 46][:3]
        for c in range(3):
            if c < len(pool):
                n, _, dg, cc, _ = _pstats(graphs[pool[c]])
                t = f"|V|={n}  ⟨deg⟩={dg:.1f}  C={cc:.2f}"
                _draw_protein(ax[r, c], graphs[pool[c]], (f"{cls_name[cl]}\n" + t) if c == 0 else t)
            else:
                ax[r, c].axis("off")
    fig.legend(handles=_sse_handles(), loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.suptitle("Porównanie klas (enzym vs nie-enzym): przykładowe grafy + statystyki", fontsize=12)
    fig.tight_layout(rect=[0, 0.04, 1, 0.94]); paths.append(savefig(fig, out, "fig_21_bialka_klasy.png"))

    # ---- fig_22: rozkłady strukturalne 2x2 wg klasy ----
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    for cl in classes:
        idx = np.where(labels == cl)[0]
        ax[0, 0].hist(sizes[idx], bins=30, alpha=0.55, label=cls_name[cl])
        degs = np.concatenate([[d for _, d in graphs[i].degree()] for i in idx])
        ax[0, 1].hist(degs, bins=range(0, 13), alpha=0.55, density=True, label=cls_name[cl])
        ccs = [nx.average_clustering(graphs[i]) for i in idx]
        ax[1, 0].hist(ccs, bins=25, alpha=0.55, density=True, label=cls_name[cl])
        comps = [nx.number_connected_components(graphs[i]) for i in idx]
        ax[1, 1].hist(comps, bins=range(1, 9), alpha=0.55, density=True, label=cls_name[cl])
    for a, t, xl in [(ax[0, 0], "Rozmiar grafu", "liczba węzłów |V|"),
                     (ax[0, 1], "Stopień węzła", "stopień"),
                     (ax[1, 0], "Współczynnik klasteryzacji", "C"),
                     (ax[1, 1], "Liczba komponentów spójnych", "komponenty")]:
        a.set_title(t); a.set_xlabel(xl); a.legend(); a.grid(alpha=.3)
    fig.suptitle("PROTEINS_full: rozkłady cech strukturalnych wg klasy", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); paths.append(savefig(fig, out, "fig_22_bialka_rozklady.png"))

    # 'hero' graf: średniej wielkości, czytelny — używany też przez fig_25
    cand = [i for i in np.argsort(sizes) if 28 <= sizes[i] <= 40]
    hi = cand[len(cand) // 2] if cand else int(np.argsort(sizes)[len(sizes) // 2])

    # ---- fig_24: pojedynczy duży graf z ramką statystyk ----
    Gh = graphs[hi]; n, m, dg, cc, comp = _pstats(Gh)
    fig, ax = plt.subplots(figsize=(7.8, 7.2))
    _draw_protein(ax, Gh, "")
    ax.text(0.02, 0.98, f"{cls_name[int(labels[hi])]}\n|V|={n}  |E|={m}\n⟨deg⟩={dg:.2f}\n"
            f"klasteryzacja C={cc:.2f}\nkomponenty={comp}", transform=ax.transAxes,
            va="top", fontsize=10, bbox=dict(boxstyle="round", fc="white", alpha=0.85))
    fig.legend(handles=_sse_handles(), loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.suptitle(f"Pojedynczy graf białka (#{hi}) — etykiety węzłów = struktura wtórna", fontsize=12)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95]); paths.append(savefig(fig, out, "fig_24_bialko_hero.png"))

    # ---- fig_25: centralność pośrednictwa (huby strukturalne) ----
    Gc = graphs[hi]; bc = nx.betweenness_centrality(Gc); pos = _protein_layout(Gc)
    fig, ax = plt.subplots(figsize=(7.6, 6.8))
    nx.draw_networkx_edges(Gc, pos, ax=ax, alpha=0.3, width=0.7, edge_color="#555555")
    nodes = nx.draw_networkx_nodes(Gc, pos, ax=ax, node_color=[bc[n] for n in Gc], cmap="viridis",
                                   node_size=[60 + 600 * bc[n] for n in Gc],
                                   linewidths=0.4, edgecolors="k")
    ax.axis("off")
    fig.colorbar(nodes, ax=ax, fraction=0.046, pad=0.04, label="centralność pośrednictwa")
    fig.suptitle(f"Strukturalne 'huby' białka (#{hi}): centralność pośrednictwa węzłów", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); paths.append(savefig(fig, out, "fig_25_bialko_centralnosc.png"))

    # ---- fig_26: macierze sąsiedztwa enzym vs nie-enzym ----
    fig, ax = plt.subplots(2, 3, figsize=(11, 7.4))
    for r, cl in enumerate(classes):
        pool = [i for i in np.argsort(sizes) if labels[i] == cl and 20 <= sizes[i] <= 50][:3]
        for c in range(3):
            if c < len(pool):
                A = nx.to_numpy_array(graphs[pool[c]])
                ax[r, c].imshow(A, cmap="Greys", interpolation="nearest")
                ax[r, c].set_title(f"{cls_name[cl]}  |V|={A.shape[0]}", fontsize=9)
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
    fig.suptitle("Macierze sąsiedztwa białek (przekątna = łańcuch, plamy = kontakty 3D)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); paths.append(savefig(fig, out, "fig_26_bialko_macierze.png"))

    # ---- fig_27: skład typów węzłów + uśredniona sygnatura spektralna wg klasy ----
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.8))
    sse_keys = sorted(SSE_COLORS.keys()); width = 0.38
    for j, cl in enumerate(classes):
        idx = np.where(labels == cl)[0]
        cnt = np.zeros(len(sse_keys))
        for i in idx:
            for nd in graphs[i]:
                l = int(graphs[i].nodes[nd].get("label", 0))
                if l in sse_keys:
                    cnt[sse_keys.index(l)] += 1
        frac = cnt / cnt.sum() if cnt.sum() else cnt
        ax[0].bar(np.arange(len(sse_keys)) + j * width, frac, width,
                  label=cls_name[cl], color=["#7aa6c2", "#c2887a"][j % 2])
    ax[0].set_xticks(np.arange(len(sse_keys)) + width / 2)
    ax[0].set_xticklabels([SSE_NAMES[k] for k in sse_keys], fontsize=8, rotation=12)
    ax[0].set_ylabel("udział węzłów"); ax[0].set_title("Skład typów struktury wtórnej wg klasy")
    ax[0].legend(); ax[0].grid(alpha=.3, axis="y")

    k = 12
    for cl in classes:
        specs = []
        for i in np.where(labels == cl)[0][:150]:
            G = graphs[i]
            if G.number_of_nodes() < 2:
                continue
            try:
                ev = np.sort(np.linalg.eigvalsh(nx.normalized_laplacian_matrix(G).toarray()))[:k]
            except Exception:
                continue
            specs.append(np.pad(ev, (0, k - len(ev))) if len(ev) < k else ev)
        if specs:
            ax[1].plot(range(1, k + 1), np.mean(specs, axis=0), "o-", label=cls_name[cl])
    ax[1].set_xlabel("indeks wartości własnej"); ax[1].set_ylabel("wartość własna (Laplasjan znorm.)")
    ax[1].set_title("Uśredniona sygnatura spektralna wg klasy"); ax[1].legend(); ax[1].grid(alpha=.3)
    fig.suptitle("PROTEINS_full: 'odcisk' strukturalny klas (skład węzłów + spektrum)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93]); paths.append(savefig(fig, out, "fig_27_bialko_sklad_spektrum.png"))

    # ---- fig_28: galeria posortowana po rozmiarze (od małych do dużych) ----
    valid = [i for i in range(len(graphs)) if graphs[i].number_of_nodes() >= 6]
    order = sorted(valid, key=lambda i: sizes[i])
    pick = [order[int(p)] for p in np.linspace(0, len(order) - 1, 6)]
    fig, ax = plt.subplots(2, 3, figsize=(11, 7)); axf = ax.ravel()
    for k2, i in enumerate(pick):
        n, _, _, _, _ = _pstats(graphs[i])
        _draw_protein(axf[k2], graphs[i], f"|V|={n}  {cls_name[int(labels[i])]}")
    fig.legend(handles=_sse_handles(), loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.suptitle("Zróżnicowanie rozmiaru grafów białek (od najmniejszych do największych)", fontsize=12)
    fig.tight_layout(rect=[0, 0.04, 1, 0.94]); paths.append(savefig(fig, out, "fig_28_bialko_rozmiary.png"))

    return paths


# ============================================================================
# INDEKS (README) — krótki opis każdej figury (PL), pod prezentację
# ============================================================================
def write_index(out, cifar_paths, protein_paths):
    lines = [
        "# Wizualizacje grafów — Lista 3",
        "",
        "Wygenerowane przez [`wizualizacja_grafow.py`](../wizualizacja_grafow.py).",
        "Nacisk na NOWY notatnik Jana `cifar_rand_graphs2_description.ipynb` (graf LOSOWY na superpikselach:",
        "probabilistyczne wstawianie krawędzi + skróty small-world + ensemble M grafów na obraz).",
        "",
        "## CIFAR-10 — graf losowy (Jan)",
        "- **fig_01_pipeline** — pełny potok: obraz -> SLIC -> RAG (wszystkie sąsiedztwa) -> graf wylosowany.",
        "- **fig_02_ensemble_losowosc** — M niezależnych losowań jednego obrazu (dlaczego potrzebny ensemble).",
        "- **fig_03_det_vs_prob** — twardy próg d<=tau vs DropEdge (rand < color_w).",
        "- **fig_04_small_world** — dalekie skróty small-world (czerwone) dokładane do grafu.",
        "- **fig_05_prawdopodobienstwo_krawedzi** — ODPOWIEDŹ na pytanie o bliskość przestrzenną: gwarantuje ją bramka RAG, nie losowanie.",
        "- **fig_06_komponenty** — rozpad na komponenty ~obiekty po przecięciu krawędzi.",
        "- **fig_07_galeria_klasy** — graf losowy dla różnych klas (topologia zależna od obiektu).",
        "- **fig_08_n_segments** — wpływ ziarnistości SLIC (30/60/120).",
        "- **fig_09_sigma** — sigma_feat steruje gęstością grafu probabilistycznego.",
        "- **fig_10_ER_baseline** — graf oparty na obrazie vs czysto losowy Erdős–Rényi.",
        "- **fig_11_wagi_krawedzi** — wagi = (granica/max) × podobieństwo koloru.",
        "- **fig_12_stopnie** — mapa stopni węzłów (struktura jako 'ciepło').",
        "- **fig_13_random_walk** — spacer losowy Node2Vec ('zdanie' węzłów) zaznaczony na grafie.",
        "- **fig_14_adjacency** — macierze sąsiedztwa: rzednięcie RAG -> próg -> losowanie.",
        "- **fig_15_galeria_10klas** — graf losowy SLIC dla wszystkich 10 klas (różnorodność kształtów).",
        "- **fig_16_ciekawe_ksztalty** — ranking obrazów po liczbie komponentów (rozdrobnione vs zwarte).",
        "- **fig_17_pixel_obiekty** — graf pikselowy: obiekt oddziela się od tła w osobne komponenty.",
        "- **fig_18_superpiksel** — czym jest superpiksel: piksele -> podział ~Voronoi -> średni kolor -> węzły grafu.",
        "",
        "## Przegląd wszystkich metod budowy grafu",
        "- **fig_23_wszystkie_metody** — WSZYSTKIE metody obok siebie na różnych klasach: nasz `.py`",
        "  (pixel / pixel+tex / patch / slic / slic+tex), graf losowy Jana (slic-prob / slic+small-world)",
        "  oraz baseline Erdős–Rényi. Kolor podpisu kolumny = rodzina metody.",
        "",
        "## Białka — PROTEINS_full",
        "- **fig_20_bialka_galeria** — galeria grafów (układ kamada-kawai, kolor = typ SSE, rozmiar ~ stopień, legenda).",
        "- **fig_21_bialka_klasy** — enzym vs nie-enzym obok siebie, ze statystykami (⟨deg⟩, klasteryzacja).",
        "- **fig_22_bialka_rozklady** — rozkłady 4 cech wg klasy: rozmiar, stopień, klasteryzacja, komponenty.",
        "- **fig_24_bialko_hero** — pojedynczy duży graf z ramką statystyk.",
        "- **fig_25_bialko_centralnosc** — węzły kolorowane centralnością pośrednictwa (huby strukturalne).",
        "- **fig_26_bialko_macierze** — macierze sąsiedztwa enzym vs nie-enzym (łańcuch + kontakty 3D).",
        "- **fig_27_bialko_sklad_spektrum** — skład typów węzłów + uśredniona sygnatura spektralna wg klasy.",
        "- **fig_28_bialko_rozmiary** — galeria posortowana po rozmiarze (od małych do dużych).",
        "",
    ]
    if not protein_paths:
        lines.append("> (Wizualizacje białek pominięte — brak danych/sieci lub --skip-proteins.)")
    path = os.path.join(out, "README.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  zapisano {path}")


def main():
    ap = argparse.ArgumentParser(description="Wizualizacje grafów (CIFAR graf losowy Jana + Białka).")
    ap.add_argument("--classes", type=int, nargs="+", default=[0, 1, 8])
    ap.add_argument("--per-class", type=int, default=8)
    ap.add_argument("--out", default="results_wiz")
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--tu-root", default="data")
    ap.add_argument("--n-segments", type=int, default=60)
    ap.add_argument("--ensemble-m", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-proteins", action="store_true")
    ap.add_argument("--skip-cifar", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg = Config(classes=args.classes, per_class=args.per_class, data_dir=args.data_dir,
                 n_segments=args.n_segments, ensemble_m=args.ensemble_m, seed=args.seed)

    cifar_paths, protein_paths = [], []
    if not args.skip_cifar:
        print("CIFAR: wczytuję obrazy ...")
        images, labels = load_cifar10(cfg)
        print(f"  {len(images)} obrazów, klasy {sorted(set(int(x) for x in labels))}")
        for fn in (fig01_pipeline, fig02_ensemble, fig03_det_vs_prob, fig04_small_world,
                   fig05_prawdopodobienstwo, fig06_komponenty, fig07_galeria_klasy,
                   fig08_n_segments, fig09_sigma, fig10_er_baseline, fig11_wagi, fig12_stopnie,
                   fig13_random_walk, fig14_adjacency, fig18_superpiksel):
            try:
                cifar_paths.append(fn(images, labels, cfg, args.out))
            except Exception as e:
                print(f"  [POMINIĘTO {fn.__name__}: {e}]")
        for fn in (fig15_galeria_10klas, fig16_ciekawe_ksztalty, fig17_pixel_obiekty,
                   fig23_wszystkie_metody):
            try:
                cifar_paths.append(fn(cfg, args.out))
            except Exception as e:
                print(f"  [POMINIĘTO {fn.__name__}: {e}]")

    if not args.skip_proteins:
        try:
            protein_paths = fig_proteins(args.out, root=args.tu_root)
        except Exception as e:
            print(f"  [POMINIĘTO białka: {e}]")

    write_index(args.out, cifar_paths, protein_paths)
    print(f"\nGotowe: {len(cifar_paths)} figur CIFAR + {len(protein_paths)} figur białek -> {args.out}/")


if __name__ == "__main__":
    main()
