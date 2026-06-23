#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Źródła grafów + obfuskacja.

Każdy graf to networkx.Graph, gdzie węzły mają:
  - 'features' : wektor deskryptora (atrybuty białka dla TU; kolor+tekstura dla obrazu),
  - 'label'    : dyskretna etykieta (typ węzła TU; lub seed-kolor, przypisywany później),
  - (opcjonalnie) 'pos', 'size'.
Krawędzie mają 'weight'. Dzięki temu embeddingi są niezależne od źródła.

Źródła:
  tu     -> ENZYMES / PROTEINS_full (gotowe grafy, auto-pobieranie z mirrora GitHub)
  cifar  -> obraz -> graf (pixel/patch/slic, warunek krawędzi Krzysztofa)   [wymaga torchvision]
  synth  -> obrazy syntetyczne -> graf (ta sama ścieżka co cifar; do testów bez pobierania)

Obfuskacja działa NA WYJŚCIU dowolnego źródła (i na obrazach, i na białkach) — to wspólne
"pokrętło siły" na osi prywatność–użyteczność (pomysł grafów losowych kolegi).
"""
from __future__ import annotations
import os
import urllib.request
import numpy as np
import networkx as nx

from .config import Config


# ============================================================================
# ŹRÓDŁO 1: TUDataset (ENZYMES / PROTEINS_full)
# ============================================================================
_GRAPHRNN = "https://raw.githubusercontent.com/snap-stanford/GraphRNN/master/dataset"


def _download_tu(name: str, root: str) -> str:
    dst = os.path.join(root, name)
    os.makedirs(dst, exist_ok=True)
    for f in ["A", "graph_indicator", "graph_labels", "node_labels", "node_attributes"]:
        path = os.path.join(dst, f"{name}_{f}.txt")
        if os.path.exists(path):
            continue
        try:
            urllib.request.urlretrieve(f"{_GRAPHRNN}/{name}/{name}_{f}.txt", path)
        except Exception as e:
            if f in ("node_labels", "node_attributes"):
                pass  # opcjonalne
            else:
                raise RuntimeError(f"Nie udało się pobrać {name}_{f}.txt: {e}")
    return dst


def load_tu(name: str, root: str):
    d = os.path.join(root, name)
    if not os.path.exists(os.path.join(d, f"{name}_A.txt")):
        d = _download_tu(name, root)
    p = lambda s: os.path.join(d, f"{name}_{s}.txt")

    A = np.loadtxt(p("A"), delimiter=",", dtype=int)
    indicator = np.loadtxt(p("graph_indicator"), dtype=int)
    glabels = np.loadtxt(p("graph_labels"), dtype=int)
    nlabels = np.loadtxt(p("node_labels"), dtype=int) if os.path.exists(p("node_labels")) else None
    nattrs = (np.loadtxt(p("node_attributes"), delimiter=",", dtype=float)
              if os.path.exists(p("node_attributes")) else None)
    if nattrs is not None and nattrs.ndim == 1:
        nattrs = nattrs.reshape(-1, 1)

    gids = np.unique(indicator)
    G = {g: nx.Graph() for g in gids}
    for i in range(indicator.shape[0]):
        g = int(indicator[i]); attrs = {}
        if nlabels is not None:
            attrs["label"] = int(nlabels[i])
        if nattrs is not None:
            attrs["features"] = nattrs[i]
        G[g].add_node(i + 1, **attrs)
    for u, v in A:
        G[int(indicator[u - 1])].add_edge(int(u), int(v), weight=1.0)

    graphs = [G[g] for g in gids]
    labels = np.array([int(glabels[g - 1]) for g in gids], dtype=int)
    return graphs, labels


# ============================================================================
# ŹRÓDŁO 2/3: obraz -> graf  (warunek krawędzi Krzysztofa: bliskość ORAZ podobieństwo)
# Buildery przeniesione z cifar_graph_clustering.py kolegów (z drobnym ujednoliceniem).
# ============================================================================
def _threshold(dists: np.ndarray, cfg: Config) -> float:
    if cfg.tau is not None:
        return cfg.tau
    if len(dists) == 0:
        return float("inf")
    return float(np.quantile(dists, cfg.edge_quantile))


def _ensure_connected(G: nx.Graph, best_neighbor: dict):
    for n in list(G.nodes()):
        if G.degree(n) == 0 and n in best_neighbor:
            nb, w = best_neighbor[n]
            G.add_edge(n, nb, weight=w)


def build_pixel_graph(image: np.ndarray, cfg: Config) -> nx.Graph:
    from skimage.color import rgb2lab
    lab = rgb2lab(image)
    H, W, _ = lab.shape
    gy, gx = np.gradient(lab[:, :, 0])
    grad = np.sqrt(gx ** 2 + gy ** 2)
    G = nx.Graph()
    for i in range(H):
        for j in range(W):
            feat = np.array([lab[i, j, 0], lab[i, j, 1], lab[i, j, 2], grad[i, j]], float)
            G.add_node(i * W + j, pos=(i, j), features=feat)
    offsets = ([(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
               if cfg.connectivity == 8 else [(-1, 0), (1, 0), (0, -1), (0, 1)])
    cand, best_neighbor = [], {}
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
    tau = _threshold(np.array([d for _, _, d, _ in cand]), cfg)
    tau_t = _threshold(np.array([dt for *_, dt in cand]), cfg) if cfg.edge_texture else None
    for u, v, d, dt in cand:
        if d <= tau and (tau_t is None or dt <= tau_t):
            G.add_edge(u, v, weight=float(np.exp(-d * d / (2 * cfg.sigma_pixel ** 2))))
    _ensure_connected(G, best_neighbor)
    return G


def build_patch_graph(image: np.ndarray, cfg: Config) -> nx.Graph:
    from scipy.spatial.distance import cdist
    H, W, _ = image.shape
    ps, st = cfg.patch_size, cfg.stride
    G = nx.Graph(); feats, pos = [], []
    idx = 0
    for i in range(0, H - ps + 1, st):
        for j in range(0, W - ps + 1, st):
            patch = image[i:i + ps, j:j + ps, :]
            f = np.concatenate([patch.mean(axis=(0, 1)), patch.std(axis=(0, 1))])
            G.add_node(idx, pos=(i, j), features=f)
            feats.append(f); pos.append((i // st, j // st)); idx += 1
    feats = np.array(feats)
    gH, gW = (H - ps) // st + 1, (W - ps) // st + 1
    cand, best_neighbor = [], {}
    for n in range(len(feats)):
        gi, gj = pos[n]
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = gi + di, gj + dj
            if 0 <= ni < gH and 0 <= nj < gW:
                m = ni * gW + nj
                if m >= len(feats):
                    continue
                d = float(np.linalg.norm(feats[n] - feats[m]))
                if n < m:
                    cand.append((n, m, d))
                w = float(np.exp(-d * d / (2 * cfg.sigma_feat ** 2)))
                if n not in best_neighbor or w > best_neighbor[n][1]:
                    best_neighbor[n] = (m, w)
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
    from skimage.segmentation import slic
    from skimage.color import rgb2lab
    labels = slic(image, n_segments=cfg.n_segments, compactness=cfg.compactness,
                  start_label=0, channel_axis=-1)
    H, W = labels.shape
    G = nx.Graph()
    boundary = {}
    for r in range(H):
        for c in range(W):
            lbl = labels[r, c]
            if c < W - 1 and labels[r, c + 1] != lbl:
                pair = tuple(sorted((int(lbl), int(labels[r, c + 1])))); boundary[pair] = boundary.get(pair, 0) + 1
            if r < H - 1 and labels[r + 1, c] != lbl:
                pair = tuple(sorted((int(lbl), int(labels[r + 1, c])))); boundary[pair] = boundary.get(pair, 0) + 1
    max_b = max(boundary.values()) if boundary else 1

    grad_lbl = {}
    if cfg.edge_texture:
        Lt = rgb2lab(image)[:, :, 0]; gyt, gxt = np.gradient(Lt)
        magt = np.sqrt(gxt ** 2 + gyt ** 2)
        for lbl in np.unique(labels):
            grad_lbl[int(lbl)] = float(magt[labels == lbl].mean())
    if cfg.rich_features:
        Lc = rgb2lab(image)[:, :, 0]; gy, gx = np.gradient(Lc)
        mag = np.sqrt(gx ** 2 + gy ** 2); nb = cfg.n_orient_bins
        obin = np.minimum((np.arctan2(gy, gx) % np.pi) / (np.pi / nb), nb - 1).astype(int)
        npx = float(H * W)

    for lbl in np.unique(labels):
        mask = labels == lbl
        f = np.concatenate([image[mask].mean(axis=0), image[mask].std(axis=0)])
        if cfg.rich_features:
            m, b = mag[mask], obin[mask]
            hist = np.bincount(b, weights=m, minlength=cfg.n_orient_bins); hist = hist / (hist.sum() + 1e-6)
            cen = np.argwhere(mask).mean(axis=0) / np.array([H, W])
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


_BUILDERS = {"pixel": build_pixel_graph, "patch": build_patch_graph, "slic": build_slic_graph}


def build_image_graphs(images, labels, cfg: Config):
    from joblib import Parallel, delayed
    from skimage.feature import hog
    builder = _BUILDERS[cfg.graph_type]

    def one(img):
        G = builder(img, cfg)
        # deskryptory wyglądu (baseline'y i hybryda) — wpięte na poziomie grafu
        G.graph["feat_rgb"] = img.reshape(-1, 3).mean(0)
        G.graph["feat_hog"] = hog(img, orientations=8, pixels_per_cell=(8, 8),
                                  cells_per_block=(2, 2), channel_axis=-1)
        return G

    graphs = Parallel(n_jobs=cfg.n2v_workers)(delayed(one)(img) for img in images)
    return graphs, np.asarray(labels, dtype=int)


# ============================================================================
# OBRAZY SYNTETYCZNE (do testów ścieżki obrazowej bez pobierania CIFAR)
# ============================================================================
def synth_images(cfg: Config):
    """3 syntetyczne 'klasy' o różnej teksturze/układzie: gładka (statek/niebo),
    teksturowana (zwierzę), pasy (auto). Pozwala testować buildery i embeddingi."""
    S = cfg.img_size
    n = cfg.per_class or 30
    rng = np.random.default_rng(cfg.seed)
    images, labels = [], []
    for cls in range(3):
        for _ in range(n):
            if cls == 0:  # gładka, niebieskawa + jasna plama
                img = np.zeros((S, S, 3), np.float32)
                img[..., 2] = 0.6 + 0.1 * rng.standard_normal((S, S))
                cx, cy = rng.integers(8, S - 8, 2)
                img[cx - 5:cx + 5, cy - 5:cy + 5, :] = 0.85
            elif cls == 1:  # silna tekstura (szum wysokiej częstotliwości)
                img = 0.4 + 0.3 * rng.standard_normal((S, S, 3)).astype(np.float32)
            else:  # poziome pasy kolorów
                img = np.zeros((S, S, 3), np.float32)
                for r in range(S):
                    img[r, :, r % 3] = 0.3 + 0.5 * (r / S)
            images.append(np.clip(img, 0, 1)); labels.append(cls)
    return images, np.array(labels, dtype=int)


# ============================================================================
# OBFUSKACJA — wspólne pokrętło siły (oś prywatność–użyteczność)
# Działa na dowolnej liście grafów. Atrybuty węzłów ZACHOWANE (zmieniamy tylko
# strukturę) — izoluje to wkład TOPOLOGII i odpowiada edge-DP (perturbacja krawędzi).
# ============================================================================
def _copy_with_nodes(G: nx.Graph) -> nx.Graph:
    H = nx.Graph()
    H.graph.update(G.graph)
    H.add_nodes_from(G.nodes(data=True))
    return H


def obfuscate_graph(G: nx.Graph, method: str, strength: float, rng) -> nx.Graph:
    if strength <= 0 or G.number_of_edges() == 0:
        return G.copy()
    if method == "rewire":
        # losowe zamiany par krawędzi zachowujące stopnie (struktura -> losowa, stopnie te same)
        H = G.copy()
        nsw = max(1, int(strength * H.number_of_edges()))
        try:
            nx.double_edge_swap(H, nswap=nsw, max_tries=nsw * 20, seed=int(rng.integers(1e9)))
        except Exception:
            pass
        return H
    if method == "dropedge":
        H = G.copy()
        for u, v in list(H.edges()):
            if rng.random() < strength:
                H.remove_edge(u, v)
        return H
    if method == "shortcuts":
        H = G.copy(); nodes = list(H.nodes())
        for _ in range(int(strength * H.number_of_nodes())):
            u, v = rng.choice(nodes, 2, replace=False)
            if not H.has_edge(u, v):
                H.add_edge(int(u), int(v), weight=1.0)
        return H
    if method == "er":
        # pełna losowość: Erdős–Rényi o tej samej liczbie węzłów i krawędzi (matched density)
        H = _copy_with_nodes(G); nodes = list(H.nodes()); m = G.number_of_edges()
        possible = len(nodes) * (len(nodes) - 1) // 2
        m = min(m, possible)
        pairs = set()
        while len(pairs) < m:
            u, v = rng.choice(nodes, 2, replace=False)
            pairs.add(tuple(sorted((int(u), int(v)))))
        H.add_edges_from((u, v, {"weight": 1.0}) for u, v in pairs)
        return H
    raise ValueError(f"Nieznana metoda obfuskacji: {method}")


def obfuscate(graphs, method: str, strength: float, seed: int = 42):
    rng = np.random.default_rng(seed)
    return [obfuscate_graph(G, method, strength, rng) for G in graphs]


# ============================================================================
# DISPATCHER
# ============================================================================
def _subsample(graphs, labels, cfg: Config):
    if cfg.per_class is None and cfg.num_samples is None:
        return graphs, labels
    rng = np.random.default_rng(cfg.seed)
    idx = []
    if cfg.per_class is not None:
        for c in np.unique(labels):
            pool = np.where(labels == c)[0]
            idx.extend(rng.choice(pool, min(cfg.per_class, len(pool)), replace=False))
    else:
        idx = rng.choice(len(graphs), min(cfg.num_samples, len(graphs)), replace=False)
    idx = sorted(int(i) for i in idx)
    return [graphs[i] for i in idx], labels[idx]


def load_cifar_images(cfg: Config):
    try:
        import torchvision
        import torchvision.transforms as transforms
    except Exception as e:
        raise RuntimeError(
            "source=cifar wymaga torchvision (pip install torchvision) oraz dostępu do sieci "
            "do pobrania CIFAR-10. Uruchom to na swojej maszynie albo użyj --source synth.") from e
    ds = torchvision.datasets.CIFAR10(root=cfg.data_dir, train=True, download=True,
                                      transform=transforms.ToTensor())
    targets = np.array(ds.targets)
    keep = cfg.classes if cfg.classes is not None else list(range(10))
    remap = {c: i for i, c in enumerate(keep)}
    rng = np.random.default_rng(cfg.seed)
    idxs = []
    for c in keep:
        pool = np.where(targets == c)[0]
        k = cfg.per_class or 100
        idxs.extend(rng.choice(pool, min(k, len(pool)), replace=False))
    idxs = sorted(int(i) for i in idxs)
    images, labels = [], []
    for i in idxs:
        img, lab = ds[i]
        images.append(img.permute(1, 2, 0).numpy().astype(np.float32)); labels.append(remap[int(lab)])
    return images, np.array(labels, dtype=int)


def get_graphs(cfg: Config):
    """Zwraca (graphs, labels, class_names)."""
    if cfg.source == "tu":
        graphs, labels = load_tu(cfg.dataset, cfg.data_dir)
        graphs, labels = _subsample(graphs, labels, cfg)
        names = [str(c) for c in sorted(np.unique(labels))]
    elif cfg.source in ("cifar", "synth"):
        if cfg.source == "cifar":
            images, labels = load_cifar_images(cfg)
            names = ["airplane", "automobile", "bird", "cat", "deer",
                     "dog", "frog", "horse", "ship", "truck"]
            names = [names[c] for c in (cfg.classes or range(10))]
        else:
            images, labels = synth_images(cfg)
            names = ["gladka", "tekstura", "pasy"]
        graphs, labels = build_image_graphs(images, labels, cfg)
    else:
        raise ValueError(f"Nieznane źródło: {cfg.source}")
    # remap etykiet do 0..K-1
    uniq = sorted(np.unique(labels))
    remap = {c: i for i, c in enumerate(uniq)}
    labels = np.array([remap[int(l)] for l in labels], dtype=int)
    return graphs, labels, names