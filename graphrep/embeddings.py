#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reprezentacje grafu (jeden wektor na graf) + fuzja strumieni.

Zasada do KLASTROWANIA: preferujemy reprezentacje całografowe, permutacyjnie
niezmiennicze (topo, wl, graph2vec, spectral). node2vec+pooling to metoda poziomu
węzła — zostaje jako KONTRAST (osobna, niewyrównana przestrzeń na graf -> słaba).

Skalowanie zależne od metody:
  topo/attr/node2vec/spectral -> StandardScaler + L2   (cechy gęste, różne jednostki)
  wl/graph2vec                -> tylko L2              (StandardScaler po SVD niszczy sygnał)
Fuzja: każdy strumień osobno -> wektor jednostkowy, mieszany wagą w (anti-dilution).
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import networkx as nx
from collections import Counter

from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import DictVectorizer

from .config import Config
from .features import attr_matrix


# ---------- pojedyncze reprezentacje ----------
def _topo(G: nx.Graph) -> np.ndarray:
    n, m = G.number_of_nodes(), G.number_of_edges()
    if n == 0:
        return np.zeros(12, float)
    degs = np.array([d for _, d in G.degree()], float)
    n_branch = int((degs >= 3).sum()); n_end = int((degs == 1).sum())
    n_cyc = m - n + nx.number_connected_components(G)
    try:
        asr = nx.degree_assortativity_coefficient(G); asr = 0.0 if np.isnan(asr) else asr
    except Exception:
        asr = 0.0
    return np.array([n, m, m / n, nx.density(G) if n > 1 else 0.0,
                     degs.mean(), degs.std(), n_branch, n_end,
                     n_cyc, n_cyc / (n + 1e-9), asr,
                     nx.average_clustering(G) if n > 2 else 0.0], float)


def _spectral(G: nx.Graph, k: int) -> np.ndarray:
    n = G.number_of_nodes()
    if n == 0 or G.number_of_edges() == 0:
        return np.zeros(k, float)
    L = nx.normalized_laplacian_matrix(G).todense()
    ev = np.sort(np.linalg.eigvalsh(np.asarray(L)))[:k]
    if len(ev) < k:
        ev = np.concatenate([ev, np.zeros(k - len(ev))])
    return ev


def _node2vec(G: nx.Graph, cfg: Config) -> np.ndarray:
    from node2vec import Node2Vec
    dim = cfg.n2v_dim
    if G.number_of_edges() == 0:
        return np.zeros(2 * dim, float)
    n2v = Node2Vec(G, dimensions=dim, walk_length=cfg.n2v_walk_length,
                   num_walks=cfg.n2v_num_walks, weight_key="weight",
                   workers=cfg.n2v_workers, quiet=True, seed=cfg.seed)
    model = n2v.fit(window=cfg.n2v_window, min_count=1, seed=cfg.seed)
    # węzły izolowane (np. po obfuskacji) mogą nie trafić do słownika -> zera
    V = np.array([model.wv[str(nd)] if str(nd) in model.wv.key_to_index else np.zeros(dim)
                  for nd in G.nodes()])
    return np.concatenate([V.mean(0), V.std(0)])


def _init_wl_labels(G: nx.Graph) -> dict:
    if G.number_of_nodes() and "label" in next(iter(G.nodes(data=True)))[1]:
        return {n: str(d["label"]) for n, d in G.nodes(data=True)}
    return {n: str(G.degree(n)) for n in G.nodes()}


def embed_wl(graphs, cfg: Config) -> np.ndarray:
    """Feature map jądra Weisfeilera–Lehmana (wspólny słownik) -> TruncatedSVD."""
    node_labels = [_init_wl_labels(G) for G in graphs]
    dicts = [Counter({f"l_{l}": c for l, c in Counter(nl.values()).items()}) for nl in node_labels]
    pat2id: dict[str, int] = {}
    for it in range(cfg.wl_iterations):
        new = []
        for gi, G in enumerate(graphs):
            nl = node_labels[gi]; nn = {}
            for nd in G.nodes():
                pat = nl[nd] + "|" + ",".join(sorted(nl[nb] for nb in G.neighbors(nd)))
                pid = pat2id.setdefault(pat, len(pat2id)); nn[nd] = str(pid)
                dicts[gi][f"wl{it}_{pid}"] += 1
            new.append(nn)
        node_labels = new
    X = DictVectorizer(sparse=True).fit_transform(dicts)
    k = min(cfg.g2v_dim, X.shape[1] - 1, X.shape[0] - 1)
    if k >= 2:
        X = TruncatedSVD(n_components=k, random_state=cfg.seed).fit_transform(X)
    else:
        X = np.asarray(X.todense())
    return X


def embed_graph2vec(graphs, cfg: Config) -> np.ndarray:
    """Graph2Vec = WL re-labeling (wspólny słownik) + Doc2Vec (PV-DBOW)."""
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument
    node_labels = [_init_wl_labels(G) for G in graphs]
    docs = [list(nl.values()) for nl in node_labels]
    pat2id: dict[str, int] = {}
    for _ in range(cfg.wl_iterations):
        new = []
        for gi, G in enumerate(graphs):
            nl = node_labels[gi]; nn = {}
            for nd in G.nodes():
                pat = nl[nd] + "|" + ",".join(sorted(nl[nb] for nb in G.neighbors(nd)))
                pid = pat2id.setdefault(pat, len(pat2id)); nn[nd] = str(pid)
            new.append(nn); docs[gi].extend(nn.values())
        node_labels = new
    tagged = [TaggedDocument(words=(d if d else ["empty"]), tags=[str(i)])
              for i, d in enumerate(docs)]
    model = Doc2Vec(tagged, vector_size=cfg.g2v_dim, dm=0, min_count=1,
                    epochs=cfg.doc2vec_epochs, workers=1, seed=cfg.seed)
    return np.vstack([model.dv[str(i)] for i in range(len(graphs))])


def embed_topo(graphs, cfg: Config) -> np.ndarray:
    return np.vstack([_topo(G) for G in graphs])


def embed_spectral(graphs, cfg: Config) -> np.ndarray:
    return np.vstack([_spectral(G, cfg.spec_k) for G in graphs])


def embed_node2vec(graphs, cfg: Config) -> np.ndarray:
    return np.vstack([_node2vec(G, cfg) for G in graphs])


# --- NetLSD: sygnatura ciepła (permutacyjnie niezmiennicza, niezależna od rozmiaru) ---
_NETLSD_TS = np.logspace(-2, 2, 32)


def _netlsd(G: nx.Graph) -> np.ndarray:
    """Ślad ciepła h(t)=Σ exp(-t·λ) po widmie znormalizowanego Laplasjanu, znormalizowany
    przez |V|. Mocny, tani deskryptor CAŁEGO grafu — dobry do porównywania/klastrowania."""
    n = G.number_of_nodes()
    if n == 0 or G.number_of_edges() == 0:
        return np.zeros(len(_NETLSD_TS))
    L = np.asarray(nx.normalized_laplacian_matrix(G).todense())
    ev = np.linalg.eigvalsh(L)
    return np.array([np.exp(-t * ev).sum() for t in _NETLSD_TS], float) / n


def embed_netlsd(graphs, cfg: Config) -> np.ndarray:
    return np.vstack([_netlsd(G) for G in graphs])


# --- Graphlety/motywy: liczności podgrafów 3- i 4-węzłowych (znormalizowane przez |V|) ---
def _graphlets(G: nx.Graph) -> np.ndarray:
    n = G.number_of_nodes()
    if n == 0:
        return np.zeros(5)
    degs = np.array([d for _, d in G.degree()], float)
    tri = sum(nx.triangles(G).values()) / 3.0                 # trójkąty (3-klika)
    wedges = float(np.sum(degs * (degs - 1) / 2.0)) - 3.0 * tri   # ścieżki 2 (otwarte)
    stars3 = float(np.sum(degs * (degs - 1) * (degs - 2) / 6.0))  # gwiazdy K1,3 (4-węzeł)
    trans = nx.transitivity(G) if n > 2 else 0.0
    sq = float(np.mean(list(nx.square_clustering(G).values()))) if n > 3 else 0.0  # 4-cykle
    inv = 1.0 / n
    return np.array([tri * inv, wedges * inv, stars3 * inv, trans, sq], float)


def embed_graphlet(graphs, cfg: Config) -> np.ndarray:
    return np.vstack([_graphlets(G) for G in graphs])


# --- Baseline'y wyglądu (bez grafu): RGB-mean i HOG; liczone z obrazu i wpięte w G.graph ---
def embed_rgb(graphs, cfg: Config) -> np.ndarray:
    return np.vstack([np.asarray(G.graph.get("feat_rgb", np.zeros(3)), float) for G in graphs])


def embed_hog(graphs, cfg: Config) -> np.ndarray:
    dim = next((len(G.graph["feat_hog"]) for G in graphs if G.graph.get("feat_hog") is not None), 0)
    if dim == 0:
        return np.zeros((len(graphs), 1))
    return np.vstack([np.asarray(G.graph["feat_hog"], float)
                      if G.graph.get("feat_hog") is not None else np.zeros(dim) for G in graphs])


STRUCT_EMBEDDERS = {
    "topo": embed_topo, "spectral": embed_spectral, "node2vec": embed_node2vec,
    "wl": embed_wl, "graph2vec": embed_graph2vec,
    "netlsd": embed_netlsd, "graphlet": embed_graphlet,
    "rgb": embed_rgb, "hog": embed_hog,
}
_SCHEME = {"topo": "standard_l2", "spectral": "standard_l2", "node2vec": "standard_l2",
           "wl": "l2", "graph2vec": "l2", "netlsd": "standard_l2", "graphlet": "standard_l2",
           "rgb": "standard_l2", "hog": "standard_l2"}


# ---------- normalizacja i fuzja ----------
def _block_norm(X: np.ndarray, scheme: str) -> np.ndarray:
    X = np.nan_to_num(np.asarray(X, float))
    if scheme == "standard_l2":
        X = StandardScaler().fit_transform(X)
    return normalize(np.nan_to_num(X))


def build_structural(graphs, names, cfg: Config) -> np.ndarray:
    """Strumień strukturalny = konkatenacja znormalizowanych embedderów -> jednostkowy."""
    blocks = [_block_norm(STRUCT_EMBEDDERS[name](graphs, cfg), _SCHEME[name]) for name in names]
    return normalize(np.nan_to_num(np.hstack(blocks)))


def embed_method(graphs, recipe: dict, w: float, cfg: Config) -> np.ndarray:
    """recipe = {'struct': [nazwy_embedderów], 'attr': bool}.
    w=1 -> tylko struktura; w=0 -> tylko atrybuty; pomiędzy -> fuzja."""
    parts = []
    struct = recipe.get("struct") or []
    if struct and w > 0:
        parts.append(w * build_structural(graphs, struct, cfg))
    if recipe.get("attr") and w < 1:
        A = normalize(_block_norm(attr_matrix(graphs), "standard_l2"))
        parts.append((1.0 - w) * A)
    if not parts:  # zabezpieczenie
        parts = [build_structural(graphs, struct or ["topo"], cfg)]
    return np.hstack(parts)