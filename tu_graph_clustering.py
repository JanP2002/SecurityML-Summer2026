#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Klasteryzacja CAŁYCH grafów (zbiór wielu grafów) — benchmark reprezentacji
================================================================================
Zbiór: TUDataset, domyślnie ENZYMES (600 grafów, 6 klas po 100; węzły = elementy
struktury drugorzędowej białka, z etykietą typu {helisa, kartka, zwrot} i 18-wym.
atrybutami fizyko-chemicznymi).

To jest ścieżka "B": jeden wektor na CAŁY graf -> klasteryzacja grafów. Pozwala
porównać Graph2Vec (natywnie) z Node2Vec+agregacja na DOKŁADNIE tym samym zbiorze.

Reprezentacje (jeden wektor na graf):
  topo      : cechy topologiczne grafu (rozmiar, cykle, stopnie...) — strukturalny baseline
  attr      : pooling atrybutów węzłów (mean⊕std 18-wym) — baseline dla grafów atrybutowanych
  wl        : feature map jądra Weisfeilera–Lehmana, inicjalizacja ETYKIETĄ węzła
  graph2vec : WL (etykieta węzła) + Doc2Vec — UCZONA, dobra przy dużym korpusie
  node2vec  : Node2Vec na węzłach + agregacja mean⊕std — metoda poziomu węzła

Skalowanie zależne od metody:
  topo / attr / node2vec -> StandardScaler ;  wl / graph2vec -> (SVD) + L2.

Ocena: KMeans (k=#klas) + HDBSCAN, metryki wewnętrzne (silhouette, DB, CH) i
zewnętrzne (ARI, NMI, V-measure), nadzorowana sonda kNN (cross-val acc/F1),
morfoprzestrzeń 2D (PCA). UWAGA: ENZYMES to znany TRUDNY problem 6-klasowy —
nawet nadzorowane SOTA bez atrybutów bywa ~30–40%, więc niskie ARI to nie błąd.

Uruchomienie:
  python tu_graph_clustering.py --all-embeddings          # auto-pobiera ENZYMES
  python tu_graph_clustering.py --name ENZYMES --embedding graph2vec
"""

from __future__ import annotations
import argparse, os, csv, warnings, urllib.request
from dataclasses import dataclass
from collections import Counter

import numpy as np
import scipy.sparse as sp
warnings.filterwarnings("ignore")

import networkx as nx
from node2vec import Node2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.feature_extraction import DictVectorizer
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score, normalized_mutual_info_score,
    homogeneity_completeness_v_measure,
)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class Config:
    name: str = "ENZYMES"
    data_dir: str = "data"
    wl_iterations: int = 3
    struct_dim: int = 64
    doc2vec_epochs: int = 100
    n2v_walk_length: int = 12
    n2v_num_walks: int = 20
    n2v_window: int = 5
    n2v_workers: int = max(1, (os.cpu_count() or 2) - 1)
    k: int | None = None
    hdbscan_min_cluster_size: int = 10
    seed: int = 42
    out_dir: str = "results"


# ============================================================================
# WCZYTYWANIE TUDataset (format Dortmund: DS_A / graph_indicator / graph_labels
# / node_labels / node_attributes)
# ============================================================================
_GRAPHRNN = "https://raw.githubusercontent.com/snap-stanford/GraphRNN/master/dataset"

def download_tu(name: str, root: str) -> str:
    """Pobiera surowe pliki TU z mirrora na GitHub (jeśli ich nie ma)."""
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
    """Zwraca (graphs: list[nx.Graph], labels: list[int]).
    Węzły mają atrybut 'label' (int) i 'x' (wektor atrybutów), jeśli dostępne."""
    d = os.path.join(root, name)
    if not os.path.exists(os.path.join(d, f"{name}_A.txt")):
        d = download_tu(name, root)
    p = lambda s: os.path.join(d, f"{name}_{s}.txt")

    A = np.loadtxt(p("A"), delimiter=",", dtype=int)            # (m,2) 1-indeksowane
    indicator = np.loadtxt(p("graph_indicator"), dtype=int)     # (n,) id grafu na węzeł
    glabels = np.loadtxt(p("graph_labels"), dtype=int)          # (N,)
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
            attrs["x"] = nattrs[i]
        G[g].add_node(i + 1, **attrs)          # node_id = pozycja+1
    for u, v in A:
        G[int(indicator[u - 1])].add_edge(int(u), int(v))

    graphs = [G[g] for g in gids]
    labels = [int(glabels[g - 1]) for g in gids]
    return graphs, labels


# ============================================================================
# REPREZENTACJE
# ============================================================================
def _init_labels(G: nx.Graph) -> dict:
    """Początkowe etykiety WL: typ węzła ('label') jeśli jest, inaczej stopień."""
    if G.number_of_nodes() and "label" in next(iter(G.nodes(data=True)))[1]:
        return {n: str(d["label"]) for n, d in G.nodes(data=True)}
    return {n: str(G.degree(n)) for n in G.nodes()}


def graph_topological_features(G: nx.Graph) -> np.ndarray:
    n, m = G.number_of_nodes(), G.number_of_edges()
    if n == 0:
        return np.zeros(12, float)
    degs = np.array([d for _, d in G.degree()], float)
    n_branch = int((degs >= 3).sum()); n_end = int((degs == 1).sum())
    n_cycles = m - n + nx.number_connected_components(G)
    try:
        asr = nx.degree_assortativity_coefficient(G); asr = 0.0 if np.isnan(asr) else asr
    except Exception:
        asr = 0.0
    return np.array([
        n, m, m / n, nx.density(G) if n > 1 else 0.0,
        degs.mean(), degs.std(), degs.max(), n_branch, n_end,
        n_cycles, asr, nx.average_clustering(G) if n > 2 else 0.0,
    ], float)


def _attr_dim(graphs) -> int:
    for G in graphs:
        for _, d in G.nodes(data=True):
            if "x" in d:
                return len(np.atleast_1d(d["x"]))
    return 0


def embed_attr(G: nx.Graph, dim: int) -> np.ndarray:
    xs = [np.atleast_1d(d["x"]) for _, d in G.nodes(data=True) if "x" in d]
    if not xs:
        return np.zeros(2 * dim, float)
    X = np.vstack(xs).astype(float)
    return np.concatenate([X.mean(0), X.std(0)])


def wl_feature_matrix(graphs, iterations) -> sp.csr_matrix:
    """Feature map jądra WL (wspólny słownik), inicjalizacja etykietą węzła."""
    node_labels = [_init_labels(G) for G in graphs]
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
            new.append(nn)
        node_labels = new
    return DictVectorizer(sparse=True).fit_transform(dicts)


def embed_graph2vec(graphs, cfg: Config) -> np.ndarray:
    node_labels = [_init_labels(G) for G in graphs]
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
    model = Doc2Vec(tagged, vector_size=cfg.struct_dim, dm=0, min_count=1,
                    epochs=cfg.doc2vec_epochs, workers=1, seed=cfg.seed)
    return np.vstack([model.dv[str(i)] for i in range(len(graphs))])


def embed_node2vec(G: nx.Graph, cfg: Config) -> np.ndarray:
    dim = cfg.struct_dim
    if G.number_of_edges() == 0:
        return np.zeros(2 * dim, float)
    n2v = Node2Vec(G, dimensions=dim, walk_length=cfg.n2v_walk_length,
                   num_walks=cfg.n2v_num_walks, workers=cfg.n2v_workers,
                   quiet=True, seed=cfg.seed)
    model = n2v.fit(window=cfg.n2v_window, min_count=1, seed=cfg.seed)
    V = np.array([model.wv[str(n)] for n in G.nodes()])
    return np.concatenate([V.mean(0), V.std(0)])


def build_embeddings(graphs, method, cfg: Config):
    if method == "topo":
        return np.vstack([graph_topological_features(G) for G in graphs])
    if method == "attr":
        dim = _attr_dim(graphs)
        if dim == 0:
            raise ValueError("Brak atrybutów węzłów ('x') — metoda 'attr' niedostępna.")
        return np.vstack([embed_attr(G, dim) for G in graphs])
    if method == "wl":
        return wl_feature_matrix(graphs, cfg.wl_iterations)
    if method == "graph2vec":
        return embed_graph2vec(graphs, cfg)
    if method == "node2vec":
        rows = []
        for i, G in enumerate(graphs):
            rows.append(embed_node2vec(G, cfg))
            if (i + 1) % 50 == 0 or i + 1 == len(graphs):
                print(f"  [node2vec] {i+1}/{len(graphs)}", end="\r")
        print()
        return np.vstack(rows)
    raise ValueError(f"Nieznana metoda: {method}")


def preprocess(X, method, cfg: Config) -> np.ndarray:
    if method in ("topo", "attr", "node2vec"):
        return StandardScaler().fit_transform(np.nan_to_num(np.asarray(X, float)))
    # wl / graph2vec -> (SVD) + L2
    if sp.issparse(X):
        k = min(cfg.struct_dim * 2, X.shape[1] - 1, X.shape[0] - 1)
        X = TruncatedSVD(n_components=max(2, k), random_state=cfg.seed).fit_transform(X)
    else:
        X = np.asarray(X, float)
        if X.shape[1] > cfg.struct_dim * 2:
            k = min(cfg.struct_dim * 2, X.shape[1] - 1, X.shape[0] - 1)
            X = TruncatedSVD(n_components=max(2, k), random_state=cfg.seed).fit_transform(X)
    return normalize(np.nan_to_num(X))


# ============================================================================
# KLASTERYZACJA / OCENA
# ============================================================================
def cluster_metrics(X, pred, y):
    res = {"n_clusters": len({c for c in set(pred) if c != -1}),
           "noise": float(np.mean(pred == -1))}
    mask = pred != -1
    if mask.sum() > 2 and len(set(pred[mask])) >= 2:
        res["silhouette"] = float(silhouette_score(X[mask], pred[mask]))
        res["davies_bouldin"] = float(davies_bouldin_score(X[mask], pred[mask]))
        res["calinski"] = float(calinski_harabasz_score(X[mask], pred[mask]))
    else:
        res.update(silhouette=float("nan"), davies_bouldin=float("nan"), calinski=float("nan"))
    res["ARI"] = float(adjusted_rand_score(y, pred))
    res["NMI"] = float(normalized_mutual_info_score(y, pred))
    _, _, v = homogeneity_completeness_v_measure(y, pred); res["Vmeasure"] = float(v)
    return res


def supervised_probe(X, y, seed):
    counts = np.bincount(y)
    n_splits = int(min(5, counts[counts > 0].min()))
    if n_splits < 2:
        return {"probe_acc": float("nan"), "probe_f1": float("nan")}
    clf = KNeighborsClassifier(n_neighbors=5)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return {"probe_acc": float(cross_val_score(clf, X, y, cv=cv, scoring="accuracy").mean()),
            "probe_f1": float(cross_val_score(clf, X, y, cv=cv, scoring="f1_macro").mean())}


def plot_2d(X, labels, title, path):
    Z = PCA(n_components=2, random_state=0).fit_transform(X) if X.shape[1] > 2 else X
    plt.figure(figsize=(6, 5))
    for lab in sorted(set(labels)):
        pts = Z[labels == lab]
        plt.scatter(pts[:, 0], pts[:, 1], s=14, alpha=0.7,
                    label=("szum" if lab == -1 else str(lab)))
    plt.title(title); plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.legend(fontsize=7, ncol=2); plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


def run_one(graphs, y, n_classes, method, cfg, plots=True):
    X = build_embeddings(graphs, method, cfg)
    Xs = preprocess(X, method, cfg)
    k = cfg.k or n_classes
    km = KMeans(n_clusters=k, n_init=10, random_state=cfg.seed).fit_predict(Xs)
    hdb = HDBSCAN(min_cluster_size=cfg.hdbscan_min_cluster_size).fit_predict(Xs)
    km_m, hdb_m = cluster_metrics(Xs, km, y), cluster_metrics(Xs, hdb, y)
    probe = supervised_probe(Xs, y, cfg.seed)
    if plots:
        os.makedirs(cfg.out_dir, exist_ok=True)
        plot_2d(Xs, y, f"Morfoprzestrzeń (klasy) · {method}",
                os.path.join(cfg.out_dir, f"{method}_morphospace.png"))
        plot_2d(Xs, km, f"KMeans · {method}", os.path.join(cfg.out_dir, f"{method}_kmeans.png"))
    return {"method": method, "dim": Xs.shape[1],
            "km_ARI": km_m["ARI"], "km_NMI": km_m["NMI"], "km_silhouette": km_m["silhouette"],
            "hdb_ARI": hdb_m["ARI"], "hdb_clusters": hdb_m["n_clusters"], "hdb_noise": hdb_m["noise"],
            "probe_acc": probe["probe_acc"], "probe_f1": probe["probe_f1"]}


def run_benchmark(graphs, labels_raw, methods, cfg):
    names = sorted(set(labels_raw))
    y = np.array([names.index(l) for l in labels_raw])
    print(f"Grafy: {len(graphs)} | klasy ({len(names)}): {names} | "
          f"liczności: {np.bincount(y).tolist()}")
    sizes = [g.number_of_nodes() for g in graphs]
    print(f"|V| min/median/max = {min(sizes)}/{int(np.median(sizes))}/{max(sizes)}\n")
    rows = [run_one(graphs, y, len(names), m, cfg) for m in methods]

    cols = ["dim", "km_ARI", "km_NMI", "km_silhouette", "hdb_ARI",
            "hdb_clusters", "hdb_noise", "probe_acc", "probe_f1"]
    head = f"{'method':<11}" + "".join(f"{c:>14}" for c in cols)
    print("=== Benchmark reprezentacji (ARI/NMI/probe: wyżej = lepiej) ===")
    print(head); print("-" * len(head))
    for r in rows:
        line = f"{r['method']:<11}"
        for c in cols:
            v = r.get(c, "")
            line += f"{v:>14.3f}" if isinstance(v, float) else f"{str(v):>14}"
        print(line)
    chance = 1.0 / len(names)
    print(f"\n(Poziom losowy sondy = {chance:.3f}; ARI losowe ≈ 0.0)")

    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "porownanie_metod.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method"] + cols); w.writeheader(); w.writerows(rows)
    print(f"Zapisano: {cfg.out_dir}/porownanie_metod.csv oraz wykresy *_morphospace.png")
    return rows


def main():
    ap = argparse.ArgumentParser(description="Klasteryzacja całych grafów (TUDataset).")
    ap.add_argument("--name", default="ENZYMES", help="nazwa zbioru TU (np. ENZYMES, PROTEINS, MUTAG)")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--embedding", choices=["topo", "attr", "wl", "graph2vec", "node2vec"], default="graph2vec")
    ap.add_argument("--all-embeddings", action="store_true")
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    cfg = Config(name=args.name, data_dir=args.data_dir, k=args.k, out_dir=args.out_dir)
    print(f"[{cfg.name}] wczytywanie...")
    graphs, labels = load_tu_dataset(cfg.name, cfg.data_dir)

    methods = (["topo", "attr", "wl", "graph2vec", "node2vec"]
               if args.all_embeddings else [args.embedding])
    # pomiń 'attr' jeśli brak atrybutów węzłów
    if "attr" in methods and _attr_dim(graphs) == 0:
        print("(brak atrybutów węzłów -> pomijam 'attr')")
        methods = [m for m in methods if m != "attr"]

    run_benchmark(graphs, labels, methods, cfg)


if __name__ == "__main__":
    main()