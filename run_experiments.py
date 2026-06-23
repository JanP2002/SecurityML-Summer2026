#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skrypt EKSPERYMENTÓW frameworka graphrep.

Wymiary eksperymentu:
  (źródło grafu) × (reprezentacja) × (waga fuzji w) × (siła obfuskacji)

Mierzymy jednocześnie UŻYTECZNOŚĆ (klastrowanie ARI/NMI = wyciek bez etykiet; sonda
nadzorowana = separowalność) i PRYWATNOŚĆ (atak rekonstrukcyjny R² = odwracalność).
Z obfuskacją (sweep siły) dostajemy krzywą prywatność–użyteczność.

Przykłady:
  # benchmark reprezentacji na ENZYMES:
  python run_experiments.py --source tu --dataset ENZYMES --per-class 50

  # białka:
  python run_experiments.py --source tu --dataset PROTEINS_full --per-class 100

  # krzywa prywatność–użyteczność (obfuskacja przez rewiring) na ENZYMES:
  python run_experiments.py --source tu --dataset ENZYMES --per-class 50 --privacy-curve

  # ścieżka obrazowa kolegów na danych syntetycznych (bez pobierania CIFAR):
  python run_experiments.py --source synth --graph-type slic --per-class 30
  # prawdziwy CIFAR (na maszynie z torchvision + siecią):
  python run_experiments.py --source cifar --graph-type slic --classes 0 1 8 --per-class 200
"""
from __future__ import annotations
import argparse, os, csv, hashlib, time
import numpy as np

from graphrep.config import Config
from graphrep import data, features, evaluate
from graphrep.embeddings import STRUCT_EMBEDDERS, _SCHEME, _block_norm, attr_matrix
from sklearn.preprocessing import normalize


# recepty metod (źródło-agnostyczne); 'sweep' -> przemiatamy wagę w
METHODS = {
    "topo":     {"struct": ["topo"],             "attr": False, "sweep": False, "w": 1.0},
    "spec":     {"struct": ["spectral"],         "attr": False, "sweep": False, "w": 1.0},
    "wl":       {"struct": ["wl"],               "attr": False, "sweep": False, "w": 1.0},
    "g2v":      {"struct": ["graph2vec"],         "attr": False, "sweep": False, "w": 1.0},
    "n2v":      {"struct": ["node2vec"],          "attr": False, "sweep": False, "w": 1.0},
    "netlsd":   {"struct": ["netlsd"],            "attr": False, "sweep": False, "w": 1.0},
    "graphlet": {"struct": ["graphlet"],          "attr": False, "sweep": False, "w": 1.0},
    "fgsd":     {"struct": ["fgsd"],              "attr": False, "sweep": False, "w": 1.0},
    "rand_ens": {"struct": ["rand_ens"],          "attr": False, "sweep": False, "w": 1.0},
    # fuzje całografowych, permutacyjnie niezmienniczych deskryptorów (czysta struktura)
    "wltopo":   {"struct": ["wl", "topo"],                       "attr": False, "sweep": False, "w": 1.0},
    "sfuse":    {"struct": ["topo", "wl", "netlsd", "graphlet"], "attr": False, "sweep": False, "w": 1.0},
    "sfuse2":   {"struct": ["topo", "netlsd", "graphlet"],       "attr": False, "sweep": False, "w": 1.0},
    "attr":     {"struct": [],                    "attr": True,  "sweep": False, "w": 0.0},
    "combo":    {"struct": ["graph2vec"],          "attr": True,  "sweep": True},
    "gnat":     {"struct": ["graph2vec", "topo"],  "attr": True,  "sweep": True},
    # tylko dla grafów z obrazu (source=cifar/synth):
    "rgb":      {"struct": ["rgb"],               "attr": False, "sweep": False, "w": 1.0},
    "hog":      {"struct": ["hog"],               "attr": False, "sweep": False, "w": 1.0},
    "hyb":      {"struct": ["graph2vec"], "second": "hog", "attr": False, "sweep": True},
}
DEFAULT_METHODS = ["topo", "wl", "g2v", "n2v", "attr", "combo", "gnat"]

CSV_COLS = ["source", "dataset", "graph_type", "obf_method", "strength", "method", "w",
            "km_ARI", "km_NMI", "silhouette", "n_clusters", "noise",
            "probe_acc", "probe_f1", "recon_r2", "recon_nmse", "recon_leak", "dim"]


def _cache_key(name, cfg: Config, strength, n) -> str:
    s = "|".join(map(str, [name, cfg.source, cfg.dataset, cfg.graph_type, cfg.obf_method,
                           round(strength, 4), n, cfg.seed, cfg.wl_iterations, cfg.g2v_dim,
                           cfg.doc2vec_epochs, cfg.n2v_dim, cfg.n2v_num_walks, cfg.spec_k,
                           cfg.edge_quantile, cfg.rich_features, cfg.label_rich,
                           cfg.rand_ens_m, cfg.rand_ens_base, cfg.rand_ens_method, cfg.rand_ens_p]))
    return hashlib.sha1(s.encode()).hexdigest()[:16]


def _raw_block(name, graphs, cfg: Config, strength, cache_dir):
    """Surowa macierz embeddera (z cache na dysku — drogie graph2vec/node2vec liczone raz)."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{name}_{_cache_key(name, cfg, strength, len(graphs))}.npy")
    if os.path.exists(path):
        return np.load(path)
    X = STRUCT_EMBEDDERS[name](graphs, cfg)
    np.save(path, X)
    return X


def run(cfg: Config, method_names, privacy_curve: bool):
    os.makedirs(cfg.out_dir, exist_ok=True)
    cache_dir = os.path.join(cfg.out_dir, "cache")

    print(f"[{cfg.source}:{cfg.dataset}] budowanie grafów...")
    graphs, y, names = data.get_graphs(cfg)
    graphs = features.ensure_labels(graphs, cfg)
    target = evaluate.content_target(graphs)          # cel ataku rekonstrukcyjnego
    k = cfg.k or len(names)
    sizes = [g.number_of_nodes() for g in graphs]
    print(f"Grafy: {len(graphs)} | klasy ({len(names)}): {names} | "
          f"|V| min/med/max = {min(sizes)}/{int(np.median(sizes))}/{max(sizes)}")

    strengths = cfg.obf_strengths if privacy_curve else [0.0]
    needed = sorted({s for m in method_names
                     for s in (METHODS[m]["struct"]
                               + ([METHODS[m]["second"]] if METHODS[m].get("second") else []))})
    use_attr = any(METHODS[m].get("attr") for m in method_names)

    rows = []
    rep_X = rep_y = None                               # do morfoprzestrzeni (strength=0)
    for strength in strengths:
        gobf = data.obfuscate(graphs, cfg.obf_method, strength, cfg.seed) if strength > 0 else graphs
        t0 = time.time()
        raw = {name: _raw_block(name, gobf, cfg, strength, cache_dir) for name in needed}
        attr_unit = normalize(_block_norm(attr_matrix(gobf), "standard_l2")) if use_attr else None

        def struct_unit(snames):
            blocks = [_block_norm(raw[s], _SCHEME[s]) for s in snames]
            return normalize(np.hstack(blocks))

        for m in method_names:
            rec = METHODS[m]
            S = struct_unit(rec["struct"]) if rec["struct"] else None
            if rec.get("second"):
                second_unit = struct_unit([rec["second"]])      # np. HOG dla hybrydy
            elif rec.get("attr"):
                second_unit = attr_unit
            else:
                second_unit = None
            ws = cfg.weights if rec["sweep"] else [rec["w"]]
            method_rows, Xs_by_w = [], {}
            for w in ws:
                parts = []
                if S is not None and w > 0:
                    parts.append(w * S)
                if second_unit is not None and w < 1:
                    parts.append((1 - w) * second_unit)
                if not parts:
                    parts = [S if S is not None else second_unit]
                X = np.hstack(parts); Xs_by_w[w] = X
                pred = evaluate.cluster(X, cfg.cluster_algo, k, cfg)
                um = evaluate.unsupervised_metrics(X, pred, y)
                pr = evaluate.supervised_probe(X, y, cfg)
                method_rows.append({
                    "source": cfg.source, "dataset": cfg.dataset, "graph_type": cfg.graph_type,
                    "obf_method": cfg.obf_method, "strength": strength, "method": m, "w": w,
                    "km_ARI": um["ARI"], "km_NMI": um["NMI"], "silhouette": um["silhouette"],
                    "n_clusters": um["n_clusters"], "noise": um["noise"],
                    "probe_acc": pr["probe_acc"], "probe_f1": pr["probe_f1"],
                    "recon_r2": float("nan"), "recon_nmse": float("nan"),
                    "recon_leak": float("nan"), "dim": X.shape[1]})
            # atak rekonstrukcyjny tylko dla najlepszego w (oszczędność)
            best = max(method_rows, key=lambda r: (r["probe_acc"] if not np.isnan(r["probe_acc"]) else -1))
            rr = evaluate.reconstruction_attack(Xs_by_w[best["w"]], target, cfg)
            best["recon_r2"] = rr["recon_r2"]
            best["recon_nmse"] = rr["recon_nmse"]
            best["recon_leak"] = rr["recon_leak"]
            rows.extend(method_rows)
            if strength == 0.0 and rep_X is None and m in ("combo", "g2v", "gnat"):
                rep_X, rep_y = Xs_by_w[best["w"]], y
            print(f"  s={strength:.2f} {m:<6} ARI={best['km_ARI']:.3f} "
                  f"acc={best['probe_acc']:.3f} recon_r2={best['recon_r2']:.3f}")
        print(f"  (siła {strength:.2f} gotowa w {time.time()-t0:.1f}s)")

    # zapis CSV
    csv_path = os.path.join(cfg.out_dir, "wyniki.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    if rep_X is not None:
        np.save(os.path.join(cfg.out_dir, "rep_embedding.npy"), rep_X)
        np.save(os.path.join(cfg.out_dir, "rep_labels.npy"), rep_y)
        with open(os.path.join(cfg.out_dir, "rep_names.txt"), "w") as f:
            f.write("\n".join(names))
    print(f"\nZapisano: {csv_path}  ({len(rows)} wierszy)")
    print(f"Wykresy:  python make_plots.py --out-dir {cfg.out_dir}")
    return rows


def main():
    ap = argparse.ArgumentParser(description="Eksperymenty: reprezentacje grafowe → klastrowanie → bezpieczeństwo.")
    ap.add_argument("--source", choices=["tu", "cifar", "synth"], default="tu")
    ap.add_argument("--dataset", default="ENZYMES", help="ENZYMES | PROTEINS_full (tu)")
    ap.add_argument("--graph-type", choices=["pixel", "patch", "slic"], default="slic")
    ap.add_argument("--classes", type=int, nargs="+", default=None)
    ap.add_argument("--per-class", type=int, default=None)
    ap.add_argument("--num-samples", type=int, default=None)
    ap.add_argument("--methods", nargs="+", default=None,
                    help=f"podzbiór z: {list(METHODS)} (domyślnie {DEFAULT_METHODS})")
    ap.add_argument("--weights", type=float, nargs="+", default=None, help="wagi fuzji do przemiatania")
    ap.add_argument("--cluster-algo", choices=["kmeans", "spectral", "agglo", "hdbscan"], default="kmeans")
    ap.add_argument("--privacy-curve", action="store_true", help="przemiataj siłę obfuskacji")
    ap.add_argument("--obf-method", choices=["rewire", "dropedge", "shortcuts", "er", "edp", "feature"], default="rewire")
    ap.add_argument("--obf-strengths", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--rich-features", action="store_true")
    ap.add_argument("--label-rich", action="store_true")
    ap.add_argument("--edge-quantile", type=float, default=0.6)
    ap.add_argument("--n-segments", type=int, default=None, help="liczba superpikseli SLIC")
    ap.add_argument("--compactness", type=float, default=None, help="zwartość SLIC")
    ap.add_argument("--rand-ens-m", type=int, default=None, help="rand_ens: liczba losowych wariantów/obiekt")
    ap.add_argument("--rand-ens-base", default=None, help="rand_ens: bazowy deskryptor (topo|wl|graph2vec|netlsd|graphlet|fgsd)")
    ap.add_argument("--rand-ens-method", default=None, help="rand_ens: dropedge|shortcuts|er")
    ap.add_argument("--rand-ens-p", type=float, default=None, help="rand_ens: siła losowania")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    cfg = Config(source=args.source, dataset=args.dataset, graph_type=args.graph_type,
                 classes=args.classes, per_class=args.per_class, num_samples=args.num_samples,
                 cluster_algo=args.cluster_algo, obf_method=args.obf_method,
                 obf_strengths=args.obf_strengths, rich_features=args.rich_features,
                 label_rich=args.label_rich, edge_quantile=args.edge_quantile,
                 data_dir=args.data_dir, out_dir=args.out_dir)
    if args.weights:
        cfg.weights = args.weights
    if args.n_segments is not None:
        cfg.n_segments = args.n_segments
    if args.compactness is not None:
        cfg.compactness = args.compactness
    if args.rand_ens_m is not None:
        cfg.rand_ens_m = args.rand_ens_m
    if args.rand_ens_base is not None:
        cfg.rand_ens_base = args.rand_ens_base
    if args.rand_ens_method is not None:
        cfg.rand_ens_method = args.rand_ens_method
    if args.rand_ens_p is not None:
        cfg.rand_ens_p = args.rand_ens_p
    methods = args.methods or DEFAULT_METHODS
    run(cfg, methods, args.privacy_curve)


if __name__ == "__main__":
    main()