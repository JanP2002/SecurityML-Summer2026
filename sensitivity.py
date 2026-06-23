#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WRAŻLIWOŚĆ kluczowych wyników na ziarno losowości / podział CV.
Dla danej (source, dataset, deskryptor) liczy km_ARI/km_NMI/recon_leak po WIELU seedach
i raportuje mean ± std. Seed wpływa na: podpróbkę (cifar), seed-labels KMeans (cifar),
init KMeans, podział CV (sonda/atak). Embedding strukturalny TU jest deterministyczny przy
braku podpróbki → wariancja ARI pochodzi tylko z initu KMeans; dla cifar także z podpróbki.

Użycie:
  python sensitivity.py --source tu --dataset PROTEINS_full --struct fgsd --seeds 42 0 1 2 7
  python sensitivity.py --source cifar --classes 0 1 8 --per-class 500 --struct wl --seeds 42 0 1 2 7
"""
from __future__ import annotations
import argparse
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from graphrep.config import Config
from graphrep import data, features, evaluate
from graphrep.embeddings import build_structural


def one_seed(source, dataset, struct, classes, per_class, seed):
    cfg = Config(source=source, dataset=dataset, classes=classes, per_class=per_class,
                 graph_type="slic", seed=seed)
    graphs, y, names = data.get_graphs(cfg)
    graphs = features.ensure_labels(graphs, cfg)
    X = build_structural(graphs, struct, cfg)
    k = len(names)
    pred = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
    ari = adjusted_rand_score(y, pred); nmi = normalized_mutual_info_score(y, pred)
    target = evaluate.content_target(graphs)
    leak = evaluate.reconstruction_attack(X, target, cfg)["recon_leak"]
    # recon_leak strumienia attr (treść) — referencja prywatności
    from graphrep.features import attr_matrix
    from graphrep.embeddings import _block_norm
    from sklearn.preprocessing import normalize
    A = normalize(_block_norm(attr_matrix(graphs), "standard_l2"))
    leak_attr = evaluate.reconstruction_attack(A, target, cfg)["recon_leak"]
    return ari, nmi, leak, leak_attr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="tu")
    ap.add_argument("--dataset", default="PROTEINS_full")
    ap.add_argument("--struct", nargs="+", default=["fgsd"])
    ap.add_argument("--classes", type=int, nargs="+", default=None)
    ap.add_argument("--per-class", type=int, default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 0, 1, 2, 7])
    args = ap.parse_args()

    rows = []
    for s in args.seeds:
        r = one_seed(args.source, args.dataset, args.struct, args.classes, args.per_class, s)
        rows.append(r)
        print(f"  seed={s:>3}  ARI={r[0]:.3f}  NMI={r[1]:.3f}  recon_leak[{'+'.join(args.struct)}]={r[2]:.3f}  recon_leak[attr]={r[3]:.3f}")
    a = np.array(rows)
    lbl = '+'.join(args.struct)
    print(f"\n  [{args.source}:{args.dataset} {lbl}]  ({len(args.seeds)} seedów)")
    for i, name in enumerate(["ARI", "NMI", f"recon_leak[{lbl}]", "recon_leak[attr]"]):
        print(f"    {name:22} = {a[:,i].mean():.3f} ± {a[:,i].std():.3f}   (min {a[:,i].min():.3f}, max {a[:,i].max():.3f})")


if __name__ == "__main__":
    main()
