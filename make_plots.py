#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skrypt WYKRESÓW. Czyta results/wyniki.csv (z run_experiments.py) i rysuje:
  - fig_benchmark.png        — ranking reprezentacji (sonda + ARI)
  - fig_privacy_utility.png  — krzywa prywatność–użyteczność (jeśli był --privacy-curve)
  - fig_weight_sweep.png     — wpływ wagi fuzji (struktura vs atrybuty)
  - fig_morphospace.png      — rzut 2D reprezentacji (kolor = klasa)

Użycie:
  python make_plots.py --out-dir results
"""
from __future__ import annotations
import argparse, csv, os
import numpy as np
from graphrep import plots


def load_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()
    od = args.out_dir
    rows = load_rows(os.path.join(od, "wyniki.csv"))
    if not rows:
        print("Brak wierszy w wyniki.csv"); return

    plots.plot_benchmark(rows, os.path.join(od, "fig_benchmark.png"))
    print("zapisano fig_benchmark.png")

    strengths = {float(r["strength"]) for r in rows}
    if len(strengths) > 1:
        plots.plot_privacy_utility(rows, os.path.join(od, "fig_privacy_utility.png"))
        print("zapisano fig_privacy_utility.png")

    if plots.plot_weight_sweep(rows, os.path.join(od, "fig_weight_sweep.png")):
        print("zapisano fig_weight_sweep.png")

    rep = os.path.join(od, "rep_embedding.npy")
    if os.path.exists(rep):
        X = np.load(rep); y = np.load(os.path.join(od, "rep_labels.npy"))
        names = open(os.path.join(od, "rep_names.txt")).read().splitlines() \
            if os.path.exists(os.path.join(od, "rep_names.txt")) else None
        plots.plot_morphospace(X, y, names, os.path.join(od, "fig_morphospace.png"))
        print("zapisano fig_morphospace.png")

    print(f"Gotowe — wykresy w {od}/")


if __name__ == "__main__":
    main()
