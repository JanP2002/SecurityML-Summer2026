#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pomocnik do eksploracji: zwięzłe podsumowanie wyniki.csv (best-w per metoda).

Użycie:
  python summarize.py <out-dir> [<out-dir2> ...]
Dla każdego pliku wypisuje tabelę (metoda, w, ARI, NMI, silhouette, probe_acc, recon_r2),
wybierając wiersz o najwyższym probe_acc per metoda (jak w pipeline). Dla porównania selekcji
bezetykietowej pokazuje też wiersz best-silhouette, jeśli różny.
"""
import csv, sys
from collections import defaultdict


def load(out_dir):
    path = f"{out_dir}/wyniki.csv"
    with open(path) as f:
        return [r for r in csv.DictReader(f)]


def fnum(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return float("nan")


def best_by(rows, key, hi=True):
    cand = [r for r in rows if r[key] not in ("", "nan")]
    if not cand:
        return rows[0]
    return (max if hi else min)(cand, key=lambda r: fnum(r[key]))


def summarize(out_dir):
    rows = load(out_dir)
    bymeth = defaultdict(list)
    for r in rows:
        bymeth[r["method"]].append(r)
    print(f"\n=== {out_dir}  ({len(rows)} wierszy) ===")
    hdr = f"{'method':9} {'w':>4} {'ARI':>7} {'NMI':>7} {'silh':>7} {'acc':>7} {'recon':>8}"
    print(hdr)
    for m, rs in bymeth.items():
        b = best_by(rs, "probe_acc")
        print(f"{m:9} {b['w']:>4} {fnum(b['km_ARI']):7.3f} {fnum(b['km_NMI']):7.3f} "
              f"{fnum(b['silhouette']):7.3f} {fnum(b['probe_acc']):7.3f} {fnum(b['recon_r2']):8.3f}")
        # wiersz wybrany po silhouette (selekcja bezetykietowa), jeśli inny w
        bs = best_by(rs, "silhouette")
        if bs["w"] != b["w"]:
            print(f"{'  ↳silh':9} {bs['w']:>4} {fnum(bs['km_ARI']):7.3f} {fnum(bs['km_NMI']):7.3f} "
                  f"{fnum(bs['silhouette']):7.3f} {fnum(bs['probe_acc']):7.3f} {fnum(bs['recon_r2']):8.3f}")


if __name__ == "__main__":
    for d in sys.argv[1:] or ["results"]:
        summarize(d)
