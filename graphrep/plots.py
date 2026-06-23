#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wykresy frameworka (czytane przez make_plots.py)."""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def _best_per_method(rows, key="probe_acc", strength=0.0):
    best = {}
    for r in rows:
        if abs(float(r["strength"])) > 1e-9 and strength == 0.0:
            continue
        m = r["method"]; v = float(r[key])
        if m not in best or v > float(best[m][key]):
            best[m] = r
    return best


def plot_benchmark(rows, path, title="Benchmark reprezentacji (strength=0)"):
    best = _best_per_method(rows, "probe_acc")
    methods = list(best.keys())
    acc = [float(best[m]["probe_acc"]) for m in methods]
    ari = [float(best[m]["km_ARI"]) for m in methods]
    x = np.arange(len(methods)); wdt = 0.38
    plt.figure(figsize=(max(7, len(methods)), 4.5))
    plt.bar(x - wdt / 2, acc, wdt, label="sonda nadzorowana (acc)")
    plt.bar(x + wdt / 2, ari, wdt, label="ARI (klastrowanie = wyciek)")
    plt.xticks(x, methods, rotation=0); plt.ylabel("wartość")
    plt.title(title); plt.legend(); plt.grid(alpha=.3, axis="y"); plt.tight_layout()
    plt.savefig(path, dpi=130); plt.close()


def plot_privacy_utility(rows, path, methods=None):
    """Oś prywatność–użyteczność: siła obfuskacji vs użyteczność (ARI, sonda) i
    odwracalność (rekonstrukcja R²). Najlepsze w po wadze, dla każdej siły."""
    if methods is None:
        methods = sorted({r["method"] for r in rows})
    strengths = sorted({float(r["strength"]) for r in rows})
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for m in methods:
        acc, ari = [], []
        for s in strengths:
            sub = [r for r in rows if r["method"] == m and abs(float(r["strength"]) - s) < 1e-9]
            if not sub:
                acc.append(np.nan); ari.append(np.nan); continue
            best = max(sub, key=lambda r: float(r["probe_acc"]))
            acc.append(float(best["probe_acc"])); ari.append(float(best["km_ARI"]))
        ax[0].plot(strengths, acc, marker="o", label=m)
        ax[1].plot(strengths, ari, marker="o", label=m)
    # rekonstrukcja: bierzemy z dowolnej metody (zależy od grafu, nie od embeddingu wprost) —
    # pokazujemy jako linia odniesienia uśredniona po metodach
    recon = []
    for s in strengths:
        vals = [float(r["recon_r2"]) for r in rows
                if abs(float(r["strength"]) - s) < 1e-9 and r["recon_r2"] not in ("", "nan")
                and not np.isnan(float(r["recon_r2"]))]
        recon.append(np.mean(vals) if vals else np.nan)
    ax2 = ax[1].twinx()
    ax2.plot(strengths, recon, color="black", ls="--", marker="s", label="rekonstrukcja R² (odwracalność)")
    ax2.set_ylabel("rekonstrukcja R² (↓ = lepsza prywatność)")
    ax[0].set_title("Użyteczność: sonda nadzorowana"); ax[0].set_xlabel("siła obfuskacji"); ax[0].set_ylabel("acc")
    ax[1].set_title("Wyciek: ARI (klastrowanie) + odwracalność"); ax[1].set_xlabel("siła obfuskacji"); ax[1].set_ylabel("ARI")
    ax[0].grid(alpha=.3); ax[1].grid(alpha=.3); ax[0].legend(fontsize=8)
    fig.suptitle("Oś prywatność–użyteczność (obfuskacja grafu)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_weight_sweep(rows, path):
    """Wpływ wagi fuzji w (0=atrybuty ... 1=struktura) na metody z fuzją."""
    fusion = {}
    for r in rows:
        if abs(float(r["strength"])) > 1e-9:
            continue
        fusion.setdefault(r["method"], {})[float(r["w"])] = float(r["probe_acc"])
    fusion = {m: d for m, d in fusion.items() if len(d) > 1}
    if not fusion:
        return False
    plt.figure(figsize=(7, 4.5))
    for m, d in fusion.items():
        ws = sorted(d); plt.plot(ws, [d[w] for w in ws], marker="o", label=m)
    plt.xlabel("waga struktury w  (0 = atrybuty, 1 = struktura)")
    plt.ylabel("sonda nadzorowana (acc)")
    plt.title("Fuzja: struktura vs atrybuty"); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()
    return True


def plot_morphospace(X, y, names, path):
    Z = PCA(n_components=2, random_state=0).fit_transform(X) if X.shape[1] > 2 else X
    plt.figure(figsize=(6, 5))
    for lab in sorted(set(y)):
        pts = Z[y == lab]
        nm = names[lab] if names and lab < len(names) else str(lab)
        plt.scatter(pts[:, 0], pts[:, 1], s=16, alpha=0.7, label=nm)
    plt.title("Morfoprzestrzeń (PCA, kolor = klasa)"); plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.legend(fontsize=7, ncol=2); plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()
