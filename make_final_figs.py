#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dwie czyste figury pod raport (oś PRYWATNOŚCI). Czyta results_FINAL_*."""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


# --- Figura 1: płaszczyzna prywatność–użyteczność na PROTEINS (ARI vs recon_leak) ---
rows = load("results_FINAL_prot/wyniki.csv")
struct = {"fgsd", "rand_ens", "topo", "wltopo", "netlsd", "wl", "graphlet", "spec", "g2v"}
fig, ax = plt.subplots(figsize=(6.2, 4.4))
for r in rows:
    m = r["method"]; x = float(r["recon_leak"]); y = float(r["km_ARI"])
    is_attr = (m == "attr")
    ax.scatter(x, y, s=90, c=("#d62728" if is_attr else "#1f77b4"),
               edgecolor="black", zorder=3, marker=("s" if is_attr else "o"))
    dx = -0.018 if m in ("fgsd", "netlsd") else 0.015
    ha = "right" if m in ("fgsd", "netlsd") else "left"
    ax.annotate(m, (x, y), xytext=(x + dx, y + 0.003), ha=ha, fontsize=9,
                color=("#d62728" if is_attr else "black"))
ax.axvspan(-0.02, 0.2, color="green", alpha=0.06)
ax.text(0.02, 0.142, "prywatne\n(struktura)", fontsize=8, color="green")
ax.text(0.66, 0.005, "wyciek\n(treść)", fontsize=8, color="#d62728")
ax.set_xlabel("recon_leak  (odwracalność — atak rekonstrukcyjny; ← lepiej)")
ax.set_ylabel("km_ARI  (klastrowanie bez etykiet; ↑ lepiej)")
ax.set_title("PROTEINS_full: użyteczność vs prywatność reprezentacji")
ax.set_xlim(-0.03, 0.85); ax.set_ylim(0.0, 0.15)
ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig("results_FINAL_prot/fig_pareto.png", dpi=140)
print("zapisano results_FINAL_prot/fig_pareto.png")

# --- Figura 3: krzywa feature-DP (recon_leak vs sigma; attr maleje, struktura płaska) ---
rows = load("results_FINAL_priv_feat/wyniki.csv")
series = {}
for r in rows:
    series.setdefault(r["method"], []).append((float(r["strength"]), float(r["recon_leak"]),
                                                float(r["probe_acc"])))
for k in series:
    series[k].sort()
fig, ax = plt.subplots(figsize=(6.2, 4.2))
col = {"attr": "#d62728", "topo": "#ff7f0e", "wl": "#2ca02c"}
for m in ["attr", "topo", "wl"]:
    s = series[m]
    xs = [a for a, _, _ in s]; ys = [b for _, b, _ in s]
    ax.plot(xs, ys, "-o", color=col[m], label=f"{m}", lw=2)
ax.annotate("0.764", (0, 0.764), xytext=(0.1, 0.79), fontsize=8, color="#d62728")
ax.annotate("0.485", (4, 0.485), xytext=(3.4, 0.44), fontsize=8, color="#d62728")
ax.set_xlabel("siła feature-DP  (σ szumu cech, ×std)")
ax.set_ylabel("recon_leak  (odwracalność treści; ↓ = prywatniej)")
ax.set_title("feature-DP: szum cech węzłów chroni TREŚĆ, nie strukturę\n(PROTEINS_full, odporny atak)")
ax.set_ylim(0, 0.9); ax.grid(alpha=0.3); ax.legend(title="strumień")
fig.tight_layout(); fig.savefig("results_FINAL_priv_feat/fig_feature_dp.png", dpi=140)
print("zapisano results_FINAL_priv_feat/fig_feature_dp.png")
