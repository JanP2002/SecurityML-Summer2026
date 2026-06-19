#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wykres-historia prób GRAF-NATYWNYCH (w duchu wskazówek Krzysztofa, BEZ globalnego HOG).
Porównuje kluczowe metody strukturalne pod trzema dźwigniami:
  - bazowa krawędź + seed WL po kolorze       (run: results_cifar/slic_spec)
  - mocniejsza krawędź: kolor ORAZ tekstura   (run: results_cifar/slic_edgetex, typy *tex)
  - seed WL po kolorze + teksturze            (run: results_cifar/slic_labelrich)
Pokazuje, że pomogła dopiero #3 (bogatszy seed WL). Zapis: results_cifar/fig_graf_natywne.png
"""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def best_acc(path):
    if not os.path.exists(path):
        return {}
    b = {}
    for r in csv.DictReader(open(path)):
        m, a = r["method"], float(r["acc"])
        if m not in b or a > b[m]:
            b[m] = a
    return b


base = best_acc("results_cifar/slic_spec/cifar_porownanie.csv")        # #0 kolor-seed, bazowa krawędź
etex = best_acc("results_cifar/slic_edgetex/cifar_porownanie.csv")     # #1 krawędź kolor+tekstura (*tex)
lbl  = best_acc("results_cifar/slic_labelrich/cifar_porownanie.csv")   # #3 seed kolor+tekstura

methods = ["g2v", "wl", "gnat", "gnat+r"]
variants = [
    ("bazowa (kolor)",          lambda m: base.get(f"{m}-slic")),
    ("#1 krawędź +tekstura",    lambda m: etex.get(f"{m}-slictex")),
    ("#3 seed +tekstura",       lambda m: lbl.get(f"{m}-slic")),
]
colors = ["#bbbbbb", "#e08a3c", "#4C9F70"]

x = np.arange(len(methods)); width = 0.26
plt.figure(figsize=(9, 5))
for i, (lab, get) in enumerate(variants):
    vals = [get(m) if get(m) is not None else 0 for m in methods]
    bars = plt.bar(x + (i - 1) * width, vals, width, label=lab, color=colors[i])
    for bx, v in zip(bars, vals):
        if v:
            plt.text(bx.get_x() + bx.get_width() / 2, v + 0.005, f"{v:.2f}", ha="center", fontsize=8)
hog = base.get("baseline-hog")
if hog:
    plt.axhline(hog, ls="--", c="#cc6600", lw=1.2, label=f"baseline HOG ({hog:.2f})")
plt.axhline(0.7, ls=":", c="#3366cc", lw=1, label="próg 0.7")
plt.xticks(x, methods); plt.ylim(0, 0.8)
plt.ylabel("najlepsza dokładność (5-fold CV)")
plt.title("Próby graf-natywne (bez HOG): co pomogło metodom strukturalnym")
plt.legend(fontsize=8, ncol=2); plt.grid(alpha=.3, axis="y")
plt.tight_layout(); plt.savefig("results_cifar/fig_graf_natywne.png", dpi=130); plt.close()
print("Zapisano: results_cifar/fig_graf_natywne.png")
