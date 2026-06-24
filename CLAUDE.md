# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A research benchmark for the question: **can class membership of an object be detected purely from the structure of its graph representation, without labels?** Each object (a molecule, a protein, an image) is turned into a graph, the graph is embedded into a vector, the vectors are clustered *unsupervised*, and labels are only used at the end to score how well clusters align with true classes. The goal is an extensible benchmark where new representations and datasets can be plugged in and compared on the same metrics.

Documentation, code comments, and printed output are mostly in **Polish**. Keep that convention when editing existing files.

## Three independent pipelines

The repo holds three separate pipelines that share the idea but no code:

1. **`tu_graph_clustering.py`** — whole-graph clustering on TUDataset (default ENZYMES). One vector per *entire graph*, then KMeans/HDBSCAN. The most mature, CLI-driven piece. See `README.md` for the write-up and headline results.
2. **`cifar_graph_clustering.py`** — the current CIFAR-10 work (the `.py` successor to the notebook; the user prefers scripts over notebooks). Images → graphs (pixel / patch / SLIC) → embedding → supervised CV + unsupervised clustering, vs RGB/HOG baselines. This is where active development happens; see its own section below.
3. **`cifar6_3graphs.ipynb`** — the original CIFAR notebook the `.py` version grew out of (kept for history; `cifar_graphs_notes.md` explains the three graph constructions). Don't edit it; port changes into `cifar_graph_clustering.py` instead.

The three do not import each other; treat them as parallel explorations of the same research question.

## Running

```bash
# Whole-graph benchmark — auto-downloads ENZYMES into data/ on first run:
python tu_graph_clustering.py --all-embeddings

# Single representation on a different TU dataset:
python tu_graph_clustering.py --name PROTEINS_full --embedding graph2vec

# CIFAR image-graph benchmark (auto-downloads CIFAR-10 into ./data on first run):
#   --classes picks a subset (>0.7 is only realistic on a few-class subset);
#   omit it for all 10. --plots writes story figures. Embeddings are cached.
python cifar_graph_clustering.py --graph-type slic --classes 0 1 8 --per-class 200 --plots
```

There is **no requirements file**. Dependencies are inferred from imports: `numpy`, `scipy`, `scikit-learn`, `networkx`, `node2vec`, `gensim`, `matplotlib`; plus `torch`, `torchvision`, `scikit-image`, `joblib` for the CIFAR pipeline (and optionally `umap-learn`, used only by the old notebook).

**Windows caveat:** the `node2vec` library's parallelism is broken on Windows; both `.py` pipelines force single-threaded Node2Vec/Doc2Vec (`workers=1`). Also, console output is Polish — `cifar_graph_clustering.py` reconfigures stdout to UTF-8 so cp1252 consoles don't crash; do the same if you add a new script that prints Polish.

## `tu_graph_clustering.py` architecture

A single-file pipeline with a clear data flow; the pieces matter more than the file layout:

- **Loader** (`load_tu_dataset` / `download_tu`) reads the Dortmund TUDataset text format (`*_A.txt`, `*_graph_indicator.txt`, `*_graph_labels.txt`, optional `*_node_labels.txt` / `*_node_attributes.txt`) into a list of `networkx.Graph`. Node attributes land as `label` (int type) and `x` (attribute vector). Missing files are auto-fetched from the GraphRNN GitHub mirror.
- **Five representations**, all producing one vector per graph, selected by the `--embedding` string in `build_embeddings`:
  - `topo` — hand-crafted structural stats (size, cycles, degree distribution).
  - `attr` — mean⊕std pooling of node attribute vectors `x` (skipped automatically if the dataset has no node attributes).
  - `wl` — Weisfeiler–Lehman subtree feature map (sparse, shared vocabulary), initialized from the node `label`.
  - `graph2vec` — the same WL patterns, but learned densely via Doc2Vec.
  - `node2vec` — node embeddings + mean⊕std aggregation (a node-level method used as a deliberate contrast; expected to be the weakest on whole-graph tasks).
- **Method-dependent preprocessing** (`preprocess`) is load-bearing: `topo`/`attr`/`node2vec` → `StandardScaler`; `wl`/`graph2vec` → optional TruncatedSVD then L2-normalize. If you add a representation, decide which branch it belongs in.
- **Evaluation** (`run_one`): KMeans (k = #classes) + HDBSCAN, internal metrics (silhouette, Davies-Bouldin, Calinski-Harabasz) and external metrics (ARI, NMI, V-measure), a supervised kNN probe via stratified CV, and 2D PCA "morphospace" plots.
- **Output** goes to `results/`: `porownanie_metod.csv` plus `<method>_morphospace.png` and `<method>_kmeans.png`.

Config lives in the `Config` dataclass (seeds, WL iterations, embedding dims, Node2Vec walk params). ENZYMES is a known-hard 6-class problem — ARI near 0 with a probe accuracy well above chance is the *expected* result, not a bug.

## `cifar_graph_clustering.py` architecture

Same spirit as the TU script, adapted to images. Data flow: CIFAR-10 → per-image graph → embedding(s) → block-weighted fusion → supervised CV + unsupervised clustering, all in `run()`.

- **Graph builders** (`build_pixel_graph` / `build_patch_graph` / `build_slic_graph`, in `GRAPH_BUILDERS`). Key design choice (instructor's hint): an edge requires **both** spatial adjacency **and** color/feature similarity — distances above an adaptive per-image threshold (`--edge-quantile`, or fixed `--tau`) are *cut*, not just down-weighted, so the graph fragments into object-like regions. Isolated nodes are reconnected to their most-similar spatial neighbor.
- **Two embedding families, kept side by side** (the user wants methods added, never replaced, so progress stays comparable):
  - Node2Vec + pooling, but the structural and attribute streams are pooled **separately** (`aggregate_streams`) and fused by `fuse(Xs, Xa, w)` — each block standardized + L2-normalized, then weighted by `w`. This `w` (0 = attributes only, 1 = structure only) is swept; flat concatenation used to let the ~128 structural dims drown the color signal ("dilution").
  - Whole-graph embeddings ported from the TU script: `embed_wl` and `embed_graph2vec` (seeded from quantized node colors via `assign_color_labels`).
- **Methods in the benchmark table** (distinct rows, all additive): `n2vflat-` (the deliberate dilution baseline), `n2v-`, `wl-`, `g2v-`, `topo-` (pure structural stats — component/region fragmentation from the edge-cutting), `spec-` (Laplacian spectral signature), `combo-` (graph2vec ⊕ attributes), `gnat-` (graph2vec ⊕ topo ⊕ attributes — the graph-native max, no HOG), `gspec-` (gnat + spectral), `hyb-` (HOG ⊕ graph2vec) — each `-<graph_type>`, `+r` suffix = rich mini-HOG node descriptor — plus `baseline-rgb` / `baseline-hog`. Pure-graph ceiling on the 3-class subset is ~0.68; `hyb` ≈ 0.74 needs HOG.
- **Graph-native levers explored (same structural spirit, no HOG), all kept as additive options:** `*tex` graph types (`make_graph`) = edge needs color **and** texture (didn't help); `spec`/`gspec` = spectral features (didn't help); `--label-rich` = seed WL/graph2vec on color+texture not color-only (**helped** g2v/wl). The pattern: strengthening the *structural representation* (better WL seed) beats changing the graph topology. Ablation: `plot_graf_natywne.py` → `results_cifar/fig_graf_natywne.png`.
- **Caching is load-bearing for iteration speed:** Node2Vec/graph2vec are computed once per config and cached as `.npy` under `<out-dir>/cache/` (keyed by a config hash); weight sweeps then re-run instantly. Pixel graphs are ~2.4 s/image, so prefer SLIC for fast iteration and run pixel/10-class in the background.
- **`--plots`** writes story figures to the out-dir (`fig_porownanie` ranked bars, `fig_waga` weight sweep, `fig_acc_vs_ari` separability-vs-clusterability, `fig_morfo` PCA morphospace).

Honest expectations: **>0.7 accuracy is only realistic on a few-class subset**; full 10-class tops out ~0.44 even for HOG. The graph methods' genuine win is *unsupervised* (ARI/silhouette ≫ baselines); `hyb` (structure + appearance) is what edges past HOG on supervised accuracy.

## Adding things

- **New TU dataset:** just pass `--name <DATASET>`; the loader downloads it. Note this mirror names the proteins set `PROTEINS_full`, not `PROTEINS`.
- **New whole-graph representation:** add a branch in `build_embeddings`, add the appropriate scaling branch in `preprocess`, and add the method name to the `--all-embeddings` list and the `--embedding` choices in `main`.
- **New CIFAR method:** add it as a *new* `add(f"<name>-{gt}", ...)` call in `run()` (alongside the existing ones — don't replace them) so it appears as a new row in the same comparison table. Reuse `fuse()` for any stream-weighted method. If it caches embeddings, extend `build_graph_reps` and bump the cache key's `extra=`.

## Branch convention

Work happens on per-person, per-challenge branches (`jpch1/2/3` = author Piotr's challenge N; `mnch1/2` = another contributor) that merge into `main`. The current branch `jpch3` is "Challenge 3, Krok 1" — the graph-clustering foundation described above. Match the existing branch when continuing someone's challenge.
