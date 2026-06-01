"""Quick sklearn ceiling probe for Pythia attack_a."""
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load_dir(d):
    imgs = []
    for p in sorted(Path(d).glob("*.png")):
        imgs.append(np.array(Image.open(p).convert("L"), dtype=np.float32) / 255.0)
    return np.stack(imgs)

clean = load_dir("pythia/clean")
atk_a = load_dir("pythia/attack_a")
print(f"clean {clean.shape}  atk_a {atk_a.shape}")

# ── Per-pixel analysis ─────────────────────────────────────────────────────
delta = atk_a.mean(0) - clean.mean(0)
print(f"\nPer-pixel |delta|  max={np.abs(delta).max():.4f}  mean={np.abs(delta).mean():.6f}")
print(f"Pixels |delta|>0.01: {(np.abs(delta)>0.01).sum()}")
print(f"Pixels |delta|>0.05: {(np.abs(delta)>0.05).sum()}")
print(f"Pixels |delta|>0.10: {(np.abs(delta)>0.10).sum()}")

# Save difference heatmap
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
im = ax.imshow(delta, cmap="RdBu_r", vmin=-0.1, vmax=0.1)
ax.set_title("Mean pixel difference\n(attack_a − clean)", fontsize=11)
plt.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.savefig("plots/step1/pythia_delta_heatmap.png", dpi=150)
plt.close()
print("  [plot] Saved → plots/step1/pythia_delta_heatmap.png")

# ── Dataset split ──────────────────────────────────────────────────────────
X = np.concatenate([clean, atk_a]).reshape(-1, 70*70)
y = np.concatenate([np.zeros(len(clean)), np.ones(len(atk_a))])

X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
X_train, _, y_train, _ = train_test_split(X_tv, y_tv, test_size=0.20, random_state=42, stratify=y_tv)
print(f"\nSplit — train {X_train.shape[0]}  test {X_test.shape[0]}")

sc = StandardScaler().fit(X_train)
Xtr = sc.transform(X_train)
Xte = sc.transform(X_test)

results = {}

# LR L2
for C in [0.0005, 0.001, 0.01]:
    m = LogisticRegression(C=C, max_iter=3000, solver='lbfgs').fit(Xtr, y_train)
    results[f"LR-L2 C={C}"] = roc_auc_score(y_test, m.predict_proba(Xte)[:,1])

# LR L1 (sparse)
for C in [0.001, 0.01, 0.1]:
    m = LogisticRegression(C=C, penalty='l1', max_iter=3000, solver='saga').fit(Xtr, y_train)
    nz = (np.abs(m.coef_[0]) > 1e-6).sum()
    results[f"LR-L1 C={C} ({nz}px)"] = roc_auc_score(y_test, m.predict_proba(Xte)[:,1])

# Shrinkage LDA
m = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(Xtr, y_train)
results["LDA shrinkage=auto"] = roc_auc_score(y_test, m.decision_function(Xte))

# Random Forest
for depth in [3, 5]:
    m = RandomForestClassifier(n_estimators=300, max_depth=depth, min_samples_leaf=10,
                               random_state=42, n_jobs=-1).fit(Xtr, y_train)
    results[f"RF depth={depth}"] = roc_auc_score(y_test, m.predict_proba(Xte)[:,1])

# GBM
m = GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                subsample=0.8, random_state=42).fit(Xtr, y_train)
results["GBM"] = roc_auc_score(y_test, m.predict_proba(Xte)[:,1])

# Patch statistics (10x10 grid of 7x7 patches → 200 features)
def patch_stats(imgs_flat):
    imgs = imgs_flat.reshape(-1, 70, 70)
    feats = []
    for r in range(0, 70, 7):
        for c in range(0, 70, 7):
            p = imgs[:, r:r+7, c:c+7]
            feats.append(p.mean(axis=(1,2)))
            feats.append(p.std(axis=(1,2)))
    return np.stack(feats, axis=1)

Fp_tr = StandardScaler().fit(patch_stats(X_train))
Fp_te_raw = patch_stats(X_test)
Fp_tr_raw = patch_stats(X_train)
sc2 = StandardScaler().fit(Fp_tr_raw)
m = LogisticRegression(C=0.01, max_iter=3000).fit(sc2.transform(Fp_tr_raw), y_train)
results["Patch-LR C=0.01"] = roc_auc_score(y_test, m.predict_proba(sc2.transform(Fp_te_raw))[:,1])

# LR on top-K pixels ranked by |delta| (oracle feature selection)
delta_flat = delta.flatten()
for K in [50, 200, 500]:
    top_k = np.argsort(np.abs(delta_flat))[-K:]
    m = LogisticRegression(C=0.1, max_iter=3000).fit(Xtr[:, top_k], y_train)
    results[f"LR top-{K}px (oracle)"] = roc_auc_score(y_test, m.predict_proba(Xte[:, top_k])[:,1])

# ── Summary ────────────────────────────────────────────────────────────────
print("\n" + "="*52)
print(f"{'Method':<37} {'AUC':>8}")
print("="*52)
for name, auc in sorted(results.items(), key=lambda x: -x[1]):
    flag = " ★" if auc >= 0.60 else ""
    print(f"{name:<37} {auc:.4f}{flag}")
print("="*52)
