import torch, numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms

data_dir = Path("pythia")
transform = transforms.ToTensor()
clean_imgs, atk_imgs = [], []
for img_path in sorted((data_dir / "clean").glob("*.png"), key=lambda p: int(p.stem))[:200]:
    clean_imgs.append(transform(Image.open(img_path).convert("L")).numpy())
for img_path in sorted((data_dir / "attack_a").glob("*.png"), key=lambda p: int(p.stem))[:200]:
    atk_imgs.append(transform(Image.open(img_path).convert("L")).numpy())

clean = np.stack(clean_imgs)
atk   = np.stack(atk_imgs)
diff  = np.abs(clean - atk).mean()

print(f"Samples in clean:    {len(list((data_dir/'clean').glob('*.png')))}")
print(f"Samples in attack_a: {len(list((data_dir/'attack_a').glob('*.png')))}")
print()
print(f"clean  mean={clean.mean():.4f}  std={clean.std():.4f}  min={clean.min():.4f}  max={clean.max():.4f}")
print(f"atk_a  mean={atk.mean():.4f}  std={atk.std():.4f}  min={atk.min():.4f}  max={atk.max():.4f}")
print(f"mean |clean-atk| per pixel = {diff:.6f}")

# Histogram of mean pixel values per image
clean_means = clean.mean(axis=(1,2,3))
atk_means   = atk.mean(axis=(1,2,3))
print(f"\nPer-image mean pixel (clean): {clean_means.mean():.4f} +/- {clean_means.std():.4f}")
print(f"Per-image mean pixel (atk_a): {atk_means.mean():.4f} +/- {atk_means.std():.4f}")

# Check if images appear to have any white pixels (high values)
print(f"\nFraction of pixels > 0.5: clean={((clean > 0.5).mean()):.4f}, atk={(atk > 0.5).mean():.4f}")
print(f"Fraction of pixels > 0.9: clean={((clean > 0.9).mean()):.4f}, atk={(atk > 0.9).mean():.4f}")
print(f"Fraction of pixels < 0.1: clean={((clean < 0.1).mean()):.4f}, atk={(atk < 0.1).mean():.4f}")
