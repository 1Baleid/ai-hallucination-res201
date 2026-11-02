"""Calibrate perceptual-hash matching on real NIH images: simulated RANZCR copies vs sibling films."""
import io, sys, random, time
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
import imagehash

img_dir, meta_csv = Path(sys.argv[1]), Path(sys.argv[2])
meta = pd.read_csv(meta_csv)
meta = meta.set_index("Image Index")
files = sorted(img_dir.glob("*.png"))
by_patient = {}
for f in files:
    by_patient.setdefault(f.name.split("_")[0], []).append(f)
rng = random.Random(7)
multi = [p for p, fs in by_patient.items() if len(fs) >= 2]
sample_patients = rng.sample(multi, 150)
t0 = time.time()
cache = {}
def h(path_or_img):
    if isinstance(path_or_img, Path):
        if path_or_img not in cache:
            cache[path_or_img] = imagehash.phash(Image.open(path_or_img).convert("L"), hash_size=16)
        return cache[path_or_img]
    return imagehash.phash(path_or_img.convert("L"), hash_size=16)
rows = []
for q in (95, 75):
    for p in sample_patients:
        fs = by_patient[p]
        src = fs[0]
        w, hgt = int(meta.loc[src.name, "OriginalImage[Width"]), int(meta.loc[src.name, "Height]"])
        im = Image.open(src).convert("L").resize((w, hgt), Image.Resampling.BICUBIC)
        buf = io.BytesIO(); im.save(buf, "JPEG", quality=q); buf.seek(0)
        jpg = Image.open(buf)
        jpg.draft("L", (w // 4, hgt // 4))
        hj = h(jpg)
        d_true = hj - h(src)
        d_sib = min(hj - h(s) for s in fs[1:])
        others = rng.sample([f for f in files if f.name.split("_")[0] != p], 200)
        d_other = min(hj - h(o) for o in others)
        rows.append((q, d_true, d_sib, d_other, len(fs)))
df = pd.DataFrame(rows, columns=["q", "d_true", "d_nearest_sibling", "d_nearest_other200", "n_films"])
print(f"{len(df)} trials in {time.time()-t0:.0f}s (256-bit pHash)")
for q, g in df.groupby("q"):
    print(f"JPEG q={q}: true match d1 median {g.d_true.median():.0f} max {g.d_true.max()} | nearest sibling median {g.d_nearest_sibling.median():.0f} min {g.d_nearest_sibling.min()} | nearest unrelated(200) min {g.d_nearest_other200.min()}")
    margin = g.d_nearest_sibling - g.d_true
    print(f"   margin (sibling - true): min {margin.min()} p05 {margin.quantile(0.05):.0f} median {margin.median():.0f}; share with margin >= 16: {(margin >= 16).mean():.3f}; share where sibling <= true: {(margin <= 0).mean():.3f}")
