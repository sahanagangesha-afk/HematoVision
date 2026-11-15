"""Generate a small SYNTHETIC dataset so you can smoke-test the whole pipeline.

These are not real blood cells - accuracy on them says nothing about medical
performance. For real results download the Kaggle "Blood Cell Images" dataset.

Usage:  python scripts/generate_sample_data.py --per-class 60
"""
import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

CLASSES = ["eosinophil", "lymphocyte", "monocyte", "neutrophil"]


def draw_cell(cls: str, size: int = 160) -> Image.Image:
    bg = (random.randint(225, 245), random.randint(200, 225), random.randint(200, 225))
    img = Image.new("RGB", (size, size), bg)
    d = ImageDraw.Draw(img)
    cx, cy = size // 2 + random.randint(-8, 8), size // 2 + random.randint(-8, 8)
    r = random.randint(46, 56)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(235, 190, 210))
    if cls == "eosinophil":           # bi-lobed nucleus + orange granules
        for dx in (-16, 16):
            d.ellipse([cx + dx - 14, cy - 16, cx + dx + 14, cy + 16], fill=(90, 60, 150))
        for _ in range(40):
            gx, gy = cx + random.randint(-r + 8, r - 8), cy + random.randint(-r + 8, r - 8)
            d.ellipse([gx - 3, gy - 3, gx + 3, gy + 3], fill=(240, 120, 60))
    elif cls == "lymphocyte":         # big round dark nucleus
        d.ellipse([cx - 34, cy - 34, cx + 34, cy + 34], fill=(60, 40, 130))
    elif cls == "monocyte":           # kidney-shaped nucleus
        d.ellipse([cx - 30, cy - 26, cx + 30, cy + 26], fill=(110, 80, 170))
        d.ellipse([cx - 8, cy - 18, cx + 22, cy + 8], fill=(235, 190, 210))
    else:                             # neutrophil: 3-5 lobes
        for _ in range(random.randint(3, 5)):
            lx, ly = cx + random.randint(-24, 24), cy + random.randint(-24, 24)
            d.ellipse([lx - 10, ly - 10, lx + 10, ly + 10], fill=(80, 50, 140))
    return img.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.2)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per-class", type=int, default=60)
    p.add_argument("--out", default="data")
    args = p.parse_args()
    random.seed(42)
    out = Path(args.out)
    for split, n in (("train", args.per_class), ("test", max(args.per_class // 4, 5))):
        for cls in CLASSES:
            folder = out / split / cls
            folder.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                draw_cell(cls).save(folder / f"{cls}_{i:03d}.jpeg", quality=92)
    print(f"Synthetic dataset written to {out.resolve()}")


if __name__ == "__main__":
    main()
