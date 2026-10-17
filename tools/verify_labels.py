"""Buat lembar pratinjau untuk memeriksa apakah bbox selaras dengan buah.

    python tools/verify_labels.py --data data --n 12
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
NAMES = {0: "tomato", 1: "banana"}
COLORS = {0: (60, 200, 90), 1: (60, 170, 250)}


def load_boxes(label_path: Path, w: int, h: int):
    if not label_path.exists():
        return []
    out = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        cx, cy, bw, bh = (float(v) for v in parts[1:5])
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        out.append((cls, x1, y1, x2, y2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--cell", type=int, default=300)
    ap.add_argument("--out", default="data/_check_preview.jpg")
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    data = Path(args.data)
    if not data.is_absolute():
        data = ROOT / data
    imgs = sorted((data / "images" / args.split).glob("*.jpg"))
    if not imgs:
        print(f"tidak ada gambar di {data / 'images' / args.split}")
        return

    random.seed(7)
    picks = random.sample(imgs, min(args.n, len(imgs)))
    cell = args.cell
    cols = 4
    rows = (len(picks) + cols - 1) // cols
    sheet = np.full((rows * cell, cols * cell, 3), 20, np.uint8)

    for i, path in enumerate(picks):
        img = cv2.imread(str(path))
        if img is None:
            continue
        h, w = img.shape[:2]
        label = data / "labels" / args.split / (path.stem + ".txt")
        boxes = load_boxes(label, w, h)

        for cls, x1, y1, x2, y2 in boxes:
            color = COLORS.get(cls, (200, 200, 200))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, max(2, w // 400))
            cv2.putText(
                img,
                NAMES.get(cls, str(cls)),
                (x1 + 4, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

        scale = cell / max(h, w)
        thumb = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        r, c = divmod(i, cols)
        y0, x0 = r * cell, c * cell
        sheet[y0 : y0 + thumb.shape[0], x0 : x0 + thumb.shape[1]] = thumb
        cv2.putText(
            sheet, path.name, (x0 + 4, y0 + cell - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1, cv2.LINE_AA,
        )

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    print(f"pratinjau: {out}  ({len(picks)} gambar)")


if __name__ == "__main__":
    main()
