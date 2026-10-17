"""Siapkan dataset YOLO dari sumber publik.

Sumber:
  tomato : Project-AgML/tomato_ripeness_detection        (804 img, 9.777 bbox)
  tomato : Project-AgML/tomato_adaptive_yolov8_detection (604 img, 2.802 bbox)
  banana : COCO 2017, kelas banana (id 46) — unduh selektif

Kelas keluaran: 0=tomato, 1=banana

Parquet dibaca dengan polars (tidak butuh library `datasets`).

Pemakaian:
    python tools/prepare_dataset.py --inspect
    python tools/prepare_dataset.py --out data
    python tools/prepare_dataset.py --out data --no-banana
    python tools/prepare_dataset.py --out data --limit 200   # uji cepat
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import polars as pl
import requests
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "_raw"

HF_SOURCES = [
    ("tomato_ripeness_detection", "ripeness", 0),
    ("tomato_adaptive_yolov8_detection", "adaptive", 0),
]

COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_IMG_URL = "http://images.cocodataset.org/train2017/{image_id:012d}.jpg"
COCO_BANANA_NAME = "banana"


# ----------------------------------------------------------------------
def split_for(key: str) -> str:
    h = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) % 100
    if h < 80:
        return "train"
    if h < 90:
        return "val"
    return "test"


def ahash(img: Image.Image) -> int:
    small = np.asarray(img.convert("L").resize((8, 8), Image.BILINEAR), dtype=np.float32)
    bits = (small > small.mean()).flatten()
    out = 0
    for bit in bits:
        out = (out << 1) | int(bit)
    return out


def is_duplicate(h: int, seen: set[int], max_dist: int = 3) -> bool:
    for other in seen:
        if (h ^ other).bit_count() <= max_dist:
            return True
    return False


def guess_format(boxes: np.ndarray, width: int, height: int, forced: str | None) -> tuple[str, bool]:
    """Tebak xyxy vs xywh.

    Penanda paling andal: pada xyxy, koordinat ke-3/ke-4 SELALU lebih besar
    dari ke-1/ke-2. Pada xywh, lebar/tinggi sering lebih kecil dari posisi.
    """
    if boxes.size == 0:
        return "xyxy", False
    if forced:
        return forced, bool(boxes.max() <= 1.5)

    normalized = bool(boxes.max() <= 1.5)
    ordered = float(
        np.mean((boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1]))
    )
    return ("xyxy" if ordered > 0.9 else "xywh"), normalized


def to_xyxy(box: np.ndarray, fmt: str, normalized: bool, width: int, height: int) -> np.ndarray:
    x1, y1, a, b = [float(v) for v in box[:4]]
    x2, y2 = (x1 + a, y1 + b) if fmt == "xywh" else (a, b)
    if normalized:
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = max(0.0, min(width - 1.0, x1))
    y1 = max(0.0, min(height - 1.0, y1))
    x2 = max(0.0, min(width - 1.0, x2))
    y2 = max(0.0, min(height - 1.0, y2))
    return np.array([x1, y1, x2, y2])


def yolo_line(cls: int, box: np.ndarray, width: int, height: int) -> str | None:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if w < 4 or h < 4:
        return None
    cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
    return f"{cls} {cx:.6f} {cy:.6f} {w / width:.6f} {h / height:.6f}"


def rows_of(path: Path, chunk: int = 32):
    lf = pl.scan_parquet(str(path))
    total = lf.select(pl.len()).collect().item()
    for off in range(0, total, chunk):
        df = lf.slice(off, chunk).collect()
        for row in df.iter_rows(named=True):
            yield row


# ----------------------------------------------------------------------
def inspect() -> None:
    for folder, _tag, _cls in [(RAW / s[0], s[1], s[2]) for s in HF_SOURCES]:
        files = sorted(folder.glob("*.parquet")) if folder.exists() else []
        print(f"\n=== {folder.name} ===  file: {len(files)}")
        if not files:
            print("  belum diunduh")
            continue
        path = files[0]
        schema = pl.scan_parquet(str(path)).collect_schema()
        print(f"  kolom: {list(schema.keys())}")
        for i, row in enumerate(rows_of(path, chunk=2)):
            objs = row["objects"]
            boxes = np.asarray(objs["bbox"], dtype=np.float64)
            cats = list(objs["categories"])
            img = Image.open(io.BytesIO(row["image"]["bytes"]))
            print(f"  contoh {i}: ukuran {img.size}  objek {len(cats)}  kategori {sorted(set(cats))}")
            print(f"           bbox[:3] = {np.round(boxes[:3], 3).tolist()}")
            print(f"           max bbox = {boxes.max():.3f}")
            if "split" in row:
                print(f"           split   = {row['split']}")
            if i >= 1:
                break


# ----------------------------------------------------------------------
def convert_source(
    folder: Path,
    tag: str,
    class_id: int,
    out: Path,
    manifest: list,
    forced_fmt: str | None,
    limit: int | None,
    dedup: bool,
    max_size: int = 1280,
) -> int:
    files = sorted(folder.glob("*.parquet"))
    if not files:
        print(f"\n>> {tag}: tidak ada parquet di {folder}")
        return 0

    print(f"\n>> {tag}  ({len(files)} file)")
    seen: set[int] = set()
    written = 0
    skipped = 0
    dup = 0
    fmt_seen: dict[str, int] = {}
    index = 0
    stop = False

    for path in files:
        if stop:
            break
        for row in rows_of(path):
            if limit and written >= limit:
                stop = True
                break
            try:
                img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
            except Exception:
                skipped += 1
                continue

            width, height = img.size
            objs = row["objects"]
            boxes = np.asarray(objs["bbox"], dtype=np.float64)
            cats = list(objs["categories"])
            if boxes.size == 0:
                skipped += 1
                continue

            fmt, normalized = guess_format(boxes, width, height, forced_fmt)
            key = f"{fmt}{'-norm' if normalized else ''}"
            fmt_seen[key] = fmt_seen.get(key, 0) + 1

            xyxy = [to_xyxy(b, fmt, normalized, width, height) for b in boxes]

            # perkecil gambar supaya training cepat & disk hemat
            scale = 1.0
            if max_size and max(width, height) > max_size:
                scale = max_size / float(max(width, height))
                img = img.resize(
                    (max(1, int(width * scale)), max(1, int(height * scale))),
                    Image.LANCZOS,
                )
                width, height = img.size
                xyxy = [b * scale for b in xyxy]

            lines = []
            for box in xyxy:
                line = yolo_line(class_id, box, width, height)
                if line:
                    lines.append(line)
            if not lines:
                skipped += 1
                continue

            if dedup:
                h = ahash(img)
                if is_duplicate(h, seen):
                    dup += 1
                    index += 1
                    continue
                seen.add(h)

            raw_split = str(row.get("split") or "").lower()
            if raw_split in {"train", "val", "valid", "validation", "test"}:
                split = {"valid": "val", "validation": "val"}.get(raw_split, raw_split)
            else:
                split = split_for(f"{tag}-{index}")

            stem = f"{tag}_{index:05d}"
            img_path = out / "images" / split / f"{stem}.jpg"
            lbl_path = out / "labels" / split / f"{stem}.txt"
            img_path.parent.mkdir(parents=True, exist_ok=True)
            lbl_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(img_path, quality=92)
            lbl_path.write_text("\n".join(lines), encoding="utf-8")

            manifest.append(
                {
                    "file": f"{stem}.jpg",
                    "source": tag,
                    "split": split,
                    "n_boxes": len(lines),
                    "width": width,
                    "height": height,
                }
            )
            written += 1
            index += 1
            if written % 100 == 0:
                print(f"   {written} gambar …")

    print(f"   selesai: {written} ditulis, {dup} duplikat, {skipped} dilewati")
    print(f"   format bbox: {fmt_seen}")
    return written


# ----------------------------------------------------------------------
def download_coco_banana(out: Path, manifest: list, limit: int = 1200, dedup: bool = True) -> int:
    print("\n>> COCO 2017 — kelas banana")
    cache = ROOT / "data" / "_coco"
    cache.mkdir(parents=True, exist_ok=True)
    ann_zip = cache / "annotations_trainval2017.zip"

    if not ann_zip.exists():
        print("   mengunduh anotasi (±241 MB) …")
        with requests.get(COCO_ANN_URL, stream=True, timeout=180) as r:
            r.raise_for_status()
            with ann_zip.open("wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)

    with zipfile.ZipFile(ann_zip) as zf:
        data = json.loads(zf.read("annotations/instances_train2017.json"))

    cat_id = next((c["id"] for c in data["categories"] if c["name"].lower() == COCO_BANANA_NAME), None)
    if cat_id is None:
        print("   kelas banana tidak ditemukan")
        return 0

    per_image: dict[int, list] = {}
    for ann in data["annotations"]:
        if ann["category_id"] != cat_id or ann.get("iscrowd"):
            continue
        per_image.setdefault(ann["image_id"], []).append(ann)

    images = {im["id"]: im for im in data["images"]}
    ids = sorted(per_image)[:limit]
    print(f"   {len(ids)} gambar berisi banana")

    def fetch(image_id: int):
        try:
            r = requests.get(COCO_IMG_URL.format(image_id=image_id), timeout=60)
            return (image_id, r.content) if r.status_code == 200 else None
        except Exception:
            return None

    written = 0
    dup = 0
    seen: set[int] = set()
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(fetch, iid): iid for iid in ids}
        for n, fut in enumerate(as_completed(futures), 1):
            got = fut.result()
            if not got:
                continue
            image_id, content = got
            arr = np.frombuffer(content, np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            h, w = bgr.shape[:2]

            if dedup:
                pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                hsh = ahash(pil)
                if is_duplicate(hsh, seen):
                    dup += 1
                    continue
                seen.add(hsh)

            lines = []
            for ann in per_image[image_id]:
                x, y, bw, bh = ann["bbox"]
                line = yolo_line(1, np.array([x, y, x + bw, y + bh], dtype=np.float64), w, h)
                if line:
                    lines.append(line)
            if not lines:
                continue

            split = split_for(f"coco-{image_id}")
            stem = f"coco_{image_id:012d}"
            img_path = out / "images" / split / f"{stem}.jpg"
            lbl_path = out / "labels" / split / f"{stem}.txt"
            img_path.parent.mkdir(parents=True, exist_ok=True)
            lbl_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(img_path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            lbl_path.write_text("\n".join(lines), encoding="utf-8")

            manifest.append(
                {
                    "file": f"{stem}.jpg",
                    "source": "coco2017",
                    "split": split,
                    "n_boxes": len(lines),
                    "width": w,
                    "height": h,
                }
            )
            written += 1
            if n % 100 == 0:
                print(f"   {n}/{len(ids)} …")

    print(f"   selesai: {written} banana, {dup} duplikat")
    return written


# ----------------------------------------------------------------------
def write_yaml(out: Path) -> None:
    (out / "data.yaml").write_text(
        f"path: {out.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: tomato\n"
        "  1: banana\n",
        encoding="utf-8",
    )


def summary(out: Path, manifest: list) -> None:
    import collections

    print("\n=== ringkasan dataset ===")
    counts = collections.Counter(
        (r["split"], "banana" if r["source"] == "coco2017" else "tomato") for r in manifest
    )
    for (split, cls), n in sorted(counts.items()):
        print(f"  {split:5s} {cls:7s} {n:5d} gambar")
    total = sum(1 for _ in (out / "images").rglob("*.jpg")) if (out / "images").exists() else 0
    boxes = sum(r["n_boxes"] for r in manifest)
    print(f"  total  : {total} gambar, {boxes} bbox")
    print(f"  yaml   : {out / 'data.yaml'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--bbox-format", choices=["xyxy", "xywh"], default=None)
    ap.add_argument("--limit", type=int, default=None, help="batasi gambar per sumber")
    ap.add_argument("--no-banana", dest="banana", action="store_false")
    ap.add_argument("--no-dedup", dest="dedup", action="store_false")
    ap.add_argument("--max-size", type=int, default=1280, help="panjang sisi terpanjang gambar")
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    if args.inspect:
        inspect()
        return

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    if args.fresh and out.exists():
        for sub in ("images", "labels"):
            shutil.rmtree(out / sub, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)

    manifest: list = []
    for name, tag, cls in HF_SOURCES:
        convert_source(
            RAW / name, tag, cls, out, manifest, args.bbox_format, args.limit,
            args.dedup, args.max_size,
        )

    if args.banana:
        try:
            download_coco_banana(out, manifest, limit=args.limit or 1200, dedup=args.dedup)
        except Exception as exc:
            print(f"!! COCO banana gagal: {exc}")

    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["file", "source", "split", "n_boxes", "width", "height"]
        )
        writer.writeheader()
        writer.writerows(manifest)

    write_yaml(out)
    summary(out, manifest)


if __name__ == "__main__":
    main()
