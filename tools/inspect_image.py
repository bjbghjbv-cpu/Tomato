"""Periksa bagaimana sebuah gambar dibaca — memakai pipeline yang SAMA
dengan dashboard (deteksi + grading), supaya hasilnya tidak berbeda.

Berguna saat rehearsal: foto tomat demo, lihat kondisi & diskon yang keluar.

Pemakaian:
    python tools/inspect_image.py tomat.jpg
    python tools/inspect_image.py foto_tomat\\ --profile tomato
    python tools/inspect_image.py tomat.jpg --roi      # paksa zona tengah
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.core.grader import analyse_crop
from backend.core.pricing import compute_pricing
from backend.core.profile import load_profile
from backend.core.shelf_life import simulate
from backend.core.vision import Pipeline

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def center_roi(frame: np.ndarray, wfrac: float = 0.34, hfrac: float = 0.52):
    h, w = frame.shape[:2]
    rw, rh = int(w * wfrac), int(h * hfrac)
    x1, y1 = w // 2 - rw // 2, h // 2 - rh // 2
    return frame[y1 : y1 + rh, x1 : x1 + rw]


def report(path: Path, profile_id: str, use_roi: bool, save: Path | None) -> None:
    frame = cv2.imread(str(path))
    if frame is None:
        print(f"  ! tidak bisa membaca {path.name}")
        return

    profile = load_profile(profile_id)

    if use_roi:
        crop = center_roi(frame)
        res = analyse_crop(crop, profile)
        state = None
        source = "roi"
    else:
        pipe = Pipeline(profile_id=profile_id, synthetic=True)
        state = pipe.analyse_image(frame)
        source = state.get("detector", "?")
        res = None

    print(f"\n{path.name}  ({frame.shape[1]}x{frame.shape[0]})  metode={source}")

    if state is not None:
        if not state.get("ok"):
            print(f"  GAGAL: {state.get('error')}")
            return
        print(f"  bbox       : {state['bbox']}")
        print(f"  tahap      : {state['stage']}  ({state['stage_label']})")
        print(f"  kematangan : {state['maturity_pct']:5.1f}%")
        print(f"  kerusakan  : {state['defect_pct']:5.1f}%")
        print(f"  KONDISI    : {state['condition_pct']:5.1f}%  ->  Grade {state['grade']}, diskon {state['discount']}%")
        print(f"  harga      : Rp {state['price']:,}  (dasar Rp {state['base_price']:,})")
        print(f"  sisa umur  : {state['remaining_hours']:.0f} jam")
        print(f"  keyakinan  : {state['confidence'] * 100:.0f}%")
        print(f"  sinyal     : {state['signals']}")
        # raw butuh crop
        x, y, w, h = state["bbox"]
        raw = analyse_crop(frame[y : y + h, x : x + w], profile).raw
        print(f"  mentah     : {raw}")
        condition = state["condition_pct"] / 100.0
        grade, discount = state["grade"], state["discount"]
        price_value = state["price"]
    else:
        if not res.ok:
            print(f"  GAGAL: {res.note}")
            return
        sim = simulate(profile, res.maturity, res.defect, temp_c=24, rh=88)
        pr = compute_pricing(profile, res.condition, sim["remaining_hours"], res.maturity, res.defect)
        print(f"  tahap      : {res.stage}  ({res.stage_label})")
        print(f"  kematangan : {res.maturity * 100:5.1f}%")
        print(f"  kerusakan  : {res.defect * 100:5.1f}%")
        print(f"  KONDISI    : {res.condition * 100:5.1f}%  ->  Grade {pr.grade}, diskon {pr.discount}%")
        print(f"  harga      : Rp {pr.price:,}  (dasar Rp {pr.base_price:,})")
        print(f"  sisa umur  : {sim['remaining_hours']:.0f} jam")
        print(f"  keyakinan  : {res.confidence * 100:.0f}%")
        print(f"  sinyal     : {res.signals}")
        print(f"  mentah     : {res.raw}")
        condition = res.condition
        grade, discount, price_value = pr.grade, pr.discount, pr.price

    if save:
        vis = frame.copy()
        text = f"{condition * 100:.0f}%  Grade {grade}  -{discount}%  Rp {price_value:,}"
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 46), (12, 18, 24), -1)
        cv2.putText(vis, text, (10, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(save), vis)
        print(f"  preview    : {save}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--profile", default="tomato")
    ap.add_argument("--roi", action="store_true", help="paksa pakai zona tengah")
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    target = Path(args.path)
    if target.is_dir():
        files = sorted(p for p in target.iterdir() if p.suffix.lower() in EXTS)
        if not files:
            print("tidak ada gambar di folder itu")
            return
        for f in files:
            report(f, args.profile, args.roi, None)
    else:
        report(target, args.profile, args.roi, Path(args.save) if args.save else None)


if __name__ == "__main__":
    main()
