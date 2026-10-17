"""Uji modul cacat pada foto asli: segar vs busuk.

Menjalankan grader yang SAMA dengan dashboard, lalu:
  1. menampilkan kondisi tiap foto + sinyal mana yang menyala
  2. menilai lulus/gagal (segar harus A, busuk harus C/D)
  3. dengan --sweep, mencari kombinasi ambang terbaik

Pemakaian:
    python tools/defect_bench.py --segar tomatsegar --busuk gambartomatbusuk
    python tools/defect_bench.py --segar tomatsegar --busuk gambartomatbusuk --sweep
"""

from __future__ import annotations

import argparse
import copy
import itertools
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

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
IMAGE = {"SEGAR": 0, "BUSUK": 1}


def gather(pipeline: Pipeline, profile, segar: Path, busuk: Path):
    items = []
    for label, folder in (("SEGAR", segar), ("BUSUK", busuk)):
        if not folder or not folder.exists():
            continue
        for f in sorted(folder.iterdir()):
            if f.suffix.lower() not in EXTS:
                continue
            frame = cv2.imread(str(f))
            if frame is None:
                print(f"  ! tidak bisa membaca {f.name}")
                continue
            bbox, conf, src = pipeline._detect(frame, profile)
            if bbox is None:
                continue
            crop, padded = pipeline._crop(frame, bbox)
            if crop is None:
                continue
            items.append({"label": label, "name": f.name, "crop": crop, "detector": src})
    return items


def evaluate(items, profile, defect_cfg) -> list[dict]:
    prof = copy.deepcopy(profile)
    prof.defect = defect_cfg
    out = []
    for it in items:
        res = analyse_crop(it["crop"], prof)
        cond = res.condition if res.ok else 0.0
        sim = simulate(prof, res.maturity, res.defect, temp_c=24, rh=88)
        price = compute_pricing(prof, cond, sim["remaining_hours"], res.maturity, res.defect)
        out.append(
            {
                **it,
                "ok": res.ok,
                "condition": cond,
                "maturity": res.maturity,
                "defect": res.defect,
                "signals": res.signals,
                "raw": res.raw,
                "grade": price.grade,
                "discount": price.discount,
            }
        )
    return out


def tally(rows) -> tuple[int, int, float]:
    fresh = [r["condition"] * 100 for r in rows if r["label"] == "SEGAR"]
    rotten = [r["condition"] * 100 for r in rows if r["label"] == "BUSUK"]
    fresh_ok = sum(1 for c in fresh if c >= 85)
    rotten_ok = sum(1 for c in rotten if c < 70)
    return fresh_ok, rotten_ok, min(fresh) if fresh else 0.0


def show(rows) -> None:
    print(f"\n{'kelas':7s} {'foto':24s} {'kondisi':>8s} {'grade':>6s} {'diskon':>7s}   sinyal")
    for r in sorted(rows, key=lambda x: (x["label"], x["condition"])):
        sig = " ".join(f"{k}={v:.2f}" for k, v in r["signals"].items() if v > 0.02)
        print(
            f"{r['label']:7s} {r['name']:24s} {r['condition'] * 100:7.1f}% "
            f"{r['grade']:>6s} {r['discount']:6d}%   {sig or '-'}"
        )


def sweep(items, profile, base_cfg):
    grid = itertools.product(
        (250,),                      # mold.v_max
        (0.05,),                     # mold.full_scale
        (0.12, 0.18),                # anomaly.lo
        (None, 0.75, 0.65, 0.55),    # bruise.relative
        (0.30,),                     # bobot anomaly
        (0.20, 0.25, 0.30),          # bobot bruise
    )
    results = []
    total = 2 * 4 * 3
    for i, (vmax, mscale, alo, brel, wanom, wbruise) in enumerate(grid, 1):
        cfg = copy.deepcopy(base_cfg)
        cfg["mold"]["v_max"] = vmax
        cfg["mold"].setdefault("s_max", 60)
        cfg["mold"]["full_scale"] = mscale
        cfg.setdefault("anomaly", {})["lo"] = alo
        cfg["anomaly"].setdefault("hi", 0.75)
        if brel is None:
            cfg["bruise"].pop("relative", None)
        else:
            cfg["bruise"]["relative"] = brel
        cfg.setdefault("weights", {})
        cfg["weights"]["anomaly"] = wanom
        cfg["weights"]["bruise"] = wbruise
        rows = evaluate(items, profile, cfg)
        fok, rok, fmin = tally(rows)
        score = rok * 10 + fok * 20 - max(0.0, 85 - fmin)
        results.append(
            {
                "score": score, "fok": fok, "rok": rok, "fmin": fmin,
                "vmax": vmax, "mscale": mscale, "alo": alo,
                "brel": brel, "wanom": wanom, "wbruise": wbruise,
                "cfg": cfg, "rows": rows,
            }
        )
        if i % 24 == 0:
            print(f"   {i}/{total} konfigurasi …")

    results.sort(key=lambda r: (-r["score"], -r["rok"]))
    print("\nTOP 6 KONFIGURASI")
    print("  skor | segar | busuk | mold_v | mold_s | anom_lo | bruise_rel | w_anom | w_bruise")
    for r in results[:6]:
        rel_txt = f"{r['brel']:.2f}" if r["brel"] is not None else "abs "
        print(
            f"  {r['score']:5.0f} | {r['fok']}/3   | {r['rok']}/10  | "
            f"{r['vmax']:6d} | {r['mscale']:.2f}   | {r['alo']:.2f}    | "
            f"{rel_txt:9s} | {r['wanom']:.2f}   | {r['wbruise']:.2f}"
        )
    return results[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--segar", default="tomatsegar")
    ap.add_argument("--busuk", default="gambartomatbusuk")
    ap.add_argument("--profile", default="tomato")
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()

    segar = Path(args.segar) if Path(args.segar).is_absolute() else ROOT / args.segar
    busuk = Path(args.busuk) if Path(args.busuk).is_absolute() else ROOT / args.busuk

    profile = load_profile(args.profile)
    profile.defect.setdefault("mold", {}).setdefault("full_scale", 0.10)

    pipe = Pipeline(profile_id=args.profile, synthetic=True)
    pipe.profile = profile
    items = gather(pipe, profile, segar, busuk)
    pipe.reader.stop()

    fresh_n = sum(1 for i in items if i["label"] == "SEGAR")
    rotten_n = sum(1 for i in items if i["label"] == "BUSUK")
    print(f"{fresh_n} foto segar, {rotten_n} foto busuk")

    rows = evaluate(items, profile, profile.defect)
    show(rows)
    fok, rok, fmin = tally(rows)
    print(f"\nSEKARANG: segar lulus {fok}/{fresh_n}, busuk lulus {rok}/{rotten_n} (minimum segar {fmin:.1f}%)")

    if args.sweep:
        best = sweep(items, profile, profile.defect)
        print("\nKONFIGURASI TERBAIK")
        print(f"  mold.v_max          : {best['vmax']}")
        print(f"  mold.s_max          : 60")
        print(f"  mold.full_scale     : {best['mscale']}")
        print(f"  anomaly.lo          : {best['alo']}")
        print(f"  bruise.relative     : {best['brel']}")
        print(f"  weights.anomaly     : {best['wanom']}")
        print(f"  weights.bruise      : {best['wbruise']}")
        show(best["rows"])
        fok, rok, fmin = tally(best["rows"])
        print(f"\nSESUDAH: segar lulus {fok}/{fresh_n}, busuk lulus {rok}/{rotten_n} (minimum segar {fmin:.1f}%)")


if __name__ == "__main__":
    main()
