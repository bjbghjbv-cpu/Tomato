"""Cetak tabel angka model: sisa umur pada berbagai suhu & pemetaan harga.

Dipakai untuk memeriksa kewajaran parameter profil sebelum demo.

    python tools/check_numbers.py --profile tomato
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.core.pricing import compute_pricing
from backend.core.profile import load_profile
from backend.core.shelf_life import simulate

PROPS = [
    ("segar", 1.00, 0.05),
    ("sedikit kusam", 1.00, 0.15),
    ("memar jelas", 1.00, 0.28),
    ("keriput + jamur", 1.00, 0.52),
    ("belum matang", 0.40, 0.02),
]

TEMPS = [5, 12, 20, 25, 35, 38]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="tomato")
    ap.add_argument("--rh", type=float, default=88)
    args = ap.parse_args()

    p = load_profile(args.profile)
    print(f"Profil: {p.display_name}  (RH {args.rh:.0f}%, acuan {p.shelf_life.temp_ref_c:.0f}°C)")
    print(f"Q10={p.shelf_life.q10}  chilling<{p.shelf_life.chilling_threshold_c}°C  "
          f"severity={p.shelf_life.chilling_severity}\n")

    header = "kondisi".ljust(20) + "".join(f"{t:>8}°C" for t in TEMPS)
    print("SISA UMUR SIMPAN (jam)")
    print(header)
    for label, m, d in PROPS:
        cells = []
        for t in TEMPS:
            sim = simulate(p, m, d, temp_c=t, rh=args.rh)
            cells.append(f"{sim['remaining_hours']:>10.0f}")
        print(label.ljust(20) + "".join(cells))

    print("\nPEMETAAN HARGA @24°C")
    print("kondisi".ljust(20) + f"{'grade':>7}{'diskon':>8}{'harga':>9}   alasan")
    for label, m, d in PROPS:
        c = m * (1 - d)
        sim = simulate(p, m, d, temp_c=24, rh=args.rh)
        pr = compute_pricing(p, c, sim["remaining_hours"], m, d)
        print(
            label.ljust(20)
            + f"{pr.grade:>7}{pr.discount:>7}%{pr.price:>9,}   {pr.reason}"
        )

    # titik kritis: apakah respons suhu berbentuk U?
    print("\nUJI U-SHAPE (tomat merah segar)")
    values = []
    for t in range(5, 39, 1):
        values.append((t, simulate(p, 1.0, 0.05, temp_c=t, rh=args.rh)["remaining_hours"]))
    best_t, best_h = max(values, key=lambda kv: kv[1])
    print(f"  umur terpanjang pada {best_t}°C ({best_h:.0f} jam)")
    print(f"  pada 5°C : {dict(values)[5]:.0f} jam")
    print(f"  pada 38°C: {dict(values)[38]:.0f} jam")


if __name__ == "__main__":
    main()
