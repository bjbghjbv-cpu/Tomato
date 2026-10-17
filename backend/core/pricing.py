"""Aturan bisnis: kondisi -> grade -> diskon -> harga.

Dua pemicu diskon:
1. Kondisi visual (warna + cacat)
2. Sisa umur simpan (pengaman agar barang yang "masih terlihat bagus"
   tapi tinggal beberapa jam tidak lolos dengan diskon kecil)

Diskon final = yang terbesar dari keduanya.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from .profile import Profile


@dataclass
class Pricing:
    grade: str
    grade_label: str
    color: str
    discount: int
    price: int
    base_price: int
    hold: bool
    reason: str
    hours_discount: int
    condition_pct: float
    next_band: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _round_price(value: float, step: int = 50) -> int:
    return int(round(value / step) * step)


def compute_pricing(
    profile: Profile,
    condition: float,
    remaining_h: float,
    maturity: float,
    defect: float,
) -> Pricing:
    condition_pct = round(max(0.0, min(1.0, condition)) * 100.0, 1)

    # --- state khusus: belum matang, jangan didiskon ---
    hold_cfg = profile.hold or {}
    hold_max_m = float(hold_cfg.get("maturity_max", -1))
    hold_max_d = float(hold_cfg.get("defect_max", 1))
    if hold_cfg and maturity <= hold_max_m and defect <= hold_max_d:
        return Pricing(
            grade="HOLD",
            grade_label=hold_cfg.get("label", "BELUM MATANG"),
            color="#7d7164",
            discount=0,
            price=profile.hold_price or profile.base_price,
            base_price=profile.base_price,
            hold=True,
            reason="belum matang — tahan dari rak, kirim ke pematangan",
            hours_discount=0,
            condition_pct=condition_pct,
            next_band=None,
        )

    bands = profile.grade_bands_sorted()
    band = bands[-1]
    band_index = len(bands) - 1
    for i, b in enumerate(bands):
        if condition_pct >= b.min_condition:
            band = b
            band_index = i
            break

    hours_discount = 0
    for rule in sorted(profile.hours_rules, key=lambda r: float(r["max_hours"])):
        if remaining_h <= float(rule["max_hours"]):
            hours_discount = int(rule["discount"])
            break

    cond_discount = int(band.discount)
    discount = max(cond_discount, int(hours_discount))

    # Grade mengikuti diskon efektif, supaya label dan harga tidak pernah
    # bertentangan (mis. "Grade A" tapi diskon 15%).
    effective = band
    for b in bands:
        if int(b.discount) == discount:
            effective = b
            break
    band = effective
    band_index = bands.index(band)

    price = _round_price(profile.base_price * (100 - discount) / 100.0)

    if hours_discount > cond_discount:
        reason = f"sisa umur {remaining_h:.0f} jam"
    else:
        reason = f"kondisi {condition_pct:.0f}%"

    next_band: dict[str, Any] | None = None
    if band_index + 1 < len(bands):
        nb = bands[band_index + 1]
        gap = max(0.1, band.min_condition - nb.min_condition)
        next_band = {
            "grade": nb.grade,
            "discount": nb.discount,
            "min_condition": nb.min_condition,
            "remaining_to_drop": round(condition_pct - nb.min_condition, 1),
            "progress": round(min(1.0, max(0.0, (condition_pct - nb.min_condition) / gap)), 3),
        }

    return Pricing(
        grade=band.grade,
        grade_label=band.label,
        color=band.color,
        discount=discount,
        price=price,
        base_price=profile.base_price,
        hold=False,
        reason=reason,
        hours_discount=hours_discount,
        condition_pct=condition_pct,
        next_band=next_band,
    )
