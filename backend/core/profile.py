"""Loader profil komoditas.

Semua parameter biologis & bisnis tinggal di file YAML, bukan di kode.
Menambah komoditas = menambah satu file YAML.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROFILE_DIR = Path(__file__).resolve().parent.parent / "profiles"


@dataclass
class Stage:
    name: str
    quality: float
    hue: list[list[int]]
    s_min: int = 0
    v_min: int = 0
    v_max: int = 255
    expected_s: float = 180.0
    label: str = ""


@dataclass
class GradeBand:
    grade: str
    min_condition: float
    discount: int
    color: str
    label: str = ""


@dataclass
class ShelfLifeCfg:
    max_hours_at_20c: float = 120.0
    q10: float = 2.2
    temp_ref_c: float = 20.0
    ripening_rate_per_hour: float = 0.026
    decay_rate_per_hour: float = 0.0038
    chilling_threshold_c: float = 12.0
    chilling_severity: float = 1.5
    rh_ideal: list[float] = field(default_factory=lambda: [90.0, 95.0])


@dataclass
class Profile:
    id: str
    display_name: str
    unit: str
    base_price: int
    detector_class: str
    stages: list[Stage]
    grades: list[GradeBand]
    shelf_life: ShelfLifeCfg
    hold_price: int = 0
    defect: dict[str, Any] = field(default_factory=dict)
    hours_rules: list[dict[str, Any]] = field(default_factory=list)
    hold: dict[str, Any] = field(default_factory=dict)

    # ---- urutan tahap, dipakai untuk label & dominasi ----
    def stage_by_name(self, name: str) -> Stage | None:
        for s in self.stages:
            if s.name == name:
                return s
        return None

    def grade_bands_sorted(self) -> list[GradeBand]:
        return sorted(self.grades, key=lambda g: g.min_condition, reverse=True)

    def public_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "display_name": self.display_name,
            "unit": self.unit,
            "base_price": self.base_price,
            "hold_price": self.hold_price,
            "detector_class": self.detector_class,
            "stages": [
                {"name": s.name, "label": s.label, "quality": s.quality}
                for s in self.stages
            ],
            "grades": [
                {
                    "grade": g.grade,
                    "label": g.label,
                    "min_condition": g.min_condition,
                    "discount": g.discount,
                    "color": g.color,
                }
                for g in self.grade_bands_sorted()
            ],
            "hours_rules": self.hours_rules,
            "shelf_life": {
                "max_hours_at_20c": self.shelf_life.max_hours_at_20c,
                "q10": self.shelf_life.q10,
                "chilling_threshold_c": self.shelf_life.chilling_threshold_c,
            },
        }


def _build(raw: dict[str, Any]) -> Profile:
    stages = [
        Stage(
            name=s["name"],
            quality=float(s["quality"]),
            hue=[[int(a), int(b)] for a, b in s["hue"]],
            s_min=int(s.get("s_min", 0)),
            v_min=int(s.get("v_min", 0)),
            v_max=int(s.get("v_max", 255)),
            expected_s=float(s.get("expected_s", 180.0)),
            label=s.get("label", s["name"]),
        )
        for s in raw["stages"]
    ]

    grades = [
        GradeBand(
            grade=g["grade"],
            min_condition=float(g["min_condition"]),
            discount=int(g["discount"]),
            color=g.get("color", "#888888"),
            label=g.get("label", g["grade"]),
        )
        for g in raw["grades"]
    ]

    sl = raw.get("shelf_life", {})
    shelf = ShelfLifeCfg(
        max_hours_at_20c=float(sl.get("max_hours_at_20c", 120)),
        q10=float(sl.get("q10", 2.2)),
        temp_ref_c=float(sl.get("temp_ref_c", 20)),
        ripening_rate_per_hour=float(sl.get("ripening_rate_per_hour", 0.026)),
        decay_rate_per_hour=float(sl.get("decay_rate_per_hour", 0.0038)),
        chilling_threshold_c=float(sl.get("chilling_threshold_c", 12)),
        chilling_severity=float(sl.get("chilling_severity", 1.5)),
        rh_ideal=[float(x) for x in sl.get("rh_ideal", [90, 95])],
    )

    return Profile(
        id=raw["id"],
        display_name=raw.get("display_name", raw["id"]),
        unit=raw.get("unit", "buah"),
        base_price=int(raw.get("base_price", 0)),
        detector_class=raw.get("detector_class", raw["id"]),
        stages=stages,
        grades=grades,
        shelf_life=shelf,
        hold_price=int(raw.get("hold_price", 0)),
        defect=raw.get("defect", {}),
        hours_rules=raw.get("hours_rules", []),
        hold=raw.get("hold", {}),
    )


@lru_cache(maxsize=32)
def load_profile(profile_id: str = "tomato") -> Profile:
    path = PROFILE_DIR / f"{profile_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Profil tidak ditemukan: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return _build(raw)


def list_profiles() -> list[str]:
    return sorted(p.stem for p in PROFILE_DIR.glob("*.yaml"))
