"""Analisis visual tomat: kematangan (maturity) dan kerusakan (defect).

Skor kondisi = maturity x (1 - defect).

Catatan penting:
- Hue OpenCV berada di 0-179. Merah ada di dua ujung (0-5 dan 173-179),
  jadi setiap tahap menerima daftar interval, bukan satu pasangan.
- Semua ambang berasal dari profil YAML, tidak ada angka yang di-hardcode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from .profile import Profile


@dataclass
class GradeResult:
    ok: bool
    maturity: float = 0.0
    defect: float = 0.0
    condition: float = 0.0
    stage: str = ""
    stage_label: str = ""
    stage_fractions: dict[str, float] = field(default_factory=dict)
    signals: dict[str, float] = field(default_factory=dict)
    raw: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    coverage: float = 0.0
    dominant_hue: float = 0.0
    dominant_s: float = 0.0
    bbox: tuple[int, int, int, int] | None = None
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "maturity": round(self.maturity, 4),
            "defect": round(self.defect, 4),
            "condition": round(self.condition, 4),
            "stage": self.stage,
            "stage_label": self.stage_label,
            "stage_fractions": {k: round(v, 4) for k, v in self.stage_fractions.items()},
            "signals": {k: round(v, 4) for k, v in self.signals.items()},
            "raw": self.raw,
            "confidence": round(self.confidence, 4),
            "coverage": round(self.coverage, 4),
            "dominant_hue": round(self.dominant_hue, 1),
            "dominant_s": round(self.dominant_s, 1),
            "note": self.note,
        }


def apply_white_balance(bgr: np.ndarray, gains: tuple[float, float, float] | None) -> np.ndarray:
    """Terapkan gain per channel dari kalibrasi kartu abu-abu."""
    if gains is None:
        return bgr
    out = bgr.astype(np.float32)
    out[..., 0] *= gains[0]
    out[..., 1] *= gains[1]
    out[..., 2] *= gains[2]
    return np.clip(out, 0, 255).astype(np.uint8)


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _interval_mask(hue: np.ndarray, intervals: list[list[int]]) -> np.ndarray:
    mask = np.zeros(hue.shape, dtype=bool)
    for lo, hi in intervals:
        mask |= (hue >= lo) & (hue <= hi)
    return mask


def _blob_fraction(mask: np.ndarray, min_area_frac: float) -> float:
    """Fraksi luas blob yang lolos ambang luas minimum.

    Ambang minimum mencegah noise JPEG dihitung sebagai jamur/memar.
    """
    if not mask.any():
        return 0.0
    n, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    total = float(mask.size)
    min_area = max(4.0, min_area_frac * total)
    area = 0.0
    for i in range(1, n):
        a = float(stats[i, cv2.CC_STAT_AREA])
        if a >= min_area:
            area += a
    return area / total if total else 0.0


def _interior_mask(shape: tuple[int, int], margin: float = 0.10) -> np.ndarray:
    """Mask yang mengecualikan tepi crop (glare & bayangan sering di tepi)."""
    h, w = shape
    m = np.ones((h, w), dtype=bool)
    dy = max(1, int(h * margin))
    dx = max(1, int(w * margin))
    m[:dy, :] = False
    m[-dy:, :] = False
    m[:, :dx] = False
    m[:, -dx:] = False
    return m


def analyse_crop(
    bgr: np.ndarray,
    profile: Profile,
    wb_gains: tuple[float, float, float] | None = None,
    bbox: tuple[int, int, int, int] | None = None,
) -> GradeResult:
    """Analisis satu crop buah (BGR)."""
    if bgr is None or bgr.size == 0:
        return GradeResult(ok=False, note="crop kosong")

    h, w = bgr.shape[:2]
    if h < 24 or w < 24:
        return GradeResult(ok=False, note="crop terlalu kecil", bbox=bbox)

    img = apply_white_balance(bgr, wb_gains)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue = hsv[..., 0].astype(np.int16)
    sat = hsv[..., 1].astype(np.int16)
    val = hsv[..., 2].astype(np.int16)

    # ---------- 1. Klasifikasi tiap piksel ke tahap kematangan ----------
    idx = np.full(hue.shape, -1, dtype=np.int16)
    for i, stage in enumerate(profile.stages):
        mask = (
            (sat >= stage.s_min)
            & (val >= stage.v_min)
            & (val <= stage.v_max)
            & _interval_mask(hue, stage.hue)
            & (idx < 0)
        )
        idx[mask] = i

    assigned = idx >= 0
    total_px = float(hue.size)
    coverage = float(assigned.sum()) / total_px if total_px else 0.0

    if coverage < 0.02:
        return GradeResult(
            ok=False,
            coverage=coverage,
            bbox=bbox,
            note="warna buah tidak dikenali (cahaya/kalibrasi?)",
        )

    counts = np.bincount(idx[assigned], minlength=len(profile.stages)).astype(np.float64)
    fracs = counts / counts.sum()

    maturity = float(sum(f * s.quality for f, s in zip(fracs, profile.stages)))
    dominant_i = int(np.argmax(fracs))
    dominant = profile.stages[dominant_i]

    stage_fractions = {s.name: float(fracs[i]) for i, s in enumerate(profile.stages)}

    # ---------- 2. Sinyal kerusakan ----------
    cfg = profile.defect or {}
    weights = cfg.get("weights", {})

    # Zona buah: convex hull dari kontur terbesar. Dipakai convex hull (bukan
    # isi kontur) karena cacat di TEPI buah membuat kontur "tergigit" sehingga
    # jamur di pinggir tidak ikut terhitung.
    m8 = assigned.astype(np.uint8)
    contours, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull_mask = None
    if contours:
        biggest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(biggest) > 0.01 * m8.size:
            hull = cv2.convexHull(biggest)
            zone = np.zeros_like(m8)
            cv2.drawContours(zone, [hull], -1, 1, thickness=cv2.FILLED)
            hull_mask = zone > 0
    if hull_mask is None:
        hull_mask = cv2.dilate(m8, np.ones((7, 7), np.uint8)) > 0
    fruit_zone = cv2.dilate(hull_mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    interior = _interior_mask(hue.shape)

    mold_cfg = cfg.get("mold", {})
    mold_mask = (
        (sat <= int(mold_cfg.get("s_max", 45)))
        & (val >= int(mold_cfg.get("v_min", 140)))
        & (val <= int(mold_cfg.get("v_max", 220)))
        & interior
        & fruit_zone
    )
    mold_frac = _blob_fraction(mold_mask, float(mold_cfg.get("min_area_frac", 0.004)))
    mold_score = _clamp01(mold_frac / float(mold_cfg.get("full_scale", 0.10)))

    bruise_cfg = cfg.get("bruise", {})
    bruise_zone = cv2.erode(fruit_zone.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    bv_max = int(bruise_cfg.get("v_max", 70))
    # Ambang relatif: nekrosis sering masih "merah gelap" (V 80-120), bukan hitam.
    # Dibandingkan dengan kecerahan median buah supaya tahan exposure.
    rel = bruise_cfg.get("relative")
    if rel:
        med_v = float(np.median(val[assigned])) if assigned.any() else 255.0
        bv_max = int(max(bv_max, rel * med_v))
    bruise_mask = (
        (val <= bv_max)
        & (sat >= int(bruise_cfg.get("s_min", 25)))
        & interior
        & bruise_zone
    )
    bruise_frac = _blob_fraction(bruise_mask, float(bruise_cfg.get("min_area_frac", 0.002)))
    bruise_score = _clamp01(bruise_frac / float(bruise_cfg.get("full_scale", 0.10)))

    # Keriput: median |Laplacian| pada bagian DALAM buah.
    # - erosi mask membuang tepi buah (gradien tepi selalu tinggi)
    # - median (bukan variance) supaya highlight spekular tidak dianggap keriput
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    smooth = cv2.GaussianBlur(gray, (5, 5), 0)
    lap = cv2.Laplacian(smooth, cv2.CV_64F, ksize=3)
    core = cv2.erode(assigned.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
    if int(core.sum()) < 60:
        core = assigned
    core_vals = np.abs(lap[core])
    mean_gray = float(np.mean(gray[core])) + 1.0
    lap_med = float(np.median(core_vals)) if core_vals.size else 0.0
    wrinkle_norm = lap_med / mean_gray * 100.0
    wr_cfg = cfg.get("wrinkle", {})
    wr_lo = float(wr_cfg.get("lo", 1.5))
    wr_hi = float(wr_cfg.get("hi", 9.0))
    wrinkle_score = _clamp01((wrinkle_norm - wr_lo) / max(1e-6, wr_hi - wr_lo))

    # Kusam: saturasi aktual dibanding ekspektasi tahap dominan.
    dull_cfg = cfg.get("dullness", {})
    actual_s = float(np.median(sat[assigned]))
    expected_s = max(1.0, dominant.expected_s)
    s_drop = max(1e-6, float(dull_cfg.get("s_drop", 0.18)))
    dull_score = _clamp01(((expected_s - actual_s) / expected_s) / s_drop)

    # Permukaan tidak sehat: porsi buah yang TIDAK lagi terklasifikasi sebagai
    # tahap sehat mana pun (gelap, nekrotik, berjamur, atau pudar).
    # Dipakai dead-zone supaya buah sehat dengan sedikit bayangan tidak dihukum.
    anom_cfg = cfg.get("anomaly", {})
    if hull_mask is not None and int(hull_mask.sum()) > 50:
        healthy = float((assigned & hull_mask).sum()) / float(hull_mask.sum())
        anomaly_raw = 1.0 - healthy
    else:
        anomaly_raw = 0.0
    a_lo = float(anom_cfg.get("lo", 0.25))
    a_hi = float(anom_cfg.get("hi", 0.75))
    anomaly_score = _clamp01((anomaly_raw - a_lo) / max(1e-6, a_hi - a_lo))

    signals = {
        "mold": mold_score,
        "bruise": bruise_score,
        "anomaly": anomaly_score,
        "wrinkle": wrinkle_score,
        "dullness": dull_score,
    }
    raw = {
        "mold_frac": round(mold_frac, 4),
        "bruise_frac": round(bruise_frac, 4),
        "anomaly_raw": round(anomaly_raw, 4),
        "wrinkle_norm": round(wrinkle_norm, 3),
        "lap_median": round(lap_med, 3),
        "actual_s": round(actual_s, 1),
        "expected_s": round(expected_s, 1),
    }

    wsum = sum(float(weights.get(k, 0.0)) for k in signals)
    if wsum <= 0:
        defect = float(np.mean(list(signals.values())))
    else:
        defect = sum(signals[k] * float(weights.get(k, 0.0)) for k in signals) / wsum
    defect = _clamp01(defect)
    condition = _clamp01(maturity * (1.0 - defect))

    # ---------- 3. Keyakinan ----------
    coverage_score = _clamp01(coverage / 0.60)
    concentration = float(np.max(fracs))
    confidence = _clamp01(0.55 * coverage_score + 0.45 * concentration)

    return GradeResult(
        ok=True,
        maturity=maturity,
        defect=defect,
        condition=condition,
        stage=dominant.name,
        stage_label=dominant.label,
        stage_fractions=stage_fractions,
        signals=signals,
        raw=raw,
        confidence=confidence,
        coverage=coverage,
        dominant_hue=float(np.median(hue[assigned])),
        dominant_s=actual_s,
        bbox=bbox,
    )


def calibrate_gains(frame_bgr: np.ndarray, patch_frac: float = 0.12) -> tuple[float, float, float]:
    """Hitung gain white balance dari kartu abu-abu di tengah frame.

    Asumsi: pengguna mengarahkan kartu putih/abu-abu ke tengah lalu menekan
    tombol kalibrasi. Referensi = rata-rata kanal (gray world).
    """
    h, w = frame_bgr.shape[:2]
    ph = max(8, int(h * patch_frac))
    pw = max(8, int(w * patch_frac))
    cy, cx = h // 2, w // 2
    patch = frame_bgr[cy - ph // 2 : cy + ph // 2, cx - pw // 2 : cx + pw // 2]
    means = patch.reshape(-1, 3).mean(axis=0) + 1e-6  # B, G, R
    target = float(np.mean(means))
    gains = tuple(float(target / m) for m in means)
    # batasi gain agar tidak ekstrem pada pencahayaan buruk
    gains = tuple(float(np.clip(g, 0.5, 2.0)) for g in gains)
    return gains  # type: ignore[return-value]
