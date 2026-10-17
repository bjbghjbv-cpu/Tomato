"""Model umur simpan: dua proses bersaing (pematangan vs pembusukan).

Mengapa dua proses?
- Pisang & tomat BUKAN sekadar "membusuk dari segar". Tomat hijau membaik
  dulu (matang) lalu memburuk. Satu kurva monoton tidak bisa mewakili itu.
- Dengan dua proses, tomat hijau otomatis punya umur lebih panjang,
  tomat matang lebih pendek, tanpa if-else per tahap.

Chilling injury: di bawah ambang tertentu, laju kerusakan NAIK lagi.
Ini yang membuat respons terhadap suhu berbentuk U, bukan linear.
"""

from __future__ import annotations

from dataclasses import dataclass

from .profile import Profile

UNSELLABLE_CONDITION = 0.50
HORIZON_H = 72.0
MAX_SIM_H = 600.0
DT = 0.5


@dataclass
class TrajectoryPoint:
    t: float
    maturity: float
    defect: float
    condition: float

    def to_dict(self) -> dict[str, float]:
        return {
            "t": round(self.t, 2),
            "maturity": round(self.maturity, 4),
            "defect": round(self.defect, 4),
            "condition": round(self.condition, 4),
        }


def _multipliers(profile: Profile, temp_c: float, rh: float) -> tuple[float, float]:
    sl = profile.shelf_life
    temp_mult = sl.q10 ** ((temp_c - sl.temp_ref_c) / 10.0)

    dec_mult = temp_mult
    if temp_c < sl.chilling_threshold_c:
        span = max(1.0, sl.chilling_threshold_c - 5.0)
        chill = (sl.chilling_threshold_c - temp_c) / span
        chill = max(0.0, min(1.0, chill))
        dec_mult *= 1.0 + sl.chilling_severity * chill

    rh_low = sl.rh_ideal[0] if sl.rh_ideal else 90.0
    rh_mult = 1.0 + 1.5 * max(0.0, (rh_low - rh) / max(1.0, rh_low))
    dec_mult *= rh_mult

    return float(temp_mult), float(dec_mult)


def _step(profile: Profile, m: float, d: float, rip: float, dec: float, dt: float) -> tuple[float, float]:
    sl = profile.shelf_life
    dm = sl.ripening_rate_per_hour * rip * max(0.0, 1.0 - m)
    dd = sl.decay_rate_per_hour * dec * (0.15 + m * m) + d * 0.004
    m = min(1.0, m + dm * dt)
    d = min(1.0, d + dd * dt)
    return m, d


def _run(
    profile: Profile, m: float, d: float, temp_c: float, rh: float
) -> tuple[list[TrajectoryPoint], float, dict[str, float]]:
    rip, dec = _multipliers(profile, temp_c, rh)

    raw: list[tuple[float, float, float, float]] = [(0.0, m, d, m * (1.0 - d))]
    t = 0.0
    while t < MAX_SIM_H:
        m, d = _step(profile, m, d, rip, dec, DT)
        t += DT
        raw.append((t, m, d, m * (1.0 - d)))

    # Puncak kualitas: untuk buah yang belum matang, kualitas NAIK dulu.
    # Sisa umur dihitung setelah puncak, bukan dari titik awal.
    peak_i = max(range(len(raw)), key=lambda i: raw[i][3])
    peak_c = raw[peak_i][3]
    peak_t = raw[peak_i][0]

    remaining = MAX_SIM_H
    if peak_c >= UNSELLABLE_CONDITION:
        for tt, _mm, _dd, cc in raw[peak_i:]:
            if cc < UNSELLABLE_CONDITION:
                remaining = tt
                break
    else:
        # tidak pernah mencapai kondisi layak jual: pakai waktu ke puncak
        remaining = peak_t

    grade_times: dict[str, float] = {}
    for band in profile.grade_bands_sorted():
        if band.min_condition <= 0:
            continue
        for tt, _mm, _dd, cc in raw[peak_i:]:
            if cc * 100.0 < band.min_condition:
                grade_times[band.grade] = tt
                break

    points = [TrajectoryPoint(*row) for row in raw]
    return points, remaining, grade_times


def simulate(
    profile: Profile,
    maturity: float,
    defect: float,
    temp_c: float = 20.0,
    rh: float = 90.0,
    horizon_h: float = HORIZON_H,
) -> dict:
    """Kembalikan trajektori (untuk grafik), sisa umur, dan waktu lintas grade."""
    m = float(max(0.0, min(1.0, maturity)))
    d = float(max(0.0, min(1.0, defect)))

    key = (profile.id, round(m, 2), round(d, 2), int(round(temp_c)), int(round(rh)))
    cached = _MEMO.get(key)
    if cached is None:
        points, remaining, grade_times = _run(profile, m, d, temp_c, rh)
        cached = (points, remaining, grade_times)
        if len(_MEMO) > 2048:
            _MEMO.clear()
        _MEMO[key] = cached

    points, remaining, grade_times = cached

    points = [p for p in points if p.t <= horizon_h + 1e-9]
    if len(points) > 80:
        step = max(1, len(points) // 80)
        points = points[::step] + [points[-1]]

    return {
        "points": [p.to_dict() for p in points],
        "remaining_hours": round(remaining, 1),
        "grade_times": {k: round(v, 1) for k, v in grade_times.items()},
        "unsellable_condition": UNSELLABLE_CONDITION,
    }


_MEMO: dict[tuple, tuple[list[TrajectoryPoint], float, dict[str, float]]] = {}


def state_at(
    profile: Profile,
    maturity: float,
    defect: float,
    hours: float,
    temp_c: float = 20.0,
    rh: float = 90.0,
) -> tuple[float, float, float]:
    """Keadaan (maturity, defect, condition) setelah `hours` jam."""
    if hours <= 0:
        return maturity, defect, maturity * (1.0 - defect)

    m, d = float(maturity), float(defect)
    rip, dec = _multipliers(profile, temp_c, rh)
    t = 0.0
    remaining = hours
    while t + DT <= hours + 1e-9:
        m, d = _step(profile, m, d, rip, dec, DT)
        t += DT
        remaining -= DT
    if remaining > 1e-6:
        m, d = _step(profile, m, d, rip, dec, remaining)
    return m, d, m * (1.0 - d)


def remaining_hours(
    profile: Profile,
    maturity: float,
    defect: float,
    temp_c: float = 20.0,
    rh: float = 90.0,
) -> float:
    return simulate(profile, maturity, defect, temp_c, rh)["remaining_hours"]
