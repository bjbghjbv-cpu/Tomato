"""Unit test logika inti — tidak butuh kamera maupun model.

Jalankan:
    .venv\\Scripts\\python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.core.grader import analyse_crop
from backend.core.pricing import compute_pricing
from backend.core.profile import list_profiles, load_profile
from backend.core.shelf_life import simulate, state_at


class TestProfile(unittest.TestCase):
    def test_profiles_load(self):
        self.assertIn("tomato", list_profiles())
        self.assertIn("banana", list_profiles())

    def test_tomato_stages_ordered_and_scored(self):
        p = load_profile("tomato")
        self.assertGreaterEqual(len(p.stages), 4)
        qualities = [s.quality for s in p.stages]
        self.assertEqual(qualities, sorted(qualities))

    def test_red_hue_wraps(self):
        p = load_profile("tomato")
        red = p.stage_by_name("red_ripe")
        self.assertIsNotNone(red)
        intervals = red.hue
        self.assertTrue(any(lo == 0 for lo, _ in intervals))
        self.assertTrue(any(hi >= 170 for _, hi in intervals))


class TestPricing(unittest.TestCase):
    def setUp(self):
        self.p = load_profile("tomato")

    def _grade(self, condition_pct: float, hours: float = 100.0):
        return compute_pricing(self.p, condition_pct / 100.0, hours, maturity=1.0, defect=0.0)

    def test_grade_bands(self):
        self.assertEqual(self._grade(95).grade, "A")
        self.assertEqual(self._grade(95).discount, 0)
        self.assertEqual(self._grade(76).grade, "B")
        self.assertEqual(self._grade(76).discount, 15)
        self.assertEqual(self._grade(60).grade, "C")
        self.assertEqual(self._grade(60).discount, 30)
        self.assertEqual(self._grade(40).grade, "D")
        self.assertEqual(self._grade(40).discount, 50)

    def test_price_math(self):
        row = self._grade(60)
        self.assertEqual(row.price, round(self.p.base_price * 0.7 / 50) * 50)

    def test_hours_override(self):
        # kondisi bagus, tapi sisa umur kritis -> diskon tetap besar
        row = self._grade(95, hours=8)
        self.assertEqual(row.discount, 50)
        self.assertIn("umur", row.reason)

    def test_hold_state(self):
        row = compute_pricing(self.p, 0.40, 200.0, maturity=0.30, defect=0.05)
        self.assertEqual(row.grade, "HOLD")
        self.assertEqual(row.discount, 0)
        self.assertEqual(row.price, self.p.hold_price)

    def test_discount_monotonic(self):
        prev = -1
        for cond in [98, 90, 80, 70, 60, 50, 30, 10]:
            d = self._grade(cond).discount
            self.assertGreaterEqual(d, prev)
            prev = d


class TestShelfLife(unittest.TestCase):
    def setUp(self):
        self.p = load_profile("tomato")

    def test_heat_shortens_life(self):
        cool = simulate(self.p, 1.0, 0.05, temp_c=20)["remaining_hours"]
        hot = simulate(self.p, 1.0, 0.05, temp_c=35)["remaining_hours"]
        self.assertLess(hot, cool)

    def test_chilling_injury(self):
        # 5C tidak lebih baik dari 20C untuk tomat
        ambient = simulate(self.p, 1.0, 0.05, temp_c=20)["remaining_hours"]
        cold = simulate(self.p, 1.0, 0.05, temp_c=5)["remaining_hours"]
        self.assertLess(cold, ambient)

    def test_green_lasts_longer(self):
        green = simulate(self.p, 0.40, 0.02, temp_c=20)["remaining_hours"]
        ripe = simulate(self.p, 1.0, 0.02, temp_c=20)["remaining_hours"]
        self.assertGreater(green, ripe)

    def test_trajectory_is_inverted_u(self):
        # tomat hijau: kualitas naik dulu, lalu turun
        sim = simulate(self.p, 0.40, 0.02, temp_c=20, horizon_h=600)
        conds = [p["condition"] for p in sim["points"]]
        peak_idx = int(np.argmax(conds))
        self.assertGreater(peak_idx, 0)
        self.assertLess(conds[-1], conds[peak_idx])

    def test_state_at_matches(self):
        m, d, c = state_at(self.p, 1.0, 0.05, 24.0, 20, 90)
        self.assertLess(c, 1.0)
        self.assertGreater(c, 0.0)


class TestGrader(unittest.TestCase):
    def setUp(self):
        self.p = load_profile("tomato")

    def _solid(self, bgr: tuple[int, int, int], size: int = 160) -> np.ndarray:
        img = np.zeros((size, size, 3), np.uint8)
        img[:, :] = bgr
        return img

    def test_ripe_red_is_high_condition(self):
        r = analyse_crop(self._solid((40, 40, 205)), self.p)
        self.assertTrue(r.ok)
        self.assertGreater(r.maturity, 0.9)
        self.assertGreater(r.condition, 0.85)

    def test_green_is_low_maturity(self):
        r = analyse_crop(self._solid((60, 170, 60)), self.p)
        self.assertTrue(r.ok)
        self.assertLess(r.maturity, 0.6)

    def test_dark_bruise_raises_defect(self):
        img = self._solid((40, 40, 205))
        img[70:95, 70:95] = (30, 30, 35)
        r = analyse_crop(img, self.p)
        self.assertTrue(r.ok)
        self.assertGreater(r.signals["bruise"], 0.15)
        self.assertGreater(r.defect, 0.03)

    def test_white_patch_reads_as_mold(self):
        img = self._solid((40, 40, 205))
        img[60:90, 60:95] = (205, 205, 205)
        r = analyse_crop(img, self.p)
        self.assertTrue(r.ok)
        self.assertGreater(r.signals["mold"], 0.2)

    def test_unknown_color_is_rejected(self):
        r = analyse_crop(self._solid((10, 10, 10)), self.p)
        self.assertFalse(r.ok)


if __name__ == "__main__":
    unittest.main()
