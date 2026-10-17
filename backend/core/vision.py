"""Pipeline kamera: capture -> deteksi -> grading -> smoothing -> pricing."""

from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .grader import GradeResult, analyse_crop, calibrate_gains
from .pricing import compute_pricing
from .profile import Profile, list_profiles, load_profile
from .shelf_life import simulate, state_at

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "models" / "best.pt"

EMA_ALPHA = 0.30
LOCK_WINDOW_S = 1.5
LOCK_TOLERANCE = 0.02  # 2 poin persentase
VERIFY_CONFIDENCE = 0.35


class CameraReader:
    """Thread pembaca kamera supaya frame selalu yang terbaru."""

    def __init__(self, index: int = 0, width: int = 1280, height: int = 720, synthetic: bool = False):
        self.index = index
        self.width = width
        self.height = height
        self.synthetic = synthetic
        self.cap: cv2.VideoCapture | None = None
        self.frame: np.ndarray | None = None
        self.ok = False
        self.error = ""
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0 = time.time()

    def start(self) -> None:
        if self.synthetic:
            self.ok = True
            self._thread = threading.Thread(target=self._loop_synthetic, daemon=True)
        else:
            self.cap = self._open_camera()
            if self.cap is None:
                self.error = "kamera tidak dapat dibuka"
                return
            self.ok = True
            self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _open_camera(self) -> cv2.VideoCapture | None:
        backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if os.name == "nt" else [cv2.CAP_ANY]
        for backend in backends:
            cap = cv2.VideoCapture(self.index, backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return cap
            cap.release()
        return None

    def _loop(self) -> None:
        while not self._stop.is_set():
            if self.cap is None:
                break
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self.frame = frame
            else:
                time.sleep(0.02)
        if self.cap is not None:
            self.cap.release()

    def _loop_synthetic(self) -> None:
        while not self._stop.is_set():
            self.frame = synthetic_frame(time.time() - self._t0)
            time.sleep(1 / 20)

    def read(self) -> np.ndarray | None:
        return self.frame

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)


def synthetic_frame(t: float) -> np.ndarray:
    """Frame buatan untuk uji UI tanpa kamera (SSE_SYNTHETIC=1)."""
    img = np.full((720, 1280, 3), 26, np.uint8)
    cv2.rectangle(img, (0, 470), (1280, 720), (42, 42, 48), -1)
    cx, cy = 640, 360
    cv2.circle(img, (cx, cy), 190, (38, 38, 205), -1)
    cv2.circle(img, (cx - 60, cy - 70), 45, (90, 90, 235), -1)      # highlight
    cv2.circle(img, (cx + 70, cy + 45), 55, (30, 30, 55), -1)       # memar
    cv2.circle(img, (cx + 70, cy + 120), 30, (150, 150, 160), -1)   # jamur
    return img


class Pipeline:
    def __init__(
        self,
        profile_id: str = "tomato",
        camera_index: int = 0,
        model_path: str | None = None,
        synthetic: bool = False,
    ):
        self.profile: Profile = load_profile(profile_id)
        self.temp_c = 24.0
        self.rh = 88.0
        self.time_offset_h = 0.0

        self.wb_gains: tuple[float, float, float] | None = None
        self._ema: dict[str, float] | None = None
        self._history: deque[tuple[float, float]] = deque(maxlen=120)
        self._stable_since: float | None = None
        self._locked = False
        self._last_state: dict[str, Any] = {}
        self._fps = 0.0
        self._last_ts = time.time()

        self.reader = CameraReader(camera_index, synthetic=synthetic)
        self.reader.start()

        self.model = None
        self._model_handle = None
        self._model_path: Path | None = None
        self.class_ids: list[int] | None = None
        self.model_note = "fallback warna (model belum ada)"
        path = Path(model_path) if model_path else DEFAULT_MODEL
        if path.exists():
            try:
                from ultralytics import YOLO  # type: ignore

                self._model_handle = YOLO(str(path))
                self._model_path = path
                self._resolve_model_classes()
            except Exception as exc:  # pragma: no cover
                self.model_note = f"model gagal dimuat: {exc}"

    def _resolve_model_classes(self) -> None:
        """Cocokkan kelas model dengan profil aktif.

        Kalau model tidak punya kelas untuk profil ini (mis. pisang belum
        dilatih), model DIMATIKAN agar tidak salah menilai objek.
        """
        if self._model_handle is None:
            return
        names = self._model_handle.names or {}
        wanted = self.profile.detector_class.lower()

        # models/classes.json mencatat kelas yang BENAR-BENAR ada di data latih.
        # Model bisa punya head 2 kelas walau hanya 1 yang pernah dilatih.
        allowed = None
        meta = DEFAULT_MODEL.parent / "classes.json"
        if meta.exists():
            try:
                allowed = {str(c).lower() for c in json.loads(meta.read_text(encoding="utf-8"))["classes"]}
            except Exception:
                allowed = None

        ids = [int(k) for k, v in names.items() if str(v).lower() == wanted]
        if allowed is not None and wanted not in allowed:
            ids = []

        if ids:
            self.model = self._model_handle
            self.class_ids = ids
            self.model_note = f"YOLO {self._model_path.name}"
        else:
            self.model = None
            self.class_ids = None
            self.model_note = (
                f"kelas '{wanted}' belum dilatih — pakai fallback warna"
            )

    # ------------------------------------------------------------------
    def set_profile(self, profile_id: str) -> None:
        self.profile = load_profile(profile_id)
        self._ema = None
        self._history.clear()
        self._locked = False
        self._stable_since = None
        self._resolve_model_classes()

    def set_env(self, temp_c: float | None = None, rh: float | None = None, time_offset_h: float | None = None) -> None:
        if temp_c is not None:
            self.temp_c = float(max(0.0, min(45.0, temp_c)))
        if rh is not None:
            self.rh = float(max(20.0, min(100.0, rh)))
        if time_offset_h is not None:
            self.time_offset_h = float(max(0.0, min(72.0, time_offset_h)))

    def calibrate(self) -> dict[str, Any]:
        frame = self.reader.read()
        if frame is None:
            return {"ok": False, "error": "belum ada frame"}
        self.wb_gains = calibrate_gains(frame)
        return {"ok": True, "gains": [round(g, 3) for g in self.wb_gains]}

    # ------------------------------------------------------------------
    def _color_blob_bbox(
        self, frame: np.ndarray, profile: Profile
    ) -> tuple[int, int, int, int] | None:
        """Deteksi cadangan tanpa model: blob warna terbesar yang cocok profil.

        Jauh lebih tahan posisi dibanding ROI tengah, dan tetap bekerja
        walau model YOLO belum dilatih.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hue = hsv[..., 0].astype(np.int16)
        sat = hsv[..., 1].astype(np.int16)
        val = hsv[..., 2].astype(np.int16)

        mask = np.zeros(hue.shape, dtype=np.uint8)
        for stage in profile.stages:
            sv = (
                (sat >= max(40, stage.s_min - 15))
                & (val >= max(25, stage.v_min - 10))
                & (val <= stage.v_max)
            )
            interval = np.zeros(hue.shape, dtype=bool)
            for lo, hi in stage.hue:
                interval |= (hue >= lo) & (hue <= hi)
            mask |= (sv & interval).astype(np.uint8)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        frame_area = float(frame.shape[0] * frame.shape[1])
        best = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(best)
        if area < 0.02 * frame_area:
            return None
        # blob raksasa biasanya berarti latar belakang ikut terdeteksi
        # (mis. dedaunan hijau pada foto kebun) -> jangan dipakai
        if area > 0.85 * frame_area:
            return None

        x, y, w, h = cv2.boundingRect(best)
        pad = int(0.05 * max(w, h))
        return (
            max(0, x - pad),
            max(0, y - pad),
            min(frame.shape[1], x + w + pad),
            min(frame.shape[0], y + h + pad),
        )

    def _detect(self, frame: np.ndarray, profile: Profile) -> tuple[tuple[int, int, int, int] | None, float, str]:
        fh, fw = frame.shape[:2]
        if self.model is not None:
            try:
                res = self.model.predict(
                    frame, conf=0.25, imgsz=640, verbose=False, classes=self.class_ids
                )[0]
                boxes = getattr(res, "boxes", None)
                if boxes is not None and len(boxes) > 0:
                    cands = []
                    for b in boxes:
                        x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
                        conf = float(b.conf[0])
                        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                        dist = ((cx - fw / 2) ** 2 + (cy - fh / 2) ** 2) ** 0.5
                        dist_n = dist / max(fw, fh)
                        area = max(1.0, (x2 - x1) * (y2 - y1))
                        cands.append([conf, area, dist_n, (int(x1), int(y1), int(x2), int(y2))])
                    max_area = max(c[1] for c in cands)
                    # mode satu item: utamakan kotak terbesar, lalu keyakinan
                    best = max(
                        cands,
                        key=lambda c: 0.55 * c[0] + 0.45 * (c[1] / max_area) - 0.30 * c[2],
                    )
                    return best[3], best[0], "yolo"
            except Exception:
                pass

        # fallback 1: blob warna terbesar (tahan posisi, tanpa model)
        blob = self._color_blob_bbox(frame, profile)
        if blob is not None:
            return blob, 0.45, "color"

        # fallback 2: zona pindai tengah (demo tidak pernah kosong)
        rw, rh = int(fw * 0.34), int(fh * 0.52)
        x1, y1 = int(fw / 2 - rw / 2), int(fh / 2 - rh / 2)
        return (x1, y1, x1 + rw, y1 + rh), 0.5, "roi"

    def _crop(self, frame: np.ndarray, bbox: tuple[int, int, int, int], pad: float = 0.04):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        px, py = int(bw * pad), int(bh * pad)
        x1 = max(0, x1 - px)
        y1 = max(0, y1 - py)
        x2 = min(w, x2 + px)
        y2 = min(h, y2 + py)
        if x2 <= x1 or y2 <= y1:
            return None, (0, 0, 0, 0)
        return frame[y1:y2, x1:x2], (x1, y1, x2 - x1, y2 - y1)

    # ------------------------------------------------------------------
    def read(self) -> tuple[np.ndarray | None, dict[str, Any]]:
        now = time.time()
        dt = max(1e-3, now - self._last_ts)
        self._last_ts = now
        self._fps = 0.92 * self._fps + 0.08 * (1.0 / dt) if self._fps else 1.0 / dt

        frame = self.reader.read()
        if frame is None:
            return None, {
                "ok": False,
                "error": self.reader.error or "menunggu kamera",
                "camera": False,
            }

        profile = self.profile
        bbox, det_conf, source = self._detect(frame, profile)
        result = GradeResult(ok=False, note="tidak ada objek")

        if bbox is not None:
            crop, padded = self._crop(frame, bbox)
            if crop is not None:
                result = analyse_crop(crop, profile, self.wb_gains, bbox=padded)
                bbox = padded

        state = self._compose_state(result, bbox, source, det_conf, now)
        self._last_state = state
        return frame, state

    def analyse_image(self, frame: np.ndarray, method: str = "auto") -> dict[str, Any]:
        profile = self.profile
        if method == "color":
            bbox = self._color_blob_bbox(frame, profile)
            det_conf, source = 0.45, "color"
            if bbox is None:
                fh, fw = frame.shape[:2]
                rw, rh = int(fw * 0.34), int(fh * 0.52)
                bbox = (fw // 2 - rw // 2, fh // 2 - rh // 2, fw // 2 + rw // 2, fh // 2 + rh // 2)
                source = "roi"
        elif method == "roi":
            fh, fw = frame.shape[:2]
            rw, rh = int(fw * 0.34), int(fh * 0.52)
            bbox = (fw // 2 - rw // 2, fh // 2 - rh // 2, fw // 2 + rw // 2, fh // 2 + rh // 2)
            det_conf, source = 0.5, "roi"
        else:
            bbox, det_conf, source = self._detect(frame, profile)

        if bbox is None:
            return {"ok": False, "error": "objek tidak terdeteksi"}
        crop, padded = self._crop(frame, bbox)
        if crop is None:
            return {"ok": False, "error": "crop gagal"}
        result = analyse_crop(crop, profile, self.wb_gains, bbox=padded)
        return self._compose_state(result, padded, source, det_conf, time.time(), smooth=False)

    # ------------------------------------------------------------------
    def _compose_state(
        self,
        res: GradeResult,
        bbox: tuple[int, int, int, int] | None,
        source: str,
        det_conf: float,
        now: float,
        smooth: bool = True,
    ) -> dict[str, Any]:
        profile = self.profile

        if not res.ok:
            self._history.clear()
            self._stable_since = None
            self._locked = False
            return {
                "ok": False,
                "camera": True,
                "error": res.note or "tidak ada objek",
                "detector": source,
                "fps": round(self._fps, 1),
                "config": self._config(),
                "profile": profile.public_dict(),
            }

        # ---- smoothing ----
        if not smooth:
            self._ema = {"maturity": res.maturity, "defect": res.defect}
        elif self._ema is None:
            self._ema = {"maturity": res.maturity, "defect": res.defect}
        else:
            a = EMA_ALPHA
            self._ema["maturity"] = (1 - a) * self._ema["maturity"] + a * res.maturity
            self._ema["defect"] = (1 - a) * self._ema["defect"] + a * res.defect

        m = float(self._ema["maturity"])
        d = float(self._ema["defect"])
        visual_condition = m * (1.0 - d)

        # ---- kestabilan & lock ----
        if smooth:
            self._history.append((now, visual_condition))
            while self._history and now - self._history[0][0] > LOCK_WINDOW_S:
                self._history.popleft()
            if len(self._history) >= 4:
                vals = [v for _, v in self._history]
                if max(vals) - min(vals) <= LOCK_TOLERANCE:
                    if self._stable_since is None:
                        self._stable_since = self._history[0][0]
                    self._locked = (now - self._stable_since) >= LOCK_WINDOW_S
                else:
                    self._stable_since = None
                    self._locked = False
            else:
                self._locked = False
        else:
            self._locked = False

        # ---- proyeksi waktu ----
        if self.time_offset_h > 0:
            m_disp, d_disp, condition = state_at(
                profile, m, d, self.time_offset_h, self.temp_c, self.rh
            )
        else:
            m_disp, d_disp, condition = m, d, visual_condition

        sim = simulate(profile, m_disp, d_disp, self.temp_c, self.rh)
        remaining = float(sim["remaining_hours"])
        pricing = compute_pricing(profile, condition, remaining, m_disp, d_disp)

        confidence = float(res.confidence) * (1.0 if (self._locked or not smooth) else 0.85)
        needs_verify = confidence < VERIFY_CONFIDENCE or not res.ok

        # rincian sinyal cacat dalam persen kontribusi
        weights = (profile.defect or {}).get("weights", {})
        wsum = sum(float(weights.get(k, 0.0)) for k in res.signals) or 1.0
        signal_contrib = {
            k: round(100.0 * float(weights.get(k, 0.0)) * v / wsum, 1)
            for k, v in res.signals.items()
        }

        return {
            "ok": True,
            "camera": True,
            "locked": self._locked,
            "needs_verify": needs_verify,
            "confidence": round(confidence, 3),
            "condition_pct": round(condition * 100.0, 1),
            "visual_condition_pct": round(visual_condition * 100.0, 1),
            "maturity_pct": round(m_disp * 100.0, 1),
            "defect_pct": round(d_disp * 100.0, 1),
            "stage": res.stage,
            "stage_label": res.stage_label,
            "signals": {k: round(v, 3) for k, v in res.signals.items()},
            "signal_contrib": signal_contrib,
            "grade": pricing.grade,
            "grade_label": pricing.grade_label,
            "color": pricing.color,
            "discount": pricing.discount,
            "price": pricing.price,
            "base_price": pricing.base_price,
            "hold": pricing.hold,
            "reason": pricing.reason,
            "hours_discount": pricing.hours_discount,
            "next_band": pricing.next_band,
            "remaining_hours": round(remaining, 1),
            "trajectory": sim["points"],
            "grade_times": sim["grade_times"],
            "unsellable_condition": sim["unsellable_condition"],
            "config": self._config(),
            "profile": profile.public_dict(),
            "detector": source,
            "det_conf": round(float(det_conf), 3),
            "bbox": list(bbox) if bbox else None,
            "fps": round(self._fps, 1),
            "ts": round(now, 3),
        }

    def _config(self) -> dict[str, float]:
        return {
            "temp_c": round(self.temp_c, 1),
            "rh": round(self.rh, 1),
            "time_offset_h": round(self.time_offset_h, 1),
        }

    def status(self) -> dict[str, Any]:
        return {
            "camera": self.reader.ok,
            "error": self.reader.error,
            "detector": self.model_note,
            "profile": self.profile.id,
            "profiles": list_profiles(),
            "calibrated": self.wb_gains is not None,
            "fps": round(self._fps, 1),
        }


def encode_jpeg(frame: np.ndarray, width: int = 960, quality: int = 72) -> bytes:
    h, w = frame.shape[:2]
    if w > width:
        scale = width / float(w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buf.tobytes() if ok else b""
