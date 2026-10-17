"""Cari indeks kamera yang benar-benar menghasilkan gambar.

    python tools/find_camera.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "camera_scan"
OUT.mkdir(exist_ok=True)


def main() -> None:
    print("memindai indeks kamera 0-5 …\n")
    found = []
    for idx in range(6):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print(f"  {idx}: tidak bisa dibuka")
            cap.release()
            continue
        ok = False
        frame = None
        for _ in range(6):
            ok, frame = cap.read()
            if ok and frame is not None:
                break
        cap.release()

        if not ok or frame is None:
            print(f"  {idx}: terbuka, tapi tidak ada frame")
            continue

        h, w = frame.shape[:2]
        mean = float(frame.mean())
        path = OUT / f"cam_{idx}.jpg"
        cv2.imwrite(str(path), frame)
        note = "gelap" if mean < 12 else "ada gambar"
        print(f"  {idx}: {w}x{h}  brightness={mean:5.1f}  {note}  -> {path.name}")
        if mean >= 12:
            found.append(idx)

    print()
    if found:
        print(f"kamera yang tampak aktif: {found}")
        print(f"jalankan:  $env:SSE_CAMERA=\"{found[0]}\"")
    else:
        print("Tidak ada kamera yang menghasilkan gambar terang.")
        print("Periksa: kamera terhalang, dipakai aplikasi lain, atau ijin kamera mati.")
        print("Untuk sementara bisa pakai:  $env:SSE_SYNTHETIC=\"1\"")


if __name__ == "__main__":
    sys.exit(main())
