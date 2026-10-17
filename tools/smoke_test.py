"""Smoke test end-to-end terhadap server yang sedang berjalan.

    python tools/smoke_test.py [--url http://127.0.0.1:8000]

Membuat gambar tomat sintetis (segar / memar / berjamur kusam), mengirimnya
ke /api/scan, lalu memeriksa apakah grade yang keluar masuk akal.
Tidak butuh kamera maupun model.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def tomato_image(kind: str, size: int = 420) -> np.ndarray:
    img = np.full((size, size, 3), 30, np.uint8)
    c = size // 2
    r = int(size * 0.40)
    cv2.circle(img, (c, c), r, (38, 38, 205), -1)
    cv2.circle(img, (c - r // 3, c - r // 3), r // 4, (92, 92, 236), -1)  # kilau

    if kind in {"memar", "busuk"}:
        cv2.circle(img, (c + r // 3, c + r // 4), int(r * 0.42), (28, 28, 52), -1)  # memar
    if kind == "busuk":
        cv2.circle(img, (c - r // 4, c + r // 2), int(r * 0.34), (152, 152, 162), -1)  # jamur
        cv2.circle(img, (c - r // 2, c - r // 4), int(r * 0.30), (30, 30, 48), -1)     # memar 2
        # seluruh buah kusam
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[..., 1] = (hsv[..., 1] * 0.45).astype(np.uint8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img


def post_json(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--save-dir", default=None, help="simpan gambar sintetis untuk diperiksa")
    args = ap.parse_args()

    try:
        with urllib.request.urlopen(f"{args.url}/api/status", timeout=10) as r:
            status = json.loads(r.read().decode())
        print(f"server  : OK  profil={status['profile']}  kamera={status['camera']}")
        print(f"detector: {status['detector']}")
    except Exception as exc:
        print(f"server  : GAGAL ({exc})")
        return 1

    # --- websocket: pastikan frame + state mengalir ---
    try:
        import asyncio

        import websockets

        ws_url = args.url.replace("http://", "ws://").replace("https://", "wss://") + "/ws"

        async def first_message():
            async with websockets.connect(ws_url) as sock:
                return json.loads(await asyncio.wait_for(sock.recv(), timeout=15))

        msg = asyncio.run(first_message())
        ok_ws = msg.get("type") == "frame" and msg.get("state", {}).get("ok") is True
        print(f"websocket: {'OK' if ok_ws else 'GAGAL'}  frame={'ada' if msg.get('image') else 'kosong'}  "
              f"kondisi={msg.get('state', {}).get('condition_pct')}%")
        if not ok_ws:
            return 2
    except Exception as exc:
        print(f"websocket: GAGAL ({exc})")
        return 2

    expect = {"segar": "A", "memar": "B", "busuk": "D"}
    ok = True
    print("\nhasil scan gambar sintetis:")
    for kind in ("segar", "memar", "busuk"):
        img = tomato_image(kind)
        if args.save_dir:
            out = Path(args.save_dir)
            out.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out / f"{kind}.jpg"), img)
        ok_enc, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not ok_enc:
            print(f"  {kind:6s}: encode gagal")
            ok = False
            continue
        b64 = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
        try:
            res = post_json(f"{args.url}/api/scan", {"image": b64, "source": "color"})
        except Exception as exc:
            print(f"  {kind:6s}: request gagal ({exc})")
            ok = False
            continue

        if not res.get("ok"):
            print(f"  {kind:6s}: tidak terbaca — {res.get('error')}")
            ok = False
            continue

        print(
            f"  {kind:6s}: kondisi {res['condition_pct']:5.1f}%  "
            f"grade {res['grade']:4s} diskon {res['discount']:2d}%  "
            f"Rp {res['price']:,}  sisa {res['remaining_hours']:.0f} jam"
        )
        want = expect[kind]
        if res["grade"] != want and not (kind == "memar" and res["grade"] == "C"):
            print(f"           ^ perkiraan {want}, dapat {res['grade']}")
            ok = False

    print("\n" + ("SEMUA OK" if ok else "ADA YANG PERLU DISETEL"))
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
