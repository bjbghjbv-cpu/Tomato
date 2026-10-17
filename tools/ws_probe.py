"""Ambil satu pesan dari WebSocket dan cetak isinya (debug cepat).

    python tools/ws_probe.py [--url ws://127.0.0.1:8000/ws]
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
from pathlib import Path

import websockets

KEYS = [
    "ok",
    "detector",
    "locked",
    "condition_pct",
    "maturity_pct",
    "defect_pct",
    "grade",
    "color",
    "discount",
    "price",
    "remaining_hours",
    "confidence",
    "signals",
    "signal_contrib",
    "next_band",
    "grade_times",
]


async def main(url: str, save: str | None) -> None:
    async with websockets.connect(url) as sock:
        raw = await asyncio.wait_for(sock.recv(), timeout=15)
    msg = json.loads(raw)
    state = msg.get("state", {})
    print(f"frame: {'ada' if msg.get('image') else 'kosong'}  ukuran={msg.get('frame')}")
    if save and msg.get("image"):
        Path(save).write_bytes(base64.b64decode(msg["image"]))
        print(f"  frame disimpan: {save}")
    for k in KEYS:
        print(f"  {k:16s}: {state.get(k)}")
    if state.get("bbox"):
        print(f"  bbox            : {state['bbox']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8000/ws")
    ap.add_argument("--save", default=None, help="simpan frame terakhir ke file")
    args = ap.parse_args()
    asyncio.run(main(args.url, args.save))
