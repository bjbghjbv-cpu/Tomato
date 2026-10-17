"""Latih YOLOv8n untuk 2 kelas: tomato, banana.

Pemakaian:
    python tools/train.py                 # 100 epoch
    python tools/train.py --quick         # 5 epoch, smoke test
    python tools/train.py --epochs 150 --batch 8
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def write_classes_meta(dest: Path) -> None:
    """Catat kelas yang benar-benar ada di data latih.

    Head model bisa punya beberapa kelas walau hanya sebagian yang pernah
    dilatih. Server memakai file ini agar tidak memakai kelas yang kosong.
    """
    import csv
    import json

    manifest = ROOT / "data" / "manifest.csv"
    classes: list[str] = []
    if manifest.exists():
        with manifest.open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                cls = "banana" if row["source"].startswith("coco") else "tomato"
                if cls not in classes:
                    classes.append(cls)
    else:
        classes = ["tomato"]

    (dest / "classes.json").write_text(
        json.dumps({"classes": classes}, indent=2), encoding="utf-8"
    )
    print(f"metadata kelas: {classes} -> {dest / 'classes.json'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "data.yaml"))
    ap.add_argument("--base", default="yolov8n.pt", help="bobot awal (transfer learning dari COCO)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--name", default="tomato_banana")
    args = ap.parse_args()

    try:
        import torch
        from ultralytics import YOLO
    except Exception as exc:
        print(f"dependency belum siap: {exc}")
        return 1

    data = Path(args.data)
    if not data.exists():
        print(f"data.yaml tidak ditemukan: {data}")
        print("jalankan dulu: python tools/prepare_dataset.py --out data")
        return 1

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU  : {name} ({vram:.1f} GB)")
    else:
        print("GPU  : tidak terdeteksi — training akan sangat lambat (CPU)")
        if args.device == "0":
            args.device = "cpu"

    epochs = 5 if args.quick else args.epochs
    print(f"data : {data}")
    print(f"epoch: {epochs}  imgsz: {args.imgsz}  batch: {args.batch}  device: {args.device}")

    model = YOLO(args.base)
    results = model.train(
        data=str(data),
        epochs=epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(ROOT / "runs"),
        name=args.name,
        exist_ok=True,
        patience=25,
        workers=4,
        cache=False,
        plots=True,
        verbose=True,
        # Detektor hanya perlu melokalisasi; warna tetap dijaga agar crop
        # yang masuk ke grader HSV tidak bergeser.
        hsv_h=0.010,
        hsv_s=0.35,
        hsv_v=0.30,
        degrees=8.0,
        fliplr=0.5,
        mosaic=1.0,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    if best.exists():
        dest = ROOT / "models"
        dest.mkdir(exist_ok=True)
        shutil.copy2(best, dest / "best.pt")
        print(f"\nmodel disalin ke: {dest / 'best.pt'}")

        write_classes_meta(dest)

        model = YOLO(str(dest / "best.pt"))
        metrics = model.val(data=str(data), imgsz=args.imgsz, device=args.device)
        try:
            print(f"mAP50   : {metrics.box.map50:.4f}")
            print(f"mAP50-95: {metrics.box.map:.4f}")
        except Exception:
            pass
    else:
        print("best.pt tidak ditemukan — cek log training")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
