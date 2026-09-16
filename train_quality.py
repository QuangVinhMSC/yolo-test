"""Train a quality-aware YOLO26 OBB model.

Example:
    python tools/make_dataset.py --root datasets/obb-quality
    python train_quality.py --data datasets/obb-quality/data.yaml --epochs 40 --imgsz 320
"""

import argparse

import obbq


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="datasets/obb-quality/data.yaml")
    p.add_argument("--scale", default="n", choices=list("nsmlx"))
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="cpu")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--quality", type=float, default=obbq.DEFAULT_QUALITY_GAIN, help="quality loss gain")
    p.add_argument("--name", default="obb-quality")
    a = p.parse_args()

    obbq.register()
    model = obbq.YOLOQuality(obbq.model_cfg(a.scale))
    model.train(
        data=a.data,
        epochs=a.epochs,
        imgsz=a.imgsz,
        batch=a.batch,
        device=a.device,
        workers=a.workers,
        quality=a.quality,
        name=a.name,
    )


if __name__ == "__main__":
    main()
