"""Run a quality-aware YOLO26 OBB model and print all three outputs.

Example:
    python predict_quality.py --weights runs/obb/trial/weights/best.pt \
        --source datasets/obb-quality/images/val
"""

import argparse
from pathlib import Path

import obbq


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", default="runs/obb/trial/weights/best.pt")
    p.add_argument("--source", default="datasets/obb-quality/images/val")
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--device", default="cpu")
    p.add_argument("--limit", type=int, default=5, help="number of images to report")
    a = p.parse_args()

    obbq.register()
    model = obbq.YOLOQuality(a.weights)

    sources = sorted(Path(a.source).glob("*.jpg")) if Path(a.source).is_dir() else [Path(a.source)]
    for src in sources[: a.limit]:
        (result,) = model.predict(str(src), imgsz=a.imgsz, conf=a.conf, device=a.device, verbose=False)
        print(f"\n{src.name}: {len(result.obb)} detection(s)")
        for box, cls, conf, quality in zip(
            result.obb.xywhr.tolist(),
            result.obb.cls.tolist(),
            result.obb.conf.tolist(),
            result.quality.tolist(),
        ):
            x, y, w, h, r = box
            print(
                f"  box=({x:6.1f},{y:6.1f},{w:6.1f},{h:6.1f},{r:+.2f}rad)  "
                f"class={result.names[int(cls)]:<5s} conf={conf:.3f}  quality={quality:.3f}"
            )


if __name__ == "__main__":
    main()
