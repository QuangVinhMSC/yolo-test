"""Train an OBB model whose third output is a defect class.

Example:
    python tools/make_defect_dataset.py --root datasets/obb-defect
    python train_defect.py --data datasets/obb-defect/data.yaml --epochs 80 --imgsz 320
"""

import argparse

import obbq
from obbq.defect import YOLODefect


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="datasets/obb-defect/data.yaml")
    p.add_argument("--scale", default="n", choices=list("nsmlx"))
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="cpu")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--defect", type=float, default=obbq.DEFAULT_DEFECT_GAIN, help="defect loss gain")
    p.add_argument(
        "--focal-gamma",
        type=float,
        default=0.0,
        dest="defect_focal_gamma",
        help="focal-loss gamma for the defect term; 0 (default) disables focal weighting",
    )
    p.add_argument(
        "--focal-alpha",
        type=float,
        default=0.25,
        dest="defect_focal_alpha",
        help="focal-loss alpha for the defect term's positive class (only used when --focal-gamma > 0)",
    )
    p.add_argument(
        "--class-weights",
        default="",
        dest="defect_class_weights",
        help="comma-separated per-defect-class weights, in defect_names order (e.g. '0.38,3.56,9.53'); "
        "empty (default) disables per-class weighting",
    )
    p.add_argument("--name", default="obb-defect")
    a = p.parse_args()

    obbq.register()
    model = YOLODefect(obbq.model_cfg(a.scale, "defect"))
    model.train(
        data=a.data,
        epochs=a.epochs,
        imgsz=a.imgsz,
        batch=a.batch,
        device=a.device,
        workers=a.workers,
        defect=a.defect,
        defect_focal_gamma=a.defect_focal_gamma,
        defect_focal_alpha=a.defect_focal_alpha,
        defect_class_weights=a.defect_class_weights,
        name=a.name,
    )


if __name__ == "__main__":
    main()
