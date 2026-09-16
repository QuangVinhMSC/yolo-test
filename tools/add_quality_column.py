"""Convert plain ultralytics OBB labels into the quality-aware format.

Stock OBB line:    cls x1 y1 x2 y2 x3 y3 x4 y4
Quality-aware line: cls x1 y1 x2 y2 x3 y3 x4 y4 quality

Use this to migrate an existing OBB dataset: every object gets `--default`
as its quality, which you then edit (or fill from your own annotation
source) before training.
"""

import argparse
from pathlib import Path


def convert(path: Path, default: float, overwrite: bool) -> int:
    changed = 0
    for txt in sorted(path.rglob("*.txt")):
        out = []
        for line in txt.read_text().splitlines():
            parts = line.split()
            if not parts:
                continue
            if len(parts) == 9:  # cls + 8 coords -> append quality
                parts.append(f"{default:.6f}")
                changed += 1
            elif len(parts) == 10 and overwrite:
                parts[9] = f"{default:.6f}"
                changed += 1
            out.append(" ".join(parts))
        txt.write_text("\n".join(out) + "\n")
    return changed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("labels", help="labels directory (searched recursively)")
    p.add_argument("--default", type=float, default=1.0, help="quality written for each object")
    p.add_argument("--overwrite", action="store_true", help="also rewrite lines that already have a quality")
    a = p.parse_args()
    n = convert(Path(a.labels), a.default, a.overwrite)
    print(f"updated {n} object(s) under {a.labels}")


if __name__ == "__main__":
    main()
