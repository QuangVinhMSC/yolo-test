"""Interactive GUI to label the defect-class (3rd output) column on top of a
plain OBB detector's boxes.

The detector (`best.pt`, a stock two-output YOLO26 OBB model) finds boxes and
object classes on its own -- this tool's only job is letting a human assign a
defect class per detected box, quickly, and write the result in the
`obbq` defect-variant label format:

    cls x1 y1 x2 y2 x3 y3 x4 y4 defect_class

Every detection defaults to "ok" (defect 0). Click a box to cycle its defect
class; only the exceptions need attention. The frames in `dataset/` are
captured upside down relative to how the model was trained, so each one is
rotated 180 degrees before inference and display -- the saved image and its
label are both in that rotated frame, ready to train on directly.

Usage:
    python tools/label_defect_gui.py --dataset dataset --weights best.pt \
        --out labeled_dataset --defects ok,uhuh,nah
"""

import argparse
import tkinter as tk
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")
BOX_COLORS = ("#2ecc71", "#f1c40f", "#e74c3c")  # ok, uhuh, nah
DELETED_COLOR = "#7f8c8d"
HOVER_WIDTH = 3
NORMAL_WIDTH = 2


@dataclass
class Detection:
    obj_cls: int
    obj_name: str
    conf: float
    poly: np.ndarray  # (4, 2) normalized [0, 1], corners of the rotated image
    defect: int = 0
    deleted: bool = False


@dataclass
class ImageState:
    detections: list = field(default_factory=list)
    dirty: bool = False
    saved: bool = False


def point_in_poly(x, y, poly):
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


class LabelGUI(tk.Tk):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.defect_names = args.defects.split(",")
        self.ignore_classes = {c.strip() for c in args.ignore_classes.split(",") if c.strip()}
        self.title("Defect-class labeling")
        self.geometry("1200x800")

        self.images = sorted(
            p for p in Path(args.dataset).iterdir() if p.suffix.lower() in IMG_EXTS
        )
        if not self.images:
            raise SystemExit(f"no images found in {args.dataset}")

        self.out_images = Path(args.out) / "images"
        self.out_labels = Path(args.out) / "labels"
        self.out_images.mkdir(parents=True, exist_ok=True)
        self.out_labels.mkdir(parents=True, exist_ok=True)

        self.state = {}  # index -> ImageState
        self.rotated_img = None  # current PIL image, rotated
        self.photo = None
        self.scale = 1.0
        self.offset = (0, 0)
        self.hover_idx = None
        self.idx = self._first_unlabeled_index()

        self._build_ui()
        self._load_model()
        self._write_data_yaml()
        self.show_image(self.idx)

    # -- setup ---------------------------------------------------------

    def _first_unlabeled_index(self):
        for i, p in enumerate(self.images):
            if not (self.out_labels / f"{p.stem}.txt").exists():
                return i
        return 0

    def _load_model(self):
        from ultralytics import YOLO

        self.status_var.set("loading model...")
        self.update_idletasks()
        self.model = YOLO(self.args.weights)
        device = self.args.device
        if device is None:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.status_var.set("model loaded")

    def _write_data_yaml(self):
        names = self.model.names if hasattr(self, "model") else {}
        lines = [f"path: {Path(self.args.out).resolve()}", "train: images", "val: images", "names:"]
        for i, n in names.items():
            lines.append(f"  {i}: '{n}'")
        lines.append("defect_names:")
        for i, n in enumerate(self.defect_names):
            lines.append(f"  {i}: {n}")
        (Path(self.args.out) / "data.yaml").write_text("\n".join(lines) + "\n")

    def _build_ui(self):
        root = ttk.Frame(self)
        root.pack(fill=tk.BOTH, expand=True)

        # left: file list
        left = ttk.Frame(root, width=220)
        left.pack(side=tk.LEFT, fill=tk.Y)
        left.pack_propagate(False)
        self.listbox = tk.Listbox(left, exportselection=False)
        self.listbox.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        for p in self.images:
            marker = "✓ " if (self.out_labels / f"{p.stem}.txt").exists() else "  "
            self.listbox.insert(tk.END, marker + p.name)
        self.listbox.bind("<<ListboxSelect>>", self._on_listbox_select)

        # center: canvas
        center = ttk.Frame(root)
        center.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(center, bg="#1e1e1e")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda e: self.render())
        self.canvas.bind("<Button-1>", self._on_left_click)
        self.canvas.bind("<Button-3>", self._on_right_click)
        self.canvas.bind("<Motion>", self._on_motion)

        # bottom: status + controls
        bottom = ttk.Frame(center)
        bottom.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Button(bottom, text="<< Prev", command=self.prev_image).pack(side=tk.LEFT, padx=4, pady=4)
        ttk.Button(bottom, text="Save", command=lambda: self.save_current(explicit=True)).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(bottom, text="Mark all OK", command=self.mark_all_ok).pack(side=tk.LEFT, padx=4)
        ttk.Button(bottom, text="Next >>", command=self.next_image).pack(side=tk.LEFT, padx=4)

        self.status_var = tk.StringVar()
        ttk.Label(bottom, textvariable=self.status_var, anchor="w").pack(
            side=tk.LEFT, padx=12, fill=tk.X, expand=True
        )

        legend = ttk.Label(
            bottom,
            text="left-click box: cycle defect  |  1/2/3: set defect  |  right-click / Del: toggle ignore box"
            "  |  ←/→: prev/next  |  s: save",
        )
        legend.pack(side=tk.RIGHT, padx=8)

        self.bind("<Left>", lambda e: self.prev_image())
        self.bind("<Right>", lambda e: self.next_image())
        self.bind("<Key-s>", lambda e: self.save_current(explicit=True))
        self.bind("<Key-1>", lambda e: self._set_hover_defect(0))
        self.bind("<Key-2>", lambda e: self._set_hover_defect(1))
        self.bind("<Key-3>", lambda e: self._set_hover_defect(2))
        self.bind("<Delete>", lambda e: self._toggle_hover_deleted())
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # -- data ------------------------------------------------------------

    def _detect(self, pil_img):
        res = self.model.predict(
            pil_img, conf=self.args.conf, device=self.device, verbose=False
        )[0]
        dets = []
        if res.obb is not None and len(res.obb):
            polys = res.obb.xyxyxyxyn.reshape(len(res.obb), 4, 2)
            for poly, cls, conf in zip(
                polys.tolist(), res.obb.cls.tolist(), res.obb.conf.tolist()
            ):
                name = self.model.names[int(cls)]
                if name in self.ignore_classes:
                    continue
                dets.append(
                    Detection(
                        obj_cls=int(cls),
                        obj_name=name,
                        conf=float(conf),
                        poly=np.array(poly, dtype=np.float64),
                    )
                )
        return dets

    def _load_existing_labels(self, stem):
        label_path = self.out_labels / f"{stem}.txt"
        if not label_path.exists():
            return None
        dets = []
        for line in label_path.read_text().strip().splitlines():
            parts = line.split()
            cls = int(parts[0])
            name = self.model.names.get(cls, str(cls))
            if name in self.ignore_classes:
                continue
            coords = list(map(float, parts[1:9]))
            defect = int(parts[9])
            poly = np.array(coords, dtype=np.float64).reshape(4, 2)
            dets.append(
                Detection(
                    obj_cls=cls,
                    obj_name=name,
                    conf=-1.0,
                    poly=poly,
                    defect=defect,
                )
            )
        return dets

    def _ensure_state(self, idx, rotated_img):
        if idx in self.state:
            return self.state[idx]
        path = self.images[idx]
        existing = self._load_existing_labels(path.stem)
        if existing is not None:
            st = ImageState(detections=existing, saved=True)
        else:
            st = ImageState(detections=self._detect(rotated_img))
        self.state[idx] = st
        return st

    # -- navigation --------------------------------------------------

    def show_image(self, idx):
        idx = max(0, min(idx, len(self.images) - 1))
        self.idx = idx
        path = self.images[idx]
        self.rotated_img = Image.open(path).convert("RGB").rotate(180)
        self._ensure_state(idx, self.rotated_img)
        self.hover_idx = None
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(idx)
        self.listbox.see(idx)
        self.render()

    def next_image(self):
        self.save_current()
        self.show_image(self.idx + 1)

    def prev_image(self):
        self.save_current()
        self.show_image(self.idx - 1)

    def _on_listbox_select(self, _event):
        sel = self.listbox.curselection()
        if not sel:
            return
        if sel[0] != self.idx:
            self.save_current()
            self.show_image(sel[0])

    # -- editing -------------------------------------------------------

    def _canvas_to_norm(self, cx, cy):
        x = (cx - self.offset[0]) / self.scale / self.rotated_img.width
        y = (cy - self.offset[1]) / self.scale / self.rotated_img.height
        return x, y

    def _hit_test(self, cx, cy):
        x, y = self._canvas_to_norm(cx, cy)
        dets = self.state[self.idx].detections
        for i in range(len(dets) - 1, -1, -1):  # topmost (last drawn) first
            if point_in_poly(x, y, dets[i].poly):
                return i
        return None

    def _on_left_click(self, event):
        i = self._hit_test(event.x, event.y)
        if i is None:
            return
        det = self.state[self.idx].detections[i]
        if det.deleted:
            return
        det.defect = (det.defect + 1) % len(self.defect_names)
        self.state[self.idx].dirty = True
        self.render()

    def _on_right_click(self, event):
        i = self._hit_test(event.x, event.y)
        if i is None:
            return
        det = self.state[self.idx].detections[i]
        det.deleted = not det.deleted
        self.state[self.idx].dirty = True
        self.render()

    def _on_motion(self, event):
        i = self._hit_test(event.x, event.y)
        if i != self.hover_idx:
            self.hover_idx = i
            self.render()

    def _set_hover_defect(self, defect_idx):
        if self.hover_idx is None or defect_idx >= len(self.defect_names):
            return
        det = self.state[self.idx].detections[self.hover_idx]
        if det.deleted:
            return
        det.defect = defect_idx
        self.state[self.idx].dirty = True
        self.render()

    def _toggle_hover_deleted(self):
        if self.hover_idx is None:
            return
        det = self.state[self.idx].detections[self.hover_idx]
        det.deleted = not det.deleted
        self.state[self.idx].dirty = True
        self.render()

    def mark_all_ok(self):
        for det in self.state[self.idx].detections:
            if not det.deleted:
                det.defect = 0
        self.state[self.idx].dirty = True
        self.render()

    # -- rendering -------------------------------------------------------

    def render(self):
        if self.rotated_img is None:
            return
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 2 or ch < 2:
            return  # not mapped yet; the <Configure> handler will re-render
        iw, ih = self.rotated_img.size
        self.scale = min(cw / iw, ch / ih)
        dw, dh = int(iw * self.scale), int(ih * self.scale)
        self.offset = ((cw - dw) // 2, (ch - dh) // 2)

        disp = self.rotated_img.resize((dw, dh), Image.BILINEAR)
        self.photo = ImageTk.PhotoImage(disp)
        self.canvas.delete("all")
        self.canvas.create_image(self.offset[0], self.offset[1], anchor="nw", image=self.photo)

        for i, det in enumerate(self.state[self.idx].detections):
            pts = det.poly * [iw, ih] * self.scale + self.offset
            flat = pts.flatten().tolist()
            if det.deleted:
                color = DELETED_COLOR
                dash = (4, 2)
            else:
                color = BOX_COLORS[det.defect]
                dash = None
            width = HOVER_WIDTH if i == self.hover_idx else NORMAL_WIDTH
            self.canvas.create_polygon(
                flat, outline=color, fill="", width=width, dash=dash
            )
            label = det.obj_name if det.deleted else f"{det.obj_name}/{self.defect_names[det.defect]}"
            tx, ty = pts[0]
            self.canvas.create_text(
                tx, ty - 8, text=label, fill=color, anchor="sw", font=("Segoe UI", 9, "bold")
            )

        self._update_status()

    def _update_status(self):
        path = self.images[self.idx]
        st = self.state[self.idx]
        active = [d for d in st.detections if not d.deleted]
        counts = {n: 0 for n in self.defect_names}
        for d in active:
            counts[self.defect_names[d.defect]] += 1
        counts_str = "  ".join(f"{n}={c}" for n, c in counts.items())
        dirty = " [unsaved]" if st.dirty and not st.saved else (" [saved]" if st.saved else "")
        self.status_var.set(
            f"[{self.idx + 1}/{len(self.images)}] {path.name} -- {len(active)} box(es)  {counts_str}{dirty}"
        )

    # -- saving -------------------------------------------------------

    def save_current(self, explicit=False):
        st = self.state.get(self.idx)
        if st is None:
            return
        if not explicit and st.saved and not st.dirty:
            return
        path = self.images[self.idx]
        active = [d for d in st.detections if not d.deleted]
        if not active:
            if explicit:
                messagebox.showinfo("Nothing to save", "No active detections in this image.")
            return

        self.rotated_img.save(self.out_images / path.name)

        lines = []
        for d in active:
            clipped = np.clip(d.poly, 0.0, 1.0)
            coords = " ".join(f"{v:.6f}" for v in clipped.flatten())
            lines.append(f"{d.obj_cls} {coords} {d.defect}")
        (self.out_labels / f"{path.stem}.txt").write_text("\n".join(lines) + "\n")

        st.dirty = False
        st.saved = True
        marker = "✓ " + path.name
        self.listbox.delete(self.idx)
        self.listbox.insert(self.idx, marker)
        self.listbox.selection_set(self.idx)
        self._update_status()

    def _on_close(self):
        self.save_current()
        self.destroy()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="dataset", help="folder of raw frames to label")
    p.add_argument("--weights", default="best.pt", help="stock OBB detector used to propose boxes")
    p.add_argument("--out", default="labeled_dataset", help="output folder (images/ + labels/ + data.yaml)")
    p.add_argument("--defects", default="ok,uhuh,nah", help="comma-separated defect class names, in index order")
    p.add_argument(
        "--ignore-classes",
        default="line2",
        help="comma-separated object class names to drop from detections (matched by name, not index)",
    )
    p.add_argument("--conf", type=float, default=0.25, help="detector confidence threshold")
    p.add_argument("--device", default=None, help="cuda / cpu; default auto-detects")
    args = p.parse_args()

    app = LabelGUI(args)
    app.mainloop()


if __name__ == "__main__":
    main()
