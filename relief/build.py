"""
Build a printable, full-color haute-relief from a News in Letters drawing script.

1. Re-run the piece's drawing script at print resolution and record every layer it paints
   (each `over(...)` / `screen(...)` call), so every object gets its own depth and color.
2. Turn the picture into letters (relief.ascii) sized to stay legible up close.
3. Height = base plate + per-object depth + raised letters.
   Color  = per-object tint, letters bright and the ground between them dark.

usage: python -m relief.build <piece id> [--width-mm 200] [--cols 133] [--out out/]
"""

import argparse
import importlib
import json
import os
import runpy
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from .ascii import CELL_ASPECT, atlas, rasterize, to_letters

HERE = Path(__file__).parent


@dataclass
class Capture:
    f: np.ndarray  # final grayscale picture 0..1
    overs: list  # masks of each over() call, in paint order
    screens: list  # second operand of each screen() call
    X: np.ndarray
    Y: np.ndarray


def capture(piece: str, width_px: int) -> Capture:
    """Run pieces/p_<piece>.py at width_px and record its layers."""
    os.environ["STUDIO_W"] = str(width_px)
    os.environ.setdefault("STUDIO_SS", "2")
    sys.modules.pop("studio", None)
    sys.path.insert(0, str(HERE))
    try:
        studio = importlib.import_module("studio")
    finally:
        sys.path.remove(str(HERE))
    overs, screens, out = [], [], {}
    real_over, real_screen = studio.over, studio.screen

    def over(base, mask, value):
        overs.append(np.broadcast_to(np.asarray(mask, np.float32), base.shape).copy())
        return real_over(base, mask, value)

    def screen(a, b):
        screens.append(np.broadcast_to(np.asarray(b, np.float32), np.shape(a)).copy())
        return real_screen(a, b)

    def save(f, pid, water_mask=None):
        out["f"] = studio.clamp(np.nan_to_num(f)).astype(np.float32)
        return {}

    studio.over, studio.screen, studio.save = over, screen, save
    old_dump = json.dump
    json.dump = lambda *a, **k: None  # pieces write their own sidecar json; not needed here
    try:
        runpy.run_path(str(HERE / "pieces" / f"p_{piece}.py"), run_name="__main__")
    finally:
        json.dump = old_dump
        sys.modules.pop("studio", None)
    return Capture(out["f"], overs, screens, studio.X, studio.Y)


def resize(a: np.ndarray, w: int, h: int) -> np.ndarray:
    return np.asarray(Image.fromarray(a.astype(np.float32)).resize((w, h), Image.BILINEAR))


@dataclass
class Relief:
    height: np.ndarray  # mm, (H, W)
    color: np.ndarray  # 0..1 RGB, (H, W, 3)
    px_per_mm: float
    letters: np.ndarray
    text: str


def build(piece: str, width_mm: float = 200, cols: int = 133, px_per_mm: float = 10,
          base_mm: float = 3.0, letter_mm: float = 0.6) -> Relief:
    spec = runpy.run_path(str(HERE / "pieces" / f"s_{piece}.py"))
    at = atlas()
    cell_w = int(round(width_mm / cols * px_per_mm))
    cell_h = int(round(cell_w * CELL_ASPECT))
    cap = capture(piece, cols * cell_w)
    letters = to_letters(cap.f, cols, at)
    ink = rasterize(letters, cell_w, at)
    H, W = ink.shape
    fit = lambda a: resize(a, W, H)

    # Per-object depth and tint, composited in the script's own paint order.
    ctx = {"X": fit(cap.X), "Y": fit(cap.Y), "f": fit(cap.f)}
    depth = np.broadcast_to(spec["sky_depth"](ctx), (H, W)).astype(np.float32).copy()
    tint = np.broadcast_to(np.asarray(spec["sky_color"](ctx), np.float32), (H, W, 3)).copy()
    layers = [(m, spec["OVERS"][i]) for i, m in enumerate(cap.overs)] + \
             [(m, spec["SCREENS"][i]) for i, m in enumerate(cap.screens)] + \
             [(fn(ctx), name) for name, fn in spec.get("REGIONS", {}).items()]
    for mask, name in layers:
        if name is None:
            continue
        m = np.clip(fit(mask), 0, 1)
        ctx["mask"] = m
        d, c = spec["DEPTH"].get(name), spec["COLOR"].get(name)
        if d is not None:
            depth = depth * (1 - m) + np.asarray(d(ctx) if callable(d) else d, np.float32) * m
        if c is not None:
            c = np.asarray(c(ctx) if callable(c) else c, np.float32)
            tint = tint * (1 - m[..., None]) + c * m[..., None]

    depth = ndi.gaussian_filter(depth, 0.15 * px_per_mm)  # soften stair-steps between objects
    ink_s = ndi.gaussian_filter(ink, 0.04 * px_per_mm)   # tiny bevel so letters print cleanly
    height = base_mm + depth + letter_mm * ink_s

    # Letters carry the picture: bright where it's bright, toward white in highlights.
    lum = ctx["f"]
    ink_col = tint * (0.6 + 0.6 * lum[..., None]) + 0.5 * (lum ** 2)[..., None]
    ground = tint * 0.22 + 0.03
    color = np.clip(ground * (1 - ink[..., None]) + ink_col * ink[..., None], 0, 1)
    text = "\n".join("".join(chr(at.codes[k]) for k in row) for row in letters)
    return Relief(height.astype(np.float32), color.astype(np.float32), px_per_mm, letters, text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("piece")
    ap.add_argument("--width-mm", type=float, default=200)
    ap.add_argument("--cols", type=int, default=133)
    ap.add_argument("--out", default="out")
    a = ap.parse_args()
    from .export import save_all
    r = build(a.piece, a.width_mm, a.cols)
    save_all(r, Path(a.out) / f"{a.piece}-{a.cols}c")


if __name__ == "__main__":
    main()
