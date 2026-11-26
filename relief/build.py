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
    height: np.ndarray  # mm, (H, W): body + raised letters
    body: np.ndarray  # mm, (H, W): base plate + object depths, without letters
    color: np.ndarray  # 0..1 RGB, (H, W, 3)
    px_per_mm: float
    letters: np.ndarray
    text: str
    ink: np.ndarray  # 0..1 letter coverage, (H, W)


FRAME_GOLD, FRAME_BRONZE = np.array([0.88, 0.68, 0.32]), np.array([0.36, 0.24, 0.12])


def frame_profile(t: np.ndarray) -> np.ndarray:
    """Molding height (mm above the base plate) across the frame: t = 0 at the picture, 1 at the outside.
    A lip at the sight edge, a rounded crest, a cove falling outward and a bead near the rim."""
    rise = np.clip(t / 0.42, 0, 1)
    h = np.where(t < 0.42, 1.2 + 9.8 * np.sin(rise * np.pi / 2) ** 0.8,
                 11.0 - 4.5 * np.clip((t - 0.42) / 0.58, 0, 1) ** 1.6)
    h = h + 0.9 * np.exp(-((t - 0.86) / 0.035) ** 2) - 0.6 * np.exp(-((t - 0.76) / 0.04) ** 2)
    return np.where(t <= 0, 0.0, h)


def load_spec(piece: str) -> dict:
    """A piece's depth/color spec: hand-written s_<id>.py if present, else s_<id>.json (written by
    the pipeline's auto-spec step), converted to the same form."""
    py, js = HERE / "pieces" / f"s_{piece}.py", HERE / "pieces" / f"s_{piece}.json"
    if py.exists():
        return runpy.run_path(str(py))
    if js.exists():
        from .spec import from_json
        return from_json(json.loads(js.read_text()))
    raise FileNotFoundError(f"no spec for {piece}: write {py.name} or run the pipeline's spec step")


def build(piece: str, width_mm: float = 200, cols: int = 133, px_per_mm: float = 10,
          base_mm: float = 3.0, letter_mm: float = 0.6, frame_mm: float = 0.0,
          spec: dict | None = None) -> Relief:
    """width_mm and cols describe the picture; frame_mm adds a sculpted letter frame around it."""
    spec = spec or load_spec(piece)
    at = atlas()
    cell_w = int(round(width_mm / cols * px_per_mm))
    cell_h = int(round(cell_w * CELL_ASPECT))
    rows = int(round(cols * 1.25 / CELL_ASPECT))  # pictures are 4:5
    cap = capture(piece, cols * cell_w)
    Hp, Wp = rows * cell_h, cols * cell_w
    fit = lambda a: resize(a, Wp, Hp)

    # Picture: per-object depth and tint, composited in the script's own paint order.
    ctx = {"X": fit(cap.X), "Y": fit(cap.Y), "f": fit(cap.f)}
    depth = np.broadcast_to(spec["sky_depth"](ctx), (Hp, Wp)).astype(np.float32).copy()
    tint = np.broadcast_to(np.asarray(spec["sky_color"](ctx), np.float32), (Hp, Wp, 3)).copy()
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
    lum = ctx["f"]

    # Frame: extra letter cells on every side, a molding profile mitered at the corners,
    # and letters dense where the molding faces the light.
    fc = int(round(frame_mm * px_per_mm / cell_w)) if frame_mm else 0
    fr = int(round(frame_mm * px_per_mm / cell_h)) if frame_mm else 0
    C, R = cols + 2 * fc, rows + 2 * fr
    H, W = R * cell_h, C * cell_w
    x0, y0 = fc * cell_w, fr * cell_h
    if fc or fr:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        dx = np.maximum(np.maximum(x0 - xx, xx - (x0 + Wp - 1)), 0) / max(x0, 1)
        dy = np.maximum(np.maximum(y0 - yy, yy - (y0 + Hp - 1)), 0) / max(y0, 1)
        t = np.maximum(dx, dy)  # 0 inside the picture, 1 at the outer edge; equal t = miter line
        fdepth = frame_profile(t).astype(np.float32)
        gy, gx = np.gradient(ndi.gaussian_filter(fdepth, 0.2 * px_per_mm), 1 / px_per_mm)
        n = np.dstack([-gx, gy, np.ones_like(gx)]); n /= np.linalg.norm(n, axis=2, keepdims=True)
        light = np.array([-0.5, 0.5, 0.71]); light /= np.linalg.norm(light)
        flum = np.clip(0.12 + 0.75 * np.clip(n @ light, 0, 1) ** 1.5, 0, 1).astype(np.float32)
        ftint = FRAME_BRONZE + (FRAME_GOLD - FRAME_BRONZE) * flum[..., None]
        D, L, T = fdepth, flum, ftint.astype(np.float32)
        D[y0:y0 + Hp, x0:x0 + Wp] = depth
        L[y0:y0 + Hp, x0:x0 + Wp] = lum
        T[y0:y0 + Hp, x0:x0 + Wp] = tint
        depth, lum, tint = D, L, T

    letters = to_letters(lum, C, at, rows=R)
    ink = rasterize(letters, cell_w, at)

    depth = ndi.gaussian_filter(depth, 0.15 * px_per_mm)  # soften stair-steps between objects
    ink_s = ndi.gaussian_filter(ink, 0.04 * px_per_mm)   # tiny bevel so letters print cleanly
    body = base_mm + depth
    height = body + letter_mm * ink_s

    # Letters carry the picture: bright where it's bright, toward white in highlights.
    ink_col = tint * (0.6 + 0.6 * lum[..., None]) + 0.5 * (lum ** 2)[..., None]
    ground = tint * 0.22 + 0.03
    color = np.clip(ground * (1 - ink[..., None]) + ink_col * ink[..., None], 0, 1)
    # Side walls between depth layers: a planar color projection would smear letters down them,
    # so paint the steep band in the plain ground color instead.
    gy, gx = np.gradient(body, 1 / px_per_mm)
    steep = ndi.binary_dilation(np.hypot(gx, gy) > 1.0, iterations=max(1, int(0.15 * px_per_mm)))
    color[steep] = np.clip(ground[steep], 0, 1)
    rim = max(2, int(0.2 * px_per_mm))  # the outer side walls take the edge pixels' color too
    for sl in (np.s_[:rim, :], np.s_[-rim:, :], np.s_[:, :rim], np.s_[:, -rim:]):
        color[sl] = np.clip(ground[sl], 0, 1)
    text = "\n".join("".join(chr(at.codes[k]) for k in row) for row in letters)
    return Relief(height.astype(np.float32), body.astype(np.float32), color.astype(np.float32),
                  px_per_mm, letters, text, ink.astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("piece")
    ap.add_argument("--width-mm", type=float, default=200)
    ap.add_argument("--cols", type=int, default=133)
    ap.add_argument("--frame-mm", type=float, default=15, help="sculpted letter frame width; 0 for none")
    ap.add_argument("--out", default="out")
    a = ap.parse_args()
    from .export import save_all
    r = build(a.piece, a.width_mm, a.cols, frame_mm=a.frame_mm)
    save_all(r, Path(a.out) / f"{a.piece}-{a.cols}c")


if __name__ == "__main__":
    main()
