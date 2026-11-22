"""
Turn a grayscale field into a grid of letters, the same way the News in Letters site does
(error-diffusion dithering over a density-sorted glyph palette), and rasterize that grid at
any physical resolution so it can become a printed relief.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-ExtraBold.ttf"
DROP = set("acemnorsuvwxz")  # same exclusions as the site: round lowercase reads as noise
CELL_ASPECT = 1.5  # cell height / width (the site's 2 x 3 pixel cells)


@dataclass
class Atlas:
    codes: np.ndarray  # (K,) char codes, sorted by density
    dens: np.ndarray  # (K,) ink coverage 0..1
    masks: dict  # (w, h) -> (K, h, w) float masks


@lru_cache(maxsize=None)
def _glyph(code: int, w: int, h: int) -> np.ndarray:
    """Glyph rendered to fill a w x h cell (stretched like the site does), antialiased."""
    ss = 4
    size = 100
    font = ImageFont.truetype(FONT, size)
    adv = font.getlength("M")
    line_h = size * 0.75  # site: 12px font on a 9px line
    im = Image.new("L", (int(adv * ss), int(line_h * ss)), 0)
    d = ImageDraw.Draw(im)
    f2 = ImageFont.truetype(FONT, size * ss)
    d.text((0, 0.8 * line_h * ss), chr(code), font=f2, fill=255, anchor="ls")
    return np.asarray(im.resize((w, h), Image.BOX), np.float32) / 255


def atlas() -> Atlas:
    codes = [c for c in range(32, 127) if chr(c) not in DROP]
    dens = np.array([_glyph(c, 40, 60).mean() for c in codes])
    order = np.argsort(dens, kind="stable")
    return Atlas(np.array(codes)[order], dens[order], {})


def _hash(x: int, y: int) -> float:
    n = (x * 374761393 + y * 668265263) & 0xFFFFFFFF
    n = ((n ^ (n >> 13)) * 1274126177) & 0xFFFFFFFF
    return ((n ^ (n >> 16)) & 0xFFFFFFFF) / 4294967296.0


def to_letters(field: np.ndarray, cols: int, at: Atlas | None = None, rows: int | None = None) -> np.ndarray:
    """field: 2D array in 0..1 (any resolution). Returns (rows, cols) array of palette indices."""
    at = at or atlas()
    h, w = field.shape
    rows = rows or int(round(cols * (h / w) / CELL_ASPECT))
    small = np.asarray(Image.fromarray(field.astype(np.float32)).resize((cols, rows), Image.BOX))
    dmax = at.dens[-1]
    tol = 0.03 * dmax
    out = np.zeros((rows, cols), np.int32)
    err = np.zeros((rows + 1, cols + 2), np.float32)
    for j in range(rows):
        rng = range(cols) if j % 2 == 0 else range(cols - 1, -1, -1)
        dirn = 1 if j % 2 == 0 else -1
        for i in rng:
            v = small[j, i] * dmax + err[j, i + 1]
            near = np.nonzero(np.abs(at.dens - v) <= tol)[0]
            k = near[int(_hash(i + 57, j + 131) * len(near))] if len(near) else int(np.abs(at.dens - v).argmin())
            out[j, i] = k
            e = np.clip(v - at.dens[k], -dmax / 2, dmax / 2)
            err[j, i + 1 + dirn] += e * 0.4375
            err[j + 1, i + 1 - dirn] += e * 0.1875
            err[j + 1, i + 1] += e * 0.3125
            err[j + 1, i + 1 + dirn] += e * 0.0625
    return out


def rasterize(letters: np.ndarray, cell_w: int, at: Atlas) -> np.ndarray:
    """Ink mask (rows*cell_h, cols*cell_w) in 0..1 for a letter grid."""
    cell_h = int(round(cell_w * CELL_ASPECT))
    stack = np.stack([_glyph(int(c), cell_w, cell_h) for c in at.codes])
    tiles = stack[letters]  # (R, C, h, w)
    r, c = letters.shape
    return tiles.transpose(0, 2, 1, 3).reshape(r * cell_h, c * cell_w)


def as_text(letters: np.ndarray, at: Atlas) -> str:
    return "\n".join("".join(chr(at.codes[k]) for k in row) for row in letters)
