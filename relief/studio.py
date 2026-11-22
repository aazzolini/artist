"""Drawing toolkit for the hourly pictures.
A picture is a float array f[H, W] in 0..1 (0 = empty ground, 1 = densest letters).
Coordinates: x in 0..1 across, y in 0..1.25 down (square units). Sizes are in the same units.
"""
import base64, io, json, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

# STUDIO_W lets the relief builder re-render a piece at print resolution (default = the site's 640 x 800).
W = int(os.environ.get("STUDIO_W", 640)); H = W * 5 // 4; SS = int(os.environ.get("STUDIO_SS", 3))
A = H / W
ROOT = os.path.dirname(os.path.abspath(__file__))
X, Y = np.meshgrid((np.arange(W) + 0.5) / W, (np.arange(H) + 0.5) / W)
X = X.astype(np.float32); Y = Y.astype(np.float32)

def zeros(): return np.zeros((H, W), np.float32)
def clamp(a, lo=0.0, hi=1.0): return np.clip(a, lo, hi)
def sm(a, b, t):
    t = np.clip((t - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)
def blur(a, r):
    """Gaussian blur, r in the picture's units (0.01 = 6.4 px)."""
    return ndi.gaussian_filter(a.astype(np.float32), r * W)
def over(base, mask, value): return base * (1 - mask) + value * mask
def screen(a, b): return 1 - (1 - a) * (1 - b)

def _hash(ix, iy, seed):
    n = (ix.astype(np.int64) * 374761393 + iy.astype(np.int64) * 668265263 + seed * 1442695041) & 0xFFFFFFFF
    n = ((n ^ (n >> 13)) * 1274126177) & 0xFFFFFFFF
    return ((n ^ (n >> 16)) & 0xFFFFFFFF) / 4294967296.0
def vnoise(x, y, seed=0):
    xi = np.floor(x); yi = np.floor(y); fx = x - xi; fy = y - yi
    u = fx * fx * (3 - 2 * fx); v = fy * fy * (3 - 2 * fy)
    a = _hash(xi, yi, seed); b = _hash(xi + 1, yi, seed); c = _hash(xi, yi + 1, seed); d = _hash(xi + 1, yi + 1, seed)
    return (a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v).astype(np.float32)
def fbm(x, y, octaves=5, seed=0, lac=2.03, gain=0.5):
    """Fractal noise in 0..1. x, y are already multiplied by the base frequency."""
    tot = zeros() if np.ndim(x) else 0.0; amp = 1.0; norm = 0.0
    for o in range(octaves):
        tot = tot + amp * vnoise(x * lac ** o + 17.3 * o, y * lac ** o + 9.1 * o, seed + o)
        norm += amp; amp *= gain
    return tot / norm

class _Pen:
    def __init__(self, d): self.d = d; self.k = W * SS
    def _p(self, pts): return [(x * self.k, y * self.k) for x, y in pts]
    def poly(self, pts, v=255): self.d.polygon(self._p(pts), fill=v)
    def line(self, pts, w, v=255): self.d.line(self._p(pts), fill=v, width=max(1, int(round(w * self.k))), joint="curve")
    def ellipse(self, cx, cy, rx, ry=None, v=255):
        ry = rx if ry is None else ry
        self.d.ellipse([(cx - rx) * self.k, (cy - ry) * self.k, (cx + rx) * self.k, (cy + ry) * self.k], fill=v)
    def rect(self, x0, y0, x1, y1, v=255): self.d.rectangle([x0 * self.k, y0 * self.k, x1 * self.k, y1 * self.k], fill=v)
    def text(self, x, y, s, size, v=255, bold=True, anchor="mm"):
        f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf" % ("-Bold" if bold else ""), int(size * self.k))
        self.d.text((x * self.k, y * self.k), s, font=f, fill=v, anchor=anchor)
def shape(fn):
    """Antialiased mask (0..1) drawn by fn(pen); pen has poly, line, ellipse, rect, text. v=0 erases."""
    im = Image.new("L", (W * SS, H * SS), 0)
    fn(_Pen(ImageDraw.Draw(im)))
    return np.asarray(im.resize((W, H), Image.BOX), np.float32) / 255
def poly(pts): return shape(lambda p: p.poly(pts))
def disc(cx, cy, r, ry=None): return shape(lambda p: p.ellipse(cx, cy, r, ry))
def rect(x0, y0, x1, y1): return shape(lambda p: p.rect(x0, y0, x1, y1))
def line(pts, w): return shape(lambda p: p.line(pts, w))

def stars(n=700, seed=1, ymax=1.0, big=8):
    """Star field: n small gaussian dots plus a few with cross flares."""
    rng = np.random.default_rng(seed); f = zeros()
    xs = rng.random(n); ys = rng.random(n) * ymax; m = rng.random(n) ** 4
    for x, y, k in zip(xs, ys, m):
        r = 0.0012 + 0.0019 * k; x0 = int(x * W); y0 = int(y * W); s = int(r * W * 3) + 2
        ya, yb, xa, xb = max(0, y0 - s), min(H, y0 + s + 1), max(0, x0 - s), min(W, x0 + s + 1)
        d2 = ((X[ya:yb, xa:xb] - x) ** 2 + (Y[ya:yb, xa:xb] - y) ** 2) / (r * r)
        f[ya:yb, xa:xb] += (0.3 + 0.7 * k) * np.exp(-d2)
    for _ in range(big):
        x, y, r = 0.03 + 0.94 * rng.random(), 0.03 + (ymax - 0.06) * rng.random(), 0.0022 + 0.0016 * rng.random()
        ax, ay = np.abs(X - x), np.abs(Y - y)
        f += np.exp(-(ax * ax + ay * ay) / (r * r)) + 0.55 * np.exp(-ax / (r * 3.2) - (ay / (r * 0.45)) ** 2) + 0.55 * np.exp(-ay / (r * 3.2) - (ax / (r * 0.45)) ** 2)
    return clamp(f)

def water(sky, yh, sheen=0.75, rough=1.0, seed=3, stretch=1.15):
    """Reflection of everything above the horizon yh into the rows below it, broken into ripples.
    stretch > 1 squeezes more of the sky into the water. Returns a full-size array; use np.where(Y >= yh, water(...), scene)."""
    rows = (np.arange(H) + 0.5) / W
    d = np.clip((rows - yh) / (A - yh), 0, 1)
    src = np.clip(((yh - (rows - yh) * stretch) * W).astype(int), 0, H - 1)
    out = sky[src, :]
    D = np.clip((Y - yh) / (A - yh), 0, 1); z = 1 / (D + 0.05)
    n1 = vnoise(X * np.minimum(z * 9, 110) + 5, z * 22, seed) - 0.5
    n2 = vnoise(X * np.minimum(z * 23, 260) + 9, z * 51 + 3, seed + 1) - 0.5
    rate = z * z * (1 / W) / (A - yh)
    rip = 0.5 + 0.65 * sm(2.6, 0.7, rate * 22) * n1 + 0.35 * sm(2.6, 0.7, rate * 51) * n2
    shift = ((rip - 0.5) * 0.03 * rough * (0.3 + D) * W).astype(int)
    cols = np.clip(np.arange(W)[None, :] + shift, 0, W - 1)
    out = np.take_along_axis(out, cols, axis=1)
    out = blur(out, 0.004) * (0.35 + 0.65 * sm(0.36, 0.64, rip)) * sheen * (1 - 0.25 * D)
    return out.astype(np.float32)

def save(f, pid, water_mask=None):
    """Write the picture, its thumbnail and a preview. Returns the fields to merge into the record."""
    f = clamp(np.nan_to_num(f)).astype(np.float32)
    im = Image.fromarray((f * 255 + 0.5).astype(np.uint8), "L")
    im.save(os.path.join(ROOT, "site", "img", pid + ".jpg"), quality=90, optimize=True)
    im.resize((176, 220), Image.LANCZOS).save(os.path.join(ROOT, "site", "thumb", pid + ".jpg"), quality=78, optimize=True)
    im.save(os.path.join(ROOT, "preview", pid + ".png"))
    out = {"img": "img/%s.jpg" % pid, "thumb": "thumb/%s.jpg" % pid}
    if water_mask is not None:
        m = Image.fromarray((np.asarray(water_mask) > 0.5).astype(np.uint8) * 255, "L").resize((80, 100), Image.BOX)
        b = io.BytesIO(); m.point(lambda v: 255 if v > 127 else 0).convert("1").save(b, "PNG", optimize=True)
        out["water"] = "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()
    kb = os.path.getsize(os.path.join(ROOT, "site", "img", pid + ".jpg")) / 1024
    print("saved %s  %.0f KB  mean %.3f  max %.2f" % (pid, kb, f.mean(), f.max()))
    return out
