"""
Depth & color specs as data.

A drawing script paints layers with over()/screen(). A spec says, for each layer in paint order,
how far it stands out (mm above the background), its surface shape and its color. Specs come from:
  - Claude (auto_spec_claude): reads the drawing script and a table of the layers it measured,
    returns JSON constrained to the AutoSpec schema. No generated code is ever executed.
  - a heuristic (auto_spec_heuristic): later layers in front, small objects domed, a fixed palette.
    Used when no API key is available or Claude declines.
  - a hand-written s_<id>.py, which always wins (see relief.build.load_spec).

JSON shape (s_<id>.json):
  {"source": "...", "sky_top": [r,g,b], "sky_bottom": [r,g,b],
   "layers": [{"index", "kind": "over"|"screen", "name", "role": "object"|"glow"|"ignore",
               "depth_mm", "shape": "flat"|"dome"|"billow", "relief_mm", "color": [r,g,b]}, ...]}
"""

import json
from typing import Literal

import numpy as np
from pydantic import BaseModel
from scipy import ndimage as ndi

MAX_DEPTH, MAX_RELIEF = 9.0, 4.0
MODEL = "claude-opus-5-5"


class Layer(BaseModel):
    index: int
    kind: Literal["over", "screen"]
    name: str
    role: Literal["object", "glow", "ignore"]
    depth_mm: float
    shape: Literal["flat", "dome", "billow"]
    relief_mm: float
    color: list[float]


class AutoSpec(BaseModel):
    sky_top: list[float]
    sky_bottom: list[float]
    layers: list[Layer]
    notes: str


# ---------------------------------------------------------------- measuring the layers


def survey(piece: str, width_px: int = 200) -> list[dict]:
    """Run the drawing script at low resolution and describe each painted layer."""
    from .build import capture
    cap = capture(piece, width_px)
    rows = []
    for kind, masks in (("over", cap.overs), ("screen", cap.screens)):
        for i, m in enumerate(masks):
            m = np.clip(m, 0, 1)
            on = m > 0.5
            ys, xs = np.nonzero(on)
            rows.append({
                "index": i, "kind": kind,
                "coverage_pct": round(100 * float(m.mean()), 2),
                "bbox_frac": [round(float(xs.min()) / m.shape[1], 2), round(float(ys.min()) / m.shape[0], 2),
                              round(float(xs.max()) / m.shape[1], 2), round(float(ys.max()) / m.shape[0], 2)]
                if len(xs) else None,
                "brightness_under": round(float(cap.f[on].mean()), 2) if on.any() else None,
            })
    return rows


def _check(spec: dict, layers: list[dict]) -> dict:
    """Validate against the measured layers and clamp to printable ranges."""
    want = {(r["kind"], r["index"]) for r in layers}
    got = {(L["kind"], L["index"]) for L in spec["layers"]}
    if want != got:
        raise ValueError(f"spec layers {sorted(got)} don't match the script's layers {sorted(want)}")
    rgb = lambda c: [float(np.clip(x, 0, 1)) for x in (list(c) + [0, 0, 0])[:3]]
    for L in spec["layers"]:
        L["depth_mm"] = float(np.clip(L["depth_mm"], 0, MAX_DEPTH))
        L["relief_mm"] = float(np.clip(L["relief_mm"], 0, MAX_RELIEF))
        if L["depth_mm"] + L["relief_mm"] > MAX_DEPTH + 0.5:
            L["relief_mm"] = MAX_DEPTH + 0.5 - L["depth_mm"]
        L["color"] = rgb(L["color"])
    spec["sky_top"], spec["sky_bottom"] = rgb(spec["sky_top"]), rgb(spec["sky_bottom"])
    spec["layers"].sort(key=lambda L: (L["kind"] != "over", L["index"]))
    return spec


# ---------------------------------------------------------------- sources


PALETTE = [[0.80, 0.82, 0.88], [0.62, 0.42, 0.30], [0.92, 0.66, 0.46], [0.45, 0.55, 0.70],
           [0.85, 0.72, 0.40], [0.55, 0.62, 0.50], [0.70, 0.50, 0.60], [0.60, 0.61, 0.66]]


def auto_spec_heuristic(layers: list[dict]) -> dict:
    overs = [r for r in layers if r["kind"] == "over"]
    out = []
    for k, r in enumerate(overs):
        rank = k / max(1, len(overs) - 1)  # painted later = in front
        small = (r["coverage_pct"] or 0) < 12
        out.append({"index": r["index"], "kind": "over", "name": f"layer {r['index']}", "role": "object",
                    "depth_mm": 1.0 + 6.5 * rank, "shape": "dome" if small else "flat",
                    "relief_mm": 1.5 if small else 0.0, "color": PALETTE[k % len(PALETTE)]})
    for r in layers:
        if r["kind"] == "screen":  # screen() adds light: glows and flames
            out.append({"index": r["index"], "kind": "screen", "name": f"light {r['index']}", "role": "glow",
                        "depth_mm": 0.0, "shape": "flat", "relief_mm": 0.0, "color": [1.0, 0.7, 0.35]})
    spec = {"source": "heuristic", "sky_top": [0.08, 0.11, 0.25], "sky_bottom": [0.20, 0.22, 0.38], "layers": out}
    return _check(spec, layers)


SYSTEM = """You turn grayscale procedural illustrations into full-color sculpted relief plaques.
The illustration is drawn by a Python script that paints layers in order with over() (opaque objects)
and screen() (added light such as flames, glows, beams). The plaque is printed in full color: every
layer gets its own depth in front of the background, a surface shape and a color. The picture itself
is rendered in raised letters on top, so keep shapes simple and readable.

Rules:
- depth_mm: 0-9 mm in front of the background. Things nearer the viewer stand further out. Objects in
  front of others (an arm in front of a ship) must be deeper than what they cover. Background bands
  such as land, sea and distant hills stay shallow (0.3-2 mm).
- shape: "dome" rounds an object outward by relief_mm at its centre (good for hulls, bodies, rocks,
  moons); "billow" makes soft masks swell by up to relief_mm (clouds, smoke, waves); "flat" otherwise.
- role "glow" for screen() light layers that should only tint color; "ignore" for layers that are
  detail marks not worth separate depth. Every layer in the table must appear exactly once.
- color: RGB 0-1, believable for the subject, saturated enough to read when printed (prints come out
  slightly darker than screens). Night scenes: deep blue skies, warm light sources.
- sky_top / sky_bottom: the background gradient.
- name: two or three words naming what the layer is, from the script's comments and code."""


def auto_spec_claude(piece: str, script: str, layers: list[dict], client=None) -> dict:
    import anthropic
    client = client or anthropic.Anthropic()
    prompt = (f"Drawing script for piece {piece}:\n\n```python\n{script}\n```\n\n"
              f"Layers measured when the script ran (index is the order within over() or screen() calls; "
              f"bbox is x0,y0,x1,y1 as fractions of width and height; brightness is 0-1 in the final picture):\n"
              f"{json.dumps(layers, indent=1)}\n\nReturn the relief spec for every layer.")
    resp = client.beta.messages.parse(
        model=MODEL, max_tokens=16000,
        betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        output_config={"effort": "high"},
        system=SYSTEM, messages=[{"role": "user", "content": prompt}],
        output_format=AutoSpec,
    )
    if resp.stop_reason == "refusal" or resp.parsed_output is None:
        raise RuntimeError(f"Claude returned no spec (stop_reason={resp.stop_reason})")
    spec = resp.parsed_output.model_dump()
    spec["source"] = f"claude ({resp.model})"
    return _check(spec, layers)


def auto_spec(piece: str, script: str, layers: list[dict], client=None, use_claude: bool = True) -> dict:
    """Claude when possible, the heuristic otherwise. The result records which was used."""
    if use_claude:
        try:
            return auto_spec_claude(piece, script, layers, client)
        except Exception as e:  # no key, network, refusal or a spec that doesn't fit the layers
            spec = auto_spec_heuristic(layers)
            spec["source"] = f"heuristic (Claude unavailable: {type(e).__name__}: {str(e)[:120]})"
            return spec
    return auto_spec_heuristic(layers)


# ---------------------------------------------------------------- JSON -> builder spec


def _dome(depth, relief):
    def f(c):
        m = c["mask"] > 0.5
        if not m.any():
            return depth
        d = ndi.distance_transform_edt(m)
        return depth + relief * np.sqrt(d / d.max())
    return f


def from_json(spec: dict) -> dict:
    overs, screens, depth, color = [], [], {}, {}
    n_over = 1 + max([L["index"] for L in spec["layers"] if L["kind"] == "over"], default=-1)
    n_screen = 1 + max([L["index"] for L in spec["layers"] if L["kind"] == "screen"], default=-1)
    overs, screens = [None] * n_over, [None] * n_screen
    for L in spec["layers"]:
        key = f"{L['kind']}{L['index']}"
        (overs if L["kind"] == "over" else screens)[L["index"]] = None if L["role"] == "ignore" else key
        if L["role"] == "ignore":
            continue
        color[key] = L["color"]
        if L["role"] == "object":
            d, r = L["depth_mm"], L["relief_mm"]
            depth[key] = {"flat": d, "dome": _dome(d, r), "billow": (lambda d, r: lambda c: d + r * c["mask"])(d, r)}[L["shape"]]
    top, bottom = np.array(spec["sky_top"]), np.array(spec["sky_bottom"])
    return {
        "OVERS": overs, "SCREENS": screens, "DEPTH": depth, "COLOR": color,
        "sky_depth": lambda c: 0.0,
        "sky_color": lambda c: top + (bottom - top) * np.clip(c["Y"] / 1.25, 0, 1)[..., None],
    }
