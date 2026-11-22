"""
From a News in Letters drawing script to a product in the store, end to end.

  python -m relief.pipeline run 2026-11-16-15 --store https://your-store.example
      spec     -> relief/pieces/s_<id>.json (Claude; heuristic fallback) unless a spec exists
      build    -> framed, ready-to-hang relief: print 3MF, web 3D model, preview
      qa       -> watertight + manifold, keyholes, size, letter size, mesh accuracy; any failure stops here
      price    -> from measured volume and pricing.json; no rates set => published as a hidden draft
      bundle   -> out/<id>/bundle.zip (manifest.json + files)
      publish  -> POST bundle.zip to <store>/admin/pieces (ADMIN_TOKEN)

Steps can also be run one at a time: spec | build | publish | changed (lists pieces changed in git).
"""

import argparse
import io
import json
import math
import os
import re
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from .build import HERE as RELIEF_DIR, build, load_spec
from .export import _img, _is_closed_manifold, previews, solid, write_3mf, write_glb

ROOT = RELIEF_DIR.parent
PIECES = RELIEF_DIR / "pieces"
ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{2,63}$")

DEFAULTS = dict(width_mm=200.0, cols=133, frame_mm=15.0)
QA_LIMITS = dict(max_mm=(300.0, 300.0, 40.0), min_letter_mm=1.4, max_error_p95_mm=0.1)
PRINT_NOTES = (
    "Full-color PolyJet (e.g. Vivid/VeroVivid colors). The 3MF is texture-mapped, units mm: "
    "print at 100% scale, do not resize. Satin or matte finish. The back is hollowed and has two "
    "keyhole slots with internal undercuts: please clear all support material from them. "
    "Colors may print slightly darker than on screen; that's expected."
)


# ---------------------------------------------------------------- steps


def step_spec(piece: str, use_claude: bool = True, force: bool = False) -> dict:
    from .spec import auto_spec, survey
    py, js = PIECES / f"s_{piece}.py", PIECES / f"s_{piece}.json"
    if py.exists() and not force:
        return {"source": f"hand-written {py.name}"}
    if js.exists() and not force:
        return {"source": json.loads(js.read_text()).get("source", js.name)}
    script = (PIECES / f"p_{piece}.py").read_text()
    spec = auto_spec(piece, script, survey(piece), use_claude=use_claude)
    js.write_text(json.dumps(spec, indent=1) + "\n")
    return {"source": spec["source"]}


def _fidelity(v, f, height, px_per_mm, window_mm=(20, 15)) -> dict:
    """Rasterize the final mesh's front surface in a window around the busiest part of the picture
    and compare with the designed height map (catches simplification damage to the letters)."""
    H, W = height.shape
    gy, gx = np.gradient(height)
    busy = np.hypot(gx, gy)
    k = 50
    small = busy[: H // k * k, : W // k * k].reshape(H // k, k, W // k, k).mean((1, 3))
    small[:2, :] = small[-2:, :] = 0; small[:, :2] = small[:, -2:] = 0  # stay off the frame edge
    cy, cx = np.unravel_index(np.argmax(small), small.shape)
    cy, cx = cy * k + k // 2, cx * k + k // 2
    hw, hh = int(window_mm[0] * px_per_mm / 2), int(window_mm[1] * px_per_mm / 2)
    x0, x1, y0, y1 = max(cx - hw, 0), min(cx + hw, W), max(cy - hh, 0), min(cy + hh, H)
    orig = height[y0:y1, x0:x1]
    px = v[:, 0] * px_per_mm; py = (H - 1) - v[:, 1] * px_per_mm
    # Every front-facing triangle that overlaps the window (flat areas have triangles far larger than it).
    n_z = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])[:, 2]
    tri = f[(n_z > 0) & (px[f].max(1) >= x0) & (px[f].min(1) <= x1) & (py[f].max(1) >= y0) & (py[f].min(1) <= y1)]
    zb = np.full(orig.shape, -1.0)
    for t in tri:
        X, Y, Z = px[t] - x0, py[t] - y0, v[t, 2]
        d = (Y[1] - Y[2]) * (X[0] - X[2]) + (X[2] - X[1]) * (Y[0] - Y[2])
        if abs(d) < 1e-12:
            continue
        xa, xb = max(int(np.floor(X.min())), 0), min(int(np.ceil(X.max())), zb.shape[1] - 1)
        ya, yb = max(int(np.floor(Y.min())), 0), min(int(np.ceil(Y.max())), zb.shape[0] - 1)
        if xa > xb or ya > yb:
            continue
        gx_, gy_ = np.meshgrid(np.arange(xa, xb + 1), np.arange(ya, yb + 1))
        l1 = ((Y[1] - Y[2]) * (gx_ - X[2]) + (X[2] - X[1]) * (gy_ - Y[2])) / d
        l2 = ((Y[2] - Y[0]) * (gx_ - X[2]) + (X[0] - X[2]) * (gy_ - Y[2])) / d
        l3 = 1 - l1 - l2
        inside = (l1 >= -1e-6) & (l2 >= -1e-6) & (l3 >= -1e-6)
        z = l1 * Z[0] + l2 * Z[1] + l3 * Z[2]
        sel = (gy_[inside], gx_[inside])
        zb[sel] = np.maximum(zb[sel], z[inside])
    covered = zb >= 0
    err = np.abs(zb[covered] - orig[covered])
    return {"median_mm": round(float(np.median(err)), 4), "p95_mm": round(float(np.percentile(err, 95)), 4),
            "covered": round(float(covered.mean()), 3)}


def step_build(piece: str, out: Path, width_mm: float, cols: int, frame_mm: float,
               target_tris: int = 1_500_000) -> dict:
    """Build the relief and write the bundle files. Returns the facts QA and pricing need."""
    r = build(piece, width_mm=width_mm, cols=cols, frame_mm=frame_mm)
    out.mkdir(parents=True, exist_ok=True)
    (out / "web").mkdir(exist_ok=True)
    pv = previews(r, out)
    os.replace(out / pv["room"], out / "preview.png")
    for name in (pv["closeup"], pv["3d"]):
        os.replace(out / name, out / name.replace("preview_", "preview-"))
    v, f, wmm, hmm, holes, plain = solid(r, target_tris)
    write_3mf(v, f, wmm, hmm, _img(r.color), out / "print.3mf", plain)
    write_glb(r, out / "web" / "model.glb")
    for extra in ("model.glb", "model.gltf.json"):  # the store serves the CSP-safe split model only
        (out / "web" / extra).unlink(missing_ok=True)
    tri = v[f].astype(np.float64)
    volume = float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6)
    return {
        "size_mm": [round(wmm, 1), round(hmm, 1), round(float(r.height.max()), 1)],
        "volume_cm3": round(volume / 1000, 1),
        "triangles": int(len(f)),
        "closed_manifold": _is_closed_manifold(f),
        "keyholes_mm": [[round(h.x, 1), round(h.y, 1)] for h in holes],
        "screw_spacing_mm": round(holes[1].x - holes[0].x, 1) if len(holes) == 2 else None,
        "letters": list(r.letters.shape[::-1]),
        "letter_mm": [round(width_mm / cols, 2), round(width_mm / cols * 1.5, 2)],
        "framed": frame_mm > 0,
        "fidelity": _fidelity(v, f, r.height, r.px_per_mm),
    }


def step_qa(facts: dict, print_file: Path) -> dict:
    import trimesh
    m = trimesh.load(print_file, force="mesh")
    checks = {
        "print file is watertight": bool(m.is_watertight),
        "faces wind consistently": bool(m.is_winding_consistent),
        "every edge joins exactly two faces": facts["closed_manifold"],
        "positive volume": m.volume > 0,
        "fits the size limits": all(a <= b for a, b in zip(facts["size_mm"], QA_LIMITS["max_mm"])),
        "letters big enough to read": facts["letter_mm"][0] >= QA_LIMITS["min_letter_mm"],
        "mesh keeps the letters (p95 error)": facts["fidelity"]["p95_mm"] <= QA_LIMITS["max_error_p95_mm"]
                                             and facts["fidelity"]["covered"] > 0.9,
    }
    if facts["framed"]:
        checks["two level keyholes"] = len(facts["keyholes_mm"]) == 2 and \
            facts["keyholes_mm"][0][1] == facts["keyholes_mm"][1][1]
    checks = {k: bool(v) for k, v in checks.items()}
    return {"passed": all(checks.values()), "checks": checks}


def load_pricing() -> dict:
    p = ROOT / "pricing.json"
    return json.loads(p.read_text()) if p.exists() else {}


def step_price(facts: dict, pricing: dict) -> dict:
    """Retail price from volume. Until real print rates are set, there is no price (draft)."""
    rate, base = pricing.get("print_cost_per_cm3"), pricing.get("print_cost_base")
    if rate is None or base is None:
        return {"price_cents": None, "reason": "set print_cost_per_cm3 and print_cost_base in pricing.json"}
    cost = base + rate * facts["volume_cm3"] + pricing.get("print_shipping_cost", 0)
    retail = max(cost * pricing.get("markup", 2.5), pricing.get("min_price", 0))
    step = pricing.get("round_to", 5)
    retail = math.ceil(retail / step) * step
    return {"price_cents": int(retail * 100), "estimated_print_cost": round(cost, 2)}


def record(piece: str) -> dict:
    p = PIECES / f"r_{piece}.json"
    return json.loads(p.read_text()) if p.exists() else {"title": piece}


def step_bundle(piece: str, out: Path, facts: dict, qa: dict, price: dict, spec_info: dict) -> Path:
    rec = record(piece)
    w, h, d = facts["size_mm"]
    size = f"{w / 10:.0f} × {h / 10:.1f} cm"
    hang = (f" Ready to hang: two keyhole slots, {facts['screw_spacing_mm']:.0f} mm apart."
            if facts.get("screw_spacing_mm") else "")
    manifest = {
        "id": piece,
        "name": f"{rec.get('title', piece)} · letter relief",
        "description": f"{rec.get('headline', '').strip()} A full-color sculpted relief made of raised "
                       f"letters, {size}, {d:.0f} mm deep.{hang}".strip(),
        "price_cents": price["price_cents"],
        "currency": load_pricing().get("currency", "usd"),
        "draft": price["price_cents"] is None or not qa["passed"],
        "fidelity": facts["fidelity"],
        "details": {**{k: facts[k] for k in ("size_mm", "volume_cm3", "letters", "letter_mm",
                                               "screw_spacing_mm", "keyholes_mm")}, "framed": facts["framed"]},
        "print_notes": PRINT_NOTES,
        "spec_source": spec_info["source"],
        "qa": qa,
        "pricing": price,
        "files": {"print": "print.3mf", "image": "preview.png", "model": "web/model.model.json"},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    zpath = out.parent / "bundle.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(out).as_posix())
    return zpath


def step_publish(zpath: Path, store: str, token: str) -> dict:
    req = urllib.request.Request(store.rstrip("/") + "/admin/pieces", data=zpath.read_bytes(), method="POST",
                                 headers={"Authorization": f"Bearer {token}", "Content-Type": "application/zip"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())


def changed_pieces(since: str) -> list[str]:
    out = subprocess.run(["git", "diff", "--name-only", since, "HEAD", "--", "relief/pieces"],
                         capture_output=True, text=True, cwd=ROOT, check=True).stdout.split()
    ids = {m.group(2) for p in out if (m := re.match(r"relief/pieces/([psr])_(.+)\.(py|json)$", p))}
    return sorted(i for i in ids if (PIECES / f"p_{i}.py").exists())


def run(piece: str, out_root: Path, store: str | None, token: str | None, use_claude: bool = True,
        **size) -> dict:
    if not ID_RE.match(piece):
        raise ValueError(f"bad piece id {piece!r}")
    size = {**DEFAULTS, **{k: v for k, v in size.items() if v is not None}}
    out = out_root / piece / "bundle"
    log = {"piece": piece}
    log["spec"] = step_spec(piece, use_claude)
    print(f"[{piece}] spec: {log['spec']['source']}", flush=True)
    facts = step_build(piece, out, **size)
    print(f"[{piece}] built: {facts['size_mm']} mm, {facts['volume_cm3']} cm3", flush=True)
    log["qa"] = qa = step_qa(facts, out / "print.3mf")
    for name, ok in qa["checks"].items():
        print(f"[{piece}]   {'PASS' if ok else 'FAIL'}  {name}", flush=True)
    log["price"] = price = step_price(facts, load_pricing())
    zpath = step_bundle(piece, out, facts, qa, price, log["spec"])
    log["bundle"] = str(zpath)
    if not qa["passed"]:
        raise SystemExit(f"[{piece}] QA failed; not publishing. Bundle kept at {zpath}")
    if store:
        log["published"] = step_publish(zpath, store, token or "")
        print(f"[{piece}] published: {log['published']}", flush=True)
    return log


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m relief.pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "spec", "build"):
        p = sub.add_parser(name)
        p.add_argument("pieces", nargs="+")
        p.add_argument("--no-claude", action="store_true", help="use the heuristic spec instead of Claude")
        p.add_argument("--out", default=str(ROOT / "out"))
        p.add_argument("--width-mm", type=float); p.add_argument("--cols", type=int); p.add_argument("--frame-mm", type=float)
        p.add_argument("--store", default=os.environ.get("STORE_URL"))
    p = sub.add_parser("publish"); p.add_argument("pieces", nargs="+"); p.add_argument("--out", default=str(ROOT / "out"))
    p.add_argument("--store", default=os.environ.get("STORE_URL"), required=False)
    p = sub.add_parser("changed"); p.add_argument("--since", required=True)
    a = ap.parse_args(argv)
    token = os.environ.get("ADMIN_TOKEN")
    if a.cmd == "changed":
        print("\n".join(changed_pieces(a.since)))
        return
    for piece in a.pieces:
        if a.cmd == "spec":
            print(piece, step_spec(piece, not a.no_claude, force=True))
        elif a.cmd == "publish":
            print(piece, step_publish(Path(a.out) / piece / "bundle.zip", a.store, token))
        else:
            run(piece, Path(a.out), a.store if a.cmd == "run" else None, token, not a.no_claude,
                width_mm=a.width_mm, cols=a.cols, frame_mm=a.frame_mm)


if __name__ == "__main__":
    main()
