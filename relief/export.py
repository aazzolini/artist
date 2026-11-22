"""Export a Relief: colored mesh files for print services, plus preview renders."""

import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from .build import Relief

# ---------------------------------------------------------------- previews


def _shade(height: np.ndarray, color: np.ndarray, px_per_mm: float) -> np.ndarray:
    gy, gx = np.gradient(height, 1 / px_per_mm)
    n = np.dstack([-gx, gy, np.ones_like(gx)])  # image rows go down; mesh y goes up
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    light = np.array([-0.45, 0.55, 0.7]) / np.linalg.norm([-0.45, 0.55, 0.7])  # from upper left
    lam = np.clip(n @ light, 0, 1)
    return np.clip(color * (0.35 + 0.8 * lam)[..., None], 0, 1)


def _img(a: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8))


def oblique(height: np.ndarray, color: np.ndarray, px_per_mm: float, tilt_deg: float = 55,
            out_w: int = 900) -> Image.Image:
    """Painter's-algorithm render of the plaque lying flat, seen from the front at tilt_deg."""
    k = out_w / height.shape[1]
    h = np.asarray(Image.fromarray(height).resize((out_w, int(height.shape[0] * k)), Image.BILINEAR))
    col = np.asarray(_img(_shade(height, color, px_per_mm)).resize((out_w, h.shape[0]), Image.LANCZOS),
                     np.float32) / 255
    t = np.radians(tilt_deg)
    cos, sin = np.cos(t), np.sin(t)
    hp = h * px_per_mm * k  # height in output pixels
    rows = h.shape[0]
    top = int(hp.max() * sin) + 4
    canvas = np.zeros((int(rows * cos) + top + 4, out_w, 3), np.float32) + 0.93
    xs = np.arange(out_w)
    for j in range(rows):  # far edge first; nearer rows paint over
        ys = (top + j * cos - hp[j] * sin).astype(int)
        nxt = hp[j + 1] if j + 1 < rows else np.zeros(out_w)
        wall = np.maximum(0, (hp[j] - nxt) * sin).astype(int) + int(np.ceil(cos)) + 1
        for d in range(int(wall.max())):
            on = d < wall
            shade = 1.0 if d <= cos else 0.55  # front-facing walls are darker
            canvas[np.clip(ys[on] + d, 0, canvas.shape[0] - 1), xs[on]] = col[j, on] * shade
    # front edge of the base plate
    return _img(canvas)


def previews(r: Relief, out: Path) -> dict:
    shaded = _shade(r.height, r.color, r.px_per_mm)
    H, W = r.height.shape
    files = {}
    room = _img(shaded).resize((600, int(600 * H / W)), Image.LANCZOS)
    room.save(out / "preview_room.png"); files["room"] = "preview_room.png"
    # 40 x 30 mm window around the brightest region, at ~0.09 mm/px (eye limit at 30 cm)
    lum = r.color.mean(2)
    cy, cx = np.unravel_index(np.argmax(np.asarray(Image.fromarray(lum).resize((W // 20, H // 20)))), (H // 20, W // 20))
    cy, cx = cy * 20, cx * 20
    hw, hh = int(20 * r.px_per_mm), int(15 * r.px_per_mm)
    cx, cy = np.clip(cx, hw, W - hw), np.clip(cy, hh, H - hh)
    _img(shaded[cy - hh:cy + hh, cx - hw:cx + hw]).resize((440, 330), Image.LANCZOS).save(out / "preview_closeup.png")
    files["closeup"] = "preview_closeup.png"
    oblique(r.height, r.color, r.px_per_mm).save(out / "preview_3d.png"); files["3d"] = "preview_3d.png"
    return files


# ---------------------------------------------------------------- mesh


def mesh(r: Relief, target_tris: int = 1_500_000, height: np.ndarray | None = None):
    """Watertight heightfield solid (top surface, flat bottom, side walls), decimated."""
    import fast_simplification
    height = r.height if height is None else height
    H, W = height.shape
    s = 1 / r.px_per_mm
    yy, xx = np.mgrid[0:H, 0:W]
    top = np.column_stack([xx.ravel() * s, (H - 1 - yy.ravel()) * s, height.ravel()])
    bot = top.copy(); bot[:, 2] = 0
    verts = np.vstack([top, bot]).astype(np.float64)
    idx = np.arange(H * W).reshape(H, W)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    # image row down = -y, so (a, c, b) is counter-clockwise seen from +z
    top_f = np.vstack([np.column_stack([a, c, b]), np.column_stack([b, c, d])])
    n = H * W
    bot_f = top_f[:, [0, 2, 1]] + n
    ring = np.concatenate([idx[0, :], idx[1:, -1], idx[-1, -2::-1], idx[-2:0:-1, 0]])  # clockwise in image = ccw from +z
    r0, r1 = ring, np.roll(ring, -1)
    walls = np.vstack([np.column_stack([r0, r1, r1 + n]), np.column_stack([r0, r1 + n, r0 + n])])
    faces = np.vstack([top_f, bot_f, walls]).astype(np.int64)
    reduction = max(0.0, 1 - target_tris / len(faces))
    v2, f2 = fast_simplification.simplify(verts, faces, target_reduction=reduction, agg=7)
    return v2.astype(np.float32), f2.astype(np.int64), (W - 1) * s, (H - 1) * s


def _uv(v: np.ndarray, wmm: float, hmm: float) -> np.ndarray:
    return np.column_stack([v[:, 0] / wmm, v[:, 1] / hmm])  # planar projection from the front


def write_obj_zip(v, f, wmm, hmm, texture: Image.Image, path: Path, name: str):
    uv = _uv(v, wmm, hmm)
    buf = io.StringIO()
    buf.write(f"mtllib {name}.mtl\nusemtl paint\n")
    np.savetxt(buf, v, fmt="v %.4f %.4f %.4f")
    np.savetxt(buf, uv, fmt="vt %.6f %.6f")
    ff = f + 1
    np.savetxt(buf, np.column_stack([ff[:, 0], ff[:, 0], ff[:, 1], ff[:, 1], ff[:, 2], ff[:, 2]]), fmt="f %d/%d %d/%d %d/%d")
    png = io.BytesIO(); texture.save(png, "PNG", optimize=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{name}.obj", buf.getvalue())
        z.writestr(f"{name}.mtl", f"newmtl paint\nKa 1 1 1\nKd 1 1 1\nmap_Kd {name}.png\n")
        z.writestr(f"{name}.png", png.getvalue())


def write_3mf(v, f, wmm, hmm, texture: Image.Image, path: Path):
    """3MF core + materials extension: one texture mapped by planar UVs, units mm."""
    uv = _uv(v, wmm, hmm)
    vs = "".join(f'<vertex x="{x:.4f}" y="{y:.4f}" z="{z:.4f}"/>' for x, y, z in v)
    uvs = "".join(f'<m:tex2coord u="{u:.6f}" v="{w:.6f}"/>' for u, w in uv)
    tris = "".join(f'<triangle v1="{a}" v2="{b}" v3="{c}" pid="2" p1="{a}" p2="{b}" p3="{c}"/>' for a, b, c in f)
    model = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<model unit="millimeter" xml:lang="en-US" '
        'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" '
        'xmlns:m="http://schemas.microsoft.com/3dmanufacturing/material/2015/02" requiredextensions="m">'
        '<resources>'
        '<m:texture2d id="1" path="/3D/Textures/paint.png" contenttype="image/png"/>'
        f'<m:texture2dgroup id="2" texid="1">{uvs}</m:texture2dgroup>'
        f'<object id="3" type="model"><mesh><vertices>{vs}</vertices><triangles>{tris}</triangles></mesh></object>'
        '</resources><build><item objectid="3"/></build></model>'
    )
    png = io.BytesIO(); texture.save(png, "PNG", optimize=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml",
                   '<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                   '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                   '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>'
                   '<Default Extension="png" ContentType="image/png"/></Types>')
        z.writestr("_rels/.rels",
                   '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                   '<Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>')
        z.writestr("3D/_rels/3dmodel.model.rels",
                   '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                   '<Relationship Target="/3D/Textures/paint.png" Id="tex0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dtexture"/></Relationships>')
        z.writestr("3D/3dmodel.model", model)
        z.writestr("3D/Textures/paint.png", png.getvalue())


def write_glb(r: Relief, path: Path, target_tris: int = 200_000, letter_mm: float = 0.6) -> dict:
    """Web viewing model (glTF binary, meters): the object depths as geometry, the raised
    letters as a full-resolution normal map, plus color and a gloss map (letters satin, ground matte)."""
    import trimesh
    from scipy import ndimage as ndi
    v, f, wmm, hmm = mesh(r, target_tris, height=r.body)
    # Letter relief -> tangent-space normal map (tangent = +x, bitangent = +y = image up).
    detail = letter_mm * ndi.gaussian_filter(r.ink, 0.04 * r.px_per_mm)
    d_row, d_col = np.gradient(detail, 1 / r.px_per_mm)
    n = np.dstack([-d_col, d_row, np.ones_like(detail)])
    # Steep side walls get planar-projected UVs; keep them smooth instead of smeared letters.
    by, bx = np.gradient(r.body, 1 / r.px_per_mm)
    steep = ndi.binary_dilation(np.hypot(bx, by) > 1.0, iterations=max(1, int(0.15 * r.px_per_mm)))
    n[steep] = (0.0, 0.0, 1.0)
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    normal = _img(n * 0.5 + 0.5)
    rough = 0.78 - 0.4 * r.ink  # glTF: G = roughness, B = metalness (0)
    mr = _img(np.dstack([np.ones_like(rough), rough, np.zeros_like(rough)]))
    uv = np.column_stack([v[:, 0] / wmm, v[:, 1] / hmm])  # trimesh flips v for glTF
    mat = trimesh.visual.material.PBRMaterial(
        name="print", baseColorTexture=_img(r.color), normalTexture=normal,
        metallicRoughnessTexture=mr, metallicFactor=0.0, roughnessFactor=1.0)
    m = trimesh.Trimesh(vertices=v * 0.001, faces=f, process=False,
                        visual=trimesh.visual.TextureVisuals(uv=uv, material=mat))
    # glTF is Y-up; our plaque faces +z with +y up already.
    path.write_bytes(trimesh.exchange.gltf.export_glb(m))
    # Same model as one self-contained .gltf JSON file, for hosts that only serve web file types.
    json_path = path.with_suffix(".gltf.json")
    json_path.write_bytes(trimesh.exchange.gltf.export_gltf(m, embed_buffers=True)["model.gltf"])
    web = write_web_model(path, path.parent, path.stem)
    return {"glb_triangles": int(len(f)), "glb_mb": round(path.stat().st_size / 1e6, 1),
            "gltf_json_mb": round(json_path.stat().st_size / 1e6, 1), "web_files": web}


def save_all(r: Relief, out: Path, target_tris: int = 1_500_000) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    name = out.name
    info = {"previews": previews(r, out)}
    texture = _img(r.color)
    texture.save(out / "texture.png", optimize=True)
    (out / "letters.txt").write_text(r.text + "\n")
    v, f, wmm, hmm = mesh(r, target_tris)
    write_3mf(v, f, wmm, hmm, texture, out / f"{name}.3mf")
    write_obj_zip(v, f, wmm, hmm, texture, out / f"{name}-obj.zip", name)
    info.update(write_glb(r, out / f"{name}.glb"))
    info.update(size_mm=[round(wmm, 1), round(hmm, 1), round(float(r.height.max()), 1)],
                triangles=int(len(f)), letters=list(r.letters.shape[::-1]))
    (out / "info.json").write_text(json.dumps(info, indent=1))
    return info


def write_web_model(glb: Path, out_dir: Path, name: str) -> dict:
    """Split a GLB into files any static host serves under a strict content policy:
    <name>.model.json  = {"gltf": glTF JSON, "bin": base64 geometry} (the page rebuilds a GLB in memory)
    <name>-<i>.png      = textures, referenced by relative URL.
    No data: or blob: URLs are fetched, which strict CSPs (like claude.ai artifacts) block."""
    import base64
    import struct
    raw = glb.read_bytes()
    jlen = struct.unpack_from("<I", raw, 12)[0]
    gj = json.loads(raw[20:20 + jlen])
    blen = struct.unpack_from("<I", raw, 20 + jlen)[0]
    bin_ = raw[28 + jlen:28 + jlen + blen]
    views = gj["bufferViews"]
    image_views = {img["bufferView"] for img in gj.get("images", [])}
    files = {}
    for i, img in enumerate(gj.get("images", [])):
        bv = views[img["bufferView"]]
        fname = f"{name}-{i}.png"
        (out_dir / fname).write_bytes(bin_[bv.get("byteOffset", 0):bv.get("byteOffset", 0) + bv["byteLength"]])
        gj["images"][i] = {"uri": fname, "mimeType": img.get("mimeType", "image/png")}
        files[fname] = (out_dir / fname).stat().st_size
    new_bin, remap, new_views = bytearray(), {}, []
    for i, bv in enumerate(views):
        if i in image_views:
            continue
        while len(new_bin) % 4:
            new_bin.append(0)
        chunk = bin_[bv.get("byteOffset", 0):bv.get("byteOffset", 0) + bv["byteLength"]]
        remap[i] = len(new_views)
        new_views.append({**bv, "buffer": 0, "byteOffset": len(new_bin)})
        new_bin += chunk
    for acc in gj.get("accessors", []):
        if "bufferView" in acc:
            acc["bufferView"] = remap[acc["bufferView"]]
    gj["bufferViews"] = new_views
    gj["buffers"] = [{"byteLength": len(new_bin)}]
    model = out_dir / f"{name}.model.json"
    model.write_text(json.dumps({"gltf": gj, "bin": base64.b64encode(bytes(new_bin)).decode()}))
    files[model.name] = model.stat().st_size
    return files
