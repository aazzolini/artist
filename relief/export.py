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


def grid_solid(height: np.ndarray, px_per_mm: float, back: np.ndarray | None = None):
    """Full-resolution watertight solid: one vertex per pixel on the front and on the back surface,
    joined by the outside walls. Returns (vertices mm, faces, width mm, height mm)."""
    H, W = height.shape
    s = 1 / px_per_mm
    yy, xx = np.mgrid[0:H, 0:W]
    top = np.column_stack([xx.ravel() * s, (H - 1 - yy.ravel()) * s, height.ravel()])
    bot = top.copy(); bot[:, 2] = 0 if back is None else back.ravel()
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
    return verts, np.vstack([top_f, bot_f, walls]).astype(np.int64), (W - 1) * s, (H - 1) * s


def _sheet(z: np.ndarray, px_per_mm: float, up: bool):
    """One heightfield surface as a triangle grid; up=True faces +z (front), False faces -z (back)."""
    H, W = z.shape
    s = 1 / px_per_mm
    yy, xx = np.mgrid[0:H, 0:W]
    v = np.column_stack([xx.ravel() * s, (H - 1 - yy.ravel()) * s, z.ravel()]).astype(np.float64)
    idx = np.arange(H * W).reshape(H, W)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    f = np.vstack([np.column_stack([a, c, b]), np.column_stack([b, c, d])])
    return v, (f if up else f[:, [0, 2, 1]]).astype(np.int64)


SIMPLIFIER_LOG: list[str] = []  # which method each simplified sheet used (reported in info.json)


def _simplify_sheet(v, f, target):
    """Quadric-simplify a heightfield sheet. If that ever folds the sheet (an edge shared by more
    than two faces), keep its well-placed vertices but reconnect them with a 2D Delaunay
    triangulation, which for a heightfield is always a clean, fold-free sheet."""
    import fast_simplification
    from scipy.spatial import Delaunay
    for attempt, agg in enumerate((7, 6, 8, 7, 5, 9, 6, 8)):
        reduction = max(0.0, 1 - target * (1 + 0.025 * attempt) / len(f))
        v2, f2 = fast_simplification.simplify(v, f, target_reduction=reduction, agg=agg)
        e = np.sort(np.vstack([f2[:, [0, 1]], f2[:, [1, 2]], f2[:, [2, 0]]]), axis=1)
        _, counts = np.unique(e, axis=0, return_counts=True)
        if counts.max() <= 2:
            SIMPLIFIER_LOG.append(f"quadric (try {attempt + 1})")
            return v2, f2.astype(np.int64)
    SIMPLIFIER_LOG.append("delaunay fallback")
    up = np.cross(v2[f2[:, 1]] - v2[f2[:, 0]], v2[f2[:, 2]] - v2[f2[:, 0]])[:, 2].sum() > 0
    keep = np.unique(f2)
    pts = v2[keep]
    # Make sure the rectangle's corners are vertices, so the sheet covers the full outline.
    x0, x1, y0, y1 = v[:, 0].min(), v[:, 0].max(), v[:, 1].min(), v[:, 1].max()
    corner_ids = [np.argmin((v[:, 0] - cx) ** 2 + (v[:, 1] - cy) ** 2) for cx, cy in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
    pts = np.vstack([pts, v[corner_ids]])
    pts = pts[np.unique(np.round(pts[:, :2], 5), axis=0, return_index=True)[1]]
    tri = Delaunay(pts[:, :2], qhull_options="Qbb Qz Q12").simplices.astype(np.int64)
    a, b, c = pts[tri[:, 0], :2], pts[tri[:, 1], :2], pts[tri[:, 2], :2]
    ccw = ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])) > 0
    tri = np.where(ccw[:, None], tri, tri[:, [0, 2, 1]])  # all counter-clockwise seen from +z
    return pts, (tri if up else tri[:, [0, 2, 1]])


def _perimeter(v, f, wmm, hmm):
    """Boundary vertices of a sheet, snapped onto the rectangle, as (indices, perimeter parameter)."""
    e = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    es = np.sort(e, axis=1)
    _, inv, counts = np.unique(es, axis=0, return_inverse=True, return_counts=True)
    ids = np.unique(e[counts[inv] == 1])
    x, y = v[ids, 0], v[ids, 1]
    side = np.argmin(np.column_stack([np.abs(y), np.abs(x - wmm), np.abs(y - hmm), np.abs(x)]), axis=1)
    v[ids[side == 0], 1] = 0; v[ids[side == 2], 1] = hmm
    v[ids[side == 1], 0] = wmm; v[ids[side == 3], 0] = 0
    x, y = v[ids, 0], v[ids, 1]
    t = np.select([side == 0, side == 1, side == 2], [x, wmm + y, wmm + hmm + (wmm - x)], 2 * wmm + hmm + (hmm - y))
    order = np.argsort(t, kind="stable")
    return ids[order], t[order]


def mesh(r: Relief, target_tris: int = 1_500_000, height: np.ndarray | None = None,
         back: np.ndarray | None = None):
    """Watertight solid: the front and back surfaces are simplified separately (so they can never be
    pinched together where the shell is thin) and stitched along the outside edge.
    back: per-pixel height of the back surface (mm); None = flat back."""
    height = r.height if height is None else height
    H, W = height.shape
    wmm, hmm = (W - 1) / r.px_per_mm, (H - 1) / r.px_per_mm
    back = np.zeros_like(height) if back is None else back
    flat_back = not back.any()
    vf, ff = _simplify_sheet(*_sheet(height, r.px_per_mm, True), target_tris * (0.97 if flat_back else 0.85))
    vb, fb = _simplify_sheet(*_sheet(back, r.px_per_mm, False), target_tris * (0.03 if flat_back else 0.15))
    pf, tf = _perimeter(vf, ff, wmm, hmm)
    pb, tb = _perimeter(vb, fb, wmm, hmm)
    off = len(vf)
    # Zipper the two perimeter loops into the outside wall.
    walls, i, j, nf, nb = [], 0, 0, len(pf), len(pb)
    while i < nf or j < nb:
        fi, fi1 = pf[i % nf], pf[(i + 1) % nf]
        bj, bj1 = pb[j % nb] + off, pb[(j + 1) % nb] + off
        ti = tf[(i + 1) % nf] + (2 * (wmm + hmm) if i + 1 >= nf else 0)
        tj = tb[(j + 1) % nb] + (2 * (wmm + hmm) if j + 1 >= nb else 0)
        if j >= nb or (i < nf and ti <= tj):
            walls.append((fi, bj, fi1)); i += 1
        else:
            walls.append((fi, bj, bj1)); j += 1
    walls = np.array(walls, np.int64)
    v = np.vstack([vf, vb]).astype(np.float32)
    f = np.vstack([ff, fb + off, walls])
    if not _is_closed_manifold(f):
        raise RuntimeError("stitched mesh is not a closed manifold")
    # Make the walls wind the same way as the sheets (outward normals => positive volume).
    tri = v[f].astype(np.float64)
    if np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() < 0:
        f[-len(walls):] = walls[:, [0, 2, 1]]
    return v, f, wmm, hmm


def _is_closed_manifold(faces: np.ndarray) -> bool:
    e = np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    return bool(np.all(counts == 2))


def _uv(v: np.ndarray, wmm: float, hmm: float) -> np.ndarray:
    return np.column_stack([v[:, 0] / wmm, v[:, 1] / hmm])  # planar projection from the front


PLAIN_UV = (0.0005, 0.9995)  # top-left texel: the plain rim color


def solid(r: Relief, target_tris: int, height: np.ndarray | None = None, hang: bool = True):
    """The printable solid: front surface, recessed back and keyhole slots (when the frame is thick
    enough). Returns vertices, faces, width/height in mm, keyholes, and the plain-colored face mask."""
    from . import hang as hg
    holes = hg.keyholes(r.body, r.px_per_mm) if hang else []
    back = hg.back_surface(r.body, r.px_per_mm, holes) if hang else None
    v, f, wmm, hmm = mesh(r, target_tris, height, back)
    v, f = hg.cut_keyholes(v, f, holes)
    if holes and not _is_closed_manifold(f):
        raise RuntimeError("keyhole cut produced a non-manifold mesh")
    return v, f, wmm, hmm, holes, hg.plain_faces(v, f, holes)


def _face_uvs(v, f, wmm, hmm, plain):
    """Per-vertex UVs plus one extra plain UV; per-face UV indices (plain faces use the extra)."""
    uv = np.vstack([_uv(v, wmm, hmm), PLAIN_UV])
    fi = f.copy()
    if plain is not None:
        fi[plain] = len(uv) - 1
    return uv, fi


def write_obj_zip(v, f, wmm, hmm, texture: Image.Image, path: Path, name: str, plain=None):
    uv, fi = _face_uvs(v, f, wmm, hmm, plain)
    buf = io.StringIO()
    buf.write(f"mtllib {name}.mtl\nusemtl paint\n")
    np.savetxt(buf, v, fmt="v %.4f %.4f %.4f")
    np.savetxt(buf, uv, fmt="vt %.6f %.6f")
    ff, ti = f + 1, fi + 1
    np.savetxt(buf, np.column_stack([ff[:, 0], ti[:, 0], ff[:, 1], ti[:, 1], ff[:, 2], ti[:, 2]]), fmt="f %d/%d %d/%d %d/%d")
    png = io.BytesIO(); texture.save(png, "PNG", optimize=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{name}.obj", buf.getvalue())
        z.writestr(f"{name}.mtl", f"newmtl paint\nKa 1 1 1\nKd 1 1 1\nmap_Kd {name}.png\n")
        z.writestr(f"{name}.png", png.getvalue())


def write_3mf(v, f, wmm, hmm, texture: Image.Image, path: Path, plain=None):
    """3MF core + materials extension: one texture mapped by planar UVs, units mm."""
    uv, fi = _face_uvs(v, f, wmm, hmm, plain)
    vs = "".join(f'<vertex x="{x:.4f}" y="{y:.4f}" z="{z:.4f}"/>' for x, y, z in v)
    uvs = "".join(f'<m:tex2coord u="{u:.6f}" v="{w:.6f}"/>' for u, w in uv)
    tris = "".join(f'<triangle v1="{a}" v2="{b}" v3="{c}" pid="2" p1="{p}" p2="{q}" p3="{t}"/>'
                   for (a, b, c), (p, q, t) in zip(f, fi))
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
    v, f, wmm, hmm, holes, plain = solid(r, target_tris, height=r.body)
    # Plain (back) faces need the plain UV. Vertices used only by them just take it; the few
    # shared with front faces (rims, keyhole edges) are duplicated for the back faces.
    plain_only = np.zeros(len(v), bool)
    if plain.any():
        shared = np.intersect1d(np.unique(f[plain]), np.unique(f[~plain]))
        remap = np.arange(len(v))
        remap[shared] = len(v) + np.arange(len(shared))
        v = np.vstack([v, v[shared]])
        f = f.copy()
        f[plain] = remap[f[plain]]
        plain_only = np.zeros(len(v), bool)
        plain_only[np.unique(f[plain])] = True
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
    uv[plain_only] = PLAIN_UV
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
    SIMPLIFIER_LOG.clear()
    v, f, wmm, hmm, holes, plain = solid(r, target_tris)
    info["print_mesh_simplifier"] = {"front": SIMPLIFIER_LOG[0], "back": SIMPLIFIER_LOG[1]}
    write_3mf(v, f, wmm, hmm, texture, out / f"{name}.3mf", plain)
    write_obj_zip(v, f, wmm, hmm, texture, out / f"{name}-obj.zip", name, plain)
    info.update(write_glb(r, out / f"{name}.glb"))
    tri = v[f].astype(np.float64)
    volume = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6
    info.update(size_mm=[round(wmm, 1), round(hmm, 1), round(float(r.height.max()), 1)],
                volume_cm3=round(volume / 1000, 1), triangles=int(len(f)),
                letters=list(r.letters.shape[::-1]),
                keyholes_mm=[[round(h.x, 1), round(h.y, 1)] for h in holes],
                screw_spacing_mm=round(holes[1].x - holes[0].x, 1) if len(holes) == 2 else None)
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
