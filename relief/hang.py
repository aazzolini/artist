"""
Ready-to-hang backs: a recessed (hollowed) back like a real frame, and two keyhole slots cut into
the back of the frame's side bands so the piece hangs level on two screws.

Keyhole (sizes in mm, for #6-#8 / 3.5-4 mm pan or round-head screws with heads up to ~8.5 mm):
  - entry hole, 9 dia: the screw head goes in here
  - shank slot, 4.6 wide: cut through a 2.4 mm lip; the plaque slides down so the shank rides up it
  - head channel, 9 wide, 4 mm deep behind the lip: holds the head, so the piece can't fall forward
"""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

SHELL_MM = 2.5  # wall thickness left behind every front surface
RIM_MM = 4.0  # solid band around the outside edge, where the piece sits on the wall
ENTRY_D, SHANK_W, LIP, HEAD_DEPTH, TRAVEL = 9.0, 4.6, 2.4, 4.0, 8.0
MIN_FRONT_OVER_CAVITY = 2.0


@dataclass
class Keyhole:
    x: float  # entry-hole centre, mm from the left edge
    y: float  # entry-hole centre, mm from the bottom edge (the slot runs upward from here)

    @property
    def bbox(self):
        r = ENTRY_D / 2
        return self.x - r, self.y - r, self.x + r, self.y + TRAVEL + r


def keyholes(body: np.ndarray, px_per_mm: float, inset_mm: float = 6.0, below_top_mm: float = 45.0):
    """Two keyholes in the side bands, or none if the piece is too thin there (e.g. no frame)."""
    H, W = body.shape
    wmm, hmm = (W - 1) / px_per_mm, (H - 1) / px_per_mm
    holes = [Keyhole(inset_mm, hmm - below_top_mm), Keyhole(wmm - inset_mm, hmm - below_top_mm)]
    need = LIP + HEAD_DEPTH + MIN_FRONT_OVER_CAVITY
    for h in holes:
        x0, y0, x1, y1 = h.bbox
        c0, c1 = int(x0 * px_per_mm), int(np.ceil(x1 * px_per_mm))
        r0, r1 = int((hmm - y1) * px_per_mm), int(np.ceil((hmm - y0) * px_per_mm))
        if c0 < 0 or c1 >= W or body[max(r0, 0):r1, c0:c1].min() < need:
            return []
    return holes


def back_surface(body: np.ndarray, px_per_mm: float, holes: list[Keyhole]) -> np.ndarray:
    """Height (mm) of the back surface: 0 where the piece is solid to the wall, higher where the
    back is hollowed out, always at least SHELL_MM below the front."""
    k = int(round(SHELL_MM * px_per_mm)) * 2 + 1
    back = ndi.minimum_filter(body, size=k) - SHELL_MM  # square footprint: at least SHELL_MM thick in all directions
    back[back < 1.0] = 0  # don't bother with recesses shallower than 1 mm
    H, W = body.shape
    rim = int(RIM_MM * px_per_mm)
    back[:rim, :] = back[-rim:, :] = back[:, :rim] = back[:, -rim:] = 0
    hmm = (H - 1) / px_per_mm
    margin = 3.0
    for h in holes:  # solid boss around each keyhole
        x0, y0, x1, y1 = h.bbox
        c0, c1 = int((x0 - margin) * px_per_mm), int(np.ceil((x1 + margin) * px_per_mm))
        r0, r1 = int((hmm - y1 - margin) * px_per_mm), int(np.ceil((hmm - y0 + margin) * px_per_mm))
        back[max(r0, 0):r1, max(c0, 0):c1] = 0
    return back.astype(np.float32)


def cutter(h: Keyhole):
    """The keyhole's empty space as one watertight solid (mm, z = 0 at the wall)."""
    import trimesh
    r = ENTRY_D / 2
    z_top = LIP + HEAD_DEPTH

    def box(x0, x1, y0, y1, z0, z1):
        b = trimesh.creation.box(extents=(x1 - x0, y1 - y0, z1 - z0))
        b.apply_translation(((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2))
        return b

    def cyl(x, y, rad, z0, z1):
        c = trimesh.creation.cylinder(radius=rad, height=z1 - z0, sections=48)
        c.apply_translation((x, y, (z0 + z1) / 2))
        return c

    parts = [
        cyl(h.x, h.y, r, -1, z_top),                                       # entry
        box(h.x - r, h.x + r, h.y, h.y + TRAVEL, LIP, z_top),              # head channel
        cyl(h.x, h.y + TRAVEL, r, LIP, z_top),
        box(h.x - SHANK_W / 2, h.x + SHANK_W / 2, h.y, h.y + TRAVEL, -1, LIP + 0.2),  # shank slot
        cyl(h.x, h.y + TRAVEL, SHANK_W / 2, -1, LIP + 0.2),
    ]
    return trimesh.boolean.union(parts, engine="manifold")


def cut_keyholes(v: np.ndarray, f: np.ndarray, holes: list[Keyhole]):
    import trimesh
    if not holes:
        return v, f
    solid = trimesh.Trimesh(vertices=v.astype(np.float64), faces=f, process=False)
    cut = trimesh.boolean.difference([solid] + [cutter(h) for h in holes], engine="manifold")
    return cut.vertices.astype(np.float32), cut.faces.astype(np.int64)


def plain_faces(v: np.ndarray, f: np.ndarray, holes: list[Keyhole]) -> np.ndarray:
    """Faces that should get the plain back color: anything facing the wall, and keyhole insides."""
    tri = v[f]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nz = n[:, 2] / np.maximum(np.linalg.norm(n, axis=1), 1e-12)
    mask = nz < -0.3
    c = tri.mean(1)
    for h in holes:
        x0, y0, x1, y1 = h.bbox
        mask |= (c[:, 0] > x0 - 0.5) & (c[:, 0] < x1 + 0.5) & (c[:, 1] > y0 - 0.5) & (c[:, 1] < y1 + 0.5) \
            & (c[:, 2] < LIP + HEAD_DEPTH + 0.5)
    return mask
