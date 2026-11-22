"""Small, fast build of a real piece through the whole relief pipeline."""
import zipfile

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from relief.ascii import atlas, rasterize, to_letters
from relief.build import build
from relief.export import save_all


def test_letters_track_brightness():
    at = atlas()
    field = np.tile(np.linspace(0, 1, 200, dtype=np.float32), (150, 1))
    letters = to_letters(field, 40, at)
    ink = rasterize(letters, 10, at)
    left, right = ink[:, : ink.shape[1] // 4].mean(), ink[:, -ink.shape[1] // 4:].mean()
    assert right > left + 0.2


def test_catch_builds_a_watertight_colored_solid(tmp_path):
    r = build("2026-11-16-15", width_mm=40, cols=27)
    assert r.letters.shape == (22, 27)
    assert 3.0 <= r.height.min() and r.height.max() <= 3.0 + 9.0 + 0.6 + 0.01
    info = save_all(r, tmp_path / "catch", target_tris=150_000)
    out = tmp_path / "catch"
    mesh = trimesh.load(out / "catch.3mf", force="mesh")
    assert mesh.is_watertight and mesh.is_winding_consistent and mesh.volume > 0
    assert abs(mesh.bounds[1][0] - info["size_mm"][0]) < 0.2
    with zipfile.ZipFile(out / "catch-obj.zip") as z:
        assert {"catch.obj", "catch.mtl", "catch.png"} <= set(z.namelist())
    for name in info["previews"].values():
        assert (out / name).stat().st_size > 1000


def test_frame_surrounds_picture_and_stands_proud():
    plain = build("2026-11-16-15", width_mm=40, cols=27)
    framed = build("2026-11-16-15", width_mm=40, cols=27, frame_mm=6)
    assert framed.letters.shape[1] > plain.letters.shape[1] and framed.letters.shape[0] > plain.letters.shape[0]
    edge = framed.body[:, : int(6 * framed.px_per_mm)]  # left frame band
    assert edge.max() > plain.body.max() - 1  # the crest rises to about the picture's tallest objects
    assert framed.height.shape[1] / framed.px_per_mm > 40 + 2 * 5
