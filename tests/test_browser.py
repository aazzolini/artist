"""Headless-browser smoke test of the product page's 3D viewer (catches JavaScript errors the
server tests can't). Skipped when Node, Playwright or npm aren't available."""
import json
import shutil
import subprocess
import tarfile
import threading
from pathlib import Path

import pytest
import uvicorn

from shop.config import load_settings
from shop.main import create_app

HERE = Path(__file__).parent


def _node_has_playwright() -> bool:
    if not shutil.which("node"):
        return False
    return subprocess.run(["node", "-e", "require('playwright')"], capture_output=True).returncode == 0


pytestmark = pytest.mark.skipif(not (_node_has_playwright() and shutil.which("npm")), reason="needs node + playwright + npm")


@pytest.fixture(scope="module")
def three_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("three")
    r = subprocess.run(["npm", "pack", "three@0.160.0", "--silent"], cwd=d, capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip("couldn't download three.js from npm")
    with tarfile.open(d / r.stdout.strip().splitlines()[-1]) as t:
        t.extractall(d, filter="data")
    return d / "package"


def test_product_page_viewer_loads_and_sways(three_dir, tmp_path, monkeypatch):
    from relief import pipeline  # build a tiny piece bundle and publish it into a temp store
    out = pipeline.run("2026-11-16-15", tmp_path / "out", store=None, token=None, use_claude=False,
                       width_mm=60, cols=40, frame_mm=15)
    products = tmp_path / "products"
    shutil.copytree(HERE.parent / "products", products)
    for k, v in {"PRODUCTS_DIR": str(products), "DB_PATH": str(tmp_path / "o.db"), "ADMIN_TOKEN": "admin",
                 "BASE_URL": "http://127.0.0.1:8899", "STRIPE_SECRET_KEY": "sk_test_x"}.items():
        monkeypatch.setenv(k, v)
    app = create_app(load_settings())
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=8899, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    try:
        import time, urllib.request
        for _ in range(50):
            if server.started:
                break
            time.sleep(0.1)
        req = urllib.request.Request("http://127.0.0.1:8899/admin/pieces", data=Path(out["bundle"]).read_bytes(),
                                     headers={"Authorization": "Bearer admin", "Content-Type": "application/zip"})
        preview = json.loads(urllib.request.urlopen(req).read())["preview_url"]
        r = subprocess.run(["node", str(HERE / "browser" / "product_page.cjs"), preview, str(three_dir)],
                           capture_output=True, text=True, timeout=240)
        result = json.loads(r.stdout.strip().splitlines()[-1])
    finally:
        server.should_exit = True
    assert result["errors"] == [], result
    assert result["loaded"] and result["moved"], result
