"""Receiving pipeline bundles (relief pieces) into the store, and signed links to print files."""

import hashlib
import hmac
import io
import json
import re
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

from .config import Settings

ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{2,63}$")
ALLOWED = re.compile(r"^(manifest\.json|print\.3mf|preview\.png|preview-[a-z0-9]+\.png|"
                     r"web/model\.model\.json|web/model-\d+\.png)$")
MAX_BUNDLE_BYTES = 300 * 1024 * 1024
WEB_ASSET = re.compile(r"^(model\.model\.json|model-\d+\.png)$")


class BundleError(ValueError):
    pass


def ingest(settings: Settings, data: bytes) -> dict:
    """Unpack a bundle into PRODUCTS_DIR/pieces/<id>/ and upsert its catalog entry.
    Returns the catalog entry. The caller reloads the catalog (and restores on failure)."""
    if len(data) > MAX_BUNDLE_BYTES:
        raise BundleError("bundle too large")
    try:
        z = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise BundleError("not a zip file")
    names = [i.filename for i in z.infolist() if not i.is_dir()]
    bad = [n for n in names if not ALLOWED.match(n)]
    if bad:
        raise BundleError(f"unexpected files in bundle: {bad[:5]}")
    for need in ("manifest.json", "print.3mf", "preview.png", "web/model.model.json"):
        if need not in names:
            raise BundleError(f"bundle is missing {need}")
    if sum(i.file_size for i in z.infolist()) > MAX_BUNDLE_BYTES * 2:
        raise BundleError("bundle expands too large")
    m = json.loads(z.read("manifest.json"))
    pid = str(m.get("id", ""))
    if not ID_RE.match(pid):
        raise BundleError(f"bad piece id {pid!r}")
    if not m.get("qa", {}).get("passed"):
        raise BundleError("bundle did not pass QA")

    pieces = settings.products_dir / "pieces"
    pieces.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(dir=pieces, prefix=f".{pid}-"))
    try:
        for n in names:  # names are whitelisted above, so they can't escape tmp
            dest = tmp / n
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(z.read(n))
        final = pieces / pid
        old = pieces / f".{pid}-old"
        shutil.rmtree(old, ignore_errors=True)
        if final.exists():
            final.rename(old)
        tmp.rename(final)
        shutil.rmtree(old, ignore_errors=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    images = settings.products_dir / "images"
    images.mkdir(exist_ok=True)
    shutil.copyfile(pieces / pid / "preview.png", images / f"{pid}.png")

    catalog = json.loads(settings.catalog_path.read_text()) if settings.catalog_path.exists() else []
    existing = next((e for e in catalog if e["id"] == pid), {})
    price = m.get("price_cents")
    entry = {
        "id": pid, "kind": "physical",
        "name": m["name"], "description": m.get("description", ""),
        # A bundle without a price keeps whatever price (and live/draft state) the owner already set.
        "price_cents": price if price is not None else existing.get("price_cents", 0),
        "draft": bool(m.get("draft")) if price is not None else existing.get("draft", True),
        "currency": m.get("currency", "usd"),
        "file": f"pieces/{pid}/print.3mf",
        "image": f"{pid}.png",
        "model": f"pieces/{pid}/web/model.model.json",
        "details": m.get("details", {}),
        "print_notes": m.get("print_notes", ""),
    }
    catalog = [e for e in catalog if e["id"] != pid] + [entry]
    write_catalog(settings, catalog)
    return entry


def write_catalog(settings: Settings, catalog: list) -> None:
    tmp = settings.catalog_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(catalog, indent=1))
    tmp.replace(settings.catalog_path)


def update_product(settings: Settings, pid: str, **fields) -> dict:
    catalog = json.loads(settings.catalog_path.read_text())
    entry = next((e for e in catalog if e["id"] == pid), None)
    if entry is None:
        raise KeyError(pid)
    if "price_cents" in fields and fields["price_cents"] is not None:
        if int(fields["price_cents"]) < 50:
            raise BundleError("price must be at least 50 cents")
        entry["price_cents"] = int(fields["price_cents"])
    if "draft" in fields and fields["draft"] is not None:
        if not fields["draft"] and entry.get("price_cents", 0) < 50:
            raise BundleError("set a price before publishing")
        entry["draft"] = bool(fields["draft"])
    write_catalog(settings, catalog)
    return entry


# ---------------------------------------------------------------- signed links


def _sig(secret: str, *parts: str) -> str:
    return hmac.new(secret.encode(), ":".join(parts).encode(), hashlib.sha256).hexdigest()[:32]


def sign(secret: str, kind: str, key: str, ttl_s: int | None = 14 * 86400) -> str:
    exp = str(int(time.time()) + ttl_s) if ttl_s else "0"
    return f"exp={exp}&sig={_sig(secret, kind, key, exp)}"


def verify(secret: str, kind: str, key: str, exp: str, sig: str) -> bool:
    if not secret or not exp.isdigit():
        return False
    if exp != "0" and int(exp) < time.time():
        return False
    return hmac.compare_digest(sig, _sig(secret, kind, key, exp))
