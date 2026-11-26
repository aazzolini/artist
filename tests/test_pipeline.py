"""The automated chain: drawing script -> spec -> relief -> QA -> bundle -> store -> order -> print file."""
import json
import shutil
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest
import stripe
from fastapi.testclient import TestClient

pytest.importorskip("trimesh")

from relief import pipeline
from relief.build import build
from relief.spec import AutoSpec, auto_spec, auto_spec_heuristic, from_json, survey
from shop.config import load_settings
from shop.main import create_app

PIECE = "2026-11-16-15"
ADMIN = {"Authorization": "Bearer admin"}


class FakeClaude:
    """Stands in for anthropic.Anthropic(): returns a fixed AutoSpec, or a refusal."""

    def __init__(self, spec=None, refuse=False):
        self.calls = []

        def parse(**kw):
            self.calls.append(kw)
            if refuse:
                return SimpleNamespace(stop_reason="refusal", parsed_output=None, model="x")
            return SimpleNamespace(stop_reason="end_turn", parsed_output=spec, model="claude-opus-5-5")

        self.beta = SimpleNamespace(messages=SimpleNamespace(parse=parse))


@pytest.fixture(scope="module")
def layers():
    return survey(PIECE, 160)


def test_claude_spec_is_validated_clamped_and_buildable(layers):
    base = auto_spec_heuristic(layers)
    raw = {**base, "notes": "test"}
    raw["layers"][0]["depth_mm"] = 40  # out of range: must be clamped
    raw["layers"][0]["color"] = [2, -1, 0.5]
    fake = FakeClaude(AutoSpec.model_validate(raw))
    spec = auto_spec(PIECE, "script", layers, client=fake)
    assert spec["source"].startswith("claude")
    kw = fake.calls[0]
    assert kw["model"] == "claude-opus-5-5" and kw["output_format"] is AutoSpec and kw["fallbacks"] == "default"
    first = next(L for L in spec["layers"] if L["kind"] == "over" and L["index"] == 0)
    assert first["depth_mm"] <= 9.5 and first["color"] == [1.0, 0.0, 0.5]
    r = build(PIECE, width_mm=40, cols=27, spec=from_json(spec))
    assert r.height.max() > 3


def test_spec_falls_back_to_heuristic_on_refusal_or_bad_layers(layers):
    spec = auto_spec(PIECE, "script", layers, client=FakeClaude(refuse=True))
    assert spec["source"].startswith("heuristic (Claude unavailable")
    wrong = AutoSpec.model_validate({**auto_spec_heuristic(layers[:-1]), "notes": ""})  # one layer short
    spec = auto_spec(PIECE, "script", layers, client=FakeClaude(wrong))
    assert spec["source"].startswith("heuristic") and len(spec["layers"]) == len(layers)


def test_pricing():
    facts = {"volume_cm3": 268.4}
    assert pipeline.step_price(facts, {"print_cost_per_cm3": None, "print_cost_base": None})["price_cents"] is None
    p = pipeline.step_price(facts, {"print_cost_per_cm3": 0.5, "print_cost_base": 20, "print_shipping_cost": 15,
                                    "markup": 2, "min_price": 79, "round_to": 5})
    assert p["price_cents"] == 34000  # (20 + 134.2 + 15) * 2 = 338.4 -> rounded up to 340


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    log = pipeline.run(PIECE, out, store=None, token=None, use_claude=False, width_mm=60, cols=40, frame_mm=15)
    assert log["qa"]["passed"], log["qa"]
    return out / PIECE / "bundle.zip"


@pytest.fixture
def store(tmp_path, monkeypatch):
    products = tmp_path / "products"
    shutil.copytree(pipeline.ROOT / "products", products)
    for k, v in {"PRODUCTS_DIR": str(products), "DB_PATH": str(tmp_path / "o.db"), "STRIPE_SECRET_KEY": "sk",
                 "STRIPE_WEBHOOK_SECRET": "wh", "BASE_URL": "http://shop.test", "ADMIN_TOKEN": "admin"}.items():
        monkeypatch.setenv(k, v)
    calls = []

    class Fulfiller:
        def submit(self, order, product, print_url=""):
            calls.append(print_url)
            return "emailed-owner"

    client = TestClient(create_app(load_settings(), fulfiller=Fulfiller()), follow_redirects=False)
    client.fulfiller_calls = calls
    return client


def local(url):
    u = urlsplit(url)
    return u.path + ("?" + u.query if u.query else "")


def test_qa_checks_are_recorded(bundle):
    import zipfile
    m = json.loads(zipfile.ZipFile(bundle).read("manifest.json"))
    assert m["qa"]["passed"] and "two level keyholes" in m["qa"]["checks"]
    assert m["draft"] is True and m["price_cents"] is None  # no print rates configured yet
    assert m["spec_source"].startswith("hand-written")


def test_bundle_to_store_to_order_to_print_file(bundle, store, monkeypatch):
    assert store.post("/admin/pieces", content=bundle.read_bytes()).status_code == 401
    r = store.post("/admin/pieces", content=bundle.read_bytes(), headers=ADMIN)
    assert r.status_code == 200, r.text
    pub = r.json()
    assert pub["draft"] is True

    # Draft: hidden and not buyable, but the owner's preview link works and shows the 3D viewer.
    assert PIECE not in store.get("/").text
    assert store.get(f"/p/{PIECE}").status_code == 404
    page = store.get(local(pub["preview_url"]))
    assert page.status_code == 200 and "model.model.json" in page.text and "Draft preview" in page.text
    model = store.get(f"/assets/{PIECE}/model.model.json")
    assert model.status_code == 200 and "gltf" in model.json()
    assert store.get(f"/assets/{PIECE}/model-0.png").status_code == 200
    assert store.get(f"/assets/{PIECE}/print.3mf").status_code == 404  # print file is never public
    assert store.post(f"/buy/{PIECE}").status_code == 404

    # Owner sets a price and takes it live.
    assert store.post(f"/admin/products/{PIECE}", json={"draft": False}, headers=ADMIN).status_code == 400
    r = store.post(f"/admin/products/{PIECE}", json={"price_cents": 18900, "draft": False}, headers=ADMIN)
    assert r.status_code == 200 and r.json()["draft"] is False
    assert f"/p/{PIECE}" in store.get("/").text and "189.00 USD" in store.get(f"/p/{PIECE}").text

    # Republishing a new build without a price keeps the owner's price and live state.
    r = store.post("/admin/pieces", content=bundle.read_bytes(), headers=ADMIN)
    assert r.json()["draft"] is False and r.json()["price_cents"] == 18900

    # A buyer pays; the print order carries a signed link to the print file.
    monkeypatch.setattr(stripe.checkout.Session, "create",
                        lambda **kw: SimpleNamespace(url="https://checkout.stripe.com/x"))
    assert store.post(f"/buy/{PIECE}").status_code == 303
    session = stripe.checkout.Session.construct_from({
        "id": "cs_p1", "object": "checkout.session", "payment_status": "paid", "amount_total": 19499,
        "currency": "usd", "metadata": {"product_id": PIECE}, "customer_details": {"email": "b@example.com"},
        "collected_information": {"shipping_details": {"name": "Ada", "address": {"line1": "1 Main", "city": "Austin",
                                                                                  "state": "TX", "postal_code": "78701",
                                                                                  "country": "US"}}}}, "sk")
    monkeypatch.setattr(stripe.checkout.Session, "retrieve", lambda sid, **kw: session)
    assert store.get("/success", params={"session_id": "cs_p1"}).status_code == 200
    (print_url,) = store.fulfiller_calls
    f = store.get(local(print_url))
    assert f.status_code == 200 and f.content[:2] == b"PK"  # a 3MF is a zip package
    assert store.get(local(print_url).replace("sig=", "sig=0")).status_code == 403


def test_store_rejects_bad_bundles(store, tmp_path):
    import io
    import zipfile
    assert store.post("/admin/pieces", content=b"not a zip", headers=ADMIN).status_code == 400
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("../evil.py", "x")
    assert store.post("/admin/pieces", content=buf.getvalue(), headers=ADMIN).status_code == 400
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for n in ("print.3mf", "preview.png", "web/model.model.json"):
            z.writestr(n, "x")
        z.writestr("manifest.json", json.dumps({"id": "ok-piece", "qa": {"passed": False}}))
    r = store.post("/admin/pieces", content=buf.getvalue(), headers=ADMIN)
    assert r.status_code == 400 and "QA" in r.text
