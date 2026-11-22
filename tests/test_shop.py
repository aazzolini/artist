import json
import time
from types import SimpleNamespace

import pytest
import stripe
from fastapi.testclient import TestClient

from shop.config import load_settings
from shop.main import create_app

ADMIN = {"Authorization": "Bearer admin"}
ADDRESS = {"line1": "1 Main St", "line2": None, "city": "Austin", "state": "TX",
           "postal_code": "78701", "country": "US"}


class FakeFulfiller:
    def __init__(self, ref="ref-1"):
        self.ref, self.calls = ref, []

    def submit(self, order, product):
        self.calls.append((order, product))
        return self.ref


@pytest.fixture
def env(tmp_path, monkeypatch):
    (tmp_path / "files").mkdir()
    (tmp_path / "images").mkdir()
    (tmp_path / "files" / "ebook.pdf").write_bytes(b"%PDF-fake")
    (tmp_path / "files" / "dragon.stl").write_text("solid x\nendsolid x\n")
    (tmp_path / "images" / "dragon.png").write_bytes(b"\x89PNG-fake")
    (tmp_path / "catalog.json").write_text(json.dumps([
        {"id": "ebook", "name": "Ebook", "description": "d",
         "price_cents": 1500, "currency": "usd", "file": "files/ebook.pdf"},
        {"id": "dragon", "kind": "physical", "name": "Dragon Figure", "description": "d",
         "price_cents": 3900, "currency": "usd", "file": "files/dragon.stl", "image": "dragon.png"},
    ]))
    for k, v in {
        "PRODUCTS_DIR": str(tmp_path), "DB_PATH": str(tmp_path / "o.db"),
        "STRIPE_SECRET_KEY": "sk_test", "STRIPE_WEBHOOK_SECRET": "whsec",
        "BASE_URL": "http://shop.test", "ADMIN_TOKEN": "admin", "MAX_DOWNLOADS": "2",
        "SHIPPING_CENTS": "599",
    }.items():
        monkeypatch.setenv(k, v)
    return tmp_path


@pytest.fixture
def fulfiller():
    return FakeFulfiller()


@pytest.fixture
def client(env, fulfiller):
    return TestClient(create_app(load_settings(), fulfiller=fulfiller), follow_redirects=False)


def stripe_session(sid="cs_1", status="paid", product="ebook", amount=1500):
    """A real stripe Session object (not a dict), like the API returns."""
    data = {"id": sid, "object": "checkout.session", "payment_status": status,
            "metadata": {"product_id": product}, "amount_total": amount, "currency": "usd",
            "customer_details": {"email": "buyer@example.com"}}
    if product == "dragon":
        data["collected_information"] = {"shipping_details": {"name": "Ada L", "address": ADDRESS}}
    return stripe.checkout.Session.construct_from(data, "sk_test")


def mock_retrieve(monkeypatch, **kw):
    monkeypatch.setattr(stripe.checkout.Session, "retrieve",
                        lambda sid, **_: stripe_session(sid, **kw))


def link_from(html):
    return html.split('href="http://shop.test')[1].split('"')[0]


def test_storefront_lists_products(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Ebook" in r.text and "15.00 USD" in r.text
    assert "Dragon Figure" in r.text and "+ shipping" in r.text and "/images/dragon.png" in r.text
    assert client.get("/images/dragon.png").status_code == 200


def test_product_files_are_not_public(client):
    assert client.get("/images/../files/dragon.stl").status_code == 404
    assert client.get("/files/dragon.stl").status_code == 404


def test_buy_digital_redirects_to_stripe(client, monkeypatch):
    seen = {}
    def fake_create(**kw):
        seen.update(kw)
        return SimpleNamespace(url="https://checkout.stripe.com/x")
    monkeypatch.setattr(stripe.checkout.Session, "create", fake_create)
    r = client.post("/buy/ebook")
    assert r.status_code == 303 and r.headers["location"] == "https://checkout.stripe.com/x"
    assert seen["line_items"][0]["price_data"]["unit_amount"] == 1500
    assert seen["metadata"] == {"product_id": "ebook"}
    assert "shipping_address_collection" not in seen
    assert client.post("/buy/nope").status_code == 404


def test_buy_physical_collects_us_shipping(client, monkeypatch):
    seen = {}
    monkeypatch.setattr(stripe.checkout.Session, "create",
                        lambda **kw: seen.update(kw) or SimpleNamespace(url="https://c/x"))
    client.post("/buy/dragon")
    assert seen["shipping_address_collection"] == {"allowed_countries": ["US"]}
    rate = seen["shipping_options"][0]["shipping_rate_data"]
    assert rate["fixed_amount"] == {"amount": 599, "currency": "usd"}


def test_digital_success_then_download_with_limit(client, monkeypatch):
    mock_retrieve(monkeypatch)
    r = client.get("/success", params={"session_id": "cs_1"})
    assert r.status_code == 200
    link = link_from(r.text)
    assert link.startswith("/download/")
    assert link in client.get("/success", params={"session_id": "cs_1"}).text  # reload: same link
    assert client.get(link).content == b"%PDF-fake"
    assert client.get(link).status_code == 200
    assert client.get(link).status_code == 410  # MAX_DOWNLOADS=2
    assert client.get("/download/bogus").status_code == 404


def test_physical_order_flow(client, monkeypatch, fulfiller):
    mock_retrieve(monkeypatch, product="dragon", amount=4499)
    r = client.get("/success", params={"session_id": "cs_d"})
    assert r.status_code == 200 and "Track your order" in r.text
    link = link_from(r.text)
    assert link.startswith("/order/")
    client.get("/success", params={"session_id": "cs_d"})  # reload must not re-submit
    assert len(fulfiller.calls) == 1
    order, product = fulfiller.calls[0]
    assert product.id == "dragon" and order.ship_name == "Ada L" and order.address["city"] == "Austin"

    page = client.get(link).text
    assert "Dragon Figure" in page and "Austin" in page
    assert client.get(link.replace("/order/", "/download/")).status_code == 404  # no STL for buyers

    data = client.get("/admin/orders", headers=ADMIN).json()
    assert data["gross_revenue_cents"] == {"usd": 4499}  # includes shipping
    assert data["orders"][0]["status"] == "submitted"
    assert data["orders"][0]["fulfillment_ref"] == "ref-1"

    r = client.post("/admin/orders/cs_d/ship", json={"tracking": "1Z999"}, headers=ADMIN)
    assert r.status_code == 200 and r.json()["status"] == "shipped"
    assert "1Z999" in client.get(link).text
    assert client.post("/admin/orders/cs_d/ship", json={}).status_code == 401
    assert client.post("/admin/orders/nope/ship", json={}, headers=ADMIN).status_code == 404


def test_physical_order_waits_when_fulfiller_cannot_send(env, monkeypatch):
    client = TestClient(create_app(load_settings(), fulfiller=FakeFulfiller(ref=None)))
    mock_retrieve(monkeypatch, product="dragon")
    client.get("/success", params={"session_id": "cs_w"})
    data = client.get("/admin/orders", params={"status": "awaiting_fulfillment"}, headers=ADMIN).json()
    assert data["count"] == 1


def test_unpaid_session_is_not_fulfilled(client, monkeypatch, fulfiller):
    mock_retrieve(monkeypatch, status="unpaid", product="dragon")
    assert client.get("/success", params={"session_id": "cs_2"}).status_code == 202
    assert client.get("/admin/orders", headers=ADMIN).json()["count"] == 0
    assert fulfiller.calls == []


def test_webhook_fulfils_and_is_idempotent(client, monkeypatch, fulfiller):
    event = stripe.Event.construct_from({
        "id": "evt_1", "object": "event", "type": "checkout.session.completed",
        "data": {"object": stripe_session("cs_3", product="dragon").to_dict()},
    }, "sk_test")
    monkeypatch.setattr(stripe.Webhook, "construct_event", lambda p, s, k: event)
    for _ in range(2):  # Stripe retries webhooks
        assert client.post("/webhook", content=b"{}", headers={"Stripe-Signature": "x"}).status_code == 200
    assert client.get("/admin/orders", headers=ADMIN).json()["count"] == 1
    assert len(fulfiller.calls) == 1


def test_webhook_rejects_bad_signature(client, monkeypatch):
    def boom(*a):
        raise stripe.SignatureVerificationError("bad", "sig")
    monkeypatch.setattr(stripe.Webhook, "construct_event", boom)
    assert client.post("/webhook", content=b"{}").status_code == 400


def test_expired_link(client, monkeypatch):
    mock_retrieve(monkeypatch)
    link = link_from(client.get("/success", params={"session_id": "cs_4"}).text)
    real = time.time
    monkeypatch.setattr(time, "time", lambda: real() + 73 * 3600)
    assert client.get(link).status_code == 410


def test_admin_requires_token(client):
    assert client.get("/admin/orders").status_code == 401
    assert client.get("/admin/orders", headers={"Authorization": "Bearer wrong"}).status_code == 401


def test_catalog_rejects_path_escape(env):
    (env / "catalog.json").write_text(json.dumps([{
        "id": "x", "name": "x", "price_cents": 100, "file": "../../etc/passwd"}]))
    with pytest.raises(ValueError):
        create_app(load_settings())


def test_repo_sample_catalog_loads(monkeypatch, tmp_path):
    monkeypatch.delenv("PRODUCTS_DIR", raising=False)
    monkeypatch.setenv("DB_PATH", str(tmp_path / "o.db"))
    assert TestClient(create_app(load_settings())).get("/").status_code == 200
