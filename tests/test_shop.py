import json
import time
from types import SimpleNamespace

import pytest
import stripe
from fastapi.testclient import TestClient

from shop.config import load_settings
from shop.main import create_app


@pytest.fixture
def env(tmp_path, monkeypatch):
    (tmp_path / "files").mkdir()
    (tmp_path / "files" / "ebook.pdf").write_bytes(b"%PDF-fake")
    (tmp_path / "catalog.json").write_text(json.dumps([{
        "id": "ebook", "name": "Ebook", "description": "d",
        "price_cents": 1500, "currency": "usd", "file": "files/ebook.pdf",
    }]))
    for k, v in {
        "PRODUCTS_DIR": str(tmp_path), "DB_PATH": str(tmp_path / "o.db"),
        "STRIPE_SECRET_KEY": "sk_test", "STRIPE_WEBHOOK_SECRET": "whsec",
        "BASE_URL": "http://shop.test", "ADMIN_TOKEN": "admin", "MAX_DOWNLOADS": "2",
    }.items():
        monkeypatch.setenv(k, v)
    return tmp_path


@pytest.fixture
def client(env):
    return TestClient(create_app(load_settings()), follow_redirects=False)


def paid_session(sid="cs_1", status="paid", product="ebook"):
    return {"id": sid, "payment_status": status, "metadata": {"product_id": product},
            "customer_details": {"email": "buyer@example.com"}}


def test_storefront_lists_products(client):
    r = client.get("/")
    assert r.status_code == 200 and "Ebook" in r.text and "15.00 USD" in r.text


def test_buy_redirects_to_stripe(client, monkeypatch):
    seen = {}
    def fake_create(**kw):
        seen.update(kw)
        return SimpleNamespace(url="https://checkout.stripe.com/x")
    monkeypatch.setattr(stripe.checkout.Session, "create", fake_create)
    r = client.post("/buy/ebook")
    assert r.status_code == 303 and r.headers["location"] == "https://checkout.stripe.com/x"
    assert seen["line_items"][0]["price_data"]["unit_amount"] == 1500
    assert seen["metadata"] == {"product_id": "ebook"}
    assert client.post("/buy/nope").status_code == 404


def test_success_then_download_with_limit(client, monkeypatch):
    monkeypatch.setattr(stripe.checkout.Session, "retrieve", lambda sid, **kw: paid_session(sid))
    r = client.get("/success", params={"session_id": "cs_1"})
    assert r.status_code == 200
    link = r.text.split('href="http://shop.test')[1].split('"')[0]
    # Reloading the success page reuses the same order/link.
    assert link in client.get("/success", params={"session_id": "cs_1"}).text
    assert client.get(link).content == b"%PDF-fake"
    assert client.get(link).status_code == 200
    assert client.get(link).status_code == 410  # MAX_DOWNLOADS=2
    assert client.get("/download/bogus").status_code == 404


def test_unpaid_session_is_not_fulfilled(client, monkeypatch):
    monkeypatch.setattr(stripe.checkout.Session, "retrieve",
                        lambda sid, **kw: paid_session(sid, status="unpaid"))
    assert client.get("/success", params={"session_id": "cs_2"}).status_code == 202
    r = client.get("/admin/orders", headers={"Authorization": "Bearer admin"})
    assert r.json()["count"] == 0


def test_webhook_fulfils_and_is_idempotent(client, monkeypatch):
    event = {"type": "checkout.session.completed", "data": {"object": paid_session("cs_3")}}
    monkeypatch.setattr(stripe.Webhook, "construct_event", lambda p, s, k: event)
    for _ in range(2):  # Stripe retries webhooks
        assert client.post("/webhook", content=b"{}", headers={"Stripe-Signature": "x"}).status_code == 200
    data = client.get("/admin/orders", headers={"Authorization": "Bearer admin"}).json()
    assert data["count"] == 1 and data["gross_revenue_cents"] == {"usd": 1500}


def test_webhook_rejects_bad_signature(client, monkeypatch):
    def boom(*a):
        raise stripe.SignatureVerificationError("bad", "sig")
    monkeypatch.setattr(stripe.Webhook, "construct_event", boom)
    assert client.post("/webhook", content=b"{}").status_code == 400


def test_expired_link(client, monkeypatch):
    monkeypatch.setattr(stripe.checkout.Session, "retrieve", lambda sid, **kw: paid_session(sid))
    r = client.get("/success", params={"session_id": "cs_4"})
    link = r.text.split('href="http://shop.test')[1].split('"')[0]
    real = time.time
    monkeypatch.setattr(time, "time", lambda: real() + 73 * 3600)
    assert client.get(link).status_code == 410


def test_admin_requires_token(client):
    assert client.get("/admin/orders").status_code == 401
    assert client.get("/admin/orders", headers={"Authorization": "Bearer wrong"}).status_code == 401


def test_catalog_rejects_path_escape(env, monkeypatch):
    (env / "catalog.json").write_text(json.dumps([{
        "id": "x", "name": "x", "price_cents": 100, "file": "../../etc/passwd"}]))
    with pytest.raises(ValueError):
        create_app(load_settings())
