"""The browser admin page: first-run password, sign-in, connecting Stripe, pricing products."""
import json

import pytest
import stripe
from fastapi.testclient import TestClient

from shop.config import load_settings
from shop.main import create_app


@pytest.fixture
def make_client(tmp_path, monkeypatch):
    (tmp_path / "files").mkdir()
    (tmp_path / "files" / "a.pdf").write_bytes(b"x")
    (tmp_path / "catalog.json").write_text(json.dumps([
        {"id": "guide", "name": "Guide", "price_cents": 0, "draft": True, "file": "files/a.pdf"}]))
    for k in ("STRIPE_SECRET_KEY", "STRIPE_WEBHOOK_SECRET", "ADMIN_TOKEN", "BASE_URL", "RENDER_EXTERNAL_URL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("PRODUCTS_DIR", str(tmp_path))
    monkeypatch.setenv("DB_PATH", str(tmp_path / "o.db"))

    def make(base_url="https://shop.example"):
        monkeypatch.setenv("RENDER_EXTERNAL_URL", base_url)
        return TestClient(create_app(load_settings()), base_url=base_url, follow_redirects=False)
    return make


def fake_stripe(monkeypatch, calls):
    acct = stripe.Account.construct_from({"id": "acct_1", "object": "account",
                                          "settings": {"dashboard": {"display_name": "Ada Prints"}}}, "k")

    def retrieve(**kw):
        calls.append(("account", kw["api_key"]))
        if kw["api_key"].endswith("bad"):
            raise stripe.AuthenticationError("Invalid API Key provided")
        return acct

    class Listing:
        def auto_paging_iter(self):
            return iter([stripe.WebhookEndpoint.construct_from(
                {"id": "we_old", "object": "webhook_endpoint", "url": "https://shop.example/webhook"}, "k")])

    monkeypatch.setattr(stripe.Account, "retrieve", retrieve)
    monkeypatch.setattr(stripe.WebhookEndpoint, "list", lambda **kw: Listing())
    monkeypatch.setattr(stripe.WebhookEndpoint, "delete", lambda wid, **kw: calls.append(("delete", wid)))

    def create(**kw):
        calls.append(("create", kw["url"], tuple(kw["enabled_events"])))
        return stripe.WebhookEndpoint.construct_from({"id": "we_new", "object": "webhook_endpoint",
                                                      "secret": "whsec_new"}, "k")
    monkeypatch.setattr(stripe.WebhookEndpoint, "create", create)


def test_first_run_setup_then_login(make_client):
    c = make_client()
    assert "Create your admin password" in c.get("/admin").text
    r = c.post("/admin/setup", data={"password": "short", "password2": "short"})
    assert "err=" in r.headers["location"]
    r = c.post("/admin/setup", data={"password": "correct horse battery", "password2": "correct horse battery"})
    assert r.status_code == 303 and "admin_session" in r.cookies
    assert "Payments" in c.get("/admin").text
    # setup can't be repeated by someone else
    other = make_client()
    assert "Sign in" in other.get("/admin").text
    assert "already+has" in other.post("/admin/setup", data={"password": "x" * 12, "password2": "x" * 12}).headers["location"]
    assert "err=" in other.post("/admin/login", data={"password": "wrong password!"}).headers["location"]
    assert other.post("/admin/login", data={"password": "correct horse battery"}).status_code == 303
    assert "Payments" in other.get("/admin").text
    # the password also works as the pipeline's bearer token
    assert make_client().get("/admin/orders", headers={"Authorization": "Bearer correct horse battery"}).status_code == 200


def test_connect_stripe_registers_webhook_and_enables_checkout(make_client, monkeypatch):
    calls = []
    fake_stripe(monkeypatch, calls)
    c = make_client()
    c.post("/admin/setup", data={"password": "correct horse battery", "password2": "correct horse battery"})
    assert "Not connected" in c.get("/admin").text
    assert c.post("/admin/payments", data={"stripe_key": "pk_live_123"}).headers["location"].startswith("/admin?err=")
    assert "err=" in c.post("/admin/payments", data={"stripe_key": "sk_live_bad"}).headers["location"]

    r = c.post("/admin/payments", data={"stripe_key": "sk_test_good"})
    assert "msg=" in r.headers["location"]
    page = c.get("/admin").text
    assert "Connected" in page and "Ada Prints" in page and "Test mode" in page
    assert ("delete", "we_old") in calls
    assert ("create", "https://shop.example/webhook",
            ("checkout.session.completed", "checkout.session.async_payment_succeeded")) in calls

    # Price the draft product from the page and take it live; checkout then uses the saved key.
    assert "err=" in c.post("/admin/products/guide/form", data={"price": "abc", "action": "save"}).headers["location"]
    c.post("/admin/products/guide/form", data={"price": "19", "action": "live"})
    assert "19.00 USD" in c.get("/").text
    seen = {}
    monkeypatch.setattr(stripe.checkout.Session, "create",
                        lambda **kw: seen.update(kw) or stripe.checkout.Session.construct_from(
                            {"id": "cs", "object": "checkout.session", "url": "https://checkout.stripe.com/x"}, "k"))
    assert c.post("/buy/guide").status_code == 303 and seen["api_key"] == "sk_test_good"

    c.post("/admin/payments/disconnect")
    assert c.post("/buy/guide").status_code == 503


def test_local_store_connects_without_webhook(make_client, monkeypatch):
    calls = []
    fake_stripe(monkeypatch, calls)
    c = make_client("http://localhost:8000")
    c.post("/admin/setup", data={"password": "correct horse battery", "password2": "correct horse battery"})
    r = c.post("/admin/payments", data={"stripe_key": "sk_test_good"})
    assert "public+https" in r.headers["location"] or "public%20https" in r.headers["location"]
    assert not any(k[0] == "create" for k in calls)


def test_admin_actions_need_login(make_client):
    c = make_client()
    c.post("/admin/setup", data={"password": "correct horse battery", "password2": "correct horse battery"})
    anon = make_client()
    assert anon.post("/admin/payments", data={"stripe_key": "sk_test_x"}).status_code == 401
    assert anon.post("/admin/products/guide/form", data={"price": "1", "action": "live"}).status_code == 401
    anon.cookies.set("admin_session", "9999999999.forged")
    assert anon.post("/admin/payments/disconnect").status_code == 401
