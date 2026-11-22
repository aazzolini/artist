"""
Owner settings managed from the /admin page: admin password, login session, payment connection.

Saved settings live in the store's database and win over environment variables, so the owner can
set everything up from the browser. Environment variables still work (useful for CI or tests).
"""

import hashlib
import hmac
import secrets
import time

import stripe

from .config import Settings
from .db import OrderStore

WEBHOOK_EVENTS = ["checkout.session.completed", "checkout.session.async_payment_succeeded"]
SESSION_TTL = 14 * 86400


class Owner:
    def __init__(self, settings: Settings, store: OrderStore):
        self.settings, self.store = settings, store

    # -- secrets
    @property
    def signing_secret(self) -> str:
        """Signs links and login cookies. Created once and kept in the database."""
        s = self.settings.admin_token or self.store.get_setting("signing_secret")
        if not s:
            s = secrets.token_urlsafe(32)
            self.store.set_setting("signing_secret", s)
        return s

    @property
    def stripe_key(self) -> str:
        return self.store.get_setting("stripe_secret_key") or self.settings.stripe_secret_key

    @property
    def webhook_secret(self) -> str:
        return self.store.get_setting("stripe_webhook_secret") or self.settings.stripe_webhook_secret

    @property
    def payments_connected(self) -> bool:
        return bool(self.stripe_key)

    @property
    def payments_info(self) -> dict:
        key = self.stripe_key
        return {"connected": bool(key), "test_mode": "_test_" in key,
                "account": self.store.get_setting("stripe_account_name"),
                "webhook": bool(self.webhook_secret)}

    @property
    def owner_email(self) -> str:
        return self.store.get_setting("owner_email") or self.settings.owner_email

    # -- password & sessions
    @property
    def needs_setup(self) -> bool:
        return not self.settings.admin_token and not self.store.get_setting("admin_password_hash")

    def set_password(self, password: str) -> None:
        if len(password) < 10:
            raise ValueError("Use at least 10 characters.")
        salt = secrets.token_bytes(16)
        h = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000)
        self.store.set_setting("admin_password_hash", salt.hex() + ":" + h.hex())

    def check_password(self, password: str) -> bool:
        if self.settings.admin_token and hmac.compare_digest(password, self.settings.admin_token):
            return True
        stored = self.store.get_setting("admin_password_hash")
        if not stored or not password:
            return False
        salt, h = stored.split(":")
        return hmac.compare_digest(hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 200_000).hex(), h)

    def session_cookie(self) -> str:
        exp = str(int(time.time()) + SESSION_TTL)
        return exp + "." + hmac.new(self.signing_secret.encode(), f"admin:{exp}".encode(), hashlib.sha256).hexdigest()

    def valid_session(self, cookie: str) -> bool:
        try:
            exp, sig = cookie.split(".", 1)
        except ValueError:
            return False
        if not exp.isdigit() or int(exp) < time.time():
            return False
        good = hmac.new(self.signing_secret.encode(), f"admin:{exp}".encode(), hashlib.sha256).hexdigest()
        return hmac.compare_digest(sig, good)

    # -- payments
    def connect_stripe(self, key: str) -> dict:
        """Check the key with Stripe, register this store's webhook, and save both.
        Raises ValueError with a message for the owner if something is wrong."""
        key = key.strip()
        if not key.startswith(("sk_live_", "sk_test_", "rk_live_", "rk_test_")):
            raise ValueError("That doesn't look like a Stripe secret key. It starts with sk_live_ or sk_test_.")
        try:
            acct = stripe.Account.retrieve(api_key=key).to_dict()  # Stripe objects aren't dicts
        except stripe.AuthenticationError:
            raise ValueError("Stripe didn't accept that key. Copy it again from your Stripe dashboard.")
        except stripe.StripeError as e:
            raise ValueError(f"Couldn't reach Stripe: {e.user_message or e}")
        name = ((acct.get("settings") or {}).get("dashboard") or {}).get("display_name") \
            or (acct.get("business_profile") or {}).get("name") or acct.get("email") or acct.get("id", "")
        url = f"{self.settings.base_url}/webhook"
        webhook_secret, note = None, ""
        if self.settings.base_url.startswith("https://"):
            try:  # replace any webhook this store registered before, then register a fresh one
                for ep in stripe.WebhookEndpoint.list(api_key=key, limit=100).auto_paging_iter():
                    if ep.to_dict().get("url") == url:
                        stripe.WebhookEndpoint.delete(ep.id, api_key=key)
                ep = stripe.WebhookEndpoint.create(api_key=key, url=url, enabled_events=WEBHOOK_EVENTS,
                                                   description="Store: record paid orders").to_dict()
                webhook_secret = ep["secret"]
            except stripe.StripeError as e:
                note = f"Payments work, but Stripe wouldn't register order notifications ({e.user_message or e})."
        else:
            note = "This store isn't on a public https address yet, so Stripe can't notify it. " \
                   "Orders are still recorded when the buyer returns to the thank-you page."
        self.store.set_setting("stripe_secret_key", key)
        self.store.set_setting("stripe_webhook_secret", webhook_secret)
        self.store.set_setting("stripe_account_name", name)
        return {"account": name, "test_mode": "_test_" in key, "note": note}

    def disconnect_stripe(self) -> None:
        for k in ("stripe_secret_key", "stripe_webhook_secret", "stripe_account_name"):
            self.store.set_setting(k, None)
