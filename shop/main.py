"""
Self-hosted digital product store.

Flow: storefront -> Stripe Checkout -> (webhook and/or success page) record order
-> buyer gets a secret, expiring, download-limited link (shown + optionally emailed).
"""

import logging
import time
from pathlib import Path

import stripe
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from .config import Settings, load_catalog, load_settings
from .db import Order, OrderStore
from .mailer import send_download_email

log = logging.getLogger("shop")
TEMPLATES = Jinja2Templates(directory=Path(__file__).parent / "templates")
FULFILL_EVENTS = {"checkout.session.completed", "checkout.session.async_payment_succeeded"}


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    catalog = load_catalog(settings)
    store = OrderStore(settings.db_path)
    app = FastAPI(title=settings.store_name, docs_url=None, redoc_url=None)

    def render(request: Request, name: str, status: int = 200, **ctx) -> HTMLResponse:
        ctx.setdefault("store_name", settings.store_name)
        return TEMPLATES.TemplateResponse(request, name, ctx, status_code=status)

    def link_for(order: Order) -> str:
        return f"{settings.base_url}/download/{order.token}"

    def fulfill(session) -> Order | None:
        """Record a paid session once; email the link on first fulfilment."""
        if session.get("payment_status") != "paid":
            return None
        product_id = (session.get("metadata") or {}).get("product_id")
        product = catalog.get(product_id)
        if product is None:
            log.error("Paid session %s for unknown product %r", session["id"], product_id)
            return None
        email = (session.get("customer_details") or {}).get("email") or ""
        order, created = store.record_paid(session["id"], product.id, email)
        if created:
            log.info("SALE %s %s %s", product.id, product.price_display, email)
            send_download_email(settings, email, product, link_for(order))
        return order

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        return render(request, "index.html", products=list(catalog.values()))

    @app.post("/buy/{product_id}")
    def buy(product_id: str):
        product = catalog.get(product_id)
        if product is None:
            raise HTTPException(404, "No such product")
        session = stripe.checkout.Session.create(
            api_key=settings.stripe_secret_key,
            mode="payment",
            line_items=[{
                "quantity": 1,
                "price_data": {
                    "currency": product.currency,
                    "unit_amount": product.price_cents,
                    "product_data": {"name": product.name},
                },
            }],
            metadata={"product_id": product.id},
            allow_promotion_codes=True,
            success_url=f"{settings.base_url}/success?session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{settings.base_url}/",
        )
        return RedirectResponse(session.url, status_code=303)

    @app.get("/success", response_class=HTMLResponse)
    def success(request: Request, session_id: str):
        try:
            session = stripe.checkout.Session.retrieve(session_id, api_key=settings.stripe_secret_key)
        except stripe.StripeError:
            raise HTTPException(404, "Unknown checkout session")
        order = fulfill(session)
        if order is None:
            # e.g. bank transfer still processing; the webhook will finish it.
            return render(request, "pending.html", status=202)
        return render(request, "success.html", product=catalog[order.product_id],
                      link=link_for(order), email=order.email, settings=settings)

    @app.post("/webhook")
    async def webhook(request: Request, stripe_signature: str = Header("")):
        payload = await request.body()
        try:
            event = stripe.Webhook.construct_event(
                payload, stripe_signature, settings.stripe_webhook_secret
            )
        except (ValueError, stripe.SignatureVerificationError):
            raise HTTPException(400, "Bad signature")
        if event["type"] in FULFILL_EVENTS:
            fulfill(event["data"]["object"])
        return {"received": True}

    @app.get("/download/{token}")
    def download(token: str):
        order = store.by_token(token)
        if order is None:
            raise HTTPException(404, "Invalid link")
        if time.time() - order.created_at > settings.link_ttl_hours * 3600:
            raise HTTPException(410, "This link has expired. Reply to your receipt email for a new one.")
        if not store.consume_download(token, settings.max_downloads):
            raise HTTPException(410, "Download limit reached. Reply to your receipt email for help.")
        product = catalog[order.product_id]
        path = settings.products_dir / product.file
        return FileResponse(path, filename=path.name)

    @app.get("/admin/orders")
    def admin_orders(authorization: str = Header("")):
        if not settings.admin_token or authorization != f"Bearer {settings.admin_token}":
            raise HTTPException(401, "Unauthorized")
        orders = store.all()
        revenue: dict[str, int] = {}
        for o in orders:
            p = catalog.get(o.product_id)
            if p:
                revenue[p.currency] = revenue.get(p.currency, 0) + p.price_cents
        return JSONResponse({
            "count": len(orders),
            "gross_revenue_cents": revenue,
            "orders": [o.__dict__ for o in orders],
        })

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    return app


def __getattr__(name: str):
    # `uvicorn shop.main:app` builds the app lazily so importing for tests needs no env.
    if name == "app":
        return create_app()
    raise AttributeError(name)
