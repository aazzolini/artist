"""
Self-hosted store for digital downloads and 3D-printed figures.

Flow: storefront -> Stripe Checkout -> (webhook and/or success page) record order once.
  digital:  buyer gets a secret, expiring, download-limited link.
  physical: order goes to the fulfiller (print-on-demand); buyer gets an order-status link.
"""

import logging
import time
from pathlib import Path

import stripe
from fastapi import Body, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .config import Product, Settings, load_catalog, load_settings
from .db import Order, OrderStore
from .fulfillment import Fulfiller, format_address, make_fulfiller
from .mailer import send_email

log = logging.getLogger("shop")
TEMPLATES = Jinja2Templates(directory=Path(__file__).parent / "templates")
FULFILL_EVENTS = {"checkout.session.completed", "checkout.session.async_payment_succeeded"}


def _as_dict(obj) -> dict:
    """Stripe objects aren't dicts in stripe-python >= 13; normalize (recursively)."""
    return obj.to_dict() if hasattr(obj, "to_dict") else obj


def _shipping(session: dict) -> tuple[str, dict]:
    # Newer Stripe API versions use collected_information; older ones shipping_details.
    details = ((session.get("collected_information") or {}).get("shipping_details")
               or session.get("shipping_details") or {})
    return details.get("name") or "", details.get("address") or {}


def create_app(settings: Settings | None = None, fulfiller: Fulfiller | None = None) -> FastAPI:
    settings = settings or load_settings()
    catalog = load_catalog(settings)
    store = OrderStore(settings.db_path)
    fulfiller = fulfiller or make_fulfiller(settings)
    app = FastAPI(title=settings.store_name, docs_url=None, redoc_url=None)

    images = settings.products_dir / "images"
    if images.is_dir():  # only images are public; product files never are
        app.mount("/images", StaticFiles(directory=images), name="images")

    def render(request: Request, name: str, status: int = 200, **ctx) -> HTMLResponse:
        ctx.setdefault("store_name", settings.store_name)
        ctx.setdefault("settings", settings)
        return TEMPLATES.TemplateResponse(request, name, ctx, status_code=status)

    def link_for(order: Order, product: Product) -> str:
        path = "order" if product.physical else "download"
        return f"{settings.base_url}/{path}/{order.token}"

    def fulfill(session) -> Order | None:
        """Record a paid session once, then deliver (download link or send to printer)."""
        session = _as_dict(session)
        if session.get("payment_status") != "paid":
            return None
        product_id = (session.get("metadata") or {}).get("product_id")
        product = catalog.get(product_id)
        if product is None:
            log.error("Paid session %s for unknown product %r", session["id"], product_id)
            return None
        email = (session.get("customer_details") or {}).get("email") or ""
        ship_name, ship_address = _shipping(session) if product.physical else ("", {})
        order, created = store.record_paid(
            session["id"], product.id, email,
            amount_total=session.get("amount_total") or product.price_cents,
            currency=session.get("currency") or product.currency,
            status="awaiting_fulfillment" if product.physical else "paid",
            ship_name=ship_name, ship_address=ship_address,
        )
        if not created:
            return order
        log.info("SALE %s %s %s", product.id, order.amount_total, email)
        link = link_for(order, product)
        if product.physical:
            ref = fulfiller.submit(order, product)
            if ref:
                order = store.update(order.session_id, status="submitted", fulfillment_ref=ref)
            send_email(settings, email, f"Order confirmed: {product.name}",
                       f"Thanks for ordering {product.name} from {settings.store_name}!\n\n"
                       f"Your figure is printed to order and ships to:\n"
                       f"{format_address(ship_name, ship_address)}\n\n"
                       f"Track your order here: {link}\n")
        else:
            send_email(settings, email, f"Your download: {product.name}",
                       f"Thanks for buying {product.name} from {settings.store_name}!\n\n"
                       f"Download it here (valid {settings.link_ttl_hours}h, "
                       f"{settings.max_downloads} downloads):\n{link}\n")
        return order

    def require_admin(authorization: str) -> None:
        if not settings.admin_token or authorization != f"Bearer {settings.admin_token}":
            raise HTTPException(401, "Unauthorized")

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        return render(request, "index.html", products=list(catalog.values()))

    @app.post("/buy/{product_id}")
    def buy(product_id: str):
        product = catalog.get(product_id)
        if product is None:
            raise HTTPException(404, "No such product")
        params: dict = dict(
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
        if product.physical:
            params["shipping_address_collection"] = {"allowed_countries": list(settings.ship_countries)}
            params["shipping_options"] = [{"shipping_rate_data": {
                "type": "fixed_amount",
                "fixed_amount": {"amount": settings.shipping_cents, "currency": product.currency},
                "display_name": "Standard shipping" if settings.shipping_cents else "Free shipping",
            }}]
        session = stripe.checkout.Session.create(api_key=settings.stripe_secret_key, **params)
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
        product = catalog[order.product_id]
        return render(request, "success.html", product=product, order=order,
                      link=link_for(order, product))

    @app.post("/webhook")
    async def webhook(request: Request, stripe_signature: str = Header("")):
        payload = await request.body()
        try:
            event = stripe.Webhook.construct_event(
                payload, stripe_signature, settings.stripe_webhook_secret
            )
        except (ValueError, stripe.SignatureVerificationError):
            raise HTTPException(400, "Bad signature")
        event = _as_dict(event)
        if event["type"] in FULFILL_EVENTS:
            fulfill(event["data"]["object"])
        return {"received": True}

    @app.get("/download/{token}")
    def download(token: str):
        order = store.by_token(token)
        product = catalog.get(order.product_id) if order else None
        if product is None or product.physical:
            raise HTTPException(404, "Invalid link")
        if time.time() - order.created_at > settings.link_ttl_hours * 3600:
            raise HTTPException(410, "This link has expired. Reply to your receipt email for a new one.")
        if not store.consume_download(token, settings.max_downloads):
            raise HTTPException(410, "Download limit reached. Reply to your receipt email for help.")
        path = settings.products_dir / product.file
        return FileResponse(path, filename=path.name)

    @app.get("/order/{token}", response_class=HTMLResponse)
    def order_status(request: Request, token: str):
        order = store.by_token(token)
        product = catalog.get(order.product_id) if order else None
        if product is None or not product.physical:
            raise HTTPException(404, "Unknown order")
        return render(request, "order.html", order=order, product=product)

    @app.get("/admin/orders")
    def admin_orders(status: str | None = None, authorization: str = Header("")):
        require_admin(authorization)
        orders = store.all(status)
        revenue: dict[str, int] = {}
        for o in orders:
            revenue[o.currency] = revenue.get(o.currency, 0) + o.amount_total
        return JSONResponse({
            "count": len(orders),
            "gross_revenue_cents": revenue,
            "orders": [o.to_dict() for o in orders],
        })

    @app.post("/admin/orders/{session_id}/ship")
    def admin_ship(session_id: str, tracking: str = Body("", embed=True),
                   authorization: str = Header("")):
        require_admin(authorization)
        order = store.by_session(session_id)
        product = catalog.get(order.product_id) if order else None
        if product is None or not product.physical:
            raise HTTPException(404, "No such physical order")
        first_time = order.status != "shipped"
        order = store.update(session_id, status="shipped", tracking=tracking)
        if first_time:
            send_email(settings, order.email, f"Shipped: {product.name}",
                       f"Your {product.name} is on its way!\n"
                       + (f"Tracking number: {tracking}\n" if tracking else "")
                       + f"\nOrder status: {link_for(order, product)}\n")
        return order.to_dict()

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    return app


def __getattr__(name: str):
    # `uvicorn shop.main:app` builds the app lazily so importing for tests needs no env.
    if name == "app":
        return create_app()
    raise AttributeError(name)
