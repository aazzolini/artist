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
from fastapi import Body, Cookie, FastAPI, Form, Header, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .config import Product, Settings, load_catalog, load_settings, seed_products
from .db import Order, OrderStore
from .fulfillment import Fulfiller, format_address, make_fulfiller
from .mailer import send_email
from .owner import Owner
from .pieces import WEB_ASSET, BundleError, ingest, sign, update_product, verify

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
    seed_products(settings)
    catalog = load_catalog(settings)  # mutated in place when the pipeline publishes a piece
    store = OrderStore(settings.db_path)
    owner = Owner(settings, store)
    fulfiller = fulfiller or make_fulfiller(settings)
    app = FastAPI(title=settings.store_name, docs_url=None, redoc_url=None)

    images = settings.products_dir / "images"
    if images.is_dir():  # only images are public; product files never are
        app.mount("/images", StaticFiles(directory=images), name="images")

    def render(request: Request, name: str, status: int = 200, **ctx) -> HTMLResponse:
        ctx.setdefault("store_name", settings.store_name)
        ctx.setdefault("settings", settings)
        return TEMPLATES.TemplateResponse(request, name, ctx, status_code=status)

    def reload_catalog() -> None:
        fresh = load_catalog(settings)
        catalog.clear()
        catalog.update(fresh)

    def print_url(order: Order) -> str:
        return f"{settings.base_url}/print-file/{order.session_id}?{sign(owner.signing_secret, 'print', order.session_id)}"

    def preview_url(pid: str) -> str:
        return f"{settings.base_url}/p/{pid}?{sign(owner.signing_secret, 'preview', pid, ttl_s=None)}"

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
            ref = fulfiller.submit(order, product, print_url(order))
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

    def is_admin(authorization: str = "", admin_session: str = "") -> bool:
        token = authorization[7:] if authorization.startswith("Bearer ") else ""
        return (bool(token) and owner.check_password(token)) or (bool(admin_session) and owner.valid_session(admin_session))

    def require_admin(authorization: str, admin_session: str = "") -> None:
        if not is_admin(authorization, admin_session):
            raise HTTPException(401, "Unauthorized")

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        return render(request, "index.html", products=[p for p in catalog.values() if not p.draft])

    @app.get("/p/{product_id}", response_class=HTMLResponse)
    def product_page(request: Request, product_id: str, exp: str = "", sig: str = ""):
        product = catalog.get(product_id)
        preview = bool(sig) and verify(owner.signing_secret, "preview", product_id, exp, sig)
        if product is None or (product.draft and not preview):
            raise HTTPException(404, "No such product")
        return render(request, "product.html", product=product, draft_preview=product.draft)

    @app.get("/assets/{product_id}/{name}")
    def asset(product_id: str, name: str):
        product = catalog.get(product_id)
        if product is None or not product.model or not WEB_ASSET.match(name):
            raise HTTPException(404, "Not found")
        path = settings.products_dir / Path(product.model).parent / name
        if not path.is_file():
            raise HTTPException(404, "Not found")
        return FileResponse(path, headers={"Cache-Control": "public, max-age=3600"})

    @app.post("/buy/{product_id}")
    def buy(product_id: str):
        product = catalog.get(product_id)
        if product is None or product.draft:
            raise HTTPException(404, "No such product")
        if not owner.payments_connected:
            raise HTTPException(503, "This store isn't taking payments yet. Please check back soon.")
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
        session = stripe.checkout.Session.create(api_key=owner.stripe_key, **params)
        return RedirectResponse(session.url, status_code=303)

    @app.get("/success", response_class=HTMLResponse)
    def success(request: Request, session_id: str):
        try:
            session = stripe.checkout.Session.retrieve(session_id, api_key=owner.stripe_key)
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
                payload, stripe_signature, owner.webhook_secret
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
    def admin_orders(status: str | None = None, authorization: str = Header(""), admin_session: str = Cookie("")):
        require_admin(authorization, admin_session)
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
                   authorization: str = Header(""), admin_session: str = Cookie("")):
        require_admin(authorization, admin_session)
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

    @app.get("/print-file/{session_id}")
    def print_file(session_id: str, exp: str = "", sig: str = ""):
        """Signed link from the print-order email: the 3MF for that order's product."""
        if not verify(owner.signing_secret, "print", session_id, exp, sig):
            raise HTTPException(403, "This link is invalid or has expired")
        order = store.by_session(session_id)
        product = catalog.get(order.product_id) if order else None
        if product is None:
            raise HTTPException(404, "Unknown order")
        path = settings.products_dir / product.file
        return FileResponse(path, filename=f"{product.id}{path.suffix}")

    @app.post("/admin/pieces")
    async def admin_publish_piece(request: Request, authorization: str = Header("")):
        """Receive a pipeline bundle (zip) and publish or update that product."""
        require_admin(authorization)  # token only: the pipeline uploads these
        before = settings.catalog_path.read_text() if settings.catalog_path.exists() else None
        try:
            entry = ingest(settings, await request.body())
            reload_catalog()
        except BundleError as e:
            raise HTTPException(400, str(e))
        except Exception as e:
            if before is not None:  # never leave the store with a catalog it can't load
                settings.catalog_path.write_text(before)
                reload_catalog()
            raise HTTPException(400, f"Bundle rejected: {e}")
        return {"id": entry["id"], "draft": entry["draft"], "price_cents": entry["price_cents"],
                "url": f"{settings.base_url}/p/{entry['id']}", "preview_url": preview_url(entry["id"])}

    @app.post("/admin/products/{product_id}")
    def admin_update_product(product_id: str, price_cents: int | None = Body(None), draft: bool | None = Body(None),
                             authorization: str = Header(""), admin_session: str = Cookie("")):
        """Set a product's price and/or take it live (draft=false) or hide it (draft=true)."""
        require_admin(authorization, admin_session)
        try:
            entry = update_product(settings, product_id, price_cents=price_cents, draft=draft)
        except KeyError:
            raise HTTPException(404, "No such product")
        except BundleError as e:
            raise HTTPException(400, str(e))
        reload_catalog()
        return entry

    # ------------------------------------------------------------ admin pages (browser)

    def admin_redirect(msg: str = "", error: bool = False) -> RedirectResponse:
        from urllib.parse import urlencode
        q = urlencode({"err" if error else "msg": msg}) if msg else ""
        return RedirectResponse("/admin" + ("?" + q if q else ""), status_code=303)

    def with_login(resp: RedirectResponse) -> RedirectResponse:
        resp.set_cookie("admin_session", owner.session_cookie(), max_age=14 * 86400, httponly=True,
                        samesite="strict", secure=settings.base_url.startswith("https://"), path="/")
        return resp

    @app.get("/admin", response_class=HTMLResponse)
    def admin_home(request: Request, msg: str = "", err: str = "", admin_session: str = Cookie("")):
        if owner.needs_setup:
            return render(request, "admin.html", state="setup", msg=msg, err=err)
        if not owner.valid_session(admin_session):
            return render(request, "admin.html", state="login", msg=msg, err=err)
        orders = store.all()
        to_print = [o for o in orders if o.status in ("awaiting_fulfillment", "submitted")]
        return render(request, "admin.html", state="dashboard", msg=msg, err=err,
                      payments=owner.payments_info, products=list(catalog.values()),
                      preview_url=preview_url, to_print=to_print, recent=orders[:20], catalog=catalog,
                      print_url=print_url, format_address=format_address,
                      revenue=sum(o.amount_total for o in orders))

    @app.post("/admin/setup")
    def admin_setup(password: str = Form(...), password2: str = Form(...)):
        if not owner.needs_setup:
            return admin_redirect("This store already has an admin password.", error=True)
        if password != password2:
            return admin_redirect("The two passwords don't match.", error=True)
        try:
            owner.set_password(password)
        except ValueError as e:
            return admin_redirect(str(e), error=True)
        return with_login(admin_redirect("Password set. Next, connect your Stripe account to take payments."))

    @app.post("/admin/login")
    def admin_login(password: str = Form(...)):
        if not owner.check_password(password):
            time.sleep(1)
            return admin_redirect("Wrong password.", error=True)
        return with_login(admin_redirect())

    @app.post("/admin/logout")
    def admin_logout():
        resp = admin_redirect("Signed out.")
        resp.delete_cookie("admin_session", path="/")
        return resp

    @app.post("/admin/payments")
    def admin_payments(stripe_key: str = Form(...), admin_session: str = Cookie("")):
        require_admin("", admin_session)
        try:
            info = owner.connect_stripe(stripe_key)
        except ValueError as e:
            return admin_redirect(str(e), error=True)
        mode = "test mode: no real money moves" if info["test_mode"] else "live: you're taking real payments"
        return admin_redirect(f"Connected to Stripe ({info['account']}, {mode}). {info['note']}".strip())

    @app.post("/admin/payments/disconnect")
    def admin_payments_disconnect(admin_session: str = Cookie("")):
        require_admin("", admin_session)
        owner.disconnect_stripe()
        return admin_redirect("Stripe disconnected. Checkout is off until you connect again.")

    @app.post("/admin/products/{product_id}/form")
    def admin_product_form(product_id: str, price: str = Form(""), action: str = Form("save"),
                           admin_session: str = Cookie("")):
        require_admin("", admin_session)
        fields: dict = {}
        if price.strip():
            try:
                fields["price_cents"] = int(round(float(price.strip().lstrip("$")) * 100))
            except ValueError:
                return admin_redirect("Enter the price as a number, like 189 or 189.00.", error=True)
        if action in ("live", "hide"):
            fields["draft"] = action == "hide"
        try:
            update_product(settings, product_id, **fields)
        except KeyError:
            return admin_redirect("No such product.", error=True)
        except BundleError as e:
            return admin_redirect(str(e), error=True)
        reload_catalog()
        return admin_redirect({"live": "It's live on the storefront.", "hide": "Hidden from the storefront."}
                              .get(action, "Saved."))

    @app.post("/admin/orders/{session_id}/ship-form")
    def admin_ship_form(session_id: str, tracking: str = Form(""), admin_session: str = Cookie("")):
        require_admin("", admin_session)
        admin_ship(session_id, tracking, authorization="", admin_session=admin_session)
        return admin_redirect("Marked as shipped. The buyer has been emailed (if email is set up).")

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    return app


def __getattr__(name: str):
    # `uvicorn shop.main:app` builds the app lazily so importing for tests needs no env.
    if name == "app":
        return create_app()
    raise AttributeError(name)
