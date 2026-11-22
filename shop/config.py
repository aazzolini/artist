"""Settings, loaded from environment variables (see .env.example)."""

import json
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Product:
    id: str
    name: str
    description: str
    price_cents: int
    currency: str
    file: str  # relative to PRODUCTS_DIR. digital: what the buyer downloads; physical: the 3D model sent to the printer
    kind: str = "digital"  # "digital" or "physical"
    image: str = ""  # filename inside PRODUCTS_DIR/images, shown on the storefront

    @property
    def physical(self) -> bool:
        return self.kind == "physical"

    @property
    def price_display(self) -> str:
        return f"{self.price_cents / 100:,.2f} {self.currency.upper()}"


@dataclass(frozen=True)
class Settings:
    stripe_secret_key: str
    stripe_webhook_secret: str
    base_url: str
    store_name: str
    products_dir: Path
    catalog_path: Path
    db_path: Path
    max_downloads: int
    link_ttl_hours: int
    smtp_host: str
    smtp_port: int
    smtp_user: str
    smtp_password: str
    smtp_from: str
    admin_token: str
    owner_email: str
    shipping_cents: int
    ship_countries: tuple[str, ...]


def load_settings() -> Settings:
    env = os.environ.get
    products_dir = Path(env("PRODUCTS_DIR", ROOT / "products"))
    return Settings(
        stripe_secret_key=env("STRIPE_SECRET_KEY", ""),
        stripe_webhook_secret=env("STRIPE_WEBHOOK_SECRET", ""),
        base_url=env("BASE_URL", "http://localhost:8000").rstrip("/"),
        store_name=env("STORE_NAME", "My Store"),
        products_dir=products_dir,
        catalog_path=Path(env("CATALOG_PATH", products_dir / "catalog.json")),
        db_path=Path(env("DB_PATH", ROOT / "orders.db")),
        max_downloads=int(env("MAX_DOWNLOADS", "5")),
        link_ttl_hours=int(env("LINK_TTL_HOURS", "72")),
        smtp_host=env("SMTP_HOST", ""),
        smtp_port=int(env("SMTP_PORT", "587")),
        smtp_user=env("SMTP_USER", ""),
        smtp_password=env("SMTP_PASSWORD", ""),
        smtp_from=env("SMTP_FROM", ""),
        admin_token=env("ADMIN_TOKEN", ""),
        owner_email=env("OWNER_EMAIL", ""),
        shipping_cents=int(env("SHIPPING_CENTS", "599")),
        ship_countries=tuple(c.strip().upper() for c in env("SHIP_COUNTRIES", "US").split(",") if c.strip()),
    )


def load_catalog(settings: Settings) -> dict[str, Product]:
    raw = json.loads(settings.catalog_path.read_text())
    catalog: dict[str, Product] = {}
    for item in raw:
        p = Product(
            id=item["id"],
            name=item["name"],
            description=item.get("description", ""),
            price_cents=int(item["price_cents"]),
            currency=item.get("currency", "usd").lower(),
            file=item["file"],
            kind=item.get("kind", "digital"),
            image=item.get("image", ""),
        )
        if p.kind not in ("digital", "physical"):
            raise ValueError(f"Product {p.id}: kind must be 'digital' or 'physical'")
        _check_file(settings.products_dir, p.file, p.id)
        if p.image:
            _check_file(settings.products_dir / "images", p.image, p.id)
        if p.price_cents < 50:
            raise ValueError(f"Product {p.id}: Stripe minimum charge is 50 cents")
        catalog[p.id] = p
    return catalog


def _check_file(base: Path, rel: str, product_id: str) -> None:
    path = (base / rel).resolve()
    if not path.is_relative_to(base.resolve()):
        raise ValueError(f"Product {product_id}: {rel} escapes {base}")
    if not path.is_file():
        raise FileNotFoundError(f"Product {product_id}: missing file {path}")
