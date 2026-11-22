"""
Getting a paid physical order printed and shipped.

A Fulfiller takes an order and either returns a reference (the order has been handed
off) or None (nothing was sent; the order stays "awaiting_fulfillment" in /admin/orders).

Only EmailOwnerFulfiller exists today: it emails the store owner the shipping address
and the 3D model so they can place the order with their print-on-demand service.
To automate that, add a class with the same `submit` method that calls the service's
API (e.g. Slant 3D) using its official docs, and select it in `make_fulfiller`.
"""

import logging
from typing import Protocol

from .config import Product, Settings
from .db import Order
from .mailer import send_email

log = logging.getLogger(__name__)


class Fulfiller(Protocol):
    def submit(self, order: Order, product: Product, print_url: str = "") -> str | None: ...


def format_address(name: str, addr: dict) -> str:
    lines = [name, addr.get("line1"), addr.get("line2"),
             " ".join(filter(None, [addr.get("city"), addr.get("state"), addr.get("postal_code")])),
             addr.get("country")]
    return "\n".join(line for line in lines if line)


class EmailOwnerFulfiller:
    def __init__(self, settings: Settings):
        self.settings = settings

    def submit(self, order: Order, product: Product, print_url: str = "") -> str | None:
        s = self.settings
        body = (
            f"New order to print and ship: {product.name}\n\n"
            f"Print file: {print_url or product.file}\n"
            + (f"  (download link valid 14 days; upload it to your print service)\n" if print_url else "")
            + (f"\nInstructions for the print service:\n{product.print_notes}\n" if product.print_notes else "")
            + "\n"
            f"Model file: {product.file} (attached if under 15 MB)\n"
            f"Order: {order.session_id}\n"
            f"Buyer email: {order.email}\n\n"
            f"Ship to:\n{format_address(order.ship_name, order.address)}\n\n"
            f"After placing it with your print service, mark it shipped:\n"
            f"curl -X POST -H 'Authorization: Bearer $ADMIN_TOKEN' -H 'Content-Type: application/json' \\\n"
            f"  -d '{{\"tracking\": \"<tracking number>\"}}' {s.base_url}/admin/orders/{order.session_id}/ship\n"
        )
        sent = send_email(s, s.owner_email, f"[{s.store_name}] Print order: {product.name}",
                          body, attachment=s.products_dir / product.file)
        if not sent:
            log.warning("Order %s awaiting fulfillment (owner email not sent)", order.session_id)
        return "emailed-owner" if sent else None


def make_fulfiller(settings: Settings) -> Fulfiller:
    return EmailOwnerFulfiller(settings)
