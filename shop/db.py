"""Tiny SQLite order store. One row per paid Stripe Checkout Session."""

import json
import secrets
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# Physical order lifecycle: awaiting_fulfillment -> submitted -> shipped
STATUSES = ("paid", "awaiting_fulfillment", "submitted", "shipped")


@dataclass
class Order:
    session_id: str
    product_id: str
    email: str
    token: str  # secret: download link (digital) or order-status link (physical)
    created_at: float
    downloads: int
    amount_total: int  # what the buyer actually paid, incl. shipping/discounts
    currency: str
    status: str
    ship_name: str
    ship_address: str  # JSON
    fulfillment_ref: str
    tracking: str

    @property
    def address(self) -> dict:
        return json.loads(self.ship_address or "{}")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ship_address"] = self.address
        return d


class OrderStore:
    def __init__(self, path: Path):
        self.path = path
        with self._conn() as c:
            c.execute(
                """CREATE TABLE IF NOT EXISTS orders (
                    session_id      TEXT PRIMARY KEY,
                    product_id      TEXT NOT NULL,
                    email           TEXT NOT NULL,
                    token           TEXT NOT NULL UNIQUE,
                    created_at      REAL NOT NULL,
                    downloads       INTEGER NOT NULL DEFAULT 0,
                    amount_total    INTEGER NOT NULL DEFAULT 0,
                    currency        TEXT NOT NULL DEFAULT '',
                    status          TEXT NOT NULL DEFAULT 'paid',
                    ship_name       TEXT NOT NULL DEFAULT '',
                    ship_address    TEXT NOT NULL DEFAULT '',
                    fulfillment_ref TEXT NOT NULL DEFAULT '',
                    tracking        TEXT NOT NULL DEFAULT ''
                )"""
            )

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, isolation_level=None)
        conn.row_factory = sqlite3.Row
        return conn

    def _get(self, c: sqlite3.Connection, where: str, value: str) -> Order | None:
        row = c.execute(f"SELECT * FROM orders WHERE {where} = ?", (value,)).fetchone()
        return Order(**row) if row else None

    def record_paid(
        self, session_id: str, product_id: str, email: str, *, amount_total: int = 0,
        currency: str = "", status: str = "paid", ship_name: str = "", ship_address: dict | None = None,
    ) -> tuple[Order, bool]:
        """Insert an order if new. Returns (order, created). Idempotent per session."""
        with self._conn() as c:
            cur = c.execute(
                """INSERT OR IGNORE INTO orders
                   (session_id, product_id, email, token, created_at, amount_total, currency,
                    status, ship_name, ship_address)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (session_id, product_id, email, secrets.token_urlsafe(32), time.time(),
                 amount_total, currency, status, ship_name, json.dumps(ship_address or {})),
            )
            return self._get(c, "session_id", session_id), cur.rowcount == 1

    def by_token(self, token: str) -> Order | None:
        with self._conn() as c:
            return self._get(c, "token", token)

    def by_session(self, session_id: str) -> Order | None:
        with self._conn() as c:
            return self._get(c, "session_id", session_id)

    def update(self, session_id: str, **fields) -> Order | None:
        allowed = {"status", "fulfillment_ref", "tracking"}
        if not fields or not set(fields) <= allowed:
            raise ValueError(f"Can only update {allowed}")
        if "status" in fields and fields["status"] not in STATUSES:
            raise ValueError(f"Unknown status {fields['status']!r}")
        sets = ", ".join(f"{k} = ?" for k in fields)
        with self._conn() as c:
            c.execute(f"UPDATE orders SET {sets} WHERE session_id = ?", (*fields.values(), session_id))
            return self._get(c, "session_id", session_id)

    def consume_download(self, token: str, max_downloads: int) -> bool:
        """Atomically count a download; False if the limit is reached."""
        with self._conn() as c:
            cur = c.execute(
                "UPDATE orders SET downloads = downloads + 1 WHERE token = ? AND downloads < ?",
                (token, max_downloads),
            )
        return cur.rowcount == 1

    def all(self, status: str | None = None) -> list[Order]:
        with self._conn() as c:
            if status:
                rows = c.execute("SELECT * FROM orders WHERE status = ? ORDER BY created_at DESC", (status,))
            else:
                rows = c.execute("SELECT * FROM orders ORDER BY created_at DESC")
            return [Order(**r) for r in rows.fetchall()]
