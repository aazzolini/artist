"""Tiny SQLite order store. One row per paid Stripe Checkout Session."""

import secrets
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Order:
    session_id: str
    product_id: str
    email: str
    token: str
    created_at: float
    downloads: int


class OrderStore:
    def __init__(self, path: Path):
        self.path = path
        with self._conn() as c:
            c.execute(
                """CREATE TABLE IF NOT EXISTS orders (
                    session_id TEXT PRIMARY KEY,
                    product_id TEXT NOT NULL,
                    email      TEXT NOT NULL,
                    token      TEXT NOT NULL UNIQUE,
                    created_at REAL NOT NULL,
                    downloads  INTEGER NOT NULL DEFAULT 0
                )"""
            )

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, isolation_level=None)
        conn.row_factory = sqlite3.Row
        return conn

    def record_paid(self, session_id: str, product_id: str, email: str) -> tuple[Order, bool]:
        """Insert an order if new. Returns (order, created). Idempotent per session."""
        with self._conn() as c:
            cur = c.execute(
                "INSERT OR IGNORE INTO orders VALUES (?, ?, ?, ?, ?, 0)",
                (session_id, product_id, email, secrets.token_urlsafe(32), time.time()),
            )
            created = cur.rowcount == 1
            row = c.execute("SELECT * FROM orders WHERE session_id = ?", (session_id,)).fetchone()
        return Order(**row), created

    def by_token(self, token: str) -> Order | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM orders WHERE token = ?", (token,)).fetchone()
        return Order(**row) if row else None

    def consume_download(self, token: str, max_downloads: int) -> bool:
        """Atomically count a download; False if the limit is reached."""
        with self._conn() as c:
            cur = c.execute(
                "UPDATE orders SET downloads = downloads + 1 WHERE token = ? AND downloads < ?",
                (token, max_downloads),
            )
        return cur.rowcount == 1

    def all(self) -> list[Order]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM orders ORDER BY created_at DESC").fetchall()
        return [Order(**r) for r in rows]
