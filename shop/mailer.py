"""Optional: email the download link to the buyer. No-op when SMTP is unset."""

import logging
import smtplib
from email.message import EmailMessage

from .config import Product, Settings

log = logging.getLogger(__name__)


def send_download_email(settings: Settings, to: str, product: Product, link: str) -> bool:
    if not (settings.smtp_host and settings.smtp_from and to):
        return False
    msg = EmailMessage()
    msg["Subject"] = f"Your download: {product.name}"
    msg["From"] = settings.smtp_from
    msg["To"] = to
    msg.set_content(
        f"Thanks for buying {product.name} from {settings.store_name}!\n\n"
        f"Download it here (valid {settings.link_ttl_hours}h, "
        f"{settings.max_downloads} downloads):\n{link}\n"
    )
    try:
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=15) as s:
            s.starttls()
            if settings.smtp_user:
                s.login(settings.smtp_user, settings.smtp_password)
            s.send_message(msg)
        return True
    except Exception:  # never fail a paid order because email broke
        log.exception("Failed to send download email to %s", to)
        return False
