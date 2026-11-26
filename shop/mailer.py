"""Optional email. Every function is a no-op (returns False) when SMTP isn't configured."""

import logging
import smtplib
from email.message import EmailMessage
from pathlib import Path

from .config import Settings

log = logging.getLogger(__name__)
MAX_ATTACHMENT_BYTES = 15 * 1024 * 1024


def send_email(settings: Settings, to: str, subject: str, body: str,
               attachment: Path | None = None) -> bool:
    if not (settings.smtp_host and settings.smtp_from and to):
        return False
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = settings.smtp_from
    msg["To"] = to
    msg.set_content(body)
    if attachment and attachment.stat().st_size <= MAX_ATTACHMENT_BYTES:
        msg.add_attachment(attachment.read_bytes(), maintype="application",
                           subtype="octet-stream", filename=attachment.name)
    try:
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=15) as s:
            s.starttls()
            if settings.smtp_user:
                s.login(settings.smtp_user, settings.smtp_password)
            s.send_message(msg)
        return True
    except Exception:  # never fail a paid order because email broke
        log.exception("Failed to send email to %s", to)
        return False
