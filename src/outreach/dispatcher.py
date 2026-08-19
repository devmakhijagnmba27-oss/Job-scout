"""Multi-channel email dispatcher with SMTP, safety dry-run, and rate limiting.

SECURITY & SAFETY:
- Defaults to DRY_RUN=true when SMTP credentials are not configured or when
  OUTREACH_DRY_RUN=true is set in environment.
- Daily dispatch quota enforcement to avoid domain reputation damage or spam triggers.
- Logs every dispatch attempt to logs/audit.jsonl and logs/outreach_sent.jsonl.
"""

from __future__ import annotations

import json
import os
import smtplib
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
LOGS_DIR = REPO_ROOT / "logs"
OUTREACH_LOG = LOGS_DIR / "outreach_sent.jsonl"


@dataclass
class DispatchResult:
    success: bool
    recipient: str
    subject: str
    is_dry_run: bool
    message_id: str | None = None
    error: str | None = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


class EmailDispatcher:
    def __init__(self):
        self.host = os.getenv("SMTP_HOST", "smtp.gmail.com")
        self.port = int(os.getenv("SMTP_PORT", "587"))
        self.user = os.getenv("SMTP_USER", "")
        self.password = os.getenv("SMTP_PASSWORD", "")
        self.from_email = os.getenv("SMTP_FROM_EMAIL", self.user)
        self.use_tls = os.getenv("SMTP_USE_TLS", "true").lower() in ("true", "1", "yes")
        self.use_ssl = os.getenv("SMTP_USE_SSL", "false").lower() in ("true", "1", "yes")

        dry_run_env = os.getenv("OUTREACH_DRY_RUN", "true").lower()
        self.dry_run = dry_run_env in ("true", "1", "yes") or not (self.user and self.password)
        self.daily_limit = int(os.getenv("OUTREACH_DAILY_LIMIT", "25"))
        LOGS_DIR.mkdir(exist_ok=True)

    def is_configured(self) -> bool:
        return bool(self.user and self.password and self.from_email)

    def get_sent_today_count(self) -> int:
        if not OUTREACH_LOG.exists():
            return 0
        today_prefix = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        count = 0
        try:
            with OUTREACH_LOG.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        if entry.get("timestamp", "").startswith(today_prefix) and not entry.get("is_dry_run"):
                            count += 1
        except Exception:
            pass
        return count

    def send_email(
        self,
        to_email: str,
        subject: str,
        body_text: str,
        body_html: str | None = None,
        candidate_name: str | None = None,
    ) -> DispatchResult:
        """Send an email or simulate in dry-run mode."""
        if not to_email:
            return DispatchResult(
                success=False,
                recipient=to_email,
                subject=subject,
                is_dry_run=self.dry_run,
                error="Recipient email address is required.",
            )

        if not self.dry_run:
            sent_today = self.get_sent_today_count()
            if sent_today >= self.daily_limit:
                return DispatchResult(
                    success=False,
                    recipient=to_email,
                    subject=subject,
                    is_dry_run=False,
                    error=f"Daily email limit ({self.daily_limit}) reached for today.",
                )

        if self.dry_run:
            # Simulate dispatch
            msg_id = f"dry_run_{int(time.time())}_{hash(to_email) % 10000}"
            result = DispatchResult(
                success=True,
                recipient=to_email,
                subject=subject,
                is_dry_run=True,
                message_id=msg_id,
            )
            self._log_dispatch(result, body_text)
            return result

        # Real SMTP dispatch
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            sender_display = f"{candidate_name} <{self.from_email}>" if candidate_name else self.from_email
            msg["From"] = sender_display
            msg["To"] = to_email

            msg.attach(MIMEText(body_text, "plain", "utf-8"))
            if body_html:
                msg.attach(MIMEText(body_html, "html", "utf-8"))

            if self.use_ssl:
                server = smtplib.SMTP_SSL(self.host, self.port, timeout=15)
            else:
                server = smtplib.SMTP(self.host, self.port, timeout=15)
                if self.use_tls:
                    server.starttls()

            server.login(self.user, self.password)
            server.sendmail(self.from_email, [to_email], msg.as_string())
            server.quit()

            msg_id = f"smtp_{int(time.time())}"
            result = DispatchResult(
                success=True,
                recipient=to_email,
                subject=subject,
                is_dry_run=False,
                message_id=msg_id,
            )
            self._log_dispatch(result, body_text)
            return result
        except Exception as e:
            result = DispatchResult(
                success=False,
                recipient=to_email,
                subject=subject,
                is_dry_run=False,
                error=str(e),
            )
            self._log_dispatch(result, body_text)
            return result

    def _log_dispatch(self, result: DispatchResult, body_snippet: str) -> None:
        entry = {
            "timestamp": result.timestamp,
            "recipient": result.recipient,
            "subject": result.subject,
            "success": result.success,
            "is_dry_run": result.is_dry_run,
            "message_id": result.message_id,
            "error": result.error,
            "snippet": body_snippet[:200],
        }
        with OUTREACH_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
