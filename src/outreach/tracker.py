"""Outreach pipeline persistence and CRM tracking.

Stores outreach records, touchpoint sequences, sent logs, and status transitions
(Drafted -> Sent -> Follow-up Due -> Replied -> Interview -> Closed).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "data"
OUTREACH_DB_PATH = DATA_DIR / "outreach_records.json"


@dataclass
class OutreachTouch:
    touch_number: int  # 1 (Initial), 2 (Value Add), 3 (Final Bump)
    subject: str
    body: str
    scheduled_date: str  # ISO date string (YYYY-MM-DD)
    sent_at: str | None = None  # ISO timestamp
    status: str = "pending"  # pending, sent, skipped, failed
    error_message: str | None = None


@dataclass
class OutreachRecord:
    id: str  # e.g. f"{job_id}_{contact_email}"
    job_id: str
    job_title: str
    company: str
    contact_name: str | None
    contact_email: str
    contact_position: str | None
    linkedin_note: str | None
    touches: list[dict[str, Any]] = field(default_factory=list)
    overall_status: str = "drafted"  # drafted, active, replied, interview, declined, archived
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OutreachRecord:
        return cls(**data)


class OutreachTracker:
    def __init__(self, db_path: Path = OUTREACH_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._records: dict[str, OutreachRecord] = {}
        self._load()

    def _load(self) -> None:
        if not self.db_path.exists():
            self._records = {}
            return
        try:
            raw = json.loads(self.db_path.read_text(encoding="utf-8"))
            self._records = {
                k: OutreachRecord.from_dict(v) for k, v in raw.items()
            }
        except Exception:
            self._records = {}

    def _save(self) -> None:
        raw = {k: v.to_dict() for k, v in self._records.items()}
        self.db_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")

    def save_record(self, record: OutreachRecord) -> None:
        record.updated_at = datetime.now(timezone.utc).isoformat()
        self._records[record.id] = record
        self._save()

    def get_record(self, record_id: str) -> OutreachRecord | None:
        return self._records.get(record_id)

    def list_records(self, status: str | None = None) -> list[OutreachRecord]:
        records = list(self._records.values())
        if status:
            records = [r for r in records if r.overall_status == status]
        return sorted(records, key=lambda r: r.updated_at, reverse=True)

    def list_due_followups(self, as_of_date: str | None = None) -> list[tuple[OutreachRecord, int, dict[str, Any]]]:
        """Find touches that are scheduled on or before `as_of_date` and still pending."""
        today = as_of_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        due: list[tuple[OutreachRecord, int, dict[str, Any]]] = []

        for record in self._records.values():
            if record.overall_status not in ("active", "drafted"):
                continue
            for idx, touch in enumerate(record.touches):
                if touch.get("status") == "pending" and touch.get("scheduled_date") <= today:
                    due.append((record, idx, touch))
                    break  # Only next pending touch per record
        return due

    def mark_touch_sent(self, record_id: str, touch_index: int) -> bool:
        record = self.get_record(record_id)
        if not record or touch_index >= len(record.touches):
            return False

        record.touches[touch_index]["status"] = "sent"
        record.touches[touch_index]["sent_at"] = datetime.now(timezone.utc).isoformat()
        record.overall_status = "active"

        # If all touches sent, keep active or await reply
        all_sent = all(t.get("status") in ("sent", "skipped") for t in record.touches)
        if all_sent:
            record.overall_status = "completed_sequence"

        self.save_record(record)
        return True

    def update_status(self, record_id: str, new_status: str, notes: str | None = None) -> bool:
        record = self.get_record(record_id)
        if not record:
            return False
        record.overall_status = new_status
        if notes:
            record.notes = f"{record.notes}\n{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}: {notes}".strip()
        self.save_record(record)
        return True
