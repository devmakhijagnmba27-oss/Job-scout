"""Outreach Sequencer: automated follow-up processing.

Identifies due follow-up touches across all active campaigns and dispatches them
through the EmailDispatcher while enforcing daily rate limits and audit logs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .dispatcher import EmailDispatcher, DispatchResult
from .tracker import OutreachTracker, OutreachRecord
from ..guardrails import audit


@dataclass
class SequenceRunResult:
    processed_count: int
    sent_count: int
    failed_count: int
    details: list[dict[str, Any]]


class OutreachSequencer:
    def __init__(self, tracker: OutreachTracker | None = None, dispatcher: EmailDispatcher | None = None):
        self.tracker = tracker or OutreachTracker()
        self.dispatcher = dispatcher or EmailDispatcher()

    def process_due_followups(self, as_of_date: str | None = None) -> SequenceRunResult:
        """Scan and dispatch all due follow-up touches."""
        due_items = self.tracker.list_due_followups(as_of_date=as_of_date)
        sent_count = 0
        failed_count = 0
        details = []

        audit("sequencer.process_due_followups", {"due_count": len(due_items)})

        for record, touch_idx, touch in due_items:
            # Don't auto-send Touch 1 (Touch 1 should be explicitly triggered via UI or Auto-Pilot)
            # Touches > 1 are follow-ups to an already started thread
            res: DispatchResult = self.dispatcher.send_email(
                to_email=record.contact_email,
                subject=touch.get("subject", f"Follow up: {record.job_title} at {record.company}"),
                body_text=touch.get("body", ""),
            )

            if res.success:
                self.tracker.mark_touch_sent(record.id, touch_idx)
                sent_count += 1
            else:
                failed_count += 1
                touch["status"] = "failed"
                touch["error_message"] = res.error
                self.tracker.save_record(record)

            details.append({
                "record_id": record.id,
                "touch_number": touch.get("touch_number"),
                "recipient": record.contact_email,
                "success": res.success,
                "is_dry_run": res.is_dry_run,
                "error": res.error,
            })

        return SequenceRunResult(
            processed_count=len(due_items),
            sent_count=sent_count,
            failed_count=failed_count,
            details=details,
        )
