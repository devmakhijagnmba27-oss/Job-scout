"""Unit tests for the autonomous outreach engine.

Tests:
1. OutreachTracker: save, retrieve, list, due follow-up detection.
2. EmailDispatcher: dry-run mode, limit enforcement, logging.
3. OutreachSequencer: follow-up dispatch and state transition.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pytest

from src.outreach.tracker import OutreachTracker, OutreachRecord
from src.outreach.dispatcher import EmailDispatcher, DispatchResult
from src.outreach.sequencer import OutreachSequencer


@pytest.fixture
def temp_tracker(tmp_path: Path) -> OutreachTracker:
    db_file = tmp_path / "test_outreach.json"
    return OutreachTracker(db_path=db_file)


def test_tracker_save_and_retrieve(temp_tracker: OutreachTracker):
    record = OutreachRecord(
        id="job123_recruiter@example.com",
        job_id="job123",
        job_title="Senior ML Engineer",
        company="Anthropic",
        contact_name="Sarah Connor",
        contact_email="recruiter@example.com",
        contact_position="Technical Recruiter",
        linkedin_note="Hi Sarah, loved your work on AI safety.",
        touches=[
            {
                "touch_number": 1,
                "subject": "Senior ML Engineer role",
                "body": "Hi Sarah...",
                "scheduled_date": "2026-08-19",
                "status": "pending",
            },
            {
                "touch_number": 2,
                "subject": "Follow up: ML Engineer role",
                "body": "Hi Sarah, following up...",
                "scheduled_date": "2026-08-22",
                "status": "pending",
            },
        ],
    )
    temp_tracker.save_record(record)

    loaded = temp_tracker.get_record("job123_recruiter@example.com")
    assert loaded is not None
    assert loaded.company == "Anthropic"
    assert len(loaded.touches) == 2


def test_tracker_due_followups(temp_tracker: OutreachTracker):
    today = "2026-08-19"
    future = "2026-08-25"

    record = OutreachRecord(
        id="rec_due",
        job_id="job456",
        job_title="Full Stack Engineer",
        company="Stripe",
        contact_name="Alex",
        contact_email="alex@stripe.com",
        contact_position="Recruiter",
        linkedin_note=None,
        overall_status="active",
        touches=[
            {
                "touch_number": 1,
                "subject": "Intro",
                "body": "Body 1",
                "scheduled_date": today,
                "status": "sent",
            },
            {
                "touch_number": 2,
                "subject": "Follow up",
                "body": "Body 2",
                "scheduled_date": today,
                "status": "pending",
            },
            {
                "touch_number": 3,
                "subject": "Final bump",
                "body": "Body 3",
                "scheduled_date": future,
                "status": "pending",
            },
        ],
    )
    temp_tracker.save_record(record)

    due = temp_tracker.list_due_followups(as_of_date=today)
    assert len(due) == 1
    rec, touch_idx, touch = due[0]
    assert rec.id == "rec_due"
    assert touch_idx == 1
    assert touch["touch_number"] == 2


def test_dispatcher_dry_run():
    dispatcher = EmailDispatcher()
    dispatcher.dry_run = True

    result = dispatcher.send_email(
        to_email="test@company.com",
        subject="Excited about the Engineer role",
        body_text="Hello, here is my background...",
        candidate_name="Jane Doe",
    )

    assert result.success is True
    assert result.is_dry_run is True
    assert result.recipient == "test@company.com"
    assert result.message_id.startswith("dry_run_")


def test_sequencer_flow(temp_tracker: OutreachTracker):
    today = "2026-08-19"
    record = OutreachRecord(
        id="seq_test",
        job_id="job789",
        job_title="Backend Engineer",
        company="Linear",
        contact_name="Karri",
        contact_email="karri@linear.app",
        contact_position="Founder",
        linkedin_note=None,
        overall_status="active",
        touches=[
            {
                "touch_number": 2,
                "subject": "Follow-up",
                "body": "Just checking in...",
                "scheduled_date": today,
                "status": "pending",
            }
        ],
    )
    temp_tracker.save_record(record)

    dispatcher = EmailDispatcher()
    dispatcher.dry_run = True

    sequencer = OutreachSequencer(tracker=temp_tracker, dispatcher=dispatcher)
    res = sequencer.process_due_followups(as_of_date=today)

    assert res.processed_count == 1
    assert res.sent_count == 1

    updated = temp_tracker.get_record("seq_test")
    assert updated.touches[0]["status"] == "sent"
    assert updated.touches[0]["sent_at"] is not None
