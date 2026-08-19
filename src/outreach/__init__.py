"""Outreach engine package initialization.

Provides cold email generation, SMTP/Gmail dispatching, follow-up sequencing,
and persistent CRM tracking for JobScout.
"""

from .generator import generate_outreach_package, OUTREACH_SCHEMA
from .dispatcher import EmailDispatcher, DispatchResult
from .tracker import OutreachTracker, OutreachRecord
from .sequencer import OutreachSequencer

__all__ = [
    "generate_outreach_package",
    "OUTREACH_SCHEMA",
    "EmailDispatcher",
    "DispatchResult",
    "OutreachTracker",
    "OutreachRecord",
    "OutreachSequencer",
]
