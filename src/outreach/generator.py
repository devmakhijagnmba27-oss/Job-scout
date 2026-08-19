"""Outreach generator: personalized cold emails + LinkedIn notes via LLM.

Uses the 'cold-email-drafting' skill with structured outputs to generate:
1. Initial cold email (Subject + Body)
2. Follow-up 1 (Value-Add - 3 days later)
3. Follow-up 2 (Polite Breakup / Bump - 7 days later)
4. LinkedIn connection request note (< 300 chars)

SECURITY & PII:
- All candidate inputs are PII-masked ({{CANDIDATE_NAME}}, {{CANDIDATE_EMAIL}}, etc.)
- Job descriptions are UNTRUSTED and enclosed in <job_posting> tags.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from typing import Any

from ..agents import MODEL, load_skill, thinking_kwargs
from ..agents.llm_client import get_llm_client
from ..guardrails import audit

OUTREACH_SCHEMA = {
    "type": "object",
    "properties": {
        "initial_email": {
            "type": "object",
            "properties": {
                "subject": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["subject", "body"],
            "additionalProperties": False,
        },
        "follow_up_1": {
            "type": "object",
            "properties": {
                "subject": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["subject", "body"],
            "additionalProperties": False,
        },
        "follow_up_2": {
            "type": "object",
            "properties": {
                "subject": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["subject", "body"],
            "additionalProperties": False,
        },
        "linkedin_note": {"type": "string"},
        "personalization_hook": {"type": "string"},
    },
    "required": [
        "initial_email",
        "follow_up_1",
        "follow_up_2",
        "linkedin_note",
        "personalization_hook",
    ],
    "additionalProperties": False,
}


def generate_outreach_package(
    skills_profile: str,
    job: dict[str, Any],
    scoring: dict[str, Any],
    contact: dict[str, Any] | None = None,
    communication_style: str = "",
    voice_profile: str = "",
) -> dict[str, Any]:
    """Generate a multi-touch outreach sequence tailored for a specific job & contact."""
    audit("llm.generate_outreach", {"job_id": job.get("id"), "company": job.get("company")})

    client = get_llm_client()
    contact_info = ""
    if contact:
        contact_info = (
            f"RECIPIENT CONTACT:\n"
            f"Name: {contact.get('name') or 'Hiring Team'}\n"
            f"Position: {contact.get('position') or 'Recruiter / Engineering Leader'}\n"
            f"Department: {contact.get('department') or 'Engineering / Talent'}\n\n"
        )

    prompt = (
        f"CANDIDATE SKILLS PROFILE (PII-masked):\n{skills_profile}\n\n"
        f"{contact_info}"
        f"WHY THIS JOB SCORED {scoring.get('score', 'N/A')}/100:\n{scoring.get('summary', '')}\n\n"
        "<job_posting>\n"
        f"Title: {job.get('title')}\nCompany: {job.get('company')}\n"
        f"Location: {job.get('location')} ({job.get('remote')})\n"
        f"Description: {job.get('description', '')[:4000]}\n"
        "</job_posting>"
    )

    system_prompt = load_skill("cold-email-drafting")
    if voice_profile:
        system_prompt += f"\n\nWrite in the candidate's authentic voice style:\n{voice_profile}"
    elif communication_style:
        system_prompt += f"\n\nCommunication style: {communication_style}"

    raw_json = client.generate_structured_json(
        system_prompt=system_prompt,
        user_prompt=prompt,
        schema=OUTREACH_SCHEMA,
        max_tokens=2500,
    )

    today = datetime.now(timezone.utc)
    day_3 = (today + timedelta(days=3)).strftime("%Y-%m-%d")
    day_7 = (today + timedelta(days=7)).strftime("%Y-%m-%d")

    touches = [
        {
            "touch_number": 1,
            "subject": raw_json["initial_email"]["subject"],
            "body": raw_json["initial_email"]["body"],
            "scheduled_date": today.strftime("%Y-%m-%d"),
            "status": "pending",
        },
        {
            "touch_number": 2,
            "subject": raw_json["follow_up_1"]["subject"],
            "body": raw_json["follow_up_1"]["body"],
            "scheduled_date": day_3,
            "status": "pending",
        },
        {
            "touch_number": 3,
            "subject": raw_json["follow_up_2"]["subject"],
            "body": raw_json["follow_up_2"]["body"],
            "scheduled_date": day_7,
            "status": "pending",
        },
    ]

    return {
        "touches": touches,
        "linkedin_note": raw_json["linkedin_note"],
        "personalization_hook": raw_json["personalization_hook"],
    }
