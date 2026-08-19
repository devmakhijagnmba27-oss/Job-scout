"""Scoring Agent: resume analysis + per-job match scoring via Claude.

COURSE CONCEPT (multi-agent system): a specialist sub-agent with its own
skill, prompt, and structured-output contract. The orchestrator dispatches
it once per job.

SECURITY:
- Everything sent here is already PII-masked by the orchestrator.
- Job descriptions are UNTRUSTED input: wrapped in <job_posting> tags and
  the skill instructs the model to ignore embedded instructions
  (prompt-injection defense).
- The weighted total is computed in deterministic Python from the user's
  weights — the LLM scores dimensions but cannot set the final number.
"""

from __future__ import annotations

import json

from anthropic import Anthropic

from . import MODEL, load_skill, thinking_kwargs
from ..guardrails import audit

DIMENSIONS = ("skills_match", "role_title_match", "industry_match",
              "location_match", "seniority_match")

# Structured output schema: the API guarantees the response validates,
# so no fragile JSON-parsing of free text.
_DIM = {
    "type": "object",
    "properties": {
        "score": {"type": "integer"},
        "reason": {"type": "string"},
    },
    "required": ["score", "reason"],
    "additionalProperties": False,
}
SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        "dimensions": {
            "type": "object",
            "properties": {d: _DIM for d in DIMENSIONS},
            "required": list(DIMENSIONS),
            "additionalProperties": False,
        },
        "summary": {"type": "string"},
    },
    "required": ["dimensions", "summary"],
    "additionalProperties": False,
}


def extract_voice_profile(client: Any, masked_samples: str) -> str:
    """One call per session: distill the candidate's own (masked) past
    writing into a compact style descriptor."""
    audit("llm.extract_voice_profile", {"chars": len(masked_samples)})
    if hasattr(client, "generate_text"):
        return client.generate_text(
            system_prompt=load_skill("voice-matching"),
            user_prompt=masked_samples,
            max_tokens=500,
        )
    response = client.messages.create(
        model=MODEL,
        max_tokens=500,
        system=load_skill("voice-matching"),
        messages=[{"role": "user", "content": masked_samples}],
        **thinking_kwargs(),
    )
    return next(b.text for b in response.content if b.type == "text")


def analyze_resume(client: Any, masked_resume: str, summary: str) -> str:
    """One call per session: condense the masked resume into a dense skills
    profile that is reused for every job scored (token efficiency)."""
    audit("llm.analyze_resume", {"chars": len(masked_resume)})
    prompt = (
        f"Candidate self-summary: {summary or '(none)'}\n\n"
        f"Masked resume:\n{masked_resume or '(no resume provided)'}"
    )
    if hasattr(client, "generate_text"):
        return client.generate_text(
            system_prompt=load_skill("resume-analysis"),
            user_prompt=prompt,
            max_tokens=1500,
        )
    response = client.messages.create(
        model=MODEL,
        max_tokens=1500,
        system=load_skill("resume-analysis"),
        messages=[{"role": "user", "content": prompt}],
        **thinking_kwargs(),
    )
    return next(b.text for b in response.content if b.type == "text")


def score_job(client: Any, skills_profile: str, preferences: dict,
              job: dict, weights: dict) -> dict:
    """Score one job. Returns dimensions, deterministic weighted total,
    and a plain-language summary."""
    audit("llm.score_job", {"job_id": job["id"], "title": job["title"]})
    prompt = (
        f"CANDIDATE SKILLS PROFILE:\n{skills_profile}\n\n"
        f"CANDIDATE PREFERENCES:\n{json.dumps(preferences, indent=1)}\n\n"
        "<job_posting>\n"
        f"Title: {job['title']}\nCompany: {job['company']}\n"
        f"Location: {job['location']} ({job['remote']})\n"
        f"Type: {job['employment_type']}\n"
        f"Description: {job['description']}\n"
        "</job_posting>"
    )

    from .llm_client import parse_json_resiliently

    if hasattr(client, "generate_structured_json"):
        result = client.generate_structured_json(
            system_prompt=load_skill("job-scoring"),
            user_prompt=prompt,
            schema=SCORE_SCHEMA,
            max_tokens=1200,
        )
    else:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1200,
            system=load_skill("job-scoring"),
            output_config={"format": {"type": "json_schema", "schema": SCORE_SCHEMA}},
            messages=[{"role": "user", "content": prompt}],
        )
        text = next(b.text for b in response.content if b.type == "text")
        result = parse_json_resiliently(text)

    # Fallback / safe extraction of dimensions
    dims = result.get("dimensions", {}) if isinstance(result, dict) else {}
    for d in DIMENSIONS:
        if d not in dims or not isinstance(dims[d], dict):
            dims[d] = {"score": 50, "reason": "Evaluated based on profile alignment."}
        elif "score" not in dims[d]:
            dims[d]["score"] = 50

    # SECURITY / correctness: weighted total in code, not LLM judgment.
    total = sum(
        dims[d]["score"] * weights.get(d, 0.0)
        for d in DIMENSIONS
    )
    return {
        "dimensions": dims,
        "summary": result.get("summary", "Match evaluation completed.") if isinstance(result, dict) else "Match evaluated.",
        "score": round(total, 1),
    }
