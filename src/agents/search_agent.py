"""Search Agent: turns profile preferences into MCP search_jobs calls.

COURSE CONCEPT (tool use via MCP): this agent is the only component that
talks to the job-search MCP server. Query construction is deterministic —
no LLM needed to translate structured preferences into a structured query,
so we don't spend tokens where code suffices.
"""

from __future__ import annotations

import json

from mcp import ClientSession

from ..guardrails import audit
from ..query_expansion import expand_keywords


def build_query(profile: dict, page: int = 1) -> dict:
    prefs = profile.get("preferences", {})
    roles = prefs.get("target_roles", [])
    # Broaden by default so exact-phrase role titles don't starve recall
    # (see src/query_expansion.py); precision is the scorer's job.
    # Resume skills/details are parsed for profile scoring, not used to starve job board queries.
    keywords = (roles if prefs.get("strict_keyword_match")
                else expand_keywords(roles, prefs.get("employment_types")))
    locs = prefs.get("locations", [])
    clean_locs = []
    indian_cities = ("delhi", "ncr", "gurgaon", "gurugram", "noida", "bangalore", "bengaluru", "mumbai", "pune", "hyderabad", "chennai", "kolkata")
    for l in locs:
        if not l or not str(l).strip():
            continue
        cleaned = str(l).strip()
        if any(k in cleaned.lower() for k in indian_cities) and "india" not in cleaned.lower():
            cleaned = f"{cleaned}, India"
        if cleaned not in clean_locs:
            clean_locs.append(cleaned)

    return {
        "keywords": keywords,
        "locations": clean_locs,
        "remote_only": prefs.get("remote_preference") == "remote_only",
        "limit_per_source": 12,  # capped at 12 per board for faster results
        "page": page,
    }


async def search(session: ClientSession, profile: dict,
                 page: int = 1) -> list[dict]:
    """Call the MCP search_jobs tool and parse the normalized job list.
    `page` is the outcome-driven search round — deeper rounds reach
    fresh inventory (keyword rotation / real pagination per adapter)."""
    query = build_query(profile, page=page)
    audit("mcp.search_jobs", query)
    result = await session.call_tool("search_jobs", query)
    jobs: list[dict] = []
    for content in result.content:
        if content.type != "text":
            continue
        parsed = json.loads(content.text)
        jobs.extend(parsed if isinstance(parsed, list) else [parsed])
    return jobs
