"""Resume Extractor Agent: Automatically parses a resume and infers optimal
candidate preferences, target job roles, seniority, and industries.
"""

from __future__ import annotations

import json
from typing import Any
from .llm_client import BaseLLMClient


RESUME_EXTRACTION_SYSTEM_PROMPT = """You are an expert executive career advisor and technical recruiter.
Analyze the candidate's resume thoroughly.
Extract the candidate's personal details, infer their career trajectory, and recommend the best-matching target job roles, seniority level, and industry categories for their job hunt.

Guidelines:
1. target_roles: Suggest 4-6 realistic, high-matching job titles that align with their actual experience level, education (e.g. MBA, engineering, etc.), and skills.
2. seniority: Determine whether they are entry, junior, mid, senior, or staff. (e.g. MBA students/recent grads should include ['junior', 'internship', 'mid']).
3. industries: Extract 3-5 relevant industries.
4. summary: Write a concise 2-3 sentence high-impact professional summary highlighting their core competencies and career goals.
5. location: If mentioned or inferred from universities/locations in the resume, format it cleanly (e.g. 'New Delhi, India', 'Delhi NCR, India').
"""

RESUME_EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "email": {"type": "string"},
        "phone": {"type": "string"},
        "location": {"type": "string"},
        "summary": {"type": "string"},
        "target_roles": {
            "type": "array",
            "items": {"type": "string"},
            "description": "4-6 prioritized target job titles matching candidate's profile",
        },
        "seniority": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Appropriate levels: 'internship', 'junior', 'mid', 'senior', 'staff'",
        },
        "industries": {
            "type": "array",
            "items": {"type": "string"},
        },
        "skills": {
            "type": "array",
            "items": {"type": "string"},
        },
        "remote_preference": {
            "type": "string",
            "enum": ["onsite", "hybrid", "remote_only", "any"],
        },
    },
    "required": ["name", "email", "summary", "target_roles", "seniority", "industries"],
}


def extract_profile_from_resume(client: BaseLLMClient, resume_text: str) -> dict[str, Any]:
    """Analyze resume text and return structured profile and recommended target roles."""
    if not resume_text.strip():
        return {}

    user_prompt = f"Candidate Resume:\n{resume_text}"
    return client.generate_structured_json(
        system_prompt=RESUME_EXTRACTION_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        schema=RESUME_EXTRACTION_SCHEMA,
    )
