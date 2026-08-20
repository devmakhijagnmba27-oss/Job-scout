"""Scout FastAPI Serverless Backend (Vercel & Local).

Provides REST endpoints for Scout's AI job-search concierge:
- Profile intake & resume parsing
- Outcome-driven search across job adapters
- Multi-dimension scoring & guardrails
- AI cover letter & resume drafting
- History Vault management
- Outreach sequencer
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Ensure repo root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

load_dotenv(REPO_ROOT / ".env")

from src.agents import MODEL, drafting_agent, insights_agent, scoring_agent
from src.agents.llm_client import get_llm_client
from src.archetype import guess_archetype
from src.contacts import find_contacts
from src.cv_pipeline import generate_cv_pdf
from src.guardrails import PIIMasker, legitimacy_check
from src.insights import aggregate_dimension_gaps
from src.intake import (
    PROFILE_PATH,
    extract_all_saved_documents,
    extract_profile_text,
    extract_writing_samples_text,
    load_profile,
)
from src.memory import Memory
from src.orchestrator import deterministic_filter
from src.outreach import (
    EmailDispatcher,
    OutreachRecord,
    OutreachSequencer,
    OutreachTracker,
    generate_outreach_package,
)
from src.pipeline import MAX_SEARCH_ROUNDS, collect_new_jobs, fetch_jobs
from src.records import Records

app = FastAPI(title="Scout Stark HUD API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

records = Records()
memory = Memory()
outreach_tracker = OutreachTracker()
email_dispatcher = EmailDispatcher()
outreach_sequencer = OutreachSequencer(tracker=outreach_tracker, dispatcher=email_dispatcher)

INDIAN_STATES = [
    "All Locations / Default",
    "Andhra Pradesh",
    "Arunachal Pradesh",
    "Assam",
    "Bihar",
    "Chhattisgarh",
    "Delhi NCR",
    "Goa",
    "Gujarat",
    "Haryana",
    "Himachal Pradesh",
    "Jharkhand",
    "Karnataka",
    "Kerala",
    "Madhya Pradesh",
    "Maharashtra",
    "Manipur",
    "Meghalaya",
    "Mizoram",
    "Nagaland",
    "Odisha",
    "Punjab",
    "Rajasthan",
    "Sikkim",
    "Tamil Nadu",
    "Telangana",
    "Tripura",
    "Uttar Pradesh",
    "Uttarakhand",
    "West Bengal",
]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ProfileUpdateRequest(BaseModel):
    profile: dict[str, Any]


class SearchRequest(BaseModel):
    work_mode: str = "Any Workplace"  # Any Workplace, Remote Only, Hybrid, On-site Only
    selected_state: str = "All Locations / Default"
    target_count: int = 6
    min_matches: int = 0
    override_roles: Optional[list[str]] = None


class DecisionRequest(BaseModel):
    job_id: str
    decision: str  # approved, rejected, skipped, undecided


class DraftRequest(BaseModel):
    job_id: str


class OutreachRequest(BaseModel):
    job_id: str
    recipient_name: Optional[str] = None
    recipient_email: Optional[str] = None


# ---------------------------------------------------------------------------
# Endpoints: System & Profile
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "system": "STARK INDUSTRIES // SCOUT MK-LXXXV",
        "model": MODEL,
        "records_count": len(records.all()),
        "vault_count": memory.approved_count,
    }


@app.get("/api/profile")
def get_profile():
    profile = load_profile()
    if profile is None:
        return {"profile": None}
    return {
        "profile": profile,
        "indian_states": INDIAN_STATES,
        "stats": {
            "reviewed": memory.seen_count,
            "approved": memory.approved_count,
            "campaigns": len(outreach_tracker.list_records()),
            "followups_due": len(outreach_tracker.list_due_followups()),
        }
    }


@app.post("/api/profile")
def update_profile(req: ProfileUpdateRequest):
    import yaml
    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PROFILE_PATH, "w", encoding="utf-8") as f:
        yaml.dump(req.profile, f, sort_keys=False, allow_unicode=True)
    return {"status": "saved", "profile": req.profile}


@app.post("/api/profile/extract-resume")
async def extract_resume(file: UploadFile = File(...)):
    try:
        from pypdf import PdfReader
        from src.agents.resume_extractor import extract_profile_from_resume

        resume_dir = REPO_ROOT / "profile"
        resume_dir.mkdir(parents=True, exist_ok=True)
        resume_pdf_path = resume_dir / "resume.pdf"

        contents = await file.read()
        resume_pdf_path.write_bytes(contents)

        reader = PdfReader(str(resume_pdf_path))
        raw_text = "\n".join(page.extract_text() or "" for page in reader.pages)

        if not raw_text.strip():
            raise HTTPException(status_code=400, detail="Uploaded PDF has no readable text.")

        client = get_llm_client()
        extracted = extract_profile_from_resume(client, raw_text)
        return {"status": "success", "extracted": extracted}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Endpoints: Search, Scoring & Drafting
# ---------------------------------------------------------------------------

@app.post("/api/search")
def run_search(req: SearchRequest):
    profile = load_profile()
    if not profile:
        raise HTTPException(status_code=400, detail="No profile found. Please set up your profile first.")

    import copy
    run_profile = copy.deepcopy(profile)
    run_prefs = run_profile.setdefault("preferences", {})

    if req.override_roles:
        run_prefs["target_roles"] = req.override_roles

    if req.work_mode == "Remote Only":
        run_prefs["remote_preference"] = "remote_only"
    elif req.work_mode == "Hybrid":
        run_prefs["remote_preference"] = "hybrid"
    elif req.work_mode == "On-site Only":
        run_prefs["remote_preference"] = "onsite"
    else:
        run_prefs["remote_preference"] = "any"

    if req.selected_state and req.selected_state != "All Locations / Default":
        run_prefs["locations"] = [f"{req.selected_state}, India", req.selected_state]

    total_dropped: dict[str, int] = {}

    def _keep(fresh: list[dict]) -> list[dict]:
        if req.selected_state and req.selected_state != "All Locations / Default":
            state_lc = req.selected_state.lower()
            def _loc_ok(j: dict) -> bool:
                loc = j.get("location", "").lower()
                if j.get("remote") == "remote" or "remote" in loc or "worldwide" in loc:
                    return True
                return state_lc in loc or "india" in loc
            fresh = [j for j in fresh if _loc_ok(j)]

        if req.work_mode == "Remote Only":
            fresh = [j for j in fresh if j.get("remote") == "remote" or "remote" in j.get("location", "").lower()]
        elif req.work_mode == "On-site Only":
            fresh = [j for j in fresh if j.get("remote") != "remote"]

        kept, dropped = deterministic_filter(fresh, run_profile, memory)
        for k, v in dropped.items():
            total_dropped[k] = total_dropped.get(k, 0) + v
        return kept

    target = max(1, req.target_count)
    jobs = collect_new_jobs(lambda p: fetch_jobs(run_profile, page=p), _keep, target=target)

    # Score jobs
    scored_packages = []
    archetypes_cfg = profile.get("archetypes")
    past_entries = records.all()
    client = get_llm_client()
    resume_text = extract_profile_text(PROFILE_PATH)
    weights = profile.get("weights", {})
    threshold = int(profile.get("draft_threshold", 70))

    cand = profile.get("candidate", {})
    masker = PIIMasker(
        name=cand.get("name", ""),
        email=cand.get("email", ""),
        phone=cand.get("phone", ""),
        address=cand.get("address", ""),
    )

    for job in jobs:
        job["archetype"] = guess_archetype(f"{job['title']} {job['description']}", archetypes_cfg)
        job["legitimacy"] = legitimacy_check(job, past_entries)

        scoring = scoring_agent.score_job(
            client=client,
            job=job,
            profile_text=resume_text,
            weights=weights,
            masker=masker,
        )
        package = {
            "job": job,
            "score": scoring.get("score", 0),
            "dimensions": scoring.get("dimensions", {}),
            "summary": scoring.get("summary", ""),
            "decision": None,
        }
        records.upsert(job, scoring=scoring)
        scored_packages.append(package)

    scored_packages.sort(key=lambda p: p["score"], reverse=True)

    return {
        "count": len(scored_packages),
        "jobs": scored_packages,
        "dropped_summary": total_dropped,
    }


@app.post("/api/draft")
def draft_cover_letter(req: DraftRequest):
    record = records.get(req.job_id)
    if not record:
        raise HTTPException(status_code=404, detail="Job record not found.")

    profile = load_profile() or {}
    cand = profile.get("candidate", {})
    masker = PIIMasker(
        name=cand.get("name", ""),
        email=cand.get("email", ""),
        phone=cand.get("phone", ""),
        address=cand.get("address", ""),
    )
    client = get_llm_client()
    resume_text = extract_profile_text(PROFILE_PATH)
    job = record["job"]

    drafts = drafting_agent.draft_application_materials(
        client=client,
        job=job,
        resume_text=resume_text,
        skills_profile="",
        masker=masker,
    )
    records.upsert(job, drafts=drafts)
    return {"status": "success", "drafts": drafts}


# ---------------------------------------------------------------------------
# Endpoints: Vault & History
# ---------------------------------------------------------------------------

@app.get("/api/vault")
def get_vault():
    all_records = records.all()
    approved = [e for e in all_records if e.get("decision") == "approved"]
    return {
        "count": len(approved),
        "items": approved,
    }


@app.post("/api/vault/decide")
def update_decision(req: DecisionRequest):
    record = records.get(req.job_id)
    if not record:
        raise HTTPException(status_code=404, detail="Job record not found.")

    job = record["job"]
    records.upsert(job, decision=req.decision)
    memory.mark_seen(job["id"], job["title"], req.decision)
    return {
        "status": "updated",
        "job_id": req.job_id,
        "decision": req.decision,
        "stats": {
            "reviewed": memory.seen_count,
            "approved": memory.approved_count,
        }
    }


@app.get("/api/history")
def get_history():
    all_records = records.all()
    gaps = aggregate_dimension_gaps(all_records) if all_records else []
    return {
        "total": len(all_records),
        "records": all_records,
        "gaps": gaps,
        "stats": {
            "reviewed": memory.seen_count,
            "approved": memory.approved_count,
        }
    }


# ---------------------------------------------------------------------------
# Endpoints: Outreach CRM
# ---------------------------------------------------------------------------

@app.get("/api/outreach")
def list_outreach():
    campaigns = outreach_tracker.list_records()
    due = outreach_tracker.list_due_followups()
    return {
        "campaigns": [c.model_dump() for c in campaigns],
        "due_followups": [d.model_dump() for d in due],
    }


@app.post("/api/outreach/generate")
def create_outreach(req: OutreachRequest):
    record = records.get(req.job_id)
    if not record:
        raise HTTPException(status_code=404, detail="Job record not found.")

    profile = load_profile() or {}
    cand = profile.get("candidate", {})
    masker = PIIMasker(
        name=cand.get("name", ""),
        email=cand.get("email", ""),
        phone=cand.get("phone", ""),
        address=cand.get("address", ""),
    )
    client = get_llm_client()
    resume_text = extract_profile_text(PROFILE_PATH)

    package = generate_outreach_package(
        client=client,
        job=record["job"],
        score_package=record,
        resume_text=resume_text,
        skills_profile="",
        masker=masker,
        recipient_name=req.recipient_name or "",
        recipient_email=req.recipient_email or "",
    )
    return {"status": "success", "package": package.model_dump()}


# ---------------------------------------------------------------------------
# Static Web Frontend Mounting (Local & fallback)
# ---------------------------------------------------------------------------

PUBLIC_DIR = REPO_ROOT / "public"
if PUBLIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(PUBLIC_DIR), html=True), name="public")
