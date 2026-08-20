"""Scout — AI job-search concierge. Profile intake, agent runs, vault, and history.

Run:  streamlit run app.py

Same pipeline as the CLI (python -m src.orchestrator): the UI reuses the
identical MCP server, guardrails, and sub-agents, so both surfaces behave
the same. The HITL gate here is the Approve / Reject / Skip buttons —
Scout has no code path that submits an application anywhere.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import pandas as pd
import streamlit as st
import yaml
from dotenv import load_dotenv

load_dotenv(override=True)

# Sync Streamlit Cloud secrets to os.environ so all sub-processes/adapters have access
try:
    if hasattr(st, "secrets"):
        for k in st.secrets:
            val = st.secrets[k]
            if isinstance(val, (str, int, float, bool)):
                os.environ[k] = str(val)
except Exception:
    pass

from src.agents import MODEL, drafting_agent, insights_agent, scoring_agent
from src.archetype import guess_archetype
from src.contacts import find_contacts
from src.contacts import available as hunter_available
from src.cv_pipeline import generate_cv_pdf
from src.guardrails import PIIMasker, legitimacy_check
from src.insights import aggregate_dimension_gaps
from src.keyword_coverage import keyword_coverage
from src.intake import (DOCUMENTS_DIR, PROFILE_PATH, WRITING_SAMPLES_DIR,
                        extract_all_saved_documents, extract_profile_text,
                        extract_writing_samples_text, list_profile_documents,
                        list_writing_samples, load_profile)
from src.memory import Memory
from src import notion_sync
from src.orchestrator import (_explain_empty_filter, _relevance_rank,
                              deterministic_filter)
from src.pipeline import (MAX_SEARCH_ROUNDS, collect_new_jobs, fetch_jobs,
                          verify_liveness)
from src.records import Records
from src.outreach import (
    OutreachTracker,
    OutreachRecord,
    EmailDispatcher,
    OutreachSequencer,
    generate_outreach_package,
)

REPO_ROOT = Path(__file__).resolve().parent
CV_OUTPUT_DIR = REPO_ROOT / "output" / "cvs"
load_dotenv(REPO_ROOT / ".env")

st.set_page_config(page_title="Scout", page_icon="🔭", layout="wide")

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

ALL_SOURCES = ["remoteok", "themuse", "remotive", "arbeitnow", "greenhouse",
               "lever", "ashby", "linkedin", "jsearch", "adzuna", "usajobs"]
REMOTE_PREFS = ["remote_or_hybrid", "remote_only", "hybrid", "onsite", "any"]
EMPLOYMENT_TYPES = ["full-time", "part-time", "internship", "contract"]
SENIORITY = ["junior", "mid", "senior", "staff"]
WEIGHT_DIMS = ["skills_match", "role_title_match", "industry_match",
               "location_match", "seniority_match"]

records = Records()
memory = Memory()
outreach_tracker = OutreachTracker()
email_dispatcher = EmailDispatcher()
outreach_sequencer = OutreachSequencer(tracker=outreach_tracker, dispatcher=email_dispatcher)


def _csv(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def score_badge(score: float, threshold: int) -> str:
    dot = "🟢" if score >= threshold else ("🟡" if score >= 50 else "🔴")
    return f"{dot} {score:.0f}"


_LEGITIMACY_ICON = {"suspicious": "🚩", "caution": "⚠️"}


def _legitimacy_tag(job: dict) -> str:
    """Icon for the card label — silent for high_confidence so the common
    case doesn't clutter every label with a checkmark."""
    tier = (job.get("legitimacy") or {}).get("tier")
    icon = _LEGITIMACY_ICON.get(tier)
    return f"{icon} {tier}  " if icon else ""


def _legitimacy_section(job: dict) -> None:
    """Block G detail: only rendered when there's actually something to
    say — high_confidence (the common case) shows nothing here at all."""
    legitimacy = job.get("legitimacy") or {}
    tier = legitimacy.get("tier")
    if tier not in ("caution", "suspicious"):
        return
    icon = _LEGITIMACY_ICON.get(tier, "")
    box = st.error if tier == "suspicious" else st.warning
    reasons = "\n".join(f"- {r}" for r in legitimacy.get("reasons", []))
    box(f"{icon} **Legitimacy check: {tier}** — a heuristic, not "
       f"certainty; use your own judgment.\n{reasons}")


def _source_label(job: dict) -> str:
    """Aggregator sources (jsearch) carry a distinct origin board in
    `publisher` — show it so Glassdoor/Indeed listings are identifiable."""
    src = job.get("source", "")
    pub = job.get("publisher")
    return f"{src} ({pub})" if pub else src


def _cv_download_button(job: dict, record: dict, key: str) -> None:
    """Reads the tailored CV back off disk for the download button — the
    PDF itself isn't kept in Records (large binary; a path is enough)."""
    path = record.get("cv_pdf_path")
    if not path:
        return
    try:
        pdf_bytes = Path(path).read_bytes()
    except FileNotFoundError:
        st.caption("📎 Tailored CV was generated but the file is missing "
                  "on disk now (moved or cleaned up).")
        return
    file_name = f"CV_{job.get('company', 'role')}.pdf".replace(" ", "_")
    st.download_button("📎 Download tailored ATS CV (PDF)", data=pdf_bytes,
                       file_name=file_name, mime="application/pdf", key=key)


def _contacts_section(job: dict, record: dict, package: dict | None = None,
                      masker: PIIMasker | None = None, skills_profile: str = "",
                      profile: dict | None = None, communication_style: str = "",
                      voice_profile: str = "", key: str = "contacts") -> None:
    """Recruiter/HR contact lookup & Autonomous Outreach Campaign center."""
    st.markdown("### 📬 Recruiter Discovery & Cold Outreach")

    contacts = record.get("contacts", [])
    
    # 1. Contact Discovery Button / Status
    if not contacts and hunter_available():
        if st.button("🔍 Find Recruiter Contacts at " + job.get("company", "Company"), key=f"find_{key}",
                     help="Searches Hunter.io for hiring managers, talent acquisition, and technical recruiters."):
            with st.spinner("Searching Hunter.io for contacts…"):
                found = find_contacts(job.get("company", ""), job.get("title", ""))
            records.upsert(job, contacts=found)
            st.rerun()

    # Manual Contact Fallback
    with st.expander("➕ Add / Manage Recruiter Contact manually", expanded=not contacts):
        c1, c2, c3 = st.columns(3)
        man_name = c1.text_input("Contact Name", key=f"man_name_{key}")
        man_email = c2.text_input("Contact Email", key=f"man_email_{key}")
        man_role = c3.text_input("Position / Title", key=f"man_role_{key}", placeholder="e.g. Engineering Manager")
        if st.button("Save Contact", key=f"save_man_{key}"):
            if man_email:
                new_c = {
                    "name": man_name or "Hiring Team",
                    "position": man_role or "Recruiter / Team Lead",
                    "email": man_email.strip(),
                    "confidence": 100,
                    "sources": ["manual_entry"],
                }
                contacts.append(new_c)
                records.upsert(job, contacts=contacts)
                st.success(f"Added {man_email} to contacts.")
                st.rerun()
            else:
                st.error("Contact email is required.")

    if not contacts:
        st.caption("No recruiter contacts linked yet. Search above or enter one manually to draft personalized cold outreach.")
        return

    st.markdown("**Discovered Contacts:**")
    for i, c in enumerate(contacts):
        c_label = c.get("name") or "(name unknown)"
        c_role = f" — {c.get('position')}" if c.get("position") else ""
        c_conf = f" · confidence {c.get('confidence')}%" if c.get('confidence') else ""
        c_email = c.get("email", "")
        
        st.markdown(f"👤 **{c_label}**{c_role} · `{c_email}`{c_conf}")
        
        # Check if an outreach record already exists in OutreachTracker
        outreach_id = f"{job['id']}_{c_email}"
        existing_outreach = outreach_tracker.get_record(outreach_id)
        
        col_gen, col_status = st.columns([2, 3])
        
        with col_gen:
            gen_label = "🔄 Re-generate Outreach Sequence" if existing_outreach else "⚡ Draft Cold Outreach Sequence"
            if st.button(gen_label, key=f"gen_out_{outreach_id}", type="secondary" if existing_outreach else "primary"):
                if not package or not masker:
                    st.error("Package context required. Run search & score first.")
                else:
                    with st.spinner(f"Generating hyper-personalized 3-step outreach for {c_label}…"):
                        # PII masked generation
                        res = generate_outreach_package(
                            skills_profile=skills_profile,
                            job=job,
                            scoring=package,
                            contact=c,
                            communication_style=communication_style,
                            voice_profile=voice_profile,
                        )
                        # Reinject candidate PII locally
                        cand_info = (profile or {}).get("candidate", {})
                        cand_name = cand_info.get("name", "Candidate")
                        cand_email = cand_info.get("email", "")
                        
                        unmasked_touches = []
                        for t in res["touches"]:
                            unmasked_touches.append({
                                "touch_number": t["touch_number"],
                                "subject": masker.unmask(t["subject"]),
                                "body": masker.unmask(t["body"]),
                                "scheduled_date": t["scheduled_date"],
                                "status": t["status"],
                            })
                        
                        outreach_rec = OutreachRecord(
                            id=outreach_id,
                            job_id=job["id"],
                            job_title=job["title"],
                            company=job["company"],
                            contact_name=c.get("name"),
                            contact_email=c_email,
                            contact_position=c.get("position"),
                            linkedin_note=masker.unmask(res.get("linkedin_note", "")),
                            touches=unmasked_touches,
                            overall_status="drafted",
                        )
                        outreach_tracker.save_record(outreach_rec)
                        st.success("✅ Outreach sequence generated and saved to Outreach CRM!")
                        st.rerun()

        # Display drafted campaign if exists
        if existing_outreach:
            with st.expander(f"✉️ Outreach Campaign for {c_label} ({existing_outreach.overall_status.upper()})", expanded=True):
                st.caption(f"Status: `{existing_outreach.overall_status}` · Recipient: `{c_email}`")
                
                # LinkedIn Note
                if existing_outreach.linkedin_note:
                    st.markdown("**🔗 LinkedIn Connection Note (<300 chars):**")
                    st.code(existing_outreach.linkedin_note, language="text")
                
                # Touchpoints Tabs
                t_labels = [f"Touch {t['touch_number']} ({t['status']})" for t in existing_outreach.touches]
                if t_labels:
                    t_tabs = st.tabs(t_labels)
                    for t_idx, (t_tab, touch) in enumerate(zip(t_tabs, existing_outreach.touches)):
                        with t_tab:
                            st.caption(f"Scheduled: `{touch.get('scheduled_date')}` · Status: `{touch.get('status')}`")
                            touch["subject"] = st.text_input(
                                "Subject", touch.get("subject", ""),
                                key=f"subj_{outreach_id}_{t_idx}"
                            )
                            touch["body"] = st.text_area(
                                "Email Body", touch.get("body", ""),
                                height=200, key=f"body_{outreach_id}_{t_idx}"
                            )
                            
                            c_send, c_save = st.columns([2, 1])
                            if c_save.button("💾 Save Edits", key=f"save_touch_{outreach_id}_{t_idx}"):
                                outreach_tracker.save_record(existing_outreach)
                                st.success("Saved edits.")
                                st.rerun()

                            send_disabled = touch.get("status") == "sent"
                            send_btn_label = "✅ Sent" if send_disabled else f"🚀 Send Touch {touch['touch_number']} Now"
                            
                            if c_send.button(send_btn_label, key=f"send_touch_{outreach_id}_{t_idx}", disabled=send_disabled, type="primary"):
                                cand_info = (profile or {}).get("candidate", {})
                                dispatch_res = email_dispatcher.send_email(
                                    to_email=c_email,
                                    subject=touch["subject"],
                                    body_text=touch["body"],
                                    candidate_name=cand_info.get("name"),
                                )
                                if dispatch_res.success:
                                    outreach_tracker.mark_touch_sent(outreach_id, t_idx)
                                    mode_text = "Simulated (Safe Dry Run Mode - set OUTREACH_DRY_RUN=false with SMTP to send live)" if dispatch_res.is_dry_run else "Live via SMTP"
                                    st.success(f"🎉 Touch {touch['touch_number']} sent! ({mode_text})")
                                    st.rerun()
                                else:
                                    st.error(f"❌ Failed to send: {dispatch_res.error}")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _goto_history(focus: str | None = None) -> None:
    """Jump to Vault (approved) or History page depending on focus flag."""
    if focus == "approved":
        st.session_state["nav"] = "🏦 Vault"
    else:
        st.session_state["nav"] = "📚 History"
        if focus:
            st.session_state["history_focus"] = focus


def _clickable_metric(label: str, value: int, name: str,
                      focus: str | None, hint: str) -> None:
    """A stat that IS its own link into History. st.metric takes no
    on_click, so a real (accessible, keyboard-reachable) button is laid
    over it and made invisible — the metric stays a metric, while the whole
    card becomes the click target."""
    with st.container(key=f"stat_{name}"):
        st.metric(label, value)
        st.button(hint, key=f"goto_{name}", on_click=_goto_history,
                  args=(focus,))


with st.sidebar:
    st.html("""
        <div style="
            background: linear-gradient(160deg, #180a0e 0%, #0d0812 50%, #07060a 100%);
            margin: -1rem -1rem 1rem -1rem;
            padding: 1.5rem 1rem 1.2rem 1rem;
            border-bottom: 2px solid #e62429;
            box-shadow: 0 4px 25px rgba(230,36,41,0.35), inset 0 1px 0 rgba(255,215,0,0.3);
            position: relative; overflow: hidden;
        ">
            <!-- HUD grid line overlay -->
            <div style="
                position:absolute; inset:0;
                background: linear-gradient(rgba(230,36,41,0.05) 1px, transparent 1px),
                            linear-gradient(90deg, rgba(230,36,41,0.05) 1px, transparent 1px);
                background-size: 16px 16px;
                pointer-events:none;
            "></div>

            <!-- Glowing Arc Reactor Logo -->
            <div style="display:flex; align-items:center; gap:0.75rem;">
                <div style="
                    width: 40px; height: 40px; border-radius: 50%;
                    background: radial-gradient(circle, #00f0ff 15%, #0088cc 45%, #051a2e 70%, #ff1a22 100%);
                    border: 2px solid #00f0ff;
                    box-shadow: 0 0 16px rgba(0,240,255,0.8), 0 0 30px rgba(0,240,255,0.4), inset 0 0 8px #ffffff;
                    display:flex; align-items:center; justify-content:center;
                    font-size: 1.2rem;
                    animation: arcPulse 3s ease-in-out infinite alternate;
                ">⚛️</div>
                <div>
                    <div style="
                        font-family: 'Orbitron', monospace;
                        font-size: 1.55rem;
                        font-weight: 900;
                        letter-spacing: 0.14em;
                        background: linear-gradient(90deg, #ff2a2a 0%, #ffd700 60%, #ff8c00 100%);
                        -webkit-background-clip: text;
                        -webkit-text-fill-color: transparent;
                        filter: drop-shadow(0 0 10px rgba(255,42,42,0.5));
                        line-height: 1.1;
                    ">SCOUT</div>
                    <div style="
                        font-family: 'Rajdhani', sans-serif;
                        font-size: 0.72rem;
                        font-weight: 700;
                        color: #ff8c00;
                        letter-spacing: 0.16em;
                        text-transform: uppercase;
                    ">STARK INDUSTRIES // HUD</div>
                </div>
            </div>

            <!-- Telemetry status -->
            <div style="
                display: flex; align-items: center; justify-content: space-between;
                margin-top: 0.8rem; padding-top: 0.6rem;
                border-top: 1px dashed rgba(255,42,42,0.25);
                font-family: 'Rajdhani', sans-serif;
                font-size: 0.7rem; font-weight: 600;
                color: #8fa0c0;
            ">
                <span style="color:#00f0ff; text-shadow:0 0 8px rgba(0,240,255,0.6);">● ARC: 100% ONLINE</span>
                <span style="color:#ffd700; font-family:'Orbitron',monospace; font-size:0.62rem;">MK-LXXXV</span>
            </div>
        </div>
    """)
    page = st.radio("Navigate",
                    ["👤 Profile", "🚀 Search Jobs", "🏦 Vault", "📚 History", "📬 Outreach"],
                    label_visibility="collapsed", key="nav")
    st.divider()
    # Reviewed spans every decision, so it opens History as-is rather than
    # forcing one tab; Vault handles Approved directly.
    _clickable_metric("Jobs reviewed", memory.seen_count, "reviewed",
                      None, "Show all reviewed jobs")
    _clickable_metric("In Vault", memory.approved_count, "approved",
                      "approved", "Open Vault")
    st.metric("Outreach Campaigns", len(outreach_tracker.list_records()))
    st.metric("Follow-ups Due", len(outreach_tracker.list_due_followups()))
    st.html("""
        <style>
        [class*="st-key-stat_"] { position: relative; }
        [class*="st-key-stat_"]:hover [data-testid="stMetricValue"] {
            text-decoration: underline;
        }
        [class*="st-key-stat_"] [class*="st-key-goto_"] {
            position: absolute; inset: 0; z-index: 2;
        }
        [class*="st-key-stat_"] [class*="st-key-goto_"] button {
            width: 100%; height: 100%; opacity: 0; cursor: pointer;
        }
        </style>""")
    st.caption(f"Model: `{MODEL}`")
    st.caption("Audit: `logs/audit.jsonl`")



# ---------------------------------------------------------------------------
# Global CSS — Premium Scout dark design system
# ---------------------------------------------------------------------------
st.html("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@600;700;800;900&family=Rajdhani:wght@500;600;700&family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">

<style>
/* ═══════════════════════════════════════════════════════════════
   SCOUT — STARK INDUSTRIES / IRON MAN HUD THEME
   ═══════════════════════════════════════════════════════════════ */

@keyframes arcPulse {
    0%   { box-shadow: 0 0 14px rgba(0,240,255,0.8), 0 0 24px rgba(0,240,255,0.4); transform: scale(1); }
    100% { box-shadow: 0 0 22px rgba(0,240,255,1), 0 0 45px rgba(0,240,255,0.7); transform: scale(1.05); }
}

@keyframes laserGlow {
    0%   { background-position: 0% 50%; }
    100% { background-position: 200% 50%; }
}

/* ── Global Typography ── */
html, body, .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Preserve Material Icons for expanders & UI controls */
[data-testid*="Icon"], [data-testid="stExpanderIcon"], .material-symbols-rounded, .material-symbols-outlined, [class*="material-symbols"], span[translate="no"] {
    font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons' !important;
}

/* ── Top Header Bar ── */
header[data-testid="stHeader"] {
    background: rgba(7, 6, 10, 0.9) !important;
    backdrop-filter: blur(12px) !important;
    border-bottom: 1px solid rgba(230, 36, 41, 0.4) !important;
    box-shadow: 0 2px 20px rgba(230, 36, 41, 0.15) !important;
}

/* ── Stark Armor HUD Background ── */
.stApp {
    background:
        radial-gradient(ellipse 90% 60% at 50% 0%, rgba(230,36,41,0.09) 0%, transparent 65%),
        radial-gradient(ellipse 70% 50% at 85% 90%, rgba(0,240,255,0.04) 0%, transparent 55%),
        linear-gradient(165deg, #07060a 0%, #0d0912 40%, #09060b 80%, #040306 100%) !important;
    background-attachment: fixed !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d070b 0%, #0e0a14 45%, #060408 100%) !important;
    border-right: 2px solid rgba(230, 36, 41, 0.45) !important;
    box-shadow: 6px 0 35px rgba(230, 36, 41, 0.12) !important;
}
[data-testid="stSidebarNav"] { display: none; }

/* ── Navigation Radio Items ── */
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    display: block !important;
    padding: 0.55rem 0.9rem !important;
    border-radius: 6px !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    color: #a4adc2 !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    border: 1px solid transparent !important;
    margin-bottom: 3px !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: rgba(230, 36, 41, 0.12) !important;
    color: #ffd700 !important;
    border-color: rgba(230, 36, 41, 0.3) !important;
    box-shadow: 0 0 12px rgba(230, 36, 41, 0.15) !important;
    transform: translateX(3px) !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb] {
    color: #ffffff !important;
    background: linear-gradient(90deg, rgba(230,36,41,0.25) 0%, rgba(255,140,0,0.1) 100%) !important;
    border-left: 3px solid #ff2a2a !important;
    border-color: rgba(230,36,41,0.5) !important;
    font-weight: 700 !important;
    box-shadow: 0 0 16px rgba(230,36,41,0.25) !important;
}

/* ── Headers (Orbitron HUD style) ── */
.stApp h1 {
    font-family: 'Orbitron', monospace !important;
    font-weight: 900 !important;
    letter-spacing: 0.06em !important;
    background: linear-gradient(90deg, #ff2a2a 0%, #ffd700 50%, #ff8c00 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    filter: drop-shadow(0 0 16px rgba(255, 42, 42, 0.45));
    text-transform: uppercase !important;
}
.stApp h2, .stApp h3 {
    font-family: 'Orbitron', monospace !important;
    font-weight: 700 !important;
    letter-spacing: 0.04em !important;
    color: #ff9900 !important;
    text-shadow: 0 0 12px rgba(255, 153, 0, 0.4);
    text-transform: uppercase !important;
}

/* ── Laser Dividers ── */
hr {
    border: none !important;
    height: 2px !important;
    background: linear-gradient(90deg, transparent, #ff1a22 25%, #ffd700 50%, #ff1a22 75%, transparent) !important;
    box-shadow: 0 0 15px rgba(255, 26, 34, 0.8), 0 0 30px rgba(255, 100, 0, 0.4) !important;
    margin: 1.8rem 0 !important;
}

/* ── Primary Action Button (Hot-Rod Crimson & Gold Armor) ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #c41217 0%, #e62429 45%, #ff4d4d 70%, #d4001a 100%) !important;
    border: 1px solid rgba(255, 215, 0, 0.6) !important;
    color: #ffffff !important;
    font-family: 'Orbitron', monospace !important;
    font-weight: 800 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    font-size: 0.82rem !important;
    border-radius: 6px !important;
    padding: 0.55rem 1.4rem !important;
    box-shadow: 0 0 20px rgba(230, 36, 41, 0.5), inset 0 1px 0 rgba(255,255,255,0.4) !important;
    transition: all 0.2s ease !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) scale(1.02) !important;
    background: linear-gradient(135deg, #e62429 0%, #ff2a2a 50%, #ffd700 100%) !important;
    box-shadow: 0 0 30px rgba(230, 36, 41, 0.85), 0 0 50px rgba(255, 215, 0, 0.5) !important;
    border-color: #ffd700 !important;
    color: #0d0812 !important;
}

/* ── Secondary Buttons ── */
.stButton > button[kind="secondary"] {
    background: rgba(18, 14, 24, 0.8) !important;
    border: 1px solid rgba(230, 36, 41, 0.35) !important;
    border-radius: 6px !important;
    color: #ffaa33 !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 0.04em !important;
    transition: all 0.2s ease !important;
}
.stButton > button[kind="secondary"]:hover {
    border-color: #ff2a2a !important;
    color: #ffffff !important;
    box-shadow: 0 0 16px rgba(230, 36, 41, 0.4) !important;
    transform: translateY(-1px) !important;
}

/* ── Apply Link Button (Arc Reactor Plasma Cyan) ── */
[data-testid="stLinkButton"] button {
    background: linear-gradient(135deg, #0099cc 0%, #00e5ff 45%, #70f3ff 75%, #0088cc 100%) !important;
    color: #031422 !important;
    font-family: 'Orbitron', monospace !important;
    font-weight: 900 !important;
    letter-spacing: 0.08em !important;
    font-size: 0.82rem !important;
    border: 1px solid #70f3ff !important;
    border-radius: 6px !important;
    box-shadow: 0 0 22px rgba(0, 229, 255, 0.65), inset 0 1px 0 #ffffff !important;
    transition: all 0.2s ease !important;
}
[data-testid="stLinkButton"] button:hover {
    transform: translateY(-2px) scale(1.03) !important;
    box-shadow: 0 0 35px rgba(0, 229, 255, 0.95), 0 0 60px rgba(0, 229, 255, 0.5) !important;
    color: #000000 !important;
}

/* ── Telemetry Metrics ── */
[data-testid="stMetricValue"] {
    font-family: 'Orbitron', monospace !important;
    font-size: 1.7rem !important;
    font-weight: 800 !important;
    color: #ffd700 !important;
    text-shadow: 0 0 16px rgba(255, 215, 0, 0.55);
}
[data-testid="stMetricLabel"] p {
    font-family: 'Rajdhani', sans-serif !important;
    color: #ff8c00 !important;
    font-size: 0.82rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    font-weight: 700 !important;
}

/* ── Tabs (History / Vault) ── */
[data-testid="stTabs"] [data-baseweb="tab"] {
    font-family: 'Orbitron', monospace !important;
    font-weight: 700 !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.06em !important;
    color: #8c9cb8 !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: #00f0ff !important;
    border-bottom: 2px solid #00f0ff !important;
    text-shadow: 0 0 12px rgba(0, 240, 255, 0.6) !important;
}

/* ── Info & Warning Alerts ── */
[data-testid="stInfo"] {
    background: rgba(0, 240, 255, 0.06) !important;
    border: 1px solid rgba(0, 240, 255, 0.35) !important;
    border-left: 4px solid #00f0ff !important;
    border-radius: 6px !important;
    color: #b0f5ff !important;
}
[data-testid="stWarning"] {
    background: rgba(255, 140, 0, 0.08) !important;
    border: 1px solid rgba(255, 140, 0, 0.4) !important;
    border-left: 4px solid #ff8c00 !important;
    border-radius: 6px !important;
}

/* ── Progress Bars (Dimension Scores) ── */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #ff1a22 0%, #ff9900 50%, #00f0ff 100%) !important;
    box-shadow: 0 0 10px rgba(0, 240, 255, 0.6);
}
[data-testid="stProgressBar"] {
    background: rgba(230, 36, 41, 0.12) !important;
    border-radius: 4px !important;
}

/* ── Input Fields & Selectboxes (HUD Tech Brackets) ── */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
[data-testid="stSelectbox"] > div > div {
    background: rgba(18, 12, 22, 0.85) !important;
    border: 1px solid rgba(230, 36, 41, 0.3) !important;
    color: #f0f4ff !important;
    border-radius: 6px !important;
    transition: all 0.2s ease !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    border-color: #00f0ff !important;
    box-shadow: 0 0 15px rgba(0, 240, 255, 0.35) !important;
}

/* ── Vault Cards (Stark Armor HUD Panels) ── */
.vault-card {
    background: linear-gradient(135deg, rgba(20, 10, 16, 0.95) 0%, rgba(10, 14, 26, 0.95) 100%);
    border: 1px solid rgba(230, 36, 41, 0.45);
    border-radius: 10px;
    padding: 1.4rem 1.6rem 1.2rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 25px rgba(0, 0, 0, 0.6), inset 0 0 20px rgba(230, 36, 41, 0.05);
    transition: all 0.25s ease;
}
.vault-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #ff1a22 0%, #ffd700 50%, #00f0ff 100%);
    background-size: 200% auto;
    animation: laserGlow 3s linear infinite;
}
.vault-card:hover {
    border-color: #00f0ff;
    box-shadow: 0 8px 40px rgba(0, 240, 255, 0.25), 0 0 0 1px rgba(0, 240, 255, 0.4);
    transform: translateY(-2px);
}
.vault-card-title {
    font-family: 'Orbitron', monospace;
    font-size: 1.15rem; font-weight: 800;
    color: #ffffff; margin-bottom: 0.2rem;
    letter-spacing: 0.03em;
}
.vault-card-company {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1rem; font-weight: 700;
    color: #ff9900; margin-bottom: 0.5rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.vault-card-meta {
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.85rem; font-weight: 600;
    color: #94a3b8;
    display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 0.8rem;
}
.vault-score-badge {
    display: inline-block;
    background: rgba(0, 240, 255, 0.12);
    border: 1px solid #00f0ff;
    color: #00f0ff; font-weight: 800; font-size: 0.82rem;
    font-family: 'Orbitron', monospace;
    padding: 0.15rem 0.65rem; border-radius: 4px;
    box-shadow: 0 0 10px rgba(0, 240, 255, 0.35);
}
.vault-approved-badge {
    display: inline-block;
    background: rgba(255, 26, 34, 0.15);
    border: 1px solid #ff1a22;
    color: #ff6b70; font-weight: 700; font-size: 0.78rem;
    font-family: 'Rajdhani', sans-serif;
    padding: 0.15rem 0.65rem; border-radius: 4px; margin-left: 0.4rem;
    text-transform: uppercase; letter-spacing: 0.06em;
}
</style>

<!-- Canvas for Hologram HUD Radar & Supersonic Plasma Streaks -->
<canvas id="stark-hud-canvas" style="
    position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    pointer-events: none; z-index: 0; opacity: 0.75;
"></canvas>

<script>
(function() {
    const canvas = document.getElementById('stark-hud-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    let width, height;
    function resize() {
        width = canvas.width = window.innerWidth;
        height = canvas.height = window.innerHeight;
    }
    resize();
    window.addEventListener('resize', resize);

    /* ── Supersonic Thruster Plasma Streaks (Image 3) ── */
    const streaks = [];
    const NUM_STREAKS = 35;
    for (let i = 0; i < NUM_STREAKS; i++) {
        streaks.push({
            x: Math.random() * width,
            y: Math.random() * height,
            length: 40 + Math.random() * 90,
            speed: 3 + Math.random() * 6,
            width: 0.8 + Math.random() * 1.8,
            color: Math.random() < 0.65 ? '#ff1a22' : (Math.random() < 0.8 ? '#ff9900' : '#00f0ff'),
            alpha: 0.15 + Math.random() * 0.45
        });
    }

    /* ── Rotating Hologram HUD Ring Parameters (Image 1) ── */
    let rotAngle1 = 0;
    let rotAngle2 = 0;
    let rotAngle3 = 0;

    function drawHologramHUD(cx, cy, radius) {
        ctx.save();
        ctx.translate(cx, cy);

        // Core Glowing Arc Center
        const glowGrad = ctx.createRadialGradient(0, 0, 5, 0, 0, radius * 0.9);
        glowGrad.addColorStop(0, 'rgba(255, 140, 0, 0.22)');
        glowGrad.addColorStop(0.35, 'rgba(230, 36, 41, 0.12)');
        glowGrad.addColorStop(0.7, 'rgba(255, 70, 0, 0.04)');
        glowGrad.addColorStop(1, 'transparent');
        ctx.fillStyle = glowGrad;
        ctx.beginPath();
        ctx.arc(0, 0, radius * 0.9, 0, Math.PI * 2);
        ctx.fill();

        // Outer Tech Ring 1 (Concentric Broken Circle)
        ctx.rotate(rotAngle1);
        ctx.strokeStyle = 'rgba(255, 140, 0, 0.35)';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([18, 8, 4, 8]);
        ctx.beginPath();
        ctx.arc(0, 0, radius * 0.82, 0, Math.PI * 2);
        ctx.stroke();

        // Ring 2 with crosshair ticks
        ctx.rotate(rotAngle2 - rotAngle1);
        ctx.strokeStyle = 'rgba(230, 36, 41, 0.4)';
        ctx.lineWidth = 1.2;
        ctx.setLineDash([35, 15, 70, 20]);
        ctx.beginPath();
        ctx.arc(0, 0, radius * 0.62, 0, Math.PI * 2);
        ctx.stroke();

        // Ring 3 (Arc Reactor Inner Nodes)
        ctx.rotate(rotAngle3 - rotAngle2);
        ctx.strokeStyle = 'rgba(255, 215, 0, 0.45)';
        ctx.lineWidth = 2;
        ctx.setLineDash([6, 12]);
        ctx.beginPath();
        ctx.arc(0, 0, radius * 0.42, 0, Math.PI * 2);
        ctx.stroke();

        // Center Arc Core
        ctx.setLineDash([]);
        ctx.strokeStyle = 'rgba(0, 240, 255, 0.5)';
        ctx.lineWidth = 1.8;
        ctx.beginPath();
        ctx.arc(0, 0, radius * 0.2, 0, Math.PI * 2);
        ctx.stroke();

        // 4 Coordinate Reticle Crosshairs
        ctx.strokeStyle = 'rgba(255, 140, 0, 0.3)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(-radius * 0.9, 0); ctx.lineTo(-radius * 0.7, 0);
        ctx.moveTo(radius * 0.7, 0);  ctx.lineTo(radius * 0.9, 0);
        ctx.moveTo(0, -radius * 0.9); ctx.lineTo(0, -radius * 0.7);
        ctx.moveTo(0, radius * 0.7);  ctx.lineTo(0, radius * 0.9);
        ctx.stroke();

        ctx.restore();
    }

    function animate() {
        ctx.clearRect(0, 0, width, height);

        // Update Angles
        rotAngle1 += 0.003;
        rotAngle2 -= 0.0045;
        rotAngle3 += 0.006;

        // Draw Holographic HUD Radar at Top-Right background (Image 1 style)
        const hudX = width > 900 ? width * 0.82 : width * 0.5;
        const hudY = height * 0.32;
        const hudRadius = width > 900 ? 180 : 130;
        drawHologramHUD(hudX, hudY, hudRadius);

        // Draw Supersonic Plasma Streaks (Image 3 style)
        streaks.forEach(s => {
            s.y -= s.speed;
            if (s.y + s.length < 0) {
                s.y = height + 20;
                s.x = Math.random() * width;
            }

            ctx.save();
            ctx.beginPath();
            const grad = ctx.createLinearGradient(s.x, s.y + s.length, s.x, s.y);
            grad.addColorStop(0, 'transparent');
            grad.addColorStop(1, s.color);
            ctx.strokeStyle = grad;
            ctx.lineWidth = s.width;
            ctx.shadowColor = s.color;
            ctx.shadowBlur = 8;
            ctx.moveTo(s.x, s.y + s.length);
            ctx.lineTo(s.x, s.y);
            ctx.stroke();
            ctx.restore();
        });

        requestAnimationFrame(animate);
    }
    requestAnimationFrame(animate);
})();
</script>
""")

# ---------------------------------------------------------------------------
# Page 1 — Profile
# ---------------------------------------------------------------------------

def page_profile() -> None:
    st.header("👤 Your profile")
    st.caption("Everything stays local (`profile/profile.yaml`, gitignored). "
               "Your name, email, and phone are masked with placeholders "
               "before any text reaches the LLM.")

    existing = load_profile() or {}
    cand = existing.get("candidate", {})
    prefs = existing.get("preferences", {})
    weights = existing.get("weights", {})
    sources = existing.get("sources", {})

    current_resume = REPO_ROOT / "profile" / "resume.pdf"
    
    with st.expander("✨ Auto-Extract Profile & Target Roles from Resume", expanded=True):
        st.write("Upload or use your existing resume (`resume.pdf`) to let the AI automatically infer your **contact details, background summary, recommended target roles, seniority, and skills**.")
        uploaded_res = st.file_uploader("Upload Resume to Auto-Extract (PDF)", type=["pdf"], key="auto_res_upload")
        if st.button("🚀 Auto-Extract & Fill Profile Now", type="primary", use_container_width=True):
            with st.spinner("🤖 Analyzing resume and discovering optimal target roles…"):
                try:
                    from pypdf import PdfReader
                    from src.agents.llm_client import get_llm_client
                    from src.agents.resume_extractor import extract_profile_from_resume
                    
                    target_pdf = current_resume
                    if uploaded_res is not None:
                        current_resume.parent.mkdir(parents=True, exist_ok=True)
                        current_resume.write_bytes(uploaded_res.read())
                        target_pdf = current_resume
                        
                    if not target_pdf.exists():
                        st.error("No resume file found. Please upload a PDF resume first.")
                    else:
                        reader = PdfReader(str(target_pdf))
                        raw_text = "\n".join(p.extract_text() or "" for p in reader.pages)
                        extracted = extract_profile_from_resume(get_llm_client(), raw_text)
                        
                        if extracted:
                            # Apply to profile.yaml directly and refresh
                            cand["name"] = extracted.get("name") or cand.get("name", "")
                            cand["email"] = extracted.get("email") or cand.get("email", "")
                            cand["phone"] = extracted.get("phone") or cand.get("phone", "")
                            cand["summary"] = extracted.get("summary") or cand.get("summary", "")
                            
                            if extracted.get("target_roles"):
                                prefs["target_roles"] = extracted["target_roles"]
                            if extracted.get("seniority"):
                                prefs["seniority"] = extracted["seniority"]
                            if extracted.get("industries"):
                                prefs["industries"] = extracted["industries"]
                            if extracted.get("location"):
                                prefs["locations"] = [extracted["location"], "Delhi NCR, India", "Remote"]
                            if extracted.get("skills"):
                                prefs["must_haves"] = extracted["skills"][:3]
                                
                            existing["candidate"] = cand
                            existing["preferences"] = prefs
                            if "sources" not in existing:
                                existing["sources"] = sources
                            if "weights" not in existing:
                                existing["weights"] = weights
                            
                            PROFILE_PATH.write_text(yaml.dump(existing, sort_keys=False))
                            st.success(f"✅ Extracted! Recommended roles: **{', '.join(prefs.get('target_roles', []))}**")
                            st.rerun()
                except Exception as e:
                    st.error(f"Extraction error: {e}")

    with st.form("profile_form"):
        st.subheader("About you")
        c1, c2, c3 = st.columns(3)
        name = c1.text_input("Full name", cand.get("name", ""))
        email = c2.text_input("Email", cand.get("email", ""))
        phone = c3.text_input("Phone", cand.get("phone", ""))
        summary = st.text_area("One-line summary of yourself (optional)",
                               cand.get("summary", ""), height=68)
        style_options = ["", "direct", "collaborative", "enthusiastic"]
        communication_style = st.selectbox(
            "Cover letter tone (optional)", style_options,
            index=style_options.index(cand.get("communication_style", ""))
            if cand.get("communication_style", "") in style_options else 0,
            help="Calibrates the drafting agent's tone. Blank = natural, "
                 "confident default. Overridden by a learned voice profile "
                 "if you upload writing samples below — a real learned "
                 "style beats a coarse preset.")

        resume_file = st.file_uploader("Replace Resume (PDF)", type=["pdf"])
        if current_resume.exists():
            st.caption(f"✅ Current resume on file: `{current_resume.name}`")

        extra_docs = st.file_uploader(
            "Additional documents (optional) — LinkedIn export, past cover "
            "letters, reference letters",
            type=["pdf", "txt", "md"], accept_multiple_files=True,
            help="Combined with your resume for richer grounding when "
                 "analyzing your skills. PII-masked the same way. A "
                 "reference letter's AUTHOR isn't detected as third-party "
                 "PII — redact their name yourself first if that matters "
                 "to you.")
        existing_docs = list_profile_documents()
        if existing_docs:
            st.caption(f"📎 Already on file: {', '.join(existing_docs)} "
                       "(uploads add to this list, don't replace it)")

        writing_samples = st.file_uploader(
            "Writing samples (optional) — past cover letters, emails, "
            "anything in your own voice",
            type=["pdf", "txt", "md"], accept_multiple_files=True,
            help="Used ONLY to learn your writing style (sentence rhythm, "
                 "formality, vocabulary) so drafted cover letters and "
                 "tailored CVs sound like you — never to extract facts or "
                 "claims (that's what the resume/documents above are "
                 "for). PII-masked the same way. Leave empty to use the "
                 "Cover letter tone preset below instead.")
        existing_samples = list_writing_samples()
        if existing_samples:
            st.caption(f"🎙️ Already on file: {', '.join(existing_samples)} "
                       "(uploads add to this list, don't replace it)")

        st.subheader("What you're looking for")
        c1, c2 = st.columns(2)
        target_roles = c1.text_input(
            "Target roles (comma-separated)",
            ", ".join(prefs.get("target_roles", ["Machine Learning Engineer"])))
        locations = c2.text_input(
            "Locations (comma-separated)",
            ", ".join(prefs.get("locations", ["Remote US"])))
        strict_keywords_pref = st.checkbox(
            "🎯 Match role titles exactly (strict) — fewer, tighter results",
            value=bool(prefs.get("strict_keyword_match", False)),
            help="Off by default. Normally JobScout broadens each target "
                 "role to surface far more postings — 'Machine Learning "
                 "Engineer' also matches 'ML Engineer', 'Machine Learning "
                 "Scientist', etc. — and lets the scorer rank relevance. "
                 "Exact-phrase titles alone match almost nothing on real "
                 "boards, so leave this OFF unless you're getting too much "
                 "noise and want only verbatim title matches.")
        c1, c2, c3 = st.columns(3)
        employment_types = c1.multiselect(
            "Employment types", EMPLOYMENT_TYPES,
            default=[t for t in prefs.get("employment_types", ["full-time"])
                     if t in EMPLOYMENT_TYPES])
        seniority = c2.multiselect(
            "Seniority", SENIORITY,
            default=[s for s in prefs.get("seniority", ["mid", "senior"])
                     if s in SENIORITY])
        remote_pref = c3.selectbox(
            "Remote preference", REMOTE_PREFS,
            index=REMOTE_PREFS.index(prefs.get("remote_preference",
                                               "remote_or_hybrid"))
            if prefs.get("remote_preference") in REMOTE_PREFS else 0)
        c1, c2 = st.columns(2)
        industries = c1.text_input("Preferred industries (comma-separated)",
                                   ", ".join(prefs.get("industries", ["AI/ML"])))
        salary_floor = c2.number_input("Salary floor USD (0 = ignore)",
                                       min_value=0, step=5000,
                                       value=int(prefs.get("salary_floor_usd", 0)))
        max_posting_age = st.number_input(
            "Only show postings from the last N days (0 = ignore)",
            min_value=0, step=1,
            value=int(prefs.get("max_posting_age_days") or 0),
            help="Deterministic filter, applied before any LLM call. "
                 "Postings with no stated date are dropped when this is "
                 "set — an unstated date isn't a reliable 'recent enough.'")
        verify_liveness_pref = st.checkbox(
            "🔎 Also verify postings are still live with a headless "
            "browser (Playwright, opt-in)",
            value=bool(prefs.get("verify_liveness", False)),
            help="Catches postings that return HTTP 200 but actually say "
                 "'no longer accepting applications' — something the "
                 "posting-age filter above can't see. Requires `pip "
                 "install playwright && playwright install chromium`. "
                 "Only checked on jobs about to be scored, not every "
                 "search result, and fails open (never wrongly drops a "
                 "job it couldn't verify) — but it does load each "
                 "posting's page in a real browser, which is slower than "
                 "everything else JobScout does and is a heavier-weight "
                 "check than a plain API call.")
        drop_suspicious_pref = st.checkbox(
            "🚩 Auto-drop postings flagged 'suspicious' by the ghost-job/"
            "scam check (Block G)",
            value=bool(prefs.get("drop_suspicious_postings", False)),
            help="Off by default: flagged postings are shown with a badge "
                 "and their reasons, not hidden, since it's a heuristic "
                 "that can misfire — better to let you judge one flagged "
                 "posting than silently remove a real job. Turn this on "
                 "only if you'd rather never see 'suspicious'-tier "
                 "postings at all. 'Caution'-tier postings are always "
                 "shown regardless.")
        c1, c2 = st.columns(2)
        must_haves = c1.text_input("Must-haves (comma-separated, optional)",
                                   ", ".join(prefs.get("must_haves", [])))
        dealbreakers = c2.text_input(
            "Dealbreakers (comma-separated, optional)",
            ", ".join(prefs.get("dealbreakers", [])),
            help="Deterministic filter — jobs containing these phrases are "
                 "dropped before any LLM call.")

        st.subheader("Scoring")
        st.caption("Relative importance of each dimension (auto-normalized "
                   "to sum to 1.0). The weighted total is computed in code, "
                   "never by the LLM.")
        cols = st.columns(len(WEIGHT_DIMS))
        raw_weights = {}
        defaults = {"skills_match": 40, "role_title_match": 20,
                    "industry_match": 15, "location_match": 15,
                    "seniority_match": 10}
        for col, dim in zip(cols, WEIGHT_DIMS):
            raw_weights[dim] = col.slider(
                dim.replace("_", " "), 0, 100,
                int(weights.get(dim, defaults[dim] / 100) * 100))
        draft_threshold = st.slider(
            "Draft threshold — jobs scoring at or above this get a drafted "
            "application package", 0, 100,
            int(existing.get("draft_threshold", 70)))

        st.subheader("Job sources")
        enabled = st.multiselect(
            "Enabled boards", ALL_SOURCES,
            default=[s for s in sources.get(
                "enabled", ["remoteok", "themuse", "remotive", "arbeitnow",
                            "greenhouse", "lever", "ashby"]) if s in ALL_SOURCES],
            help="JSearch (Google for Jobs — includes Indeed/Glassdoor "
                 "postings), Adzuna, and USAJOBS also need free API keys "
                 "in .env")
        gh_companies = st.text_input(
            "Greenhouse companies to watch (board tokens, comma-separated)",
            ", ".join(sources.get("greenhouse_companies", ["anthropic"])))
        c1, c2 = st.columns(2)
        lever_companies = c1.text_input(
            "Lever companies to watch (board tokens, comma-separated)",
            ", ".join(sources.get("lever_companies", [])),
            help="Startup-heavy ATS. Postings label internships explicitly, "
                 "so JobScout detects them exactly instead of guessing.")
        ashby_companies = c2.text_input(
            "Ashby companies to watch (org names, comma-separated)",
            ", ".join(sources.get("ashby_companies", [])),
            help="The dominant ATS among recent YC-batch startups — good "
                 "coverage if you're hunting for internships there.")

        linkedin_ack = st.checkbox(
            "Enable LinkedIn public search",
            value=bool(sources.get("linkedin_tos_acknowledged", True)),
            help="Accesses public guest job postings without needing any login or cookies.")

        saved = st.form_submit_button("💾 Save profile", type="primary")

    if saved:
        total = sum(raw_weights.values()) or 1
        profile = {
            "candidate": {
                "name": name, "email": email, "phone": phone,
                "resume_path": "./profile/resume.pdf", "summary": summary,
                "communication_style": communication_style,
            },
            "preferences": {
                "employment_types": employment_types or ["full-time"],
                "target_roles": _csv(target_roles),
                "strict_keyword_match": bool(strict_keywords_pref),
                "seniority": seniority,
                "industries": _csv(industries),
                "locations": _csv(locations),
                "remote_preference": remote_pref,
                "salary_floor_usd": int(salary_floor),
                "max_posting_age_days": int(max_posting_age) or None,
                "verify_liveness": bool(verify_liveness_pref),
                "drop_suspicious_postings": bool(drop_suspicious_pref),
                "visa_sponsorship_required": bool(
                    prefs.get("visa_sponsorship_required", False)),
                "must_haves": _csv(must_haves),
                "dealbreakers": _csv(dealbreakers),
            },
            "weights": {d: round(v / total, 4) for d, v in raw_weights.items()},
            "draft_threshold": int(draft_threshold),
            "sources": {
                "enabled": enabled,
                "greenhouse_companies": _csv(gh_companies),
                "lever_companies": _csv(lever_companies),
                "ashby_companies": _csv(ashby_companies),
                "linkedin_tos_acknowledged": bool(linkedin_ack),
            },
        }
        if resume_file is not None:
            current_resume.parent.mkdir(exist_ok=True)
            current_resume.write_bytes(resume_file.getvalue())
        if extra_docs:
            DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
            for f in extra_docs:
                # sanitize: strip any path component from the browser-supplied
                # filename before writing, so an upload can't traverse out of
                # profile/documents/
                safe_name = Path(f.name).name
                if safe_name:
                    (DOCUMENTS_DIR / safe_name).write_bytes(f.getvalue())
        if writing_samples:
            WRITING_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
            for f in writing_samples:
                safe_name = Path(f.name).name
                if safe_name:
                    (WRITING_SAMPLES_DIR / safe_name).write_bytes(f.getvalue())
        PROFILE_PATH.parent.mkdir(exist_ok=True)
        PROFILE_PATH.write_text(yaml.safe_dump(profile, sort_keys=False))
        # A changed resume/profile/documents/samples invalidates the caches
        st.session_state.pop("skills_profile", None)
        st.session_state.pop("voice_profile", None)
        st.success(f"Profile saved to `{PROFILE_PATH.relative_to(REPO_ROOT)}` "
                   "(gitignored). Head to **🚀 Run JobScout**.")

    st.divider()
    with st.expander("🔍 Preview PDF text extraction"):
        st.caption("Runs the same extraction used before scoring, on whatever "
                   "resume/documents are currently saved — catches a scanned "
                   "or image-only PDF with no real text layer before it "
                   "silently produces an empty or garbled profile.")
        if st.button("Extract now"):
            extracted = extract_all_saved_documents()
            if not extracted.strip():
                st.warning("No text extracted. Either no resume/documents are "
                           "saved yet, or the PDF has no selectable text layer "
                           "(common with scanned/image-only PDFs) — try "
                           "re-exporting it from a text editor instead.")
            else:
                st.text_area("Extracted text", extracted, height=300)
                st.caption(f"{len(extracted)} characters extracted.")


# ---------------------------------------------------------------------------
# Page 2 — Run
# ---------------------------------------------------------------------------

def _draft_for(package: dict, masker: PIIMasker, skills_profile: str,
               profile: dict, communication_style: str = "",
               voice_profile: str = "") -> None:
    job = package["job"]
    client = st.session_state["client"]
    with st.spinner(f"Drafting application package for {job['title']}…"):
        drafts = drafting_agent.draft_package(client, skills_profile, job,
                                              package, communication_style,
                                              voice_profile)
    with st.spinner("Second pass: reviewing the draft for a fresh critique…"):
        review = drafting_agent.review_draft(
            client, skills_profile, job, drafts["cover_letter"],
            communication_style, voice_profile)
        # SECURITY: unmask ONLY here — final local render for human eyes;
        # the unmasked text never goes back through the model.
        drafts["cover_letter"] = masker.unmask(review["revised_cover_letter"])
        drafts["review_notes"] = review["revision_summary"]
        drafts["review_issues"] = review["issues_found"]
        drafts["keyword_coverage"] = keyword_coverage(job, drafts["cover_letter"])
    with st.spinner("Tailoring an ATS-optimized CV…"):
        try:
            resume_text = masker.mask(extract_profile_text(profile))
            drafts["cv_pdf_path"] = generate_cv_pdf(
                client, resume_text, skills_profile, job,
                profile.get("candidate", {}), masker, CV_OUTPUT_DIR,
                voice_profile)
        except Exception as exc:
            st.warning(f"Cover letter ready, but CV generation failed: {exc}")
    records.upsert(job, drafts=drafts)
    st.rerun()


def _decide(job: dict, decision: str) -> None:
    records.upsert(job, decision=decision)
    memory.mark_seen(job["id"], job["title"], decision)
    st.rerun()


def render_package(package: dict, threshold: int, masker: PIIMasker,
                   skills_profile: str, profile: dict,
                   communication_style: str = "", voice_profile: str = "") -> None:
    job = package["job"]
    record = records.get(job["id"]) or {}
    decision = record.get("decision")
    badge = {"approved": "✅ approved", "rejected": "❌ rejected",
             "skipped": "⏭ skipped"}.get(decision, "")
    tag = f"🏷️ {job['archetype']}  " if job.get("archetype") else ""
    legitimacy_tag = _legitimacy_tag(job)
    label = (f"{score_badge(package['score'], threshold)} — "
             f"**{job['title']}** @ {job['company']}  {tag}{legitimacy_tag}{badge}")

    with st.expander(label, expanded=package["score"] >= threshold and not decision):
        _legitimacy_section(job)
        meta, dims = st.columns([1, 2])
        with meta:
            st.markdown(f"**{job['company']}**")
            st.caption(f"{job['location'] or '—'} · {job['remote']} · "
                       f"via {_source_label(job)}")
            st.link_button("Open job posting ↗", job["url"])
            st.metric("Weighted score", f"{package['score']:.0f}/100")
        with dims:
            for dim, d in package["dimensions"].items():
                # structured outputs can't enforce a 0-100 range, so clamp
                # before st.progress (which raises outside [0, 1])
                clamped = max(0, min(int(d["score"]), 100))
                st.progress(clamped / 100,
                            text=f"**{dim.replace('_', ' ')} — {d['score']}**"
                                 f"  ·  {d['reason']}")
        st.markdown(f"*{package['summary']}*")
        st.divider()

        _contacts_section(job, record, package=package, masker=masker,
                          skills_profile=skills_profile, profile=profile,
                          communication_style=communication_style,
                          voice_profile=voice_profile, key=f"contacts_{job['id']}")

        # Draft package (auto-suggested above threshold; on-demand below)
        if record.get("cover_letter"):
            st.text_area("✉️ Cover letter (edit before sending)",
                         record["cover_letter"], height=320,
                         key=f"letter_{job['id']}")
            st.markdown("**📄 Suggested resume tweaks**")
            st.markdown(record.get("resume_tweaks", ""))
            if record.get("review_notes"):
                with st.expander("🔍 Reviewer notes (second-pass critique)"):
                    st.caption(record["review_notes"])
                    for issue in record.get("review_issues") or []:
                        st.markdown(f"- **{issue['category'].replace('_', ' ')}** "
                                    f"— {issue['detail']}")
            kw = record.get("keyword_coverage")
            if kw and (kw["covered"] or kw["missing"]):
                total = len(kw["covered"]) + len(kw["missing"])
                st.caption(f"🔑 Keyword coverage: {len(kw['covered'])}/{total}")
                c1, c2 = st.columns(2)
                c1.markdown("✅ " + (", ".join(kw["covered"]) or "—"))
                c2.markdown("⚠️ " + (", ".join(kw["missing"]) or "—"))
            _cv_download_button(job, record, key=f"cv_{job['id']}")
        else:
            hint = ("" if package["score"] >= threshold else
                    " (below your draft threshold — drafting is optional)")
            if st.button(f"✍️ Draft cover letter + resume tweaks{hint}",
                         key=f"draft_{job['id']}"):
                _draft_for(package, masker, skills_profile, profile,
                          communication_style, voice_profile)

        # HITL gate — the human decides; Scout never submits.
        st.info("🔒 Scout never submits applications. If you approve, "
                "apply manually at the job URL above.")
        c1, c2, c3, _ = st.columns([1, 1, 1, 3])
        if c1.button("✅ Approve", key=f"approve_{job['id']}",
                     type="primary", disabled=decision == "approved"):
            _decide(job, "approved")
        if c2.button("❌ Reject", key=f"reject_{job['id']}",
                     disabled=decision == "rejected"):
            _decide(job, "rejected")
        if c3.button("⏭ Skip", key=f"skip_{job['id']}",
                     disabled=decision == "skipped"):
            _decide(job, "skipped")


def page_run() -> None:
    st.header("🚀 Search Jobs")
    profile = load_profile()
    if profile is None:
        st.warning("No profile yet — create one on the **👤 Profile** page first.")
        return
    ai_keys = ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "OMNIROUTE_BASE_URL", "OPENAI_BASE_URL")
    if not any(os.getenv(k) for k in ai_keys) and os.getenv("LLM_PROVIDER") != "omniroute":
        st.error("No AI key or local proxy found in `.env`. Add `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or `OMNIROUTE_BASE_URL` to `.env`.")
        return

    prefs = profile.get("preferences", {})
    threshold = int(profile.get("draft_threshold", 70))
    st.caption(f"Searching for **{', '.join(prefs.get('target_roles', []))}** "
               f"across **{', '.join(profile.get('sources', {}).get('enabled', []))}** "
               f"· draft threshold **{threshold}** · 🔭 Scout")

    st.markdown("##### ⚙️ Search Filters & Options")
    f1, f2 = st.columns([1, 1])
    work_mode = f1.selectbox(
        "💼 Workplace Type",
        ["Any Workplace", "Remote Only", "Hybrid", "On-site Only"],
        index=0,
        help="Filter jobs by work setup: Remote, Hybrid, On-site, or Any."
    )
    selected_state = f2.selectbox(
        "📍 Location / State (All 28 States)",
        INDIAN_STATES,
        index=0,
        help="Search and filter jobs across all 28 states of India or Pan-India/Remote."
    )

    c1, c2, c3 = st.columns([1, 1, 1])
    max_score = c1.number_input(
        "Jobs to score (target & cost cap)", 1, 40, 6,
        help="The search is outcome-driven: it keeps deepening (keyword "
             "rotation and further pages, up to 3 rounds) until this many "
             "NEW jobs survive the deterministic filters — not just "
             "however many one fixed fetch happens to yield.")
    min_matches = c2.number_input(
        f"Stop early once this many ≥ {threshold} found (0 = off)",
        0, 20, 0,
        help="Keeps scoring more jobs, up to the cost cap above, instead "
             "of stopping after a fixed batch.")
    run = c3.button("🔎 Search & score", type="primary", use_container_width=True)

    if run:
        load_dotenv(override=True)
        import copy
        run_profile = copy.deepcopy(profile)
        run_prefs = run_profile.setdefault("preferences", {})

        # Apply dynamic Workplace filter
        if work_mode == "Remote Only":
            run_prefs["remote_preference"] = "remote_only"
        elif work_mode == "Hybrid":
            run_prefs["remote_preference"] = "hybrid"
        elif work_mode == "On-site Only":
            run_prefs["remote_preference"] = "onsite"
        else:
            run_prefs["remote_preference"] = "any"

        # Apply dynamic State / Location filter
        if selected_state and selected_state != "All Locations / Default":
            run_prefs["locations"] = [f"{selected_state}, India", selected_state]

        cand = profile.get("candidate", {})
        masker = PIIMasker(name=cand.get("name", ""),
                           email=cand.get("email", ""),
                           phone=cand.get("phone", ""),
                           address=cand.get("address", ""))
        from src.agents.llm_client import get_llm_client
        st.session_state["client"] = get_llm_client()
        st.session_state.pop("skills_profile", None)
        st.session_state.pop("voice_profile", None)

        with st.status("Running the agent pipeline…", expanded=True) as status:
            target = int(max_score)
            st.write(f"🔎 Searching job boards via MCP — deepening until "
                     f"**{target}** new jobs found (up to "
                     f"{MAX_SEARCH_ROUNDS} rounds)…")

            # Deterministic filters run inside each round, so the loop
            # deepens on the OUTCOME (new jobs worth scoring), not on raw
            # fetch counts. Drop reasons accumulate across rounds for the
            # empty-result explainer.
            total_dropped: dict[str, int] = {}

            def _keep(fresh: list[dict]) -> list[dict]:
                # ── Strict location filter ──────────────────────────────────
                # When a specific state is selected, only keep jobs whose
                # location mentions that state, India broadly, or "remote".
                # This stops EU/Russian/US-only listings from slipping through.
                if selected_state and selected_state != "All Locations / Default":
                    state_lc = selected_state.lower()
                    def _location_ok(j: dict) -> bool:
                        loc = j.get("location", "").lower()
                        # Always keep remote-flagged or explicitly remote location
                        if j.get("remote") == "remote" or "remote" in loc or "worldwide" in loc:
                            return True
                        # Keep if location mentions this state or India
                        return state_lc in loc or "india" in loc
                    fresh = [j for j in fresh if _location_ok(j)]

                # ── Workplace mode filter ───────────────────────────────────
                if work_mode == "Remote Only":
                    fresh = [j for j in fresh
                             if j.get("remote") == "remote"
                             or "remote" in j.get("location", "").lower()]
                elif work_mode == "On-site Only":
                    fresh = [j for j in fresh if j.get("remote") != "remote"]

                kept, dropped = deterministic_filter(fresh, run_profile, memory)
                for k, v in dropped.items():
                    total_dropped[k] = total_dropped.get(k, 0) + v
                return kept

            def _narrate(page, found, kept_round, kept_total):
                st.write(f"Round {page}: **{found}** found → "
                         f"**{kept_round}** new kept "
                         f"({kept_total}/{target}).")

            jobs = collect_new_jobs(
                lambda p: fetch_jobs(run_profile, page=p),
                _keep, target=target, on_round=_narrate)
            if not jobs:
                status.update(label="Nothing new to review", state="complete")
                st.session_state["scored"] = []
                st.warning(_explain_empty_filter(total_dropped, run_profile))
                return

            archetypes_config = profile.get("archetypes")
            past_entries = records.all()
            for job in jobs:
                job["archetype"] = guess_archetype(
                    f"{job['title']} {job['description']}", archetypes_config)
                job["legitimacy"] = legitimacy_check(job, past_entries)

            if prefs.get("drop_suspicious_postings"):
                before = len(jobs)
                jobs = [j for j in jobs
                       if j["legitimacy"]["tier"] != "suspicious"]
                if before - len(jobs):
                    st.write(f"Dropped **{before - len(jobs)}** posting(s) "
                            "flagged suspicious (Block G legitimacy check).")
                if not jobs:
                    status.update(label="Nothing left after the legitimacy "
                                  "filter", state="complete")
                    st.session_state["scored"] = []
                    st.warning("Nothing left after the legitimacy filter — "
                              "try disabling it on the Profile page to "
                              "review flagged postings yourself.")
                    return

            jobs.sort(key=lambda j: _relevance_rank(
                j, prefs.get("target_roles", [])), reverse=True)
            to_score = jobs[:int(max_score)]

            if prefs.get("verify_liveness"):
                st.write("🔎 Verifying postings are still live (Playwright)…")
                to_score, dead_count = verify_liveness(to_score)
                if dead_count:
                    st.write(f"Dropped **{dead_count}** posting(s) that "
                            "appear closed/filled/expired.")

            if "skills_profile" not in st.session_state:
                st.write("🧠 Analyzing your resume + supplementary "
                        "documents (PII-masked)…")
                resume_text = masker.mask(extract_profile_text(profile))
                summary = masker.mask(cand.get("summary", ""))
                st.session_state["skills_profile"] = scoring_agent.analyze_resume(
                    st.session_state["client"], resume_text, summary)

            if "voice_profile" not in st.session_state:
                samples_text = masker.mask(extract_writing_samples_text())
                st.session_state["voice_profile"] = ""
                if samples_text:
                    st.write("🎙️ Learning your writing style from uploaded "
                            "samples (PII-masked)…")
                    st.session_state["voice_profile"] = \
                        scoring_agent.extract_voice_profile(
                            st.session_state["client"], samples_text)

            goal = f", stopping early at {int(min_matches)} matches" if min_matches else ""
            st.write(f"⚖️ Scoring up to {len(to_score)} jobs{goal}…")
            bar = st.progress(0.0)
            scored = []
            hits = 0
            for i, job in enumerate(to_score):
                import time
                if i > 0:
                    time.sleep(0.4)
                try:
                    result = scoring_agent.score_job(
                        st.session_state["client"],
                        st.session_state["skills_profile"],
                        prefs, job, profile.get("weights", {}))
                except Exception as exc:  # one bad job must not kill the run
                    st.write(f"⚠️ Scoring failed for {job['title']!r}: {exc}")
                    continue
                package = {"job": job, **result}
                scored.append(package)
                records.upsert(job, scoring=result)
                memory.mark_seen(job["id"], job["title"], "scored")
                bar.progress((i + 1) / len(to_score),
                             text=f"{result['score']:.0f} — {job['title']}")
                if result["score"] >= threshold:
                    hits += 1
                    if min_matches and hits >= int(min_matches):
                        st.write(f"✅ Reached the goal: {hits} matches "
                                f"≥ {threshold} after scoring {len(scored)}.")
                        break
            if min_matches and hits < int(min_matches):
                st.warning(
                    f"Only {hits}/{int(min_matches)} matches ≥ {threshold} "
                    f"after scoring all {len(scored)} available jobs (capped "
                    "by the cost cap above). Raise the cost cap, broaden "
                    "target roles/locations, or loosen the posting-age "
                    "limit on the Profile page to find more.")

            scored.sort(key=lambda p: p["score"], reverse=True)
            st.session_state["scored"] = scored
            st.session_state["masker_fields"] = {
                "name": cand.get("name", ""), "email": cand.get("email", ""),
                "phone": cand.get("phone", ""), "address": cand.get("address", ""),
            }
            status.update(label="Pipeline complete ✅", state="complete",
                          expanded=False)

    scored = st.session_state.get("scored")
    if not scored:
        return

    matches = sum(1 for p in scored if p["score"] >= threshold)
    m1, m2, m3 = st.columns(3)
    m1.metric("Jobs scored", len(scored))
    m2.metric(f"Matches ≥ {threshold}", matches)
    m3.metric("Top score", f"{scored[0]['score']:.0f}" if scored else "—")

    masker = PIIMasker(**st.session_state.get("masker_fields", {}))
    if "client" not in st.session_state:
        from anthropic import Anthropic
        st.session_state["client"] = Anthropic()
    communication_style = profile.get("candidate", {}).get("communication_style", "")
    for package in scored:
        render_package(package, threshold, masker,
                       st.session_state.get("skills_profile", ""), profile,
                       communication_style, st.session_state.get("voice_profile", ""))


# ---------------------------------------------------------------------------
# Page 3 — History
# ---------------------------------------------------------------------------

def render_skill_gaps(entries: list[dict]) -> None:
    """Recurring dimension gaps across every job ever scored — pure
    aggregation, no LLM call, always available for free. The narrative
    suggestion below it is a separate, explicit, on-demand LLM call."""
    gaps = aggregate_dimension_gaps(entries)
    if not gaps:
        return
    st.subheader("📈 Recurring gaps")
    st.caption("Where your scores consistently land, across every job "
              "JobScout has ever scored for you.")
    for row in gaps:
        clamped = max(0, min(int(row["avg_score"]), 100))
        st.progress(clamped / 100,
                   text=f"**{row['dimension'].replace('_', ' ')} — "
                        f"{row['avg_score']}/100 avg**  ·  scored on "
                        f"{row['count']} jobs  ·  weakest link in "
                        f"{row['weakest_count']} of them")

    worst = gaps[0]
    if st.button(f"🎯 Get suggestions for {worst['dimension'].replace('_', ' ')}",
                key="suggest_focus"):
        if "client" not in st.session_state:
            from anthropic import Anthropic
            st.session_state["client"] = Anthropic()
        with st.spinner("Thinking about what would actually move this number…"):
            suggestion = insights_agent.suggest_focus(st.session_state["client"], worst)
        st.info(suggestion)


DECISION_CATEGORIES = [
    ("undecided", "🕓 Undecided"), ("approved", "✅ Approved"),
    ("rejected", "❌ Rejected"), ("skipped", "⏭ Skipped"),
]


SORT_FIELDS = {
    # Leading "has a value" flag keeps records missing the field grouped at
    # the bottom on the default (descending) view instead of masquerading
    # as a zero score / oldest posting.
    "Match rating": lambda e: (e.get("score") is not None, e.get("score") or 0),
    # posted_at is a stored ISO-8601 string, so plain string order IS
    # chronological order — no parsing needed.
    "Date posted": lambda e: (bool(e["job"].get("posted_at")),
                              e["job"].get("posted_at") or ""),
}


def _history_decide(job: dict, decision: str) -> None:
    records.upsert(job, decision=decision)
    memory.mark_seen(job["id"], job["title"], decision)
    st.rerun()


def render_history_entry(e: dict) -> None:
    """One record's full detail — score breakdown, draft, and Approve /
    Reject / Skip. Same decide buttons as the Run page, so a job you left
    undecided there isn't stranded once its buttons scroll out of that
    session — you can always come back and decide from here."""
    job = e["job"]
    score = e.get("score")
    decision = e.get("decision") or "undecided"
    badge = {"approved": "✅ approved", "rejected": "❌ rejected",
             "skipped": "⏭ skipped"}.get(decision, "🕓 undecided")
    score_label = f"{score:.0f}/100" if score is not None else "—"
    tag = f"🏷️ {job['archetype']}  " if job.get("archetype") else ""
    legitimacy_tag = _legitimacy_tag(job)
    with st.expander(f"{score_label} — **{job['title']}** @ "
                     f"{job['company']}  {tag}{legitimacy_tag}{badge}"):
        _legitimacy_section(job)
        meta, dims = st.columns([1, 2])
        with meta:
            st.caption(f"{job.get('location') or '—'} · "
                      f"{job.get('remote', '')} · via {_source_label(job)}")
            st.link_button("Open job posting ↗", job["url"],
                           key=f"hist_link_{job['id']}")
            st.metric("Weighted score", score_label)
            st.caption(f"Decision: **{decision}**")
            if e.get("decided_at"):
                st.caption(f"Date applied: {e['decided_at'][:10]}")
        with dims:
            for dim, d in (e.get("dimensions") or {}).items():
                clamped = max(0, min(int(d["score"]), 100))
                st.progress(clamped / 100,
                           text=f"**{dim.replace('_', ' ')} — {d['score']}**"
                                f"  ·  {d['reason']}")
        if e.get("summary"):
            st.markdown(f"*{e['summary']}*")

        _contacts_section(job, e, key=f"hist_contacts_{job['id']}")

        if e.get("cover_letter"):
            st.divider()
            st.text_area("✉️ Cover letter", e["cover_letter"], height=280,
                         key=f"hist_letter_{job['id']}")
            if e.get("resume_tweaks"):
                st.markdown("**Resume tweaks**")
                st.markdown(e["resume_tweaks"])
            if e.get("review_notes"):
                st.markdown("**🔍 Reviewer notes**")
                st.caption(e["review_notes"])
            kw = e.get("keyword_coverage")
            if kw and (kw["covered"] or kw["missing"]):
                total = len(kw["covered"]) + len(kw["missing"])
                st.markdown(f"**🔑 Keyword coverage: "
                           f"{len(kw['covered'])}/{total}**")
                st.caption(f"✅ {', '.join(kw['covered']) or '—'}")
                st.caption(f"⚠️ {', '.join(kw['missing']) or '—'}")
            _cv_download_button(job, e, key=f"hist_cv_{job['id']}")

        st.divider()
        c1, c2, c3 = st.columns(3)
        if c1.button("✅ Approve", key=f"hist_approve_{job['id']}",
                     type="primary", disabled=decision == "approved"):
            _history_decide(job, "approved")
        if c2.button("❌ Reject", key=f"hist_reject_{job['id']}",
                     disabled=decision == "rejected"):
            _history_decide(job, "rejected")
        if c3.button("⏭ Skip", key=f"hist_skip_{job['id']}",
                     disabled=decision == "skipped"):
            _history_decide(job, "skipped")


# ---------------------------------------------------------------------------
# History Vault page
# ---------------------------------------------------------------------------

def page_vault() -> None:
    """Dedicated Vault page — shows only Approved jobs as premium cards
    with full details and a prominent Apply link."""
    st.markdown("""
    <h1 style="
        font-family:'Inter',sans-serif; font-weight:800;
        background: linear-gradient(90deg, #a78bfa 0%, #f5c842 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-size: 2rem; margin-bottom: 0;
    ">🏦 Your Vault</h1>
    <p style="color:#6b7a99; font-size:0.88rem; margin-top:0.3rem; font-family:'Inter',sans-serif;">
        Jobs you’ve approved. Apply at your own pace — Scout never submits on your behalf.
    </p>
    """, unsafe_allow_html=True)

    entries = records.all()
    approved = [e for e in entries if e.get("decision") == "approved"]

    if not approved:
        st.markdown("""
        <div style="
            text-align:center; padding: 4rem 2rem;
            background: rgba(124,106,255,0.06);
            border: 1px dashed rgba(167,139,250,0.25);
            border-radius: 18px; margin-top: 2rem;
        ">
            <div style="font-size:3rem; margin-bottom:1rem;">🔭</div>
            <div style="font-size:1.1rem; font-weight:700; color:#c4b5fd; font-family:'Inter',sans-serif;">
                Your vault is empty
            </div>
            <div style="font-size:0.88rem; color:#6b7a99; margin-top:0.5rem; font-family:'Inter',sans-serif;">
                Go to 🚀 Search Jobs, score some jobs, and click ✅ Approve on the ones you want to pursue.
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Summary bar
    m1, m2, m3 = st.columns(3)
    m1.metric("🏦 Approved", len(approved))
    pending_apply = sum(1 for e in approved
                       if not e.get("decided_at") or
                       not e["job"].get("url"))
    m2.metric("📎 Has Apply Link",
              sum(1 for e in approved if e["job"].get("url")))
    m3.metric("📅 Latest Approved",
              (approved[0].get("decided_at") or "")[:10] or "—")
    st.divider()

    # Sort: newest approved first
    approved_sorted = sorted(
        approved,
        key=lambda e: e.get("decided_at") or e.get("updated") or "",
        reverse=True
    )

    for idx, e in enumerate(approved_sorted):
        job = e["job"]
        score = e.get("score")
        approved_on = (e.get("decided_at") or "")[:10] or "—"
        location = job.get("location") or "—"
        remote = job.get("remote") or ""
        remote_label = {
            "remote": "🏠 Remote",
            "hybrid": "↔️ Hybrid",
            "onsite": "🏢 On-site",
        }.get(remote, remote.capitalize() if remote else "")
        source = _source_label(job)
        score_str = f"{score:.0f}/100" if score is not None else "—"

        # Render vault card HTML header
        st.markdown(f"""
        <div class="vault-card">
            <div class="vault-card-title">{job['title']}</div>
            <div class="vault-card-company">{job['company']}</div>
            <div class="vault-card-meta">
                <span>📍 {location}</span>
                {f'<span>{remote_label}</span>' if remote_label else ''}
                <span>📊 <span class="vault-score-badge">{score_str}</span></span>
                <span class="vault-approved-badge">✅ Approved {approved_on}</span>
                <span style="color:#4b5680">via {source}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Action row
        col_apply, col_letter, col_remove, _ = st.columns([1.4, 1.2, 1, 3])

        if job.get("url"):
            col_apply.link_button(
                "🚀 Apply Now ↗", job["url"],
                use_container_width=True,
                help="Opens the original job posting in a new tab."
            )
        else:
            col_apply.caption("No apply link saved")

        has_letter = bool(e.get("cover_letter"))
        show_letter_key = f"vault_show_letter_{idx}"
        if has_letter:
            if col_letter.button(
                "✉️ Cover Letter",
                key=f"vault_letter_btn_{idx}",
                use_container_width=True,
                help="Toggle cover letter preview"
            ):
                st.session_state[show_letter_key] = \
                    not st.session_state.get(show_letter_key, False)

        if col_remove.button(
            "❌ Remove",
            key=f"vault_remove_{idx}",
            help="Move back to Undecided. You can re-approve from History."
        ):
            records.upsert(job, decision="undecided")
            memory.mark_seen(job["id"], job["title"], "undecided")
            st.rerun()

        # Cover letter (toggle)
        if has_letter and st.session_state.get(show_letter_key):
            with st.container():
                st.text_area(
                    "✉️ Cover letter",
                    e["cover_letter"],
                    height=260,
                    key=f"vault_letter_text_{idx}",
                    help="Your AI-drafted cover letter for this role."
                )
                if e.get("resume_tweaks"):
                    with st.expander("📄 Resume tweaks suggested for this role"):
                        st.markdown(e["resume_tweaks"])

        # Keyword coverage mini-section
        kw = e.get("keyword_coverage")
        if kw and (kw.get("covered") or kw.get("missing")):
            total_kw = len(kw["covered"]) + len(kw["missing"])
            with st.expander(
                f"🔑 Keyword coverage: {len(kw['covered'])}/{total_kw}",
                expanded=False
            ):
                c1, c2 = st.columns(2)
                c1.markdown("**✅ Covered**\n" + (', '.join(kw['covered']) or '—'))
                c2.markdown("**⚠️ Missing**\n" + (', '.join(kw['missing']) or '—'))

        st.markdown("<div style='margin-bottom:0.5rem'></div>",
                    unsafe_allow_html=True)

    st.divider()
    st.download_button(
        "⬇️ Export Vault (JSON)",
        data=json.dumps(approved_sorted, indent=2, default=str),
        file_name="scout_vault.json",
        mime="application/json",
        help="Download your approved jobs as a JSON file."
    )


def page_history() -> None:
    st.header("📚 History")
    entries = records.all()
    if not entries:
        st.info("No records yet — run a Search first.")
        return

    render_skill_gaps(entries)
    st.divider()

    df = pd.DataFrame([{
        "updated": e.get("updated", ""),
        "score": e.get("score"),
        "title": e["job"]["title"],
        "company": e["job"]["company"],
        "archetype": e["job"].get("archetype") or "Unclassified",
        "legitimacy": (e["job"].get("legitimacy") or {}).get(
            "tier", "high_confidence"),
        "location": e["job"].get("location", ""),
        "source": e["job"].get("source", ""),
        "publisher": e["job"].get("publisher") or "",
        "decision": e.get("decision") or "undecided",
        "date_applied": (e.get("decided_at") or "")[:10],
        "url": e["job"]["url"],
    } for e in entries])
    st.dataframe(
        df, width="stretch", hide_index=True,
        column_config={
            "url": st.column_config.LinkColumn("posting", display_text="open ↗"),
            "score": st.column_config.NumberColumn(format="%.0f"),
            "date_applied": st.column_config.TextColumn("date applied"),
            "publisher": st.column_config.TextColumn(
                "publisher", help="Origin board for aggregator sources "
                "(e.g. Glassdoor via jsearch)"),
            "archetype": st.column_config.TextColumn(
                "archetype", help="Deterministic role-category tag from "
                "your profile.yaml `archetypes` (or the built-in default)"),
            "legitimacy": st.column_config.TextColumn(
                "legitimacy", help="Block G ghost-job/scam heuristic — "
                "high_confidence / caution / suspicious. A heuristic, "
                "not certainty."),
        })

    archetypes_present = sorted(
        {e["job"].get("archetype") or "Unclassified" for e in entries})
    pick = st.selectbox("🏷️ Filter by archetype",
                        ["All"] + archetypes_present)
    if pick != "All":
        entries = [e for e in entries
                  if (e["job"].get("archetype") or "Unclassified") == pick]

    st.subheader("🔍 Job details")
    st.caption("Grouped by decision. Undecided is where a job lands if "
              "you never clicked Approve/Reject/Skip on the Search page — "
              "decide on it here any time. Approved jobs also appear in the "
              "🏦 Vault with their apply links.")

    with st.container(horizontal=True):
        sort_by = st.multiselect(
            "Sort by", list(SORT_FIELDS), default=["Match rating"],
            key="hist_sort",
            help="Pick more than one to sort like SQL's ORDER BY: the "
                 "first field decides, the next one breaks its ties.")
        order = st.selectbox("Order", ["Descending", "Ascending"],
                             key="hist_order")
    if sort_by:
        entries = sorted(
            entries, key=lambda e: tuple(SORT_FIELDS[f](e) for f in sort_by),
            reverse=order == "Descending")

    buckets: dict[str, list[dict]] = {key: [] for key, _ in DECISION_CATEGORIES}
    for e in entries:
        buckets[e.get("decision") or "undecided"].append(e)

    labels = [f"{label} ({len(buckets[key])})"
              for key, label in DECISION_CATEGORIES]
    # Tab labels carry live counts, so the archetype filter rewrites them —
    # remap the stored selection onto the new label instead of letting an
    # unmatched value silently bounce the user back to the first tab.
    stored = st.session_state.get("hist_tabs")
    if stored and stored not in labels:
        stem = stored.rsplit(" (", 1)[0]
        st.session_state["hist_tabs"] = next(
            (l for l in labels if l.startswith(stem)), labels[0])
    # Sidebar "Show approved jobs →" lands here: preselect its tab by
    # writing the tab widget's state BEFORE st.tabs() instantiates it, and
    # pop the flag so a later manual tab switch isn't overridden.
    focus = st.session_state.pop("history_focus", None)
    keys = [key for key, _ in DECISION_CATEGORIES]
    if focus in keys:
        st.session_state["hist_tabs"] = labels[keys.index(focus)]
    tabs = st.tabs(labels, key="hist_tabs")
    for tab, (key, _) in zip(tabs, DECISION_CATEGORIES):
        with tab:
            if not buckets[key]:
                st.caption("Nothing here.")
            for e in buckets[key]:
                render_history_entry(e)

    c1, c2, c3 = st.columns(3)
    c1.download_button(
        "⬇️ Export all records (JSON)",
        data=json.dumps(records.all(), indent=2, default=str),
        file_name="scout_records.json", mime="application/json")
    if notion_sync.available():
        if c2.button("📤 Sync history to Notion",
                     help="Mirrors job metadata, scores, and decisions "
                          "into your own Notion database (set NOTION_API_KEY "
                          "and NOTION_DATABASE_ID). Cover letters and your "
                          "personal details never leave this machine. "
                          "Re-syncing updates existing pages, no duplicates."):
            with st.spinner("Syncing records to Notion…"):
                result = notion_sync.sync_records(records)
            if result is None:
                st.error("Couldn't read your Notion database — check the "
                         "key, the database id, and that the database is "
                         "shared with your integration (••• → Connections).")
            else:
                created, updated, failed = result
                msg = f"Notion: {created} created, {updated} updated"
                if failed:
                    st.warning(f"{msg}, {failed} failed.")
                else:
                    st.success(f"{msg}.")
        if c3.button("🔁 Pull decisions from Notion",
                     help="If you changed a job's Decision cell directly in "
                          "Notion (e.g. from your phone), this reads that "
                          "back and applies it here. Notion has no way to "
                          "it requires a public server JobScout doesn't "
                          "run — so pulling on demand is the way to pick "
                          "up decisions made there."):
            with st.spinner("Checking Notion for decisions…"):
                pulled = notion_sync.pull_decisions(records, memory)
            if pulled is None:
                st.error("Couldn't read your Notion database — check the "
                         "key, the database id, and that it's shared with "
                         "your integration.")
            else:
                applied, checked = pulled
                if applied:
                    st.success(f"Applied {applied} decision(s) made in "
                              f"Notion (checked {checked} synced record(s)).")
                    st.rerun()
                else:
                    st.info(f"No new decisions found ({checked} synced "
                           "record(s) checked).")


# ---------------------------------------------------------------------------
# Page 3 — Outreach CRM
# ---------------------------------------------------------------------------

def page_outreach_crm() -> None:
    st.header("📬 Outreach CRM & Follow-up Sequencer")
    st.caption("Manage personalized outreach campaigns, track responses, and automate multi-touch follow-ups.")

    all_campaigns = outreach_tracker.list_records()
    due_items = outreach_tracker.list_due_followups()
    active_count = len([c for c in all_campaigns if c.overall_status == "active"])
    replied_count = len([c for c in all_campaigns if c.overall_status in ("replied", "interview")])

    # Metric Banner
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Campaigns", len(all_campaigns))
    m2.metric("Active Sequences", active_count)
    m3.metric("Follow-ups Due Today", len(due_items))
    m4.metric("Replied / Interviews", replied_count)

    st.divider()

    # Dispatcher Mode Banner
    is_dry = email_dispatcher.dry_run
    mode_badge = "🟡 Safe Dry Run Mode (Simulating dispatches to logs/outreach_sent.jsonl)" if is_dry else "🟢 Live SMTP Dispatch Mode"
    st.info(f"**Email Engine Status:** {mode_badge}\n\n*Configure SMTP_USER, SMTP_PASSWORD, and OUTREACH_DRY_RUN=false in `.env` for live sending.*")

    # Follow-up Automation Action
    c_proc, c_filt = st.columns([1, 2])
    with c_proc:
        if st.button("⚡ Process All Due Follow-ups Now", type="primary", disabled=len(due_items) == 0):
            with st.spinner("Dispatching due follow-ups…"):
                res = outreach_sequencer.process_due_followups()
            st.success(f"Processed {res.processed_count} follow-ups: {res.sent_count} sent successfully, {res.failed_count} failed.")
            st.rerun()

    status_options = ["all", "drafted", "active", "completed_sequence", "replied", "interview", "declined", "archived"]
    selected_status = c_filt.selectbox("Filter by Status", status_options, index=0)

    displayed_campaigns = all_campaigns if selected_status == "all" else [c for c in all_campaigns if c.overall_status == selected_status]

    if not displayed_campaigns:
        st.caption("No outreach campaigns found matching the filter. Run JobScout and click 'Draft Cold Outreach Sequence' on any job card.")
        return

    for camp in displayed_campaigns:
        with st.expander(f"**{camp.company}** — {camp.job_title} · `{camp.contact_email}` [{camp.overall_status.upper()}]"):
            c_meta, c_actions = st.columns([2, 1])
            with c_meta:
                c_name = camp.contact_name or "Hiring Team"
                c_pos = f" ({camp.contact_position})" if camp.contact_position else ""
                st.markdown(f"**Contact:** {c_name}{c_pos} · `{camp.contact_email}`")
                st.caption(f"Created: {camp.created_at[:10]} · Last Updated: {camp.updated_at[:10]}")
                if camp.linkedin_note:
                    st.caption(f"**LinkedIn Note:** {camp.linkedin_note}")

            with c_actions:
                new_st = st.selectbox("Status", status_options[1:], 
                                      index=status_options[1:].index(camp.overall_status) if camp.overall_status in status_options[1:] else 0,
                                      key=f"status_sel_{camp.id}")
                if new_st != camp.overall_status:
                    outreach_tracker.update_status(camp.id, new_st)
                    st.success(f"Updated status to {new_st}")
                    st.rerun()

            st.markdown("**Sequence Touchpoints:**")
            for t_idx, t in enumerate(camp.touches):
                t_num = t.get("touch_number", t_idx + 1)
                t_status = t.get("status", "pending")
                t_date = t.get("scheduled_date", "N/A")
                t_sent = f" (sent at {t.get('sent_at')[:16]})" if t.get("sent_at") else ""
                
                col_t_info, col_t_act = st.columns([3, 1])
                with col_t_info:
                    st.markdown(f"- **Touch {t_num}:** *{t.get('subject')}* — `{t_status}` [Sched: {t_date}]{t_sent}")
                    st.caption(f"Snippet: {t.get('body', '')[:120]}…")
                with col_t_act:
                    if t_status == "pending":
                        if st.button(f"Send Touch {t_num}", key=f"send_crm_{camp.id}_{t_idx}"):
                            d_res = email_dispatcher.send_email(
                                to_email=camp.contact_email,
                                subject=t.get("subject", ""),
                                body_text=t.get("body", ""),
                            )
                            if d_res.success:
                                outreach_tracker.mark_touch_sent(camp.id, t_idx)
                                st.success(f"Touch {t_num} dispatched!")
                                st.rerun()
                            else:
                                st.error(f"Error: {d_res.error}")

            # Notes section
            st.markdown("**Campaign Notes:**")
            notes_val = st.text_area("Notes", camp.notes, key=f"notes_{camp.id}", height=68)
            if st.button("Save Notes", key=f"save_notes_{camp.id}"):
                camp.notes = notes_val
                outreach_tracker.save_record(camp)
                st.success("Notes saved.")


# ---------------------------------------------------------------------------

if page == "👤 Profile":
    page_profile()
elif page == "🚀 Search Jobs":
    page_run()
elif page == "🏦 Vault":
    page_vault()
elif page == "📬 Outreach":
    page_outreach_crm()
else:
    page_history()
