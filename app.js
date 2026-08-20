/* ═══════════════════════════════════════════════════════════════
   SCOUT — STARK INDUSTRIES / IRON MAN HUD ENGINE (app.js)
   ═══════════════════════════════════════════════════════════════ */

document.addEventListener('DOMContentLoaded', () => {
    initCanvasAnimation();
    initNavigation();
    loadProfileData();
    loadVaultData();
    loadHistoryData();
    initEventListeners();
});

// ---------------------------------------------------------------------------
// 1. Navigation & View Routing
// ---------------------------------------------------------------------------

function initNavigation() {
    const navItems = document.querySelectorAll('.nav-item');
    const panes = document.querySelectorAll('.tab-pane');

    navItems.forEach(item => {
        item.addEventListener('click', () => {
            const targetTab = item.getAttribute('data-tab');
            navItems.forEach(n => n.classList.remove('active'));
            panes.forEach(p => p.classList.remove('active'));

            item.classList.add('active');
            const activePane = document.getElementById(`tab-${targetTab}`);
            if (activePane) activePane.classList.add('active');

            if (targetTab === 'vault') loadVaultData();
            if (targetTab === 'history') loadHistoryData();
            if (targetTab === 'outreach') loadOutreachData();
        });
    });

    // Sidebar quick stats click
    document.getElementById('stat-reviewed-card')?.addEventListener('click', () => {
        document.querySelector('[data-tab="history"]')?.click();
    });
    document.getElementById('stat-vault-card')?.addEventListener('click', () => {
        document.querySelector('[data-tab="vault"]')?.click();
    });
}

// ---------------------------------------------------------------------------
// 2. Profile Management & Auto-Extract
// ---------------------------------------------------------------------------

let currentProfile = null;

async function loadProfileData() {
    try {
        const res = await fetch('/api/profile');
        const data = await res.json();
        if (!data || !data.profile) return;

        currentProfile = data.profile;
        const cand = currentProfile.candidate || {};
        const prefs = currentProfile.preferences || {};

        document.getElementById('cand-name').value = cand.name || '';
        document.getElementById('cand-email').value = cand.email || '';
        document.getElementById('cand-phone').value = cand.phone || '';
        document.getElementById('cand-location').value = cand.address || '';
        document.getElementById('pref-roles').value = (prefs.target_roles || []).join(', ');
        document.getElementById('cand-skills').value = (cand.skills || []).join(', ');
        document.getElementById('cand-bio').value = cand.background_summary || '';

        // Populate Indian States Dropdown
        const stateSelect = document.getElementById('search-state');
        if (stateSelect && data.indian_states) {
            stateSelect.innerHTML = data.indian_states
                .map(s => `<option value="${s}">${s}</option>`)
                .join('');
        }

        // Update stats
        if (data.stats) {
            document.getElementById('stat-reviewed').textContent = data.stats.reviewed || 0;
            document.getElementById('stat-vault').textContent = data.stats.approved || 0;
        }
    } catch (err) {
        console.error('Failed to load profile:', err);
    }
}

async function saveProfileData() {
    if (!currentProfile) currentProfile = {};
    const cand = currentProfile.candidate || {};
    const prefs = currentProfile.preferences || {};

    cand.name = document.getElementById('cand-name').value.trim();
    cand.email = document.getElementById('cand-email').value.trim();
    cand.phone = document.getElementById('cand-phone').value.trim();
    cand.address = document.getElementById('cand-location').value.trim();
    cand.background_summary = document.getElementById('cand-bio').value.trim();
    cand.skills = document.getElementById('cand-skills').value.split(',').map(s => s.trim()).filter(Boolean);

    prefs.target_roles = document.getElementById('pref-roles').value.split(',').map(s => s.trim()).filter(Boolean);

    currentProfile.candidate = cand;
    currentProfile.preferences = prefs;

    try {
        const res = await fetch('/api/profile', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ profile: currentProfile }),
        });
        const data = await res.json();
        alert('✅ Profile data saved to Stark Intelligence database.');
    } catch (err) {
        alert('❌ Error saving profile: ' + err.message);
    }
}

async function handleResumeExtract() {
    const fileInput = document.getElementById('resume-upload');
    const statusBox = document.getElementById('extract-status');

    if (!fileInput.files || !fileInput.files[0]) {
        alert('Please select a PDF resume first.');
        return;
    }

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    statusBox.textContent = '🤖 J.A.R.V.I.S. analyzing resume & extracting skills...';
    statusBox.classList.remove('hidden');

    try {
        const res = await fetch('/api/profile/extract-resume', {
            method: 'POST',
            body: formData,
        });
        const data = await res.json();
        if (data.status === 'success' && data.extracted) {
            const ext = data.extracted;
            if (ext.candidate) {
                if (ext.candidate.name) document.getElementById('cand-name').value = ext.candidate.name;
                if (ext.candidate.email) document.getElementById('cand-email').value = ext.candidate.email;
                if (ext.candidate.phone) document.getElementById('cand-phone').value = ext.candidate.phone;
                if (ext.candidate.background_summary) document.getElementById('cand-bio').value = ext.candidate.background_summary;
                if (ext.candidate.skills) document.getElementById('cand-skills').value = ext.candidate.skills.join(', ');
            }
            if (ext.recommended_target_roles) {
                document.getElementById('pref-roles').value = ext.recommended_target_roles.join(', ');
            }
            statusBox.textContent = '✨ Profile extracted successfully! Review fields and click Save.';
            setTimeout(() => statusBox.classList.add('hidden'), 5000);
        } else {
            statusBox.textContent = '❌ Extraction failed: ' + (data.detail || 'Unknown error');
        }
    } catch (err) {
        statusBox.textContent = '❌ Network error during resume extraction: ' + err.message;
    }
}

// ---------------------------------------------------------------------------
// 3. Search & Scoring Engine
// ---------------------------------------------------------------------------

async function handleSearch() {
    const workMode = document.getElementById('search-work-mode').value;
    const state = document.getElementById('search-state').value;
    const targetCount = parseInt(document.getElementById('search-target-count').value, 10) || 6;

    const loadingState = document.getElementById('search-loading');
    const cardsList = document.getElementById('search-cards-list');

    loadingState.classList.remove('hidden');
    cardsList.innerHTML = '';

    try {
        const res = await fetch('/api/search', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                work_mode: workMode,
                selected_state: state,
                target_count: targetCount,
            }),
        });
        const data = await res.json();
        loadingState.classList.add('hidden');

        if (!data || !data.jobs || data.jobs.length === 0) {
            cardsList.innerHTML = `<div class="hud-card"><p>No fresh jobs matching criteria. Try broadening state or workplace filters.</p></div>`;
            return;
        }

        renderSearchJobs(data.jobs);
        loadProfileData(); // refresh counts
    } catch (err) {
        loadingState.classList.add('hidden');
        cardsList.innerHTML = `<div class="hud-card"><p class="text-crimson">❌ Search failed: ${err.message}</p></div>`;
    }
}

function renderSearchJobs(packages) {
    const cardsList = document.getElementById('search-cards-list');
    cardsList.innerHTML = packages.map(pkg => {
        const job = pkg.job;
        const score = Math.round(pkg.score || 0);
        const remoteLabel = job.remote === 'remote' ? '🏠 Remote' : (job.remote === 'hybrid' ? '↔️ Hybrid' : '🏢 On-site');
        const archetypeTag = job.archetype ? `<span class="hud-tag">🏷️ ${job.archetype}</span>` : '';

        return `
            <div class="job-card" id="card-${job.id}">
                <div class="job-card-top">
                    <div>
                        <div class="job-card-title">${job.title}</div>
                        <div class="job-card-company">${job.company} ${archetypeTag}</div>
                        <div class="job-card-meta">
                            <span>📍 ${job.location || 'India'}</span>
                            <span>${remoteLabel}</span>
                            <span>📡 ${job.source || 'Aggregator'}</span>
                        </div>
                    </div>
                    <div class="score-badge">${score}/100</div>
                </div>
                <div class="card-desc" style="margin-top:0.8rem;">
                    <em>${pkg.summary || ''}</em>
                </div>
                <div class="card-actions">
                    <button class="btn btn-primary" onclick="decideJob('${job.id}', 'approved')">
                        ✅ APPROVE TO VAULT
                    </button>
                    <button class="btn btn-outline" onclick="draftMaterials('${job.id}')">
                        ✉️ DRAFT COVER LETTER
                    </button>
                    <a href="${job.url}" target="_blank" class="btn btn-cyan">
                        🚀 APPLY LINK ↗
                    </a>
                    <button class="btn btn-outline" onclick="decideJob('${job.id}', 'skipped')">
                        ⏭ SKIP
                    </button>
                </div>
            </div>
        `;
    }).join('');
}

// ---------------------------------------------------------------------------
// 4. Vault & History Handlers
// ---------------------------------------------------------------------------

async function loadVaultData() {
    try {
        const res = await fetch('/api/vault');
        const data = await res.json();
        const list = document.getElementById('vault-list');
        const countBadge = document.getElementById('vault-count-badge');
        const linksBadge = document.getElementById('vault-links-badge');

        const items = data.items || [];
        countBadge.textContent = items.length;
        linksBadge.textContent = items.filter(i => i.job && i.job.url).length;
        document.getElementById('stat-vault').textContent = items.length;

        if (items.length === 0) {
            list.innerHTML = `
                <div class="hud-card" style="text-align:center; padding:3rem;">
                    <div style="font-size:2.5rem; margin-bottom:0.5rem;">⚛️</div>
                    <h3>VAULT IS EMPTY</h3>
                    <p class="card-desc">Search and approve jobs to save them with direct apply links.</p>
                </div>
            `;
            return;
        }

        list.innerHTML = items.map(entry => {
            const job = entry.job;
            const score = Math.round(entry.score || 0);
            const decidedOn = (entry.decided_at || '').substring(0, 10) || 'Recently';

            return `
                <div class="job-card">
                    <div class="job-card-top">
                        <div>
                            <div class="job-card-title">${job.title}</div>
                            <div class="job-card-company">${job.company}</div>
                            <div class="job-card-meta">
                                <span>📍 ${job.location || 'India'}</span>
                                <span class="score-badge">${score}/100</span>
                                <span style="color:#4ade80;">✅ Approved: ${decidedOn}</span>
                            </div>
                        </div>
                        <a href="${job.url}" target="_blank" class="btn btn-cyan" style="font-size:0.9rem; padding:0.7rem 1.4rem;">
                            🚀 APPLY NOW ↗
                        </a>
                    </div>
                    <div class="card-actions">
                        ${entry.cover_letter ? `
                            <button class="btn btn-outline" onclick="viewLetterModal('${job.title}', \`${encodeURIComponent(entry.cover_letter)}\`, \`${encodeURIComponent(entry.resume_tweaks || '')}\`)">
                                ✉️ VIEW COVER LETTER
                            </button>
                        ` : `
                            <button class="btn btn-outline" onclick="draftMaterials('${job.id}')">
                                ✍️ GENERATE COVER LETTER
                            </button>
                        `}
                        <button class="btn btn-outline" onclick="decideJob('${job.id}', 'undecided')">
                            ❌ REMOVE FROM VAULT
                        </button>
                    </div>
                </div>
            `;
        }).join('');
    } catch (err) {
        console.error('Failed to load vault:', err);
    }
}

async function loadHistoryData() {
    try {
        const res = await fetch('/api/history');
        const data = await res.json();
        const gapsContainer = document.getElementById('gaps-container');
        const list = document.getElementById('history-list');

        // Render recurring gaps
        if (data.gaps && data.gaps.length > 0) {
            gapsContainer.innerHTML = data.gaps.map(g => `
                <div style="margin-bottom:0.8rem;">
                    <div style="display:flex; justify-content:space-between; font-family:var(--font-tech); font-weight:700; font-size:0.85rem; color:var(--amber);">
                        <span>${g.dimension.replace('_', ' ').toUpperCase()}</span>
                        <span>${g.avg_score}/100 AVG (SCORED ON ${g.count} JOBS)</span>
                    </div>
                    <div style="background:rgba(230,36,41,0.15); height:6px; border-radius:3px; overflow:hidden; margin-top:4px;">
                        <div style="background:linear-gradient(90deg, #e62429, #ffd700, #00f0ff); width:${Math.min(100, Math.max(0, g.avg_score))}%; height:100%;"></div>
                    </div>
                </div>
            `).join('');
        }

        // Render records list
        const records = data.records || [];
        document.getElementById('stat-reviewed').textContent = records.length;

        if (records.length === 0) {
            list.innerHTML = `<p class="card-desc">No evaluated postings in ledger yet.</p>`;
            return;
        }

        list.innerHTML = records.map(entry => {
            const job = entry.job;
            const score = Math.round(entry.score || 0);
            const decision = entry.decision || 'undecided';
            const decisionColor = decision === 'approved' ? '#4ade80' : (decision === 'rejected' ? '#ff4d4d' : '#ffd700');

            return `
                <div class="job-card" style="padding:1rem 1.4rem;">
                    <div class="job-card-top">
                        <div>
                            <div class="job-card-title" style="font-size:1rem;">${job.title}</div>
                            <div class="job-card-company">${job.company} · 📍 ${job.location || 'India'}</div>
                        </div>
                        <div style="display:flex; align-items:center; gap:0.6rem;">
                            <span class="score-badge">${score}/100</span>
                            <span style="font-family:var(--font-tech); font-weight:700; color:${decisionColor}; text-transform:uppercase;">${decision}</span>
                            <a href="${job.url}" target="_blank" class="btn btn-cyan" style="padding:0.35rem 0.75rem; font-size:0.75rem;">↗</a>
                        </div>
                    </div>
                </div>
            `;
        }).join('');
    } catch (err) {
        console.error('Failed to load history:', err);
    }
}

async function decideJob(jobId, decision) {
    try {
        const res = await fetch('/api/vault/decide', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ job_id: jobId, decision }),
        });
        const data = await res.json();
        loadVaultData();
        loadHistoryData();
        loadProfileData();

        // If on search page, update button visually
        const card = document.getElementById(`card-${jobId}`);
        if (card && decision === 'approved') {
            card.style.borderColor = '#00f0ff';
        }
    } catch (err) {
        alert('Error updating decision: ' + err.message);
    }
}

async function draftMaterials(jobId) {
    try {
        alert('✍️ AI is drafting your tailored cover letter and resume tweaks...');
        const res = await fetch('/api/draft', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ job_id: jobId }),
        });
        const data = await res.json();
        if (data.status === 'success' && data.drafts) {
            viewLetterModal('Tailored Application', encodeURIComponent(data.drafts.cover_letter || ''), encodeURIComponent(data.drafts.resume_tweaks || ''));
            loadVaultData();
        }
    } catch (err) {
        alert('Drafting error: ' + err.message);
    }
}

function viewLetterModal(title, letterEncoded, tweaksEncoded) {
    const modal = document.getElementById('letter-modal');
    document.getElementById('modal-job-title').textContent = decodeURIComponent(title);
    document.getElementById('modal-letter-text').value = decodeURIComponent(letterEncoded);
    document.getElementById('modal-tweaks-text').innerHTML = decodeURIComponent(tweaksEncoded)
        .replace(/\n/g, '<br>');
    modal.classList.remove('hidden');
}

// ---------------------------------------------------------------------------
// 5. Outreach CRM Loader
// ---------------------------------------------------------------------------

async function loadOutreachData() {
    const container = document.getElementById('outreach-container');
    try {
        const res = await fetch('/api/outreach');
        const data = await res.json();
        const campaigns = data.campaigns || [];

        if (campaigns.length === 0) {
            container.innerHTML = `
                <div class="hud-card" style="text-align:center; padding:3rem;">
                    <div style="font-size:2.5rem; margin-bottom:0.5rem;">📬</div>
                    <h3>NO ACTIVE OUTREACH CAMPAIGNS</h3>
                    <p class="card-desc">Approve jobs and draft sequences to track recruiter contacts.</p>
                </div>
            `;
            return;
        }

        container.innerHTML = campaigns.map(c => `
            <div class="job-card">
                <div class="job-card-top">
                    <div>
                        <div class="job-card-title">${c.job_title}</div>
                        <div class="job-card-company">${c.company}</div>
                        <div class="job-card-meta">
                            <span>Status: ${c.overall_status}</span>
                            <span>Touches: ${c.touches.length}</span>
                        </div>
                    </div>
                </div>
            </div>
        `).join('');
    } catch (err) {
        container.innerHTML = `<p>Error loading outreach: ${err.message}</p>`;
    }
}

// ---------------------------------------------------------------------------
// 6. Global Event Listeners
// ---------------------------------------------------------------------------

function initEventListeners() {
    document.getElementById('btn-save-profile')?.addEventListener('click', saveProfileData);
    document.getElementById('btn-extract-resume')?.addEventListener('click', handleResumeExtract);
    document.getElementById('btn-run-search')?.addEventListener('click', handleSearch);
    document.getElementById('btn-close-modal')?.addEventListener('click', () => {
        document.getElementById('letter-modal').classList.add('hidden');
    });
    document.querySelector('.modal-backdrop')?.addEventListener('click', () => {
        document.getElementById('letter-modal').classList.add('hidden');
    });
}

// ---------------------------------------------------------------------------
// 7. Background Canvas: Hologram Radar & Supersonic Plasma Streaks
// ---------------------------------------------------------------------------

function initCanvasAnimation() {
    const canvas = document.getElementById('stark-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    let width, height;
    function resize() {
        width = canvas.width = window.innerWidth;
        height = canvas.height = window.innerHeight;
    }
    resize();
    window.addEventListener('resize', resize);

    const streaks = [];
    const NUM_STREAKS = 32;
    for (let i = 0; i < NUM_STREAKS; i++) {
        streaks.push({
            x: Math.random() * width,
            y: Math.random() * height,
            length: 45 + Math.random() * 85,
            speed: 3 + Math.random() * 5,
            width: 0.8 + Math.random() * 1.6,
            color: Math.random() < 0.6 ? '#ff1a22' : (Math.random() < 0.85 ? '#ff9900' : '#00f0ff'),
        });
    }

    let rot1 = 0;
    let rot2 = 0;
    let rot3 = 0;

    function drawHologram(cx, cy, radius) {
        ctx.save();
        ctx.translate(cx, cy);

        // Ambient Arc Pulse Core
        const grad = ctx.createRadialGradient(0, 0, 4, 0, 0, radius * 0.85);
        grad.addColorStop(0, 'rgba(255, 140, 0, 0.2)');
        grad.addColorStop(0.4, 'rgba(230, 36, 41, 0.1)');
        grad.addColorStop(1, 'transparent');
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(0, 0, radius * 0.85, 0, Math.PI * 2);
        ctx.fill();

        // Ring 1 (Broken Tech Ring)
        ctx.rotate(rot1);
        ctx.strokeStyle = 'rgba(255, 140, 0, 0.35)';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([16, 8, 4, 8]);
        ctx.beginPath();
        ctx.arc(0, 0, radius * 0.8, 0, Math.PI * 2);
        ctx.stroke();

        // Ring 2 (Concentric Coordinate Reticle)
        ctx.rotate(rot2 - rot1);
        ctx.strokeStyle = 'rgba(230, 36, 41, 0.4)';
        ctx.lineWidth = 1.2;
        ctx.setLineDash([30, 15, 60, 20]);
        ctx.beginPath();
        ctx.arc(0, 0, radius * 0.6, 0, Math.PI * 2);
        ctx.stroke();

        // Ring 3 (Arc Core Inner Nodes)
        ctx.rotate(rot3 - rot2);
        ctx.strokeStyle = 'rgba(255, 215, 0, 0.45)';
        ctx.lineWidth = 1.8;
        ctx.setLineDash([6, 12]);
        ctx.beginPath();
        ctx.arc(0, 0, radius * 0.4, 0, Math.PI * 2);
        ctx.stroke();

        ctx.restore();
    }

    function animate() {
        ctx.clearRect(0, 0, width, height);

        rot1 += 0.003;
        rot2 -= 0.004;
        rot3 += 0.0055;

        // Draw Hologram Radar at top right
        const hudX = width > 900 ? width * 0.84 : width * 0.5;
        const hudY = height * 0.35;
        const radius = width > 900 ? 170 : 120;
        drawHologram(hudX, hudY, radius);

        // Draw Supersonic Plasma Streaks
        streaks.forEach(s => {
            s.y -= s.speed;
            if (s.y + s.length < 0) {
                s.y = height + 15;
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
}

// Global scope helpers for inline onclicks
window.decideJob = decideJob;
window.draftMaterials = draftMaterials;
window.viewLetterModal = viewLetterModal;
