"""
streamlit_app.py — WorkPulse AI

Competition demo for Team 19 — Dallas AI Summer Program.

Five views:
  Overview   : system architecture, Human-AI gap, one-sentence pitch
  Live Demo  : check-in → cluster routing → decision trace → intervention
  AI Patterns: cluster explorer showing learned workforce stress archetypes
  Interventions: human-designed intervention library with evidence basis
  Team Review: project proposal summary for team voting

Session state:
  active_view          str   current view (default: "overview")
  active_track_id      str | None
  active_activity_id   str | None
  track_source         str | None
  selected_cluster_id  int | None
  progress_log         list[dict]
  demo_result          dict | None   output of route_user_input()
  checkin              CheckInResponse | None
  burnout              BurnoutScore | None
"""

import ast
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# ── Project root on sys.path so src.* imports resolve ─────────────────────────
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.checkin_engine import CheckInResponse          # noqa: E402
from src.compound_scorer import compute_burnout_score   # noqa: E402
from src.labeler import label_checkin_debrief           # noqa: E402
from src.analysis.personas import (                     # noqa: E402
    load_tracks, get_track, cluster_to_track_ids,
)

# Cluster router is optional — falls back gracefully if deps are missing
try:
    from src.cluster_router import route_user_input as _route_user_input
    _CLUSTER_ROUTER_OK = True
except Exception:
    _CLUSTER_ROUTER_OK = False


# ── Constants ─────────────────────────────────────────────────────────────────

_RISK_LABELS = {
    "low":      "Low — you're managing well today",
    "moderate": "Moderate — some strain is showing",
    "high":     "High — your tank is running low",
    "critical": "Critical — recovery should be a priority",
}
_RISK_COLORS = {
    "low":      "#27ae60",
    "moderate": "#e67e22",
    "high":     "#d35400",
    "critical": "#c0392b",
}
_RISK_EMOJI = {"low": "🟢", "moderate": "🟡", "high": "🟠", "critical": "🔴"}

_SETTING_LABELS = {
    "break_room": "Break room",
    "commute":    "Commute",
    "home":       "Home",
    "other":      "Other",
}

_VIEWS = ["overview", "demo", "explorer", "library", "proposal"]
_VIEW_LABELS = {
    "overview":  "🏠 Overview",
    "demo":      "⚡ Live Demo",
    "explorer":  "🔍 AI Patterns",
    "library":   "🎨 Interventions",
    "proposal":  "📋 Team Review",
}

_SOURCE_LABELS = {
    "check_in":   "Recommended from your check-in score",
    "text_route": "Matched from your text description",
    "cluster":    "Mapped from cluster selection",
    "manual":     "Manually selected",
}

# Dominant dimension → default track when no text input is given
_DIM_TO_TRACK = {"body": "restore", "mind": "ground", "life": "rebalance"}

_DATA_DIR = _ROOT / "data" / "processed"


# ── Cached data loaders ───────────────────────────────────────────────────────

@st.cache_data
def _load_cluster_data():
    report_path = _DATA_DIR / "cluster_report.csv"
    reps_path   = _DATA_DIR / "cluster_representatives.csv"
    if not report_path.exists() or not reps_path.exists():
        return None, None
    return pd.read_csv(report_path), pd.read_csv(reps_path)


@st.cache_data
def _load_tracks() -> list:
    return load_tracks()


# ── Session state ─────────────────────────────────────────────────────────────

_SS_DEFAULTS: dict[str, Any] = {
    "active_view":         "overview",
    "active_track_id":     None,
    "active_activity_id":  None,
    "track_source":        None,
    "selected_cluster_id": None,
    "progress_log":        [],
    "demo_result":         None,
    "checkin":             None,
    "burnout":             None,
}


def _init_state() -> None:
    for k, v in _SS_DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── Navigation helpers ────────────────────────────────────────────────────────

def _go(view: str) -> None:
    st.session_state["active_view"] = view
    st.rerun()


def _on_nav_change() -> None:
    st.session_state["active_view"] = st.session_state["_radio_nav"]


def _first_activity_id(track: dict) -> str | None:
    ivs = track.get("interventions", [])
    return ivs[0]["id"] if ivs else None


def _next_activity_id(track: dict, current_id: str) -> str | None:
    ids = [iv["id"] for iv in track.get("interventions", [])]
    if not ids:
        return None
    try:
        return ids[(ids.index(current_id) + 1) % len(ids)]
    except ValueError:
        return ids[0]


def _set_context(track_id: str, source: str) -> None:
    track = get_track(track_id)
    st.session_state["active_track_id"]    = track_id
    st.session_state["active_activity_id"] = _first_activity_id(track) if track else None
    st.session_state["track_source"]       = source


def _activate(track_id: str, source: str) -> None:
    _set_context(track_id, source)
    _go("library")


# ── Callbacks ─────────────────────────────────────────────────────────────────

def _on_cluster_change() -> None:
    cid = st.session_state.get("_cluster_sel")
    if cid is None:
        return
    st.session_state["selected_cluster_id"] = cid
    mapped = cluster_to_track_ids(int(cid))
    if mapped:
        _set_context(mapped[0], "cluster")


def _on_activity_track_change() -> None:
    tid = st.session_state.get("_act_track_sel")
    if tid:
        _set_context(tid, "manual")


def _on_activity_sel_change() -> None:
    aid = st.session_state.get("_act_sel")
    if aid:
        st.session_state["active_activity_id"] = aid


# ── Shared render helpers ─────────────────────────────────────────────────────

def _dim_progress(label: str, value: float) -> None:
    st.caption(f"{label}  —  {value:.0f} / 100")
    st.progress(int(value) / 100)


def _risk_banner(risk_level: str) -> None:
    color = _RISK_COLORS[risk_level]
    emoji = _RISK_EMOJI[risk_level]
    label = _RISK_LABELS[risk_level]
    st.markdown(
        f'<div style="background:{color};color:white;padding:12px 16px;'
        f'border-radius:8px;font-size:1.05em;font-weight:600;margin:8px 0 12px 0;">'
        f"{emoji}&nbsp; {label}</div>",
        unsafe_allow_html=True,
    )


def _delta_str(val: int) -> str:
    if val == 0:
        return "→"
    return f"{'↑' if val > 0 else '↓'}{abs(val)}"


def _ai_badge() -> None:
    st.markdown(
        '<div style="background:#1a3a5c;color:white;padding:8px 14px;'
        'border-radius:6px;margin-bottom:12px;font-size:0.9em;">'
        "🤖 <b>AI Layer</b> — pattern detection and routing"
        "</div>",
        unsafe_allow_html=True,
    )


def _human_badge() -> None:
    st.markdown(
        '<div style="background:#1a4a2e;color:white;padding:8px 14px;'
        'border-radius:6px;margin-bottom:12px;font-size:0.9em;">'
        "🧠 <b>Human Design Layer</b> — interventions designed by humans, not generated by AI"
        "</div>",
        unsafe_allow_html=True,
    )


def _render_now_doing_panel() -> None:
    track_id = st.session_state.get("active_track_id")
    if not track_id:
        return
    track = get_track(track_id)
    if not track:
        return
    activity_id = st.session_state.get("active_activity_id")
    activity = next(
        (iv for iv in track.get("interventions", []) if iv["id"] == activity_id), None
    )
    source_label = _SOURCE_LABELS.get(st.session_state.get("track_source") or "", "")

    with st.container(border=True):
        left, right = st.columns([5, 2])
        with left:
            st.markdown(f"**{track['title']}** &nbsp;·&nbsp; *{track['archetype']}*")
            if source_label:
                st.caption(source_label)
            if activity:
                env_str = ", ".join(
                    _SETTING_LABELS.get(e, e.title()) for e in activity.get("environment", [])
                )
                st.caption(
                    f"Next: **{activity['title']}** · "
                    f"⏱ {activity['duration_minutes']} min · 📍 {env_str}"
                )
        with right:
            if st.button("Continue →", type="primary", use_container_width=True, key="_now_doing_cta"):
                _go("library")


def _render_session_summary() -> None:
    log = st.session_state.get("progress_log", [])
    if not log:
        return
    st.markdown("---")
    st.markdown("##### Session progress")
    for entry in log:
        d_e = entry["after_energy"]  - entry["before_energy"]
        d_t = entry["after_tension"] - entry["before_tension"]
        d_c = entry["after_clarity"] - entry["before_clarity"]
        st.caption(
            f"**{entry['activity_title']}** ({entry['track_title']}) — "
            f"Energy {_delta_str(d_e)} · "
            f"Tension {_delta_str(-d_t)} · "
            f"Clarity {_delta_str(d_c)}"
        )


# ── Activity execution card ───────────────────────────────────────────────────

def _render_activity_card(track: dict, iv: dict, all_ivs: list) -> None:
    env_str   = ", ".join(_SETTING_LABELS.get(e, e.title()) for e in iv.get("environment", []))
    done_key  = f"_done_{track['track_id']}_{iv['id']}"
    saved_key = f"_saved_{track['track_id']}_{iv['id']}"

    with st.container(border=True):
        st.markdown(f"### {iv['title']}")
        st.caption(f"⏱ {iv['duration_minutes']} min  ·  📍 {env_str}")
        st.markdown("---")

        st.markdown("**Before you begin — how are you right now?**")
        st.caption("1 = very low  ·  5 = high  ·  takes 10 seconds")
        bc1, bc2, bc3 = st.columns(3)
        with bc1:
            st.select_slider("Energy", options=[1, 2, 3, 4, 5], value=3,
                             key=f"_be_{track['track_id']}_{iv['id']}")
        with bc2:
            st.select_slider("Tension", options=[1, 2, 3, 4, 5], value=3,
                             key=f"_bt_{track['track_id']}_{iv['id']}")
        with bc3:
            st.select_slider("Mental clarity", options=[1, 2, 3, 4, 5], value=3,
                             key=f"_bc_{track['track_id']}_{iv['id']}")
        st.markdown("---")

        st.markdown("**What to do:**")
        _steps = [ln.strip() for ln in iv["instructions"].strip().splitlines() if ln.strip()]
        st.markdown("  \n".join(_steps))

        with st.expander("Why this works"):
            st.write(iv["why_it_works"].strip())

        st.markdown("---")

        if not st.session_state.get(done_key):
            if st.button("Mark as done ✓", key=f"_btn_done_{iv['id']}",
                         type="primary", use_container_width=True):
                st.session_state[done_key] = True
                st.rerun()

        elif not st.session_state.get(saved_key):
            st.success("Done! How do you feel now?")
            ac1, ac2, ac3 = st.columns(3)
            with ac1:
                after_energy = st.select_slider("Energy ", options=[1, 2, 3, 4, 5], value=3,
                                                key=f"_ae_{track['track_id']}_{iv['id']}")
            with ac2:
                after_tension = st.select_slider("Tension ", options=[1, 2, 3, 4, 5], value=3,
                                                 key=f"_at_{track['track_id']}_{iv['id']}")
            with ac3:
                after_clarity = st.select_slider("Mental clarity ", options=[1, 2, 3, 4, 5], value=3,
                                                 key=f"_ac_{track['track_id']}_{iv['id']}")

            if st.button("Save rating", key=f"_btn_save_{iv['id']}",
                         type="primary", use_container_width=True):
                st.session_state["progress_log"].append({
                    "timestamp":      datetime.now().isoformat(timespec="seconds"),
                    "track_id":       track["track_id"],
                    "track_title":    track["title"],
                    "activity_id":    iv["id"],
                    "activity_title": iv["title"],
                    "before_energy":  st.session_state.get(f"_be_{track['track_id']}_{iv['id']}", 3),
                    "before_tension": st.session_state.get(f"_bt_{track['track_id']}_{iv['id']}", 3),
                    "before_clarity": st.session_state.get(f"_bc_{track['track_id']}_{iv['id']}", 3),
                    "after_energy":   after_energy,
                    "after_tension":  after_tension,
                    "after_clarity":  after_clarity,
                })
                st.session_state[saved_key] = True
                next_aid = _next_activity_id(track, iv["id"])
                if next_aid:
                    st.session_state["active_activity_id"] = next_aid
                st.rerun()

        else:
            log = st.session_state.get("progress_log", [])
            entry = next((e for e in reversed(log) if e["activity_id"] == iv["id"]), None)
            if entry:
                d_e = entry["after_energy"]  - entry["before_energy"]
                d_t = entry["after_tension"] - entry["before_tension"]
                d_c = entry["after_clarity"] - entry["before_clarity"]
                st.caption(
                    f"Rating saved — "
                    f"Energy {_delta_str(d_e)} · "
                    f"Tension {_delta_str(-d_t)} · "
                    f"Clarity {_delta_str(d_c)}"
                )
            next_aid = st.session_state.get("active_activity_id")
            if next_aid and next_aid != iv["id"]:
                next_iv = next((i for i in all_ivs if i["id"] == next_aid), None)
                if next_iv:
                    st.info(f"Next up: **{next_iv['title']}** ({next_iv['duration_minutes']} min)")
                    if st.button("Do next activity →", key=f"_btn_next_{iv['id']}",
                                 use_container_width=True):
                        st.rerun()


# ── Decision trace ────────────────────────────────────────────────────────────

def _render_decision_trace(result: dict, time_min: int, setting: str) -> None:
    """Display every routing step explicitly — the core explainability moment."""
    cm        = result.get("cluster_match", {})
    track_ids = result.get("track_ids", [])
    ivs       = result.get("interventions", [])

    # Navigate nested structure: ivs[i] = {category_id, intervention: {name, ...}, is_relevant, ...}
    first_cat = next((iv for iv in ivs if iv.get("is_relevant")), ivs[0] if ivs else None)
    first_iv  = first_cat.get("intervention", {}) if first_cat else {}
    first_name = first_iv.get("name", "—")
    first_dur  = first_iv.get("duration_min", "?")

    st.markdown("---")
    st.markdown("#### 🔍 Decision Trace — how this output was selected")
    st.caption("Every routing step is visible. No black boxes.")

    steps = [
        ("📝 Input", "Your text was embedded with sentence-BERT (all-MiniLM-L6-v2)"),
        ("🧠 Pattern matched",
         f"Cluster **{cm.get('cluster_id', '?')}** — *\"{cm.get('cluster_label', 'unknown')}\"*"),
        ("📊 Confidence",
         f"{cm.get('similarity', 0):.0%} cosine similarity · method: **{cm.get('method', 'unknown')}**"),
        ("⚙️ Constraints",
         f"{time_min} min available · {_SETTING_LABELS.get(setting, setting.replace('_', ' ').title())}"),
        ("🗺️ Track",
         ", ".join(f"**{t}**" for t in track_ids) if track_ids else "No track configured for this cluster"),
        ("🎯 Intervention",
         f"**{first_name}** ({first_dur} min)"),
    ]

    for label, detail in steps:
        col_l, col_r = st.columns([1.4, 3])
        with col_l:
            st.markdown(f"**{label}**")
        with col_r:
            st.markdown(detail)

    if cm.get("top_matches") and len(cm["top_matches"]) > 1:
        with st.expander("Alternative pattern matches"):
            for m in cm["top_matches"][1:]:
                st.caption(
                    f"Cluster {m['cluster_id']} — \"{m['cluster_label']}\" "
                    f"({m['similarity']:.0%} similarity)"
                )


def _render_vs_generic(cluster_label: str, track_ids: list, first_name: str, time_min: int) -> None:
    """Side-by-side: generic AI vs WorkPulse AI — the differentiation moment."""
    with st.expander("📊 Generic AI vs WorkPulse AI — what's the difference?"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Generic AI**")
            st.warning(
                "\"It sounds like you're stressed. Try taking a break, "
                "practice mindfulness, or speak to someone you trust. "
                "Self-care is important.\""
            )
            st.caption("No pattern detection · No constraint filtering · Same response for everyone")
        with col2:
            st.markdown("**WorkPulse AI**")
            track_str = " + ".join(track_ids) if track_ids else "matched track"
            st.success(
                f"Pattern: *\"{cluster_label}\"*\n\n"
                f"Track: **{track_str}**\n\n"
                f"Activity: **{first_name}** · fits {time_min} min"
            )
            st.caption("Cluster matched · Constraints applied · Explainable routing")


# ═══════════════════════════════════════════════════════════════════════════════
# View A — Overview
# ═══════════════════════════════════════════════════════════════════════════════

def _view_overview() -> None:
    st.markdown("## What is WorkPulse AI?")
    st.info(
        "**One-sentence pitch:** WorkPulse AI detects workforce stress archetypes from public "
        "narrative data, then routes frontline workers to short, constraint-aware recovery "
        "interventions — in under 60 seconds, explainably, without a therapist."
    )

    st.markdown("### The problem")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            "**Most wellness AI stops at detection.**\n\n"
            "It tells you you're stressed. It suggests 'take a break' or 'talk to someone.' "
            "That advice is generic, non-actionable, and ignores real constraints:\n"
            "- 5 minutes on a break\n"
            "- a loud warehouse floor\n"
            "- physical exhaustion, not anxiety\n"
            "- no access to a therapist"
        )
    with col2:
        st.markdown(
            "**WorkPulse AI bridges detection → action.**\n\n"
            "It doesn't just score your burnout. It:\n"
            "1. Identifies your stress *archetype* from 20,000+ workforce posts\n"
            "2. Applies your real constraints (time, setting, energy level)\n"
            "3. Routes you to a specific evidence-backed micro-intervention\n"
            "4. Shows you exactly why — every step"
        )

    st.markdown("---")
    st.markdown("### System architecture")
    st.code(
        "STATIC LAYER (pre-computed offline)\n"
        "  Reddit + Glassdoor posts (20,000+)\n"
        "      ↓  Clean · Label · Embed (sentence-BERT)\n"
        "      ↓  KMeans clustering\n"
        "      ↓  30 stress archetype clusters\n"
        "\n"
        "RUNTIME LAYER (per user, no retraining)\n"
        "  User text input\n"
        "      ↓  Embed with same model\n"
        "      ↓  Cosine similarity → nearest archetype\n"
        "      ↓  Apply constraints (time · setting · dimension)\n"
        "      ↓  Route to evidence-based micro-intervention\n"
        "      ↓  Show explainable decision trace",
        language=None,
    )

    st.markdown("---")
    st.markdown("### Human vs AI — who does what?")
    ai_col, human_col = st.columns(2)
    with ai_col:
        st.markdown(
            '<div style="background:#1a3a5c;color:white;padding:14px;border-radius:8px;">'
            "<b>🤖 AI Layer</b><br><br>"
            "• Detects patterns in 20,000+ workforce posts<br>"
            "• Embeds user input in real time<br>"
            "• Computes cosine similarity to archetypes<br>"
            "• Filters interventions by constraints<br>"
            "• Produces explainable routing decisions"
            "</div>",
            unsafe_allow_html=True,
        )
    with human_col:
        st.markdown(
            '<div style="background:#1a4a2e;color:white;padding:14px;border-radius:8px;">'
            "<b>🧠 Human Design Layer</b><br><br>"
            "• Defined what stress archetypes mean<br>"
            "• Designed all 5 intervention tracks<br>"
            "• Wrote evidence-backed activity instructions<br>"
            "• Set constraint rules and safety boundaries<br>"
            "• Decides when AI should not act alone"
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("")
    st.warning(
        "**Why AI alone is not enough here:** "
        "AI identifies patterns and routes efficiently. But deciding *what* a healthy recovery "
        "looks like for a warehouse worker on a 10-minute break requires human judgment, "
        "occupational health research, and ethical design. "
        "The AI surfaces options; humans defined what those options are and what their limits should be."
    )

    st.markdown("---")
    st.markdown("### What's already built vs what the team could extend")
    built_col, extend_col = st.columns(2)
    with built_col:
        st.markdown("**✅ Already built (this demo)**")
        st.markdown(
            "- Data pipeline (Reddit + Glassdoor → 30 clusters)\n"
            "- Sentence-BERT embedding + KMeans clustering\n"
            "- Runtime text → cluster matching\n"
            "- 3 intervention tracks, 5 activity types\n"
            "- Constraint-aware routing engine\n"
            "- Compound burnout scoring (body × mind × life)\n"
            "- Before / after measurement"
        )
    with extend_col:
        st.markdown("**🔧 Team could extend**")
        st.markdown(
            "- More subreddits / job sectors\n"
            "- Additional intervention tracks\n"
            "- Evaluation benchmark vs generic AI\n"
            "- Industry-specific personas\n"
            "- Fine-tuned embedding model\n"
            "- Mobile-first redesign\n"
            "- Longitudinal tracking"
        )

    st.markdown("---")
    if st.button("▶ Try the Live Demo →", type="primary", use_container_width=True, key="_ov_cta"):
        _go("demo")


# ═══════════════════════════════════════════════════════════════════════════════
# View B — Live Demo
# ═══════════════════════════════════════════════════════════════════════════════

def _view_demo() -> None:
    st.markdown("## ⚡ Live Demo")
    st.caption(
        "Describe how you're feeling, fill in the sliders, and set your constraints. "
        "WorkPulse AI will match your input to a stress archetype and show you the full "
        "routing decision."
    )

    report, _ = _load_cluster_data()
    cluster_data_ok = report is not None

    if not cluster_data_ok:
        st.info(
            "Cluster data not found — text-based routing is unavailable. "
            "Slider-based routing still works. "
            "Run the clustering pipeline to enable full routing: "
            "`python -m src.analysis.cluster_embeddings`"
        )

    with st.form("demo_form"):
        st.markdown("#### Step 1 — Describe what you're experiencing")
        free_text = st.text_area(
            "Free text (optional — enables cluster matching)",
            placeholder=(
                'e.g. "Exhausted after a 12-hour shift, back hurts, can\'t switch off." '
                'or "Anxious about a manager review, barely slept."'
            ),
            height=90,
        )

        st.markdown("#### Step 2 — Rate your current state (0–10)")
        physical = st.slider("Physical worn-down", 0, 10, 0)
        soreness = st.slider("Pain or soreness", 0, 10, 0)
        drained  = st.slider("Mentally drained", 0, 10, 0)
        sleep    = st.slider("Sleep quality last night", 0, 10, 10,
                             help="0 = terrible · 10 = great (reversed scale)")
        stress   = st.slider("Work stress right now", 0, 10, 0)
        safety   = st.radio(
            "Enough energy to finish shift safely?",
            ["yes", "somewhat", "no"],
            format_func=lambda x: {"yes": "Yes ✓", "somewhat": "Somewhat", "no": "No"}[x],
            horizontal=True,
        )

        st.markdown("#### Step 3 — Your real-world constraints")
        c1, c2 = st.columns(2)
        with c1:
            time_available = st.radio(
                "Time available", [5, 10, 15],
                format_func=lambda x: f"{x} min", horizontal=True, index=1,
            )
        with c2:
            setting = st.selectbox(
                "Where are you?", list(_SETTING_LABELS.keys()),
                format_func=lambda x: _SETTING_LABELS[x],
            )

        submitted = st.form_submit_button(
            "▶ Run WorkPulse AI", use_container_width=True, type="primary"
        )

    if not submitted:
        prior_burnout = st.session_state.get("burnout")
        prior_result  = st.session_state.get("demo_result")
        if prior_burnout is None and prior_result is None:
            return

    if submitted:
        checkin = CheckInResponse(
            physical_worn_down=physical,
            pain_soreness=soreness,
            mentally_drained=drained,
            sleep_quality=sleep,
            work_stress=stress,
            shift_safety_energy=safety,
            time_available_min=time_available,
            setting=setting,
            free_text=free_text.strip() or None,
        )
        errors = checkin.validate()
        if errors:
            st.error("Check-in issues:\n" + "\n".join(f"• {e}" for e in errors))
            return

        burnout = compute_burnout_score(checkin)
        st.session_state["checkin"] = checkin
        st.session_state["burnout"] = burnout
        st.session_state.pop("debrief", None)

        demo_result = None
        if free_text.strip() and cluster_data_ok and _CLUSTER_ROUTER_OK:
            with st.spinner("Matching your input to stress archetypes…"):
                try:
                    demo_result = _route_user_input(
                        user_text=free_text.strip(),
                        time_available_min=time_available,
                        setting=setting,
                    )
                    track_ids = demo_result.get("track_ids", [])
                    if track_ids:
                        _set_context(track_ids[0], "text_route")
                    elif free_text.strip():
                        st.session_state["debrief"] = label_checkin_debrief(free_text)
                except Exception as exc:
                    st.warning(f"Cluster routing unavailable: {exc}")
        else:
            # Dimension-based fallback
            rec_tid = _DIM_TO_TRACK.get(burnout.dominant_dimension, "restore")
            _set_context(rec_tid, "check_in")
            if free_text.strip():
                st.session_state["debrief"] = label_checkin_debrief(free_text)

        st.session_state["demo_result"] = demo_result

    # ── Results ───────────────────────────────────────────────────────────────
    burnout = st.session_state.get("burnout")
    if burnout is None:
        return

    st.divider()
    st.markdown("### Results")

    _risk_banner(burnout.risk_level)

    score_col, dim_col = st.columns([1, 2])
    with score_col:
        st.metric("Strain score", f"{burnout.score:.0f} / 100")
        st.caption(f"Dominant: **{burnout.dominant_dimension}**")
        if burnout.compound_multiplier > 1.0:
            pairs = ", ".join(burnout.interaction_pairs)
            st.caption(f"Compound ×{burnout.compound_multiplier:.2f} ({pairs})")
    with dim_col:
        st.caption("**Dimension breakdown** (0 = no strain · 100 = maximum)")
        _dim_progress("Body", burnout.body_score)
        _dim_progress("Mind", burnout.mind_score)
        _dim_progress("Life", burnout.life_score)

    if "debrief" in st.session_state:
        debrief = st.session_state["debrief"]
        with st.expander("What your notes suggest"):
            sentiment = debrief.get("sentiment") or {}
            lbl = sentiment.get("sentiment_label", "neutral")
            emoji_map = {"positive": "😊", "neutral": "😐", "negative": "😔"}
            st.write(f"Tone: {emoji_map.get(lbl, '😐')} {lbl.title()}")
            signals = debrief.get("domain_signals", {})
            flagged = {k.replace("_", " "): v for k, v in signals.items() if v.get("count", 0) > 0}
            if flagged:
                st.markdown("Themes: " + "  ·  ".join(f"**{t}** ({d['count']})" for t, d in flagged.items()))

    # Decision trace
    demo_result = st.session_state.get("demo_result")
    checkin_obj = st.session_state.get("checkin")
    t_min = checkin_obj.time_available_min if checkin_obj else 10
    s_key = checkin_obj.setting if checkin_obj else "break_room"

    if demo_result:
        _render_decision_trace(demo_result, t_min, s_key)
        cm        = demo_result.get("cluster_match", {})
        ivs       = demo_result.get("interventions", [])
        first_cat = next((iv for iv in ivs if iv.get("is_relevant")), ivs[0] if ivs else None)
        first_iv  = first_cat.get("intervention", {}) if first_cat else {}
        _render_vs_generic(
            cm.get("cluster_label", "unknown"),
            demo_result.get("track_ids", []),
            first_iv.get("name", "—"),
            t_min,
        )
    else:
        rec_track = get_track(st.session_state.get("active_track_id") or "")
        if rec_track:
            st.markdown("---")
            st.markdown("#### 🔍 Routing decision (dimension-based)")
            st.markdown(
                f"Dominant dimension: **{burnout.dominant_dimension}** "
                f"→ Track: **{rec_track['title']}** ({rec_track['archetype']})"
            )
            st.caption(
                "💡 Add a free-text description to enable full cluster-based routing "
                "with similarity score and decision trace."
            )

    rec_track = get_track(st.session_state.get("active_track_id") or "")
    if rec_track:
        st.divider()
        st.info(f"Recommended: **{rec_track['title']}** — {rec_track['archetype']}")
        if st.button("Start this intervention →", type="primary", use_container_width=True, key="_demo_cta"):
            _go("library")


# ═══════════════════════════════════════════════════════════════════════════════
# View C — AI Pattern Explorer
# ═══════════════════════════════════════════════════════════════════════════════

def _view_explorer() -> None:
    st.markdown("## 🔍 AI Pattern Explorer")
    _ai_badge()
    st.caption(
        "30 stress archetypes extracted from 20,000 Reddit and Glassdoor posts via "
        "sentence-BERT embeddings + KMeans clustering. Browse what the AI learned from the data."
    )

    with st.expander("How the AI learned these patterns"):
        st.markdown(
            "1. **20,000+ posts** from r/antiwork, r/jobs, r/warehouse, and Glassdoor were cleaned and filtered\n"
            "2. **Sentence-BERT** (all-MiniLM-L6-v2) converted each post to a 384-dimensional vector\n"
            "3. **KMeans** grouped similar posts; k was selected by silhouette score (best from k ∈ {10, 15, 20, 30})\n"
            "4. **TF-IDF keywords** per cluster produced human-readable archetype labels\n"
            "5. The 5 posts closest to each centroid became **cluster representatives** — "
            "used at runtime to match new user input via cosine similarity"
        )

    report, reps = _load_cluster_data()
    if report is None:
        st.warning(
            "Cluster data not found in `data/processed/`. "
            "Run: `python -m src.analysis.cluster_embeddings --input data/processed/workforce_all_text.parquet`"
        )
        return

    cluster_options = {
        int(row["cluster_id"]): (
            f"{int(row['cluster_id']):02d} — {row['cluster_label']}  ({int(row['size']):,} posts)"
        )
        for _, row in report.sort_values("cluster_id").iterrows()
    }
    cid_list = list(cluster_options.keys())
    preselected = st.session_state.get("selected_cluster_id")
    default_idx = cid_list.index(preselected) if preselected in cid_list else 0

    st.selectbox(
        "Select a stress archetype",
        options=cid_list,
        format_func=lambda x: cluster_options[x],
        index=default_idx,
        key="_cluster_sel",
        on_change=_on_cluster_change,
    )
    selected_id = st.session_state.get("_cluster_sel") or cid_list[0]
    st.session_state["selected_cluster_id"] = selected_id

    row = report[report["cluster_id"] == selected_id].iloc[0]
    st.divider()

    mapped_tracks = cluster_to_track_ids(int(selected_id))
    if mapped_tracks:
        primary_track = get_track(mapped_tracks[0])
        if primary_track:
            left, right = st.columns([4, 2])
            with left:
                st.markdown(
                    f"🗺️ **AI routing →** Human-designed track: "
                    f"**{primary_track['title']}** — {primary_track['archetype']}"
                )
                st.caption("AI identified this pattern · Humans designed the recovery track")
            with right:
                if st.button("Start this track →", type="primary",
                             use_container_width=True, key="_cluster_start"):
                    _activate(mapped_tracks[0], "cluster")
    else:
        st.caption("This cluster is not yet mapped to a track.")

    ov_col, sz_col = st.columns([3, 1])
    with ov_col:
        st.markdown(f"**Archetype:** {row['cluster_label']}")
        kws = [kw.strip() for kw in str(row["top_keywords"]).split(",")]
        st.markdown("**Top keywords:** " + "  ·  ".join(f"`{kw}`" for kw in kws[:8]))
    with sz_col:
        st.metric("Posts", f"{int(row['size']):,}")

    try:
        source_counts = ast.literal_eval(str(row["source_counts"]))
        type_counts   = ast.literal_eval(str(row["type_counts"]))
    except (ValueError, SyntaxError):
        source_counts, type_counts = {}, {}

    sc_col, tc_col = st.columns(2)
    with sc_col:
        st.caption("**By source**")
        for src, count in source_counts.items():
            pct = count / max(int(row["size"]), 1) * 100
            st.caption(f"{'Reddit' if src == 'reddit' else 'Glassdoor'}: {count:,} ({pct:.0f}%)")
    with tc_col:
        st.caption("**By type**")
        for ctype, count in type_counts.items():
            pct = count / max(int(row["size"]), 1) * 100
            st.caption(f"{ctype.title()}: {count:,} ({pct:.0f}%)")

    st.divider()
    st.markdown("**Representative posts** — 5 posts closest to this cluster's centroid")
    cluster_reps = reps[reps["cluster_id"] == selected_id].sort_values("rank")
    if cluster_reps.empty:
        st.info("No representative texts for this cluster.")
    else:
        for _, rep_row in cluster_reps.iterrows():
            snippet = str(rep_row["text_snippet"])
            preview = snippet[:90].replace("\n", " ")
            src     = str(rep_row.get("source", "")).title()
            ctype   = str(rep_row.get("type", "")).title()
            rank    = int(rep_row["rank"])
            with st.expander(f"#{rank + 1}  [{src} {ctype}]  {preview}…"):
                st.write(snippet)
                meta = [src, ctype]
                sub  = rep_row.get("subreddit")
                if pd.notna(sub) and str(sub).strip():
                    meta.append(f"r/{sub}")
                st.caption(" · ".join(p for p in meta if p))


# ═══════════════════════════════════════════════════════════════════════════════
# View D — Intervention Library
# ═══════════════════════════════════════════════════════════════════════════════

def _view_library() -> None:
    st.markdown("## 🎨 Intervention Library")
    _human_badge()
    st.caption(
        "5 evidence-backed micro-intervention categories. Each mapped to specific stress archetypes, "
        "time constraints, and physical settings. AI routes here — humans designed what happens."
    )

    with st.expander("Why human-designed interventions?"):
        st.markdown(
            "**AI is good at pattern matching. It is not good at designing recovery experiences.**\n\n"
            "Every intervention here is grounded in published research:\n"
            "- **Interoception drawing** — body awareness (Mehling et al.)\n"
            "- **Expressive writing** — Pennebaker method for emotional processing\n"
            "- **Tempo-matched audio** — 60 BPM for parasympathetic activation\n"
            "- **3-thing notice** — psychological detachment technique\n"
            "- **Gamified creative micro-task** — attention redirection\n\n"
            "AI identifies *which category* to recommend. "
            "Humans decided *what* each category does, *why* it works, and *what boundaries* it has. "
            "This distinction is essential for trust and safety."
        )

    tracks = _load_tracks()
    track_map = {t["track_id"]: f"{t['title']}  —  {t['archetype']}" for t in tracks}
    active_tid = st.session_state.get("active_track_id") or tracks[0]["track_id"]
    source = st.session_state.get("track_source")

    if source and source in _SOURCE_LABELS:
        st.info(f"📍 {_SOURCE_LABELS[source]}")

    st.selectbox(
        "Track",
        options=list(track_map.keys()),
        format_func=lambda x: track_map[x],
        index=list(track_map.keys()).index(active_tid) if active_tid in track_map else 0,
        key="_act_track_sel",
        on_change=_on_activity_track_change,
    )
    selected_tid = st.session_state.get("_act_track_sel") or active_tid
    active_track = get_track(selected_tid)
    if not active_track:
        st.warning("Track not found.")
        return

    with st.expander(f"About: {active_track['title']}", expanded=False):
        st.write(active_track["description"])
        st.caption(f"Techniques: {', '.join(active_track['techniques'])}")
        cids = active_track.get("cluster_ids", [])
        if cids:
            st.caption(f"Mapped to stress archetype clusters: {', '.join(str(c) for c in cids)}")

    interventions = active_track.get("interventions", [])
    if not interventions:
        st.info("No activities in this track yet.")
        return

    iv_map = {iv["id"]: f"{iv['title']}  ({iv['duration_minutes']} min)" for iv in interventions}
    active_aid = st.session_state.get("active_activity_id") or interventions[0]["id"]
    if active_aid not in iv_map:
        active_aid = interventions[0]["id"]

    st.selectbox(
        "Activity",
        options=list(iv_map.keys()),
        format_func=lambda x: iv_map[x],
        index=list(iv_map.keys()).index(active_aid),
        key="_act_sel",
        on_change=_on_activity_sel_change,
    )
    selected_aid = st.session_state.get("_act_sel") or active_aid
    active_iv = next((iv for iv in interventions if iv["id"] == selected_aid), interventions[0])

    st.markdown("---")
    _render_activity_card(active_track, active_iv, interventions)

    others = [iv for iv in interventions if iv["id"] != active_iv["id"]]
    if others:
        with st.expander("Other activities in this track"):
            for iv in others:
                env_str = ", ".join(_SETTING_LABELS.get(e, e.title()) for e in iv.get("environment", []))
                c1, c2 = st.columns([5, 1])
                with c1:
                    st.markdown(f"**{iv['title']}** — {iv['duration_minutes']} min · {env_str}")
                    st.caption(iv["instructions"].strip().splitlines()[0] + "…")
                with c2:
                    if st.button("Do this", key=f"_pick_{iv['id']}", use_container_width=True):
                        st.session_state["active_activity_id"] = iv["id"]
                        st.rerun()
                st.divider()

    _render_session_summary()


# ═══════════════════════════════════════════════════════════════════════════════
# View E — Team Review / Project Proposal
# ═══════════════════════════════════════════════════════════════════════════════

def _view_proposal() -> None:
    st.markdown("## 📋 Team Review — Should we build WorkPulse AI?")
    st.caption("Dallas AI Summer Program · Team 19 · Project proposal summary for team vote")

    st.info(
        "**Pitch:** WorkPulse AI detects workforce stress archetypes from public narrative data "
        "and routes frontline workers to short, constraint-aware interventions — "
        "explainably, in under 60 seconds, without a therapist or a generic chatbot."
    )

    # ── Problem + User ────────────────────────────────────────────────────────
    st.markdown("---")
    p_col, u_col = st.columns(2)
    with p_col:
        st.markdown("#### 🎯 The problem")
        st.markdown(
            "Frontline workers (warehouse, logistics, manufacturing, healthcare support) "
            "experience burnout at high rates but have almost no access to context-aware support. "
            "Generic wellness apps give everyone the same advice and completely ignore real-world "
            "constraints: 5 minutes on a noisy break room floor, no phone signal, physical exhaustion "
            "not anxiety, and no capacity to engage with a 15-step program."
        )
    with u_col:
        st.markdown("#### 👷 The user")
        st.markdown(
            "Shift workers — warehouse pickers, delivery drivers, nurses' aides, line workers — "
            "during or immediately after a shift. They need something that:\n"
            "- works in 5 minutes on a phone\n"
            "- requires no clinical vocabulary\n"
            "- gives a specific action, not a suggestion to 'take care of yourself'\n"
            "- respects where they are and how much energy they have left"
        )

    # ── AI + Human ────────────────────────────────────────────────────────────
    st.markdown("---")
    ai_col, human_col = st.columns(2)
    with ai_col:
        st.markdown("#### 🤖 AI element")
        st.markdown(
            "- NLP clustering of 20,000+ real workforce posts (Reddit + Glassdoor)\n"
            "- Sentence-BERT embeddings for real-time input matching\n"
            "- Cosine similarity → nearest stress archetype\n"
            "- Constraint-aware routing (time, setting, dominant dimension)\n"
            "- Compound burnout scoring: body × mind × life interaction effects\n"
            "- Explainable decision trace — every routing step is visible"
        )
    with human_col:
        st.markdown("#### 🧠 Human element")
        st.markdown(
            "- Defined what each stress archetype means for workers\n"
            "- Designed 5 evidence-based intervention tracks\n"
            "- Wrote activity instructions grounded in published research\n"
            "- Set constraint rules, safety boundaries, and ethical limits\n"
            "- Curated and filtered source data for quality\n"
            "- Decided what AI should and should not decide alone"
        )

    # ── MVP scope ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📦 MVP scope — what this demo already does")
    scope_col, beta_col = st.columns([3, 2])
    with scope_col:
        st.markdown(
            "| Component | Status |\n"
            "|-----------|--------|\n"
            "| Data pipeline (Reddit + Glassdoor → clusters) | ✅ Built |\n"
            "| 30 stress archetype clusters | ✅ Built |\n"
            "| Runtime text → cluster matching | ✅ Built |\n"
            "| Constraint-aware routing engine | ✅ Built |\n"
            "| 3 intervention tracks + 5 activity types | ✅ Built |\n"
            "| Compound burnout scoring | ✅ Built |\n"
            "| Before / after measurement | ✅ Built |\n"
            "| Explainable decision trace | ✅ Built |\n"
            "| Evaluation benchmark vs generic AI | 🔧 Stub only |\n"
            "| FastAPI backend | 🔧 Designed, not built |\n"
            "| Mobile-first / React UI | 🔧 Future stage |"
        )
    with beta_col:
        st.markdown("**What 'beta demo' means honestly:**")
        st.warning(
            "This is a working prototype — not a product. "
            "All processing runs locally or on a free Cloud server. "
            "No user accounts. No persistent data. No clinical validation. "
            "English only. Static dataset, not live ingestion.\n\n"
            "It is demo-ready and team-reviewable. "
            "It is not production-ready."
        )

    # ── Business value ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 💼 Business value — why would anyone actually want this?")
    bw_col, be_col, bp_col = st.columns(3)
    with bw_col:
        st.markdown("**Worker-facing**")
        st.markdown(
            "A tool that actually fits a 5-minute break. "
            "Not a journal, not a hotline, not a meditation app that assumes you have 20 minutes "
            "and a quiet room. Workers get a specific, validated activity matched to what they're "
            "experiencing right now — not generic advice.\n\n"
            "*Value: reduced friction, improved recovery, autonomy*"
        )
    with be_col:
        st.markdown("**Employer / workforce support**")
        st.markdown(
            "High burnout = high turnover. Warehouses and logistics companies lose billions "
            "annually to absenteeism and attrition. An evidence-based, low-cost digital tool "
            "that measurably reduces acute strain during shifts is a compelling EAP (Employee "
            "Assistance Program) addition — especially as workforce mental health becomes a "
            "compliance and liability concern.\n\n"
            "*Value: retention, safety, EAP differentiation*"
        )
    with bp_col:
        st.markdown("**API / Platform (future)**")
        st.markdown(
            "The routing engine and intervention library are already structured as backend "
            "functions that could be served via API. A future platform layer could allow "
            "HR systems, workforce management tools, or occupational health providers to "
            "integrate WorkPulse AI as a service — not just a standalone app.\n\n"
            "*Value: B2B SaaS, API licensing, white-label*"
        )

    # ── Risks ─────────────────────────────────────────────────────────────────
    st.markdown("---")
    r1, r2 = st.columns(2)
    with r1:
        st.markdown("#### ⚠️ Technical risks")
        st.markdown(
            "- `sentence-transformers` loads slowly on first run (free Cloud tier)\n"
            "- Static dataset — no live data ingestion built yet\n"
            "- TF-IDF cluster labels are rough keywords, not curated archetype names\n"
            "- English only — no multilingual support\n"
            "- Cluster quality depends on sample diversity (current: Reddit + Glassdoor)"
        )
    with r2:
        st.markdown("#### ⚠️ Scope risks")
        st.markdown(
            "- Not a clinical tool — this must be clearly communicated\n"
            "- Intervention efficacy is evidence-informed but not formally validated in this app\n"
            "- Requires Kaggle source datasets (not real-time ingestion)\n"
            "- AI routing accuracy depends on how representative the training data is\n"
            "- Low engagement is a known risk for digital wellness tools (Linardon, 2020)"
        )

    # ── Team roles ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🧑‍💻 Team roles — what each person would actually own")
    st.caption(
        "Each role below has a clear workstream. No one is a passive contributor. "
        "These roles can overlap — the goal is that everyone has a specific thing they can build, improve, or own."
    )

    roles = [
        (
            "🤖 AI / NLP — Clustering & Pattern Detection",
            "**What you'd own:** The intelligence layer of the system.\n\n"
            "**What you'd actually do:**\n"
            "- Evaluate and improve the 30 cluster archetypes (manual review + silhouette analysis)\n"
            "- Experiment with different sentence-BERT models or fine-tune on workforce-specific text\n"
            "- Add new data sources (additional subreddits, industry-specific forums, survey data)\n"
            "- Improve the cluster → archetype label quality (from TF-IDF keywords to curated names)\n"
            "- Explore hybrid approaches: keyword matching + semantic similarity\n\n"
            "**Initiative opportunity:** Build a cluster quality scorecard — a way to objectively "
            "measure whether the 30 archetypes are meaningful, distinct, and cover the right problems.",
        ),
        (
            "⚙️ Routing / Decision Logic",
            "**What you'd own:** The constraint engine and routing rules.\n\n"
            "**What you'd actually do:**\n"
            "- Extend the constraint filtering (add cognitive load, noise tolerance, physical capability)\n"
            "- Build an A/B test framework for routing strategies\n"
            "- Add fallback logic when cluster similarity is low (low-confidence routing)\n"
            "- Design the cluster → track mapping (currently sparse — most clusters are unmapped)\n"
            "- Document and test edge cases (what happens at constraint boundaries?)\n\n"
            "**Initiative opportunity:** Design a 'routing confidence' indicator — show users when "
            "the match is strong vs when it's a best guess.",
        ),
        (
            "📚 Intervention Research & Archetype Design",
            "**What you'd own:** The content layer — what the app actually tells workers to do.\n\n"
            "**What you'd actually do:**\n"
            "- Add 2–3 new intervention tracks with full activity instructions\n"
            "- Validate and improve research citations for existing activities\n"
            "- Map the 30 stress archetypes to specific intervention strategies (clinical framing)\n"
            "- Write archetype descriptions that explain each cluster in plain language\n"
            "- Collaborate with the routing team to improve cluster → track accuracy\n\n"
            "**Initiative opportunity:** Design a set of 5–10 worker personas grounded in both "
            "cluster data and occupational health literature — not made-up characters, but evidence-based archetypes.",
        ),
        (
            "🎨 Front-End / UX + Demo",
            "**What you'd own:** Everything the user sees and touches.\n\n"
            "**What you'd actually do:**\n"
            "- Improve the Streamlit app layout, visual hierarchy, and mobile readability\n"
            "- Design the demo flow for judges (which views, in what order, with what narrative)\n"
            "- Create pitch slides that complement the live demo\n"
            "- Record a 60–90 second demo video as a fallback and sharing asset\n"
            "- Prototype a mobile-first version (even as a Figma mockup, not just code)\n\n"
            "**Initiative opportunity:** Design what the React/mobile app would look like in Stage 3 "
            "of the productization path — a forward-looking UI concept that shows where the product is headed.",
        ),
        (
            "🔌 Backend / API Transition",
            "**What you'd own:** The path from Streamlit prototype to real product.\n\n"
            "**What you'd actually do:**\n"
            "- Wrap existing `src/` functions as FastAPI endpoints (already designed in productization_plan.md)\n"
            "- Write API documentation and test suite for the backend layer\n"
            "- Set up a simple deployment (Railway, Render, or Fly.io) for the API\n"
            "- Update Streamlit to call the API instead of importing src/ directly\n"
            "- Design the user session and data persistence model\n\n"
            "**Initiative opportunity:** Build one fully working FastAPI endpoint (e.g. `/checkin/score`) "
            "end-to-end — proving the backend layer works before the full transition.",
        ),
        (
            "📊 Evaluation / Benchmarking",
            "**What you'd own:** Proving the system works better than the alternative.\n\n"
            "**What you'd actually do:**\n"
            "- Build the formal Generic AI vs WorkPulse AI comparison (currently a stub)\n"
            "- Design a user testing protocol (5–10 workers, structured input scenarios)\n"
            "- Define and measure 3–5 concrete success metrics (route accuracy, before/after deltas, etc.)\n"
            "- Run silhouette analysis and manual cluster review\n"
            "- Write an honest limitations section for the final report\n\n"
            "**Initiative opportunity:** Create a 'benchmark deck' — 10 example inputs, "
            "side-by-side outputs from a generic LLM vs WorkPulse AI, with commentary on the difference.",
        ),
        (
            "📝 Docs / Product / Presentation",
            "**What you'd own:** How the project is understood and communicated.\n\n"
            "**What you'd actually do:**\n"
            "- Write and maintain the final project report\n"
            "- Keep the README, proposal, and productization plan aligned with what's actually built\n"
            "- Prepare the judge-facing presentation (10-minute pitch + Q&A prep)\n"
            "- Coordinate the demo walkthrough with the UX team\n"
            "- Maintain a 'what changed' log so the team always knows the current state\n\n"
            "**Initiative opportunity:** Write a one-page 'product brief' — what WorkPulse AI is, "
            "who it's for, what it does, and what it would need to become a real product.",
        ),
        (
            "🎙️ Voice / Accessibility (Future Workstream)",
            "**Status: future work — not built yet. Could become its own sub-initiative.**\n\n"
            "**Why voice matters here:**\n"
            "Many frontline workers can't type on a break. They're wearing gloves, standing, moving. "
            "A voice interface would let them describe how they feel in 10 seconds — 'I'm exhausted, "
            "my back hurts, I've got 5 minutes' — and get a routed response without touching a screen.\n\n"
            "**What this workstream would involve:**\n"
            "- Integrate a speech-to-text layer (Whisper API or browser Web Speech API)\n"
            "- Pass transcribed text through the existing cluster router (no routing changes needed)\n"
            "- Design a voice-first UX: minimal screen, audio feedback for instructions\n"
            "- Address privacy concerns: is voice data stored? Who has access?\n"
            "- Consider accessibility: hearing impairment, noisy environments, accent diversity\n\n"
            "**Scenarios where this is highest value:**\n"
            "Warehouse floor with gloves on · post-shift fatigue with low screen tolerance · "
            "commute on public transit · accessibility for lower literacy users\n\n"
            "**Honest constraint:** This adds a real dependency (speech API, audio playback, "
            "privacy policy) that is out of scope for the competition MVP. "
            "Flag it as 'Stage 4' of the productization path.",
        ),
    ]

    for role_title, role_body in roles:
        with st.expander(role_title):
            st.markdown(role_body)

    # ── Evaluation ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📊 How we would evaluate success")
    st.markdown(
        "| Metric | How to measure |\n"
        "|--------|----------------|\n"
        "| Cluster match quality | Silhouette score + manual review: do the 30 clusters feel distinct and meaningful? |\n"
        "| Routing clarity | % of test inputs that produce a valid, relevant track (not 'no track configured') |\n"
        "| vs Generic AI | Side-by-side comparison on 10 worker input scenarios — judges evaluate which response is more actionable |\n"
        "| User actionability | Before/after ratings across 5 completed activities — does Energy/Tension/Clarity shift? |\n"
        "| Demo completeness | Can a judge follow the full routing path (input → cluster → track → activity) in under 2 minutes? |"
    )

    # ── Future paths ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🚀 Future paths — where this could go")
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown("**Stage 2 — API backend**")
        st.markdown(
            "Wrap `src/` functions as FastAPI endpoints. "
            "Streamlit becomes an API client. "
            "Enables persistent user data, session history, and separation of frontend from logic. "
            "Already designed in `productization_plan.md`."
        )
    with f2:
        st.markdown("**Stage 3 — Mobile / React**")
        st.markdown(
            "Replace Streamlit with a React web app or React Native mobile app. "
            "Same FastAPI backend. "
            "Unlock: offline support, push notifications, native performance, "
            "and the UX needed for workers who are standing on a factory floor."
        )
    with f3:
        st.markdown("**Stage 4 — Voice + Accessibility**")
        st.markdown(
            "Add speech-to-text input so workers can describe their state verbally. "
            "Transcribed text routes through the same cluster engine. "
            "No changes to the routing logic — just a new input modality. "
            "Highest value for physical workers who can't type during a shift."
        )

    # ── Open questions ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🗳️ Open questions for team vote")
    st.markdown(
        "1. **Data scope:** Should we expand beyond Reddit/Glassdoor? (industry surveys, healthcare data, union reports)\n"
        "2. **Cluster quality:** Do the 30 archetypes feel meaningful? Should we manually curate the labels?\n"
        "3. **Clinical framing:** How prominently do we label this 'not a clinical tool'? Does it affect trust?\n"
        "4. **Team focus:** Polish the MVP for the competition, or demonstrate more technical depth?\n"
        "5. **Voice path:** Is a voice input prototype within reach for the timeline? Does it strengthen the pitch?\n"
        "6. **Business framing:** Should we present this as a product pitch or a research demo?"
    )

    # ── Why this fits ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div style="background:#1a3a5c;color:white;padding:18px 24px;border-radius:8px;">'
        "<b>Why this fits the Dallas AI Summer Program</b><br><br>"
        "NLP + applied AI + measurable social impact. Six distinct team roles with clear ownership. "
        "A live, working demo at a shareable URL. Honest evaluation criteria. "
        "A clear productization path from Streamlit prototype to API to mobile. "
        "A problem that affects millions of workers and that generic AI tools handle poorly.<br><br>"
        "<b>The MVP already runs. The team's job is to make it better, evaluate it honestly, "
        "and present it clearly.</b>"
        "</div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Main render
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="WorkPulse AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

_init_state()

st.title("🛡️ WorkPulse AI")
st.caption(
    "Constraint-aware AI intervention system for frontline workers"
)

_render_now_doing_panel()

_current_view = st.session_state.get("active_view", "overview")
st.radio(
    "Navigate",
    options=_VIEWS,
    format_func=lambda v: _VIEW_LABELS[v],
    horizontal=True,
    index=_VIEWS.index(_current_view) if _current_view in _VIEWS else 0,
    key="_radio_nav",
    on_change=_on_nav_change,
    label_visibility="collapsed",
)

st.divider()

{
    "overview":  _view_overview,
    "demo":      _view_demo,
    "explorer":  _view_explorer,
    "library":   _view_library,
    "proposal":  _view_proposal,
}.get(st.session_state.get("active_view", "overview"), _view_overview)()
