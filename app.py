# app.py - Streamlit UI for MetaAgent.
#
# UI only: collects input, shows progress and renders results. All pipeline
# logic (planning, tool selection, generation, testing, self-correction,
# deployment) lives in the metaagent package and is reached through
# metaagent.builds.service.BuildService.

import html

import streamlit as st

from metaagent.ai.llm import describe_error
from metaagent.builds.service import BuildService
from metaagent.core.config import get_settings
from metaagent.rag.service import DocumentService
from metaagent.schemas import BuildEvent, BuildResult, Constraints

settings = get_settings()

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Meta-Agent",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def get_build_service() -> BuildService:
    return BuildService(settings)


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
defaults = {
    "agent_request": "",
    "build_result": None,
    "doc_service": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================
# DESIGN SYSTEM (unchanged)
# ============================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,400;0,9..144,500;1,9..144,300;1,9..144,400&family=JetBrains+Mono:wght@300;400;500&family=Outfit:wght@300;400;500;600&display=swap');

/* ── HIDE CHROME ── */
header[data-testid="stHeader"],
footer, #MainMenu { display:none!important; }
.block-container {
    padding: 0 2.5rem 5rem !important;
    max-width: 1280px !important;
}

/* ── TOKENS ── */
:root {
    --bg:       #0e0e10;
    --s1:       #141416;
    --s2:       #1b1b1e;
    --s3:       #242428;
    --line:     rgba(255,255,255,0.06);
    --line-2:   rgba(255,255,255,0.11);
    --tx:       #edebe6;
    --tx-2:     #97948f;
    --tx-3:     #5c5a57;
    --ac:       #d03d2f;
    --ac-dim:   rgba(208,61,47,0.12);
    --ac-hover: #b83327;
    --ok:       #4a9e7a;
    --warn:     #c98b2a;
    --r:        5px;
    --r2:       9px;
    --sans:     'Outfit', sans-serif;
    --serif:    'Fraunces', Georgia, serif;
    --mono:     'JetBrains Mono', 'Courier New', monospace;
}

/* ── BASE ── */
.stApp { background:var(--bg)!important; font-family:var(--sans); color:var(--tx); }
* { box-sizing:border-box; }

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background:var(--s1)!important;
    border-right:1px solid var(--line)!important;
}
[data-testid="stSidebar"] .stMarkdown p {
    font-family:var(--mono)!important;
    font-size:0.65rem!important;
    letter-spacing:0.09em!important;
    text-transform:uppercase!important;
    color:var(--tx-3)!important;
}
[data-testid="stSidebar"] h2 {
    font-family:var(--serif)!important;
    font-weight:400!important;
    font-size:1.05rem!important;
    color:var(--tx)!important;
    letter-spacing:-0.01em!important;
    padding-bottom:0.6rem!important;
    border-bottom:1px solid var(--line)!important;
    margin-bottom:1.25rem!important;
}

/* ── HERO ── */
.hero {
    padding: 4rem 0 3rem;
    border-bottom: 1px solid var(--line);
    margin-bottom: 3rem;
}
.hero-kicker {
    font-family: var(--mono);
    font-size: 0.62rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--ac);
    margin-bottom: 1.25rem;
    display: flex;
    align-items: center;
    gap: 0.8rem;
}
.hero-kicker::after {
    content:'';
    width:24px; height:1px;
    background:var(--ac);
    opacity:0.55;
    flex-shrink:0;
}
.hero-h1 {
    font-family: var(--serif);
    font-size: 3.5rem;
    font-weight: 300;
    color: var(--tx);
    line-height: 1.08;
    letter-spacing: -0.03em;
    margin: 0 0 1.25rem;
}
.hero-h1 em {
    font-style: italic;
    color: var(--ac);
}
.hero-body {
    font-family: var(--sans);
    font-size: 0.95rem;
    font-weight: 300;
    color: var(--tx-2);
    line-height: 1.75;
    max-width: 460px;
    margin: 0;
}

/* ── FEATURE STRIP ── */
.feat-strip {
    display:grid;
    grid-template-columns:repeat(5,1fr);
    border:1px solid var(--line);
    border-radius:var(--r2);
    overflow:hidden;
    margin-bottom:3.25rem;
    background:var(--line);
    gap:1px;
}
.feat-cell {
    background:var(--s1);
    padding:1.5rem 1.125rem;
    transition:background 0.15s;
}
.feat-cell:hover { background:var(--s2); }
.feat-num {
    font-family:var(--mono);
    font-size:0.56rem;
    color:var(--ac);
    letter-spacing:0.14em;
    margin-bottom:0.65rem;
    opacity:0.75;
}
.feat-name {
    font-family:var(--serif);
    font-size:0.92rem;
    font-weight:400;
    color:var(--tx);
    line-height:1.3;
    margin-bottom:0.4rem;
}
.feat-desc {
    font-family:var(--mono);
    font-size:0.6rem;
    color:var(--tx-3);
    line-height:1.55;
    letter-spacing:0.02em;
}

/* ── SECTION LABEL ── */
.slabel {
    font-family:var(--mono);
    font-size:0.6rem;
    letter-spacing:0.16em;
    text-transform:uppercase;
    color:var(--tx-3);
    margin-bottom:1rem;
    display:flex;
    align-items:center;
    gap:0.65rem;
}
.slabel::before {
    content:'';
    width:12px; height:1px;
    background:var(--tx-3);
    flex-shrink:0;
}

/* ── TEXT AREA / INPUT ── */
.stTextArea textarea, .stTextInput input {
    background:var(--s1)!important;
    color:var(--tx)!important;
    border:1px solid var(--line-2)!important;
    border-radius:var(--r)!important;
    font-family:var(--sans)!important;
    font-size:0.9rem!important;
    font-weight:300!important;
    padding:0.875rem 1rem!important;
    line-height:1.65!important;
    transition:border-color 0.18s,box-shadow 0.18s!important;
    resize:none!important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color:var(--ac)!important;
    box-shadow:0 0 0 3px var(--ac-dim)!important;
    outline:none!important;
}
.stTextArea label, .stTextInput label {
    font-family:var(--mono)!important;
    font-size:0.62rem!important;
    letter-spacing:0.1em!important;
    text-transform:uppercase!important;
    color:var(--tx-3)!important;
}

/* ── BUTTONS — default ── */
.stButton > button {
    background:var(--s2)!important;
    color:var(--tx-2)!important;
    border:1px solid var(--line-2)!important;
    border-radius:var(--r)!important;
    font-family:var(--mono)!important;
    font-size:0.64rem!important;
    letter-spacing:0.1em!important;
    text-transform:uppercase!important;
    padding:0.575rem 1rem!important;
    font-weight:400!important;
    width:100%!important;
    transition:all 0.15s ease!important;
}
.stButton > button:hover {
    background:var(--s1)!important;
    border-color:var(--ac)!important;
    color:var(--tx)!important;
    transform:none!important;
    box-shadow:none!important;
}

/* ── BUILD BUTTON (primary): st.button(type="primary") ── */
.stButton > button[kind="primary"] {
    background:var(--ac)!important;
    color:#fff!important;
    border-color:var(--ac)!important;
    font-size:0.68rem!important;
    letter-spacing:0.16em!important;
    padding:0.8rem 2rem!important;
}
.stButton > button[kind="primary"]:hover {
    background:var(--ac-hover)!important;
    border-color:var(--ac-hover)!important;
    color:#fff!important;
}

/* ── SELECT / SLIDER ── */
.stSelectbox div[data-baseweb="select"] > div {
    background:var(--s1)!important;
    border-color:var(--line-2)!important;
    border-radius:var(--r)!important;
}
.stSelectbox span { color:var(--tx)!important; font-family:var(--sans)!important; font-size:0.85rem!important; }
div[data-testid="stSlider"] label {
    font-family:var(--mono)!important;
    font-size:0.62rem!important;
    text-transform:uppercase!important;
    letter-spacing:0.1em!important;
    color:var(--tx-3)!important;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background:transparent!important;
    border-bottom:1px solid var(--line)!important;
    gap:0!important;
    padding:0!important;
    margin-bottom:2.25rem!important;
}
.stTabs [data-baseweb="tab"] {
    font-family:var(--mono)!important;
    font-size:0.65rem!important;
    letter-spacing:0.1em!important;
    text-transform:uppercase!important;
    color:var(--tx-3)!important;
    padding:0.8rem 1.5rem!important;
    border-radius:0!important;
    border-bottom:2px solid transparent!important;
    margin-bottom:-1px!important;
    background:transparent!important;
    transition:color 0.15s!important;
}
.stTabs [aria-selected="true"] {
    color:var(--tx)!important;
    border-bottom-color:var(--ac)!important;
    background:transparent!important;
}

/* ── METRICS ── */
[data-testid="stMetric"] {
    background:var(--s1)!important;
    border:1px solid var(--line)!important;
    border-radius:var(--r)!important;
    padding:1.25rem 1.375rem!important;
}
[data-testid="stMetric"] label {
    font-family:var(--mono)!important;
    font-size:0.58rem!important;
    text-transform:uppercase!important;
    letter-spacing:0.13em!important;
    color:var(--tx-3)!important;
}
[data-testid="stMetricValue"] {
    font-family:var(--serif)!important;
    font-size:1.6rem!important;
    font-weight:300!important;
    color:var(--tx)!important;
    line-height:1.2!important;
}

/* ── ALERTS ── */
.stAlert {
    background:var(--s1)!important;
    border:1px solid var(--line)!important;
    border-left:2px solid var(--ac)!important;
    border-radius:var(--r)!important;
    font-family:var(--sans)!important;
    font-size:0.85rem!important;
    color:var(--tx-2)!important;
}
[data-testid="stNotificationContentSuccess"] { border-left-color:var(--ok)!important; }
[data-testid="stNotificationContentWarning"] { border-left-color:var(--warn)!important; }

/* ── CODE BLOCK ── */
.stCodeBlock {
    background:var(--s1)!important;
    border:1px solid var(--line)!important;
    border-radius:var(--r)!important;
    width:100%!important;
    max-width:100%!important;
}
.stCodeBlock > div { width:100%!important; overflow-x:auto!important; }
.stCodeBlock pre {
    font-family:var(--mono)!important;
    font-size:0.78rem!important;
    line-height:1.8!important;
    white-space:pre!important;
    overflow-x:auto!important;
    color:var(--tx)!important;
    padding:1.25rem 1.5rem!important;
}

/* ── PROGRESS ── */
.stProgress > div > div > div > div {
    background:var(--ac)!important;
    border-radius:2px!important;
}
.stProgress > div > div > div {
    background:var(--s2)!important;
    border-radius:2px!important;
    height:2px!important;
}

/* ── DATAFRAME ── */
.stDataFrame {
    border:1px solid var(--line)!important;
    border-radius:var(--r)!important;
    overflow:hidden!important;
}
.stDataFrame th {
    font-family:var(--mono)!important;
    font-size:0.6rem!important;
    text-transform:uppercase!important;
    letter-spacing:0.1em!important;
    color:var(--tx-3)!important;
    background:var(--s1)!important;
    padding:0.75rem 1rem!important;
    border:none!important;
}
.stDataFrame td {
    color:var(--tx)!important;
    font-family:var(--sans)!important;
    font-size:0.85rem!important;
    background:var(--bg)!important;
    padding:0.6rem 1rem!important;
    border-color:var(--line)!important;
}

/* ── EXPANDER ── */
.streamlit-expanderHeader {
    font-family:var(--mono)!important;
    font-size:0.65rem!important;
    text-transform:uppercase!important;
    letter-spacing:0.1em!important;
    color:var(--tx-3)!important;
    background:var(--s1)!important;
    border:1px solid var(--line)!important;
    border-radius:var(--r)!important;
    padding:0.875rem 1rem!important;
}

/* ── HR ── */
hr {
    border:none!important;
    border-top:1px solid var(--line)!important;
    margin:2.5rem 0!important;
}

/* ── CAPTION ── */
.stCaption, [data-testid="stCaptionContainer"] {
    font-family:var(--mono)!important;
    font-size:0.6rem!important;
    color:var(--tx-3)!important;
    letter-spacing:0.06em!important;
}

/* ── STATUS BOX ── */
.status-box {
    background:var(--s1);
    border:1px solid var(--line);
    border-left:2px solid var(--s3);
    border-radius:var(--r);
    padding:0.875rem 1.25rem;
    font-family:var(--mono);
    font-size:0.72rem;
    color:var(--tx-2);
    letter-spacing:0.04em;
    line-height:1.5;
}

/* ── RESPONSE BOX ── */
.resp-box {
    background:var(--s1);
    border:1px solid var(--line);
    border-left:2px solid var(--ac);
    border-radius:var(--r);
    padding:1.375rem 1.5rem;
    margin-top:1rem;
    color:var(--tx);
    font-family:var(--sans);
    font-size:0.9rem;
    font-weight:300;
    line-height:1.7;
}
.resp-label {
    font-family:var(--mono);
    font-size:0.58rem;
    text-transform:uppercase;
    letter-spacing:0.13em;
    color:var(--ac);
    display:block;
    margin-bottom:0.6rem;
}

/* ── MARKDOWN ── */
.stMarkdown p {
    color:var(--tx-2);
    font-family:var(--sans);
    font-size:0.9rem;
    font-weight:300;
    line-height:1.7;
}
.stMarkdown strong { color:var(--tx); font-weight:600; }
.stMarkdown code {
    font-family:var(--mono);
    font-size:0.75rem;
    background:var(--s2);
    padding:0.15em 0.45em;
    border-radius:3px;
    color:var(--tx-2);
}

/* ── SPINNER ── */
.stSpinner > div { border-top-color:var(--ac)!important; }

/* ── WELCOME SCREEN ── */
.welcome-wrap {
    border:1px solid var(--line);
    border-radius:var(--r2);
    padding:5rem 3rem;
    text-align:center;
    background:var(--s1);
    margin-top:2rem;
}
.welcome-h {
    font-family:var(--serif);
    font-size:2rem;
    font-weight:300;
    color:var(--tx);
    letter-spacing:-0.015em;
    margin-bottom:0.875rem;
}
.welcome-p {
    font-family:var(--sans);
    font-size:0.88rem;
    font-weight:300;
    color:var(--tx-2);
    line-height:1.75;
    margin-bottom:2.75rem;
}
.feat-pills {
    display:flex;
    justify-content:center;
    gap:1px;
    background:var(--line);
    border:1px solid var(--line);
    border-radius:var(--r);
    overflow:hidden;
    max-width:660px;
    margin:0 auto;
}
.feat-pill {
    flex:1;
    background:var(--s2);
    padding:0.875rem 0.875rem;
}
.feat-pill span {
    font-family:var(--mono);
    font-size:0.58rem;
    color:var(--tx-3);
    letter-spacing:0.09em;
    text-transform:uppercase;
    line-height:1.45;
    display:block;
}

/* ── SPACERS ── */
.sp-sm { height:1rem; }
.sp-md { height:1.75rem; }
.sp-lg { height:2.5rem; }

/* ── FILE UPLOADER ── */
.stFileUploader {
    background:var(--s1);
    border:1px dashed var(--ac);
    border-radius:var(--r);
    padding:0.5rem;
}
</style>
""", unsafe_allow_html=True)


def esc(text) -> str:
    """Escape model/agent output before it goes inside our HTML wrappers."""
    return html.escape(str(text)).replace("\n", "<br>")


# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-kicker">Meta-Agent System</div>
    <h1 class="hero-h1">An AI that builds<br><em>AI agents.</em></h1>
    <p class="hero-body">
        Describe what you need in plain English. The pipeline plans,
        selects tools, writes code, tests, and deploys — automatically.
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FEATURE STRIP
# ─────────────────────────────────────────────
st.markdown("""
<div class="feat-strip">
    <div class="feat-cell">
        <div class="feat-num">01</div>
        <div class="feat-name">What-if Simulation</div>
        <div class="feat-desc">Compares every tool option before committing</div>
    </div>
    <div class="feat-cell">
        <div class="feat-num">02</div>
        <div class="feat-name">Self-Correction</div>
        <div class="feat-desc">Test failures are fed back to the LLM for a retry</div>
    </div>
    <div class="feat-cell">
        <div class="feat-num">03</div>
        <div class="feat-name">Build Cache</div>
        <div class="feat-desc">Reuses generations that already passed testing</div>
    </div>
    <div class="feat-cell">
        <div class="feat-num">04</div>
        <div class="feat-name">Hybrid Architecture</div>
        <div class="feat-desc">LLM generation with tested template fallback</div>
    </div>
    <div class="feat-cell">
        <div class="feat-num">05</div>
        <div class="feat-name">Gated Testing</div>
        <div class="feat-desc">Only agents that pass every check are deployed</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Configuration")
    selected_model = st.selectbox(
        "Model",
        list(settings.available_models),
        index=list(settings.available_models).index(settings.default_model),
        help="Used for planning, tool selection and code generation.",
    )
    st.markdown("---")
    st.markdown("## Constraints")
    budget      = st.select_slider("Budget",      options=["free", "paid"],                 value="free")
    privacy     = st.select_slider("Privacy",     options=["strict", "moderate", "none"],   value="moderate")
    performance = st.select_slider("Performance", options=["fast", "balanced"],             value="balanced")
    st.markdown("---")
    st.markdown("## Build history")
    # Filled at the end of the script so the counts include a build that just ran.
    stats_slot = st.container()
    st.markdown("---")
    st.caption("Meta-Agent · Nexus")

# ─────────────────────────────────────────────
# INPUT SECTION
# ─────────────────────────────────────────────
QUICK_STARTS = {
    "PDF QA Agent":     "Build a PDF QA system that answers questions from documents",
    "Customer Chatbot": "Build a friendly chatbot for customer service",
    "Web Search Agent": "Build a web search assistant",
    "Math Calculator":  "Build a calculator that can do basic math",
}


def use_quick_start(text: str) -> None:
    # on_click callbacks run before the rerun, so the text area shows the new value.
    st.session_state.agent_request = text


st.markdown('<div class="slabel">Build a new agent</div>', unsafe_allow_html=True)

col_in, col_ex = st.columns([3, 1], gap="large")

with col_in:
    st.text_area(
        "Describe the agent",
        key="agent_request",
        placeholder="e.g. Build a PDF QA system that answers questions from uploaded documents…",
        height=128,
        max_chars=settings.max_request_chars,
        label_visibility="collapsed",
    )

with col_ex:
    st.markdown('<div class="slabel">Quick start</div>', unsafe_allow_html=True)
    for label, text in QUICK_STARTS.items():
        st.button(label, key=f"qs_{label}", on_click=use_quick_start, args=(text,))

st.markdown('<div class="sp-md"></div>', unsafe_allow_html=True)

_, col_btn, _ = st.columns([1, 1.1, 1])
with col_btn:
    build_button = st.button("BUILD AGENT", type="primary", use_container_width=True)

# ─────────────────────────────────────────────
# PIPELINE EXECUTION
# ─────────────────────────────────────────────
STAGE_PROGRESS = {
    "planning":       ("Stage 1 / 5 · Planning",        10),
    "tool_selection": ("Stage 2 / 5 · Tool Selection",  30),
    "generation":     ("Stage 3 / 5 · Code Generation", 45),
    "testing":        ("Stage 4 / 5 · Testing",         70),
    "deployment":     ("Stage 5 / 5 · Deployment",      90),
}

request_text = st.session_state.agent_request.strip()

if build_button and not request_text:
    st.warning("Please describe the agent you want to build.")

elif build_button:
    st.session_state.build_result = None
    st.markdown("---")
    st.markdown('<div class="slabel">Pipeline</div>', unsafe_allow_html=True)
    progress_bar = st.progress(0)
    status = st.empty()

    def show_status(msg: str) -> None:
        status.markdown(f'<div class="status-box">{esc(msg)}</div>', unsafe_allow_html=True)

    def on_event(event: BuildEvent) -> None:
        label, pct = STAGE_PROGRESS.get(event.stage, (event.stage, None))
        if pct is not None and event.event in ("started", "completed"):
            progress_bar.progress(min(100, pct + (10 if event.event == "completed" else 0)))
        detail = f" — {event.message}" if event.message else ""
        show_status(f"{label} · {event.event}{detail}")

    show_status("Starting build…")
    result = get_build_service().start_build(
        request_text,
        Constraints(budget=budget, privacy=privacy, performance=performance),
        selected_model,
        on_event=on_event,
    )
    st.session_state.build_result = result
    if result.succeeded:
        progress_bar.progress(100)
        show_status(f"Pipeline complete — agent deployed ({result.duration_s:.0f}s)")
    else:
        show_status(f"Pipeline stopped at {result.failed_stage} ({result.duration_s:.0f}s)")


# ─────────────────────────────────────────────
# RESULT RENDERING
# ─────────────────────────────────────────────
def render_plan(result: BuildResult) -> None:
    pr = result.plan
    if pr is None:
        st.info("Planning did not run.")
        return
    if not pr.ok:
        st.error(f"Planning failed: {pr.error.message}")
        return
    plan = pr.plan
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Agent Type",     plan.agent_type)
    with c2: st.metric("Confidence",     f"{plan.confidence * 100:.0f}%")
    with c3: st.metric("Tools Required", len(plan.tools))
    with c4: st.metric("Flow Steps",     len(plan.flow))
    for w in pr.warnings:
        st.warning(w)
    st.markdown('<div class="sp-md"></div>', unsafe_allow_html=True)
    st.markdown('<div class="slabel">Execution Flow</div>', unsafe_allow_html=True)
    for i, step in enumerate(plan.flow, 1):
        st.markdown(f"`{i:02d}` &nbsp; {esc(step)}")


def render_simulation(result: BuildResult) -> None:
    sel = result.tool_selection
    if sel is None:
        st.info("Tool selection did not run.")
        return
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Budget",      sel.constraints.budget)
    with c2: st.metric("Privacy",     sel.constraints.privacy)
    with c3: st.metric("Performance", sel.constraints.performance)
    if not sel.ok:
        st.error(f"Tool selection failed: {sel.error.message}")
    for w in sel.warnings:
        st.warning(w)
    if not sel.simulations:
        st.info("This agent needs no tools.")
        return
    st.markdown('<div class="sp-md"></div>', unsafe_allow_html=True)
    for tool_name, sim in sel.simulations.items():
        st.markdown(f'<div class="slabel">{esc(tool_name.upper())} — tool analysis</div>', unsafe_allow_html=True)
        rows = [{"Tool": o.name, "Cost": o.cost, "Score": f"{o.score}/100",
                 "API Key": "Yes" if o.api_key_required else "No",
                 "Available": "Yes" if o.available else "No"}
                for o in sim.options]
        st.dataframe(rows, use_container_width=True, hide_index=True)
        choice = sel.selections.get(tool_name)
        if choice:
            st.success(f"Selected: **{choice.name}**" + ("" if choice.was_recommended else " (differs from simulation)"))
            st.info(f"Reasoning: {choice.reasoning[:300]}")
        st.markdown("---")


def render_code(result: BuildResult) -> None:
    gen = result.generation
    if gen is None:
        st.info("Code generation did not run.")
        return
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Lines of Code",    gen.lines)
    with c2: st.metric("Quality Score",    f"{gen.quality_score * 100:.0f}%")
    with c3: st.metric("Method",           gen.method or "—")
    with c4: st.metric("Self-corrections", gen.corrections)
    if gen.attempts:
        st.dataframe(
            [{"Attempt": a.attempt, "Source": a.method, "Passed": "Yes" if a.passed else "No",
              "Error": a.error or ""} for a in gen.attempts],
            use_container_width=True, hide_index=True,
        )
    if gen.error and not gen.code:
        st.error(f"Generation failed: {gen.error.message}")
    st.markdown('<div class="sp-sm"></div>', unsafe_allow_html=True)
    st.code(gen.code or "# No code generated", language="python", line_numbers=True)


def render_tests(result: BuildResult) -> None:
    test = result.test
    if test is None:
        st.info("Testing did not run.")
        return
    if test.passed:
        st.success("All tests passed.")
    else:
        st.error(f"Tests failed: {test.error.message if test.error else 'unknown error'}")
    st.markdown('<div class="sp-sm"></div>', unsafe_allow_html=True)
    for check in test.checks:
        mark = "✓" if check.passed else "✗"
        st.markdown(f"{mark} &nbsp; **{esc(check.name)}** — {esc(check.message)}")
    for warn in test.warnings:
        st.warning(warn)
    if test.output_preview:
        st.caption("Sample output (test mode, canned LLM reply)")
        st.code(test.output_preview, language="text")
    if test.error and test.error.detail:
        with st.expander("Failure details"):
            st.code(test.error.detail, language="text")


def render_deploy(result: BuildResult) -> None:
    dep = result.deployment
    if not result.deployed:
        reason = result.error.message if result.error else "unknown reason"
        st.error(f"Not deployed — the build stopped at **{result.failed_stage or 'an earlier stage'}**: {reason}")
        if dep is not None and dep.error:
            st.error(f"Deployment error: {dep.error.message}")
        return

    st.success("Agent deployed and ready.")
    c1, c2 = st.columns(2)
    with c1: st.metric("Deployment", dep.deployment_id)
    with c2: st.metric("Status", "Ready")
    st.markdown('<div class="sp-sm"></div>', unsafe_allow_html=True)
    st.markdown('<div class="slabel">Run command</div>', unsafe_allow_html=True)
    st.code(dep.run_command, language="bash")
    st.markdown("---")
    st.markdown('<div class="slabel">Live agent test</div>', unsafe_allow_html=True)
    st.caption("Runs the deployed agent once in a separate process. Model calls are made by "
               "MetaAgent on the agent's behalf; the agent never receives your API key.")
    test_q = st.text_input("Question", key="live_q", placeholder="Type a message…",
                           label_visibility="collapsed")
    if st.button("Send", key="send_btn"):
        if not test_q:
            st.warning("Enter a question first.")
            return
        with st.spinner("Running…"):
            try:
                run = get_build_service().run_deployed_agent(result, test_q)
            except Exception as e:
                st.error(f"Error: {describe_error(e)}")
                return
        for problem in dict.fromkeys(run.proxy_errors):
            st.warning(f"A model call made for the agent failed: {problem}")
        if run.ok:
            st.markdown(f'<div class="resp-box"><span class="resp-label">Agent response</span>'
                        f'{esc(run.output)}</div>', unsafe_allow_html=True)
            st.caption(f"{run.duration_s:.1f}s · {run.llm_calls} model call(s)")
        else:
            st.error(f"Agent run failed: {run.error_type or ''} {run.error or ''}".strip())
            if run.traceback:
                with st.expander("Traceback"):
                    st.code(run.traceback, language="text")


def get_doc_service() -> DocumentService:
    if st.session_state.doc_service is None:
        st.session_state.doc_service = DocumentService(settings)
    return st.session_state.doc_service


def render_doc_qa(_result: BuildResult) -> None:
    st.markdown('<div class="slabel">Document Upload & Q&A</div>', unsafe_allow_html=True)
    st.markdown(f"Upload PDF or TXT documents (max {settings.max_upload_mb} MB each) and ask questions "
                "about their content. Note: the knowledge base is shared by everyone using this app.")
    try:
        docs = get_doc_service()
        chunk_count = docs.document_count()
    except Exception as e:
        st.error(f"Document store unavailable: {e}")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**1. Upload Documents**")
        uploaded_files = st.file_uploader(
            "Choose PDF or TXT files",
            type=list(settings.allowed_upload_types),
            accept_multiple_files=True,
            key="rag_uploader",
            label_visibility="collapsed",
        )
        if uploaded_files and st.button("Process Documents", key="process_docs"):
            with st.spinner("Processing documents..."):
                try:
                    ingest = docs.ingest([(f.name, f.getvalue()) for f in uploaded_files])
                except Exception as e:
                    st.error(f"Error: {describe_error(e)}")
                    ingest = None
            if ingest:
                for msg in ingest.rejected:
                    st.warning(msg)
                if ingest.chunks:
                    st.success(f"Indexed {ingest.files} file(s) into {ingest.chunks} chunks. "
                               "Re-uploading a file replaces its chunks instead of duplicating them.")
                    chunk_count = docs.document_count()
                elif not ingest.rejected:
                    st.error("No text could be extracted from the uploaded files.")
        if chunk_count:
            st.success(f"{chunk_count} chunks in the knowledge base")
        else:
            st.info("The knowledge base is empty.")

    with col2:
        st.markdown("**2. Ask Questions**")
        query = st.text_area("Your question:", placeholder="What is this document about?",
                             key="rag_query", label_visibility="collapsed")
        if st.button("Ask", key="ask_rag") and query:
            if not chunk_count:
                st.warning("No documents loaded. Please upload files first.")
                return
            with st.spinner("Searching for answer..."):
                try:
                    answer = docs.answer(query)
                except Exception as e:
                    st.error(f"Error: {describe_error(e)}")
                    return
            st.markdown(f'<div class="resp-box"><span class="resp-label">Answer</span>{esc(answer.text)}</div>',
                        unsafe_allow_html=True)
            if answer.sources:
                with st.expander("View source documents"):
                    for i, (source, excerpt) in enumerate(answer.sources, 1):
                        st.markdown(f"**Source {i} ({esc(source)}):** {esc(excerpt)}…")


result: BuildResult = st.session_state.build_result

if result is not None:
    st.markdown("---")
    st.markdown('<div class="slabel">Build results</div>', unsafe_allow_html=True)
    if result.succeeded:
        st.success(f"Build {result.build_id} succeeded in {result.duration_s:.0f}s — agent deployed.")
    else:
        message = result.error.message if result.error else "unknown error"
        st.error(f"Build {result.build_id} failed at **{result.failed_stage}**: {message}")

    agent_type = result.plan.plan.agent_type if result.plan and result.plan.ok else None
    tab_specs = [("Plan", render_plan), ("Simulation", render_simulation), ("Code", render_code),
                 ("Tests", render_tests), ("Deploy", render_deploy)]
    if agent_type == "rag":
        tab_specs.append(("Document Q&A", render_doc_qa))

    for tab, (_, render) in zip(st.tabs([name for name, _ in tab_specs]), tab_specs):
        with tab:
            render(result)

    st.markdown("---")
    with st.expander("Build record (JSON)"):
        st.json(result.model_dump(mode="json"))

elif not build_button:
    # ─── WELCOME / EMPTY STATE ───
    st.markdown("""
    <div class="welcome-wrap">
        <div class="welcome-h">Start by describing an agent.</div>
        <p class="welcome-p">
            Enter a plain-English description above.<br>
            The pipeline handles planning, tool selection,<br>
            code generation, testing, and deployment.
        </p>
        <div class="feat-pills">
            <div class="feat-pill"><span>What-if<br>Simulation</span></div>
            <div class="feat-pill"><span>Self<br>Correction</span></div>
            <div class="feat-pill"><span>Build<br>Cache</span></div>
            <div class="feat-pill"><span>Gated<br>Testing</span></div>
            <div class="feat-pill"><span>Versioned<br>Deploy</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR STATS (rendered last so they include this run's build)
# ─────────────────────────────────────────────
with stats_slot:
    try:
        stats = get_build_service().stats()
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Builds",    stats.total)
            st.metric("Failed",    stats.failed)
        with c2:
            st.metric("Deployed",  stats.succeeded)
            st.metric("Corrected", stats.self_corrected)
        st.caption("Counted from saved build records")
    except Exception as e:
        st.caption(f"Build history unavailable: {e}")
