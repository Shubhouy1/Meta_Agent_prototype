# app.py - Complete UI with Document Q&A for RAG Agents

import streamlit as st
import json
import time
import pandas as pd
import sys
import os
import importlib.util
import tempfile

sys.path.insert(0, os.path.dirname(__file__))

from planner import generate_plan
from TOOL_generator.tool_generator import ToolSelector
from code_generator.stage_3_code_generator import CodeGenerator
from Tester_agent.stage_4_tester import Stage4Tester
from Delpoy_agent.stage_5_deployer import Stage5Deployer

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Meta-Agent",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# SESSION STATE (ADD RAG STATE)
# ─────────────────────────────────────────────
defaults = {
    "deployment": None,
    "code_result": None,
    "test_result": None,
    "plan": None,
    "tool_result": None,
    "agent_built": False,
    "rag_vectorstore": None,
    "rag_embeddings": None,
    "rag_llm": None,
    "rag_ready": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================
# DESIGN SYSTEM (KEPT EXACTLY AS YOU HAVE)
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

/* ── BUILD BUTTON (primary) ── */
.primary-btn .stButton > button {
    background:var(--ac)!important;
    color:#fff!important;
    border-color:var(--ac)!important;
    font-size:0.68rem!important;
    letter-spacing:0.16em!important;
    padding:0.8rem 2rem!important;
}
.primary-btn .stButton > button:hover {
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
        <div class="feat-desc">LLM detects and repairs its own errors</div>
    </div>
    <div class="feat-cell">
        <div class="feat-num">03</div>
        <div class="feat-name">Pattern Learning</div>
        <div class="feat-desc">Caches and reuses successful patterns</div>
    </div>
    <div class="feat-cell">
        <div class="feat-num">04</div>
        <div class="feat-name">Hybrid Architecture</div>
        <div class="feat-desc">LLM + prebuilt — the best of both worlds</div>
    </div>
    <div class="feat-cell">
        <div class="feat-num">05</div>
        <div class="feat-name">Auto Testing</div>
        <div class="feat-desc">4-stage validation on every build</div>
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
        ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-1.5-flash"],
        index=0
    )
    st.markdown("---")
    st.markdown("## Constraints")
    budget      = st.select_slider("Budget",      options=["free","low","medium","high"],       value="free")
    privacy     = st.select_slider("Privacy",     options=["strict","moderate","none"],         value="moderate")
    performance = st.select_slider("Performance", options=["fast","balanced","accurate"],       value="balanced")
    st.markdown("---")
    st.markdown("## Stats")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Patterns",   "12")
        st.metric("Cache Hits", "8")
    with c2:
        st.metric("LLM Success",  "94%")
        st.metric("Corrections",  "3")
    st.markdown("---")
    st.caption("Meta-Agent · Nexus")

# ─────────────────────────────────────────────
# INPUT SECTION
# ─────────────────────────────────────────────
st.markdown('<div class="slabel">Build a new agent</div>', unsafe_allow_html=True)

col_in, col_ex = st.columns([3, 1], gap="large")

with col_in:
    user_input = st.text_area(
        "Describe the agent",
        placeholder="e.g. Build a PDF QA system that answers questions from uploaded documents…",
        height=128,
        label_visibility="collapsed",
    )

with col_ex:
    st.markdown('<div class="slabel">Quick start</div>', unsafe_allow_html=True)
    if st.button("PDF QA Agent"):
        user_input = "Build a PDF QA system that answers questions from documents"
    if st.button("Customer Chatbot"):
        user_input = "Build a friendly chatbot for customer service"
    if st.button("Web Search Agent"):
        user_input = "Build a web search assistant"
    if st.button("Math Calculator"):
        user_input = "Build a calculator that can do basic math"

st.markdown('<div class="sp-md"></div>', unsafe_allow_html=True)

_, col_btn, _ = st.columns([1, 1.1, 1])
with col_btn:
    st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
    build_button = st.button("BUILD AGENT", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PIPELINE EXECUTION
# ─────────────────────────────────────────────
if build_button and user_input:
    st.session_state.agent_built = False
    st.markdown("---")
    st.markdown('<div class="slabel">Pipeline</div>', unsafe_allow_html=True)

    progress_bar = st.progress(0)
    status       = st.empty()

    def show_status(msg):
        status.markdown(f'<div class="status-box">{msg}</div>', unsafe_allow_html=True)

    try:
        show_status("Stage 1 / 5 &nbsp;·&nbsp; Planning — analyzing your request…")
        progress_bar.progress(10)
        time.sleep(0.3)

        plan = generate_plan(user_input)
        st.session_state.plan = plan
        progress_bar.progress(20)
        show_status("Stage 1 complete — agent type identified")

        show_status("Stage 2 / 5 &nbsp;·&nbsp; Tool Selection — running what-if simulation…")
        selector    = ToolSelector()
        tool_result = selector.select_tools(plan, user_input, {
            "budget": budget, "privacy": privacy, "performance": performance
        })
        st.session_state.tool_result = tool_result
        progress_bar.progress(40)
        show_status("Stage 2 complete — tools selected")

        show_status("Stage 3 / 5 &nbsp;·&nbsp; Code Generation — writing agent code…")
        generator   = CodeGenerator()
        code_result = generator.generate(plan, tool_result, user_input)
        st.session_state.code_result = code_result
        progress_bar.progress(60)
        show_status(f"Stage 3 complete — {code_result['lines']} lines generated")

        show_status("Stage 4 / 5 &nbsp;·&nbsp; Testing — validating agent…")
        tester      = Stage4Tester()
        test_result = tester.run_tests(code_result["filename"])
        st.session_state.test_result = test_result
        progress_bar.progress(80)
        label = "all tests passed" if test_result["passed"] else "tests passed with warnings"
        show_status(f"Stage 4 complete — {label}")

        show_status("Stage 5 / 5 &nbsp;·&nbsp; Deployment — packaging agent…")
        deployer   = Stage5Deployer()
        deployment = deployer.deploy(
            tested_file=code_result["filename"],
            agent_name=plan.get("agent_type", "agent"),
            create_api=False
        )
        st.session_state.deployment = deployment
        progress_bar.progress(100)
        show_status("Pipeline complete — agent deployed and ready")

        st.session_state.agent_built = True
        time.sleep(0.4)

    except Exception as e:
        st.error(f"Build failed: {str(e)}")
        progress_bar.empty()
        status.empty()

# ─────────────────────────────────────────────
# RESULTS - CONDITIONAL TABS BASED ON AGENT TYPE
# ─────────────────────────────────────────────
if st.session_state.agent_built:
    st.markdown("---")
    st.markdown('<div class="slabel">Build results</div>', unsafe_allow_html=True)

    agent_type = st.session_state.plan.get("agent_type", "chatbot")

    if agent_type == "rag":
        # 6 tabs for RAG agents (including Document Q&A)
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Plan", "Simulation", "Code", "Tests", "Deploy", "Document Q&A"])

        # Tab 1: Plan
        with tab1:
            if st.session_state.plan:
                c1, c2, c3, c4 = st.columns(4)
                with c1: st.metric("Agent Type",     st.session_state.plan.get("agent_type", "—"))
                with c2: st.metric("Confidence",     f"{st.session_state.plan.get('confidence', 0)*100:.0f}%")
                with c3: st.metric("Tools Required", len(st.session_state.plan.get("tools", [])))
                with c4: st.metric("Flow Steps",     len(st.session_state.plan.get("flow",  [])))
                st.markdown('<div class="sp-md"></div>', unsafe_allow_html=True)
                st.markdown('<div class="slabel">Execution Flow</div>', unsafe_allow_html=True)
                for i, step in enumerate(st.session_state.plan.get("flow", []), 1):
                    st.markdown(f"`{i:02d}` &nbsp; {step}")

        # Tab 2: Simulation
        with tab2:
            if st.session_state.tool_result:
                con = st.session_state.tool_result.get("constraints_applied", {})
                c1, c2, c3 = st.columns(3)
                with c1: st.metric("Budget",      con.get("budget",      "—"))
                with c2: st.metric("Privacy",     con.get("privacy",     "—"))
                with c3: st.metric("Performance", con.get("performance", "—"))
                st.markdown('<div class="sp-md"></div>', unsafe_allow_html=True)
                for tool_name, sim in st.session_state.tool_result.get("what_if_simulations", {}).items():
                    st.markdown(f'<div class="slabel">{tool_name.upper()} — tool analysis</div>', unsafe_allow_html=True)
                    rows = [{"Tool": o["name"], "Cost": o["cost"],
                             "Score": f"{o['score']}/100",
                             "API Key": "No" if not o.get("api_key_required") else "Yes"}
                            for o in sim.get("simulated_options", [])]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                    st.success(f"Recommended: **{sim.get('recommended_name')}**")
                    st.info(f"Reasoning: {sim.get('reasoning', '')[:300]}…")
                    st.markdown("---")

        # Tab 3: Code
        with tab3:
            if st.session_state.code_result:
                method = st.session_state.code_result.get("generation_method", "—")
                c1, c2, c3 = st.columns(3)
                with c1: st.metric("Lines of Code",   st.session_state.code_result.get("lines", 0))
                with c2: st.metric("Quality Score",   f"{st.session_state.code_result.get('quality_score', 0)*100:.0f}%")
                with c3: st.metric("Method",          method)
                if "self_corrected" in method:
                    st.success("Self-correction applied — LLM detected and fixed its own errors.")
                st.markdown('<div class="sp-sm"></div>', unsafe_allow_html=True)
                st.code(
                    st.session_state.code_result.get("code", "# No code generated"),
                    language="python",
                    line_numbers=True
                )

        # Tab 4: Tests
        with tab4:
            if st.session_state.test_result:
                if st.session_state.test_result.get("passed"):
                    st.success("All tests passed.")
                st.markdown('<div class="sp-sm"></div>', unsafe_allow_html=True)
                for res in st.session_state.test_result.get("test_results", []):
                    st.markdown(f"✓ &nbsp; {res}")
                for warn in st.session_state.test_result.get("warnings", []):
                    st.warning(warn)

        # Tab 5: Deploy
        with tab5:
            dep = st.session_state.deployment
            if dep and dep.get("success"):
                st.success("Agent deployed and ready.")
                c1, c2 = st.columns(2)
                with c1: st.metric("Deployed File", dep.get("deployed_file", "—"))
                with c2: st.metric("Status", "Ready")
                st.markdown('<div class="sp-sm"></div>', unsafe_allow_html=True)
                st.markdown('<div class="slabel">Run command</div>', unsafe_allow_html=True)
                st.code(dep.get("run_command", "—"), language="bash")
                st.markdown("---")
                st.markdown('<div class="slabel">Live agent test</div>', unsafe_allow_html=True)
                test_q = st.text_input("Question", key="live_q",
                                       placeholder="Type a message…",
                                       label_visibility="collapsed")
                if st.button("Send", key="send_btn"):
                    if test_q:
                        with st.spinner("Running…"):
                            try:
                                af = dep.get("deployed_file")
                                if af and os.path.exists(af):
                                    spec   = importlib.util.spec_from_file_location("test_agent", af)
                                    module = importlib.util.module_from_spec(spec)
                                    spec.loader.exec_module(module)
                                    resp = module.Agent().run(test_q)
                                    st.markdown(
                                        f'<div class="resp-box"><span class="resp-label">Agent response</span>{resp}</div>',
                                        unsafe_allow_html=True
                                    )
                                else:
                                    st.error(f"File not found: {af}")
                            except Exception as e:
                                st.error(f"Error: {e}")
                    else:
                        st.warning("Enter a question first.")

        # Tab 6: Document Q&A (RAG only)
        with tab6:
            st.markdown('<div class="slabel">Document Upload & Q&A</div>', unsafe_allow_html=True)
            st.markdown("Upload PDF or TXT documents and ask questions about their content")

            # Initialize RAG components
            if st.session_state.rag_ready is False:
                try:
                    from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
                    from langchain_chroma import Chroma

                    st.session_state.rag_embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
                    st.session_state.rag_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

                    if os.path.exists("./chroma_db"):
                        try:
                            st.session_state.rag_vectorstore = Chroma(
                                persist_directory="./chroma_db",
                                embedding_function=st.session_state.rag_embeddings
                            )
                            st.session_state.rag_ready = True
                        except:
                            st.session_state.rag_vectorstore = None
                    else:
                        st.session_state.rag_vectorstore = None
                except Exception as e:
                    st.warning(f"RAG components not available: {e}")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**1. Upload Documents**")
                uploaded_files = st.file_uploader(
                    "Choose PDF or TXT files",
                    type=["pdf", "txt"],
                    accept_multiple_files=True,
                    key="rag_uploader",
                    label_visibility="collapsed"
                )

                if uploaded_files and st.button("Process Documents", key="process_docs"):
                    with st.spinner("Processing documents..."):
                        try:
                            from langchain_community.document_loaders import PyPDFLoader, TextLoader
                            from langchain_text_splitters import RecursiveCharacterTextSplitter
                            from langchain_chroma import Chroma

                            all_docs = []
                            for uploaded_file in uploaded_files:
                                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
                                    tmp.write(uploaded_file.getvalue())
                                    tmp_path = tmp.name

                                if uploaded_file.name.endswith('.pdf'):
                                    loader = PyPDFLoader(tmp_path)
                                else:
                                    loader = TextLoader(tmp_path, encoding='utf-8')

                                docs = loader.load()
                                all_docs.extend(docs)
                                os.unlink(tmp_path)

                            if all_docs:
                                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                                chunks = text_splitter.split_documents(all_docs)

                                if st.session_state.rag_vectorstore:
                                    st.session_state.rag_vectorstore.add_documents(chunks)
                                    st.session_state.rag_vectorstore.persist()
                                else:
                                    st.session_state.rag_vectorstore = Chroma.from_documents(
                                        documents=chunks,
                                        embedding=st.session_state.rag_embeddings,
                                        persist_directory="./chroma_db"
                                    )

                                st.session_state.rag_ready = True
                                st.success(f"✅ Processed {len(uploaded_files)} files into {len(chunks)} chunks")
                                st.rerun()
                            else:
                                st.error("No documents could be loaded")
                        except Exception as e:
                            st.error(f"Error: {e}")

                if st.session_state.rag_ready and st.session_state.rag_vectorstore:
                    st.success("✅ Documents ready for questions")

            with col2:
                st.markdown("**2. Ask Questions**")
                query = st.text_area("Your question:", placeholder="What is this document about?", key="rag_query", label_visibility="collapsed")

                if st.button("Ask", key="ask_rag") and query:
                    with st.spinner("Searching for answer..."):
                        try:
                            if st.session_state.rag_vectorstore:
                                docs = st.session_state.rag_vectorstore.similarity_search(query, k=3)
                                if docs:
                                    context = "\n\n".join([d.page_content for d in docs])
                                    prompt = f"Based on the context, answer:\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
                                    response = st.session_state.rag_llm.invoke(prompt)
                                    st.markdown(f'<div class="resp-box"><span class="resp-label">Answer</span>{response.content}</div>', unsafe_allow_html=True)

                                    with st.expander("View source documents"):
                                        for i, doc in enumerate(docs):
                                            st.write(f"**Source {i+1}:** {doc.page_content[:300]}...")
                                else:
                                    st.warning("No relevant information found. Try a different question.")
                            else:
                                st.warning("No documents loaded. Please upload files first.")
                        except Exception as e:
                            st.error(f"Error: {e}")

    else:
        # 5 tabs for non-RAG agents
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Plan", "Simulation", "Code", "Tests", "Deploy"])

        # Tab 1: Plan
        with tab1:
            if st.session_state.plan:
                c1, c2, c3, c4 = st.columns(4)
                with c1: st.metric("Agent Type",     st.session_state.plan.get("agent_type", "—"))
                with c2: st.metric("Confidence",     f"{st.session_state.plan.get('confidence', 0)*100:.0f}%")
                with c3: st.metric("Tools Required", len(st.session_state.plan.get("tools", [])))
                with c4: st.metric("Flow Steps",     len(st.session_state.plan.get("flow",  [])))
                st.markdown('<div class="sp-md"></div>', unsafe_allow_html=True)
                st.markdown('<div class="slabel">Execution Flow</div>', unsafe_allow_html=True)
                for i, step in enumerate(st.session_state.plan.get("flow", []), 1):
                    st.markdown(f"`{i:02d}` &nbsp; {step}")

        # Tab 2: Simulation
        with tab2:
            if st.session_state.tool_result:
                con = st.session_state.tool_result.get("constraints_applied", {})
                c1, c2, c3 = st.columns(3)
                with c1: st.metric("Budget",      con.get("budget",      "—"))
                with c2: st.metric("Privacy",     con.get("privacy",     "—"))
                with c3: st.metric("Performance", con.get("performance", "—"))
                st.markdown('<div class="sp-md"></div>', unsafe_allow_html=True)
                for tool_name, sim in st.session_state.tool_result.get("what_if_simulations", {}).items():
                    st.markdown(f'<div class="slabel">{tool_name.upper()} — tool analysis</div>', unsafe_allow_html=True)
                    rows = [{"Tool": o["name"], "Cost": o["cost"],
                             "Score": f"{o['score']}/100",
                             "API Key": "No" if not o.get("api_key_required") else "Yes"}
                            for o in sim.get("simulated_options", [])]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                    st.success(f"Recommended: **{sim.get('recommended_name')}**")
                    st.info(f"Reasoning: {sim.get('reasoning', '')[:300]}…")
                    st.markdown("---")

        # Tab 3: Code
        with tab3:
            if st.session_state.code_result:
                method = st.session_state.code_result.get("generation_method", "—")
                c1, c2, c3 = st.columns(3)
                with c1: st.metric("Lines of Code",   st.session_state.code_result.get("lines", 0))
                with c2: st.metric("Quality Score",   f"{st.session_state.code_result.get('quality_score', 0)*100:.0f}%")
                with c3: st.metric("Method",          method)
                if "self_corrected" in method:
                    st.success("Self-correction applied — LLM detected and fixed its own errors.")
                st.markdown('<div class="sp-sm"></div>', unsafe_allow_html=True)
                st.code(
                    st.session_state.code_result.get("code", "# No code generated"),
                    language="python",
                    line_numbers=True
                )

        # Tab 4: Tests
        with tab4:
            if st.session_state.test_result:
                if st.session_state.test_result.get("passed"):
                    st.success("All tests passed.")
                st.markdown('<div class="sp-sm"></div>', unsafe_allow_html=True)
                for res in st.session_state.test_result.get("test_results", []):
                    st.markdown(f"✓ &nbsp; {res}")
                for warn in st.session_state.test_result.get("warnings", []):
                    st.warning(warn)

        # Tab 5: Deploy
        with tab5:
            dep = st.session_state.deployment
            if dep and dep.get("success"):
                st.success("Agent deployed and ready.")
                c1, c2 = st.columns(2)
                with c1: st.metric("Deployed File", dep.get("deployed_file", "—"))
                with c2: st.metric("Status", "Ready")
                st.markdown('<div class="sp-sm"></div>', unsafe_allow_html=True)
                st.markdown('<div class="slabel">Run command</div>', unsafe_allow_html=True)
                st.code(dep.get("run_command", "—"), language="bash")
                st.markdown("---")
                st.markdown('<div class="slabel">Live agent test</div>', unsafe_allow_html=True)
                test_q = st.text_input("Question", key="live_q",
                                       placeholder="Type a message…",
                                       label_visibility="collapsed")
                if st.button("Send", key="send_btn"):
                    if test_q:
                        with st.spinner("Running…"):
                            try:
                                af = dep.get("deployed_file")
                                if af and os.path.exists(af):
                                    spec   = importlib.util.spec_from_file_location("test_agent", af)
                                    module = importlib.util.module_from_spec(spec)
                                    spec.loader.exec_module(module)
                                    resp = module.Agent().run(test_q)
                                    st.markdown(
                                        f'<div class="resp-box"><span class="resp-label">Agent response</span>{resp}</div>',
                                        unsafe_allow_html=True
                                    )
                                else:
                                    st.error(f"File not found: {af}")
                            except Exception as e:
                                st.error(f"Error: {e}")
                    else:
                        st.warning("Enter a question first.")

    # Show stats
    if st.session_state.code_result and "stats" in st.session_state.code_result:
        st.markdown("---")
        with st.expander("Generator Statistics & Pattern Learning Data"):
            st.json(st.session_state.code_result["stats"])

elif build_button and not user_input:
    st.warning("Please describe the agent you want to build.")

else:
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
            <div class="feat-pill"><span>Pattern<br>Learning</span></div>
            <div class="feat-pill"><span>Auto<br>Testing</span></div>
            <div class="feat-pill"><span>One-click<br>Deploy</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)