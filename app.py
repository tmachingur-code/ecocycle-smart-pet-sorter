"""
EcoCycle Energy Hub - Smart PET Sorter (V1 prototype)
Offline Edge-AI identification and sorting of PET bottles.

Run:  streamlit run app.py
"""
import os
import time
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from ultralytics import YOLO

# =====================================================================
# SETTINGS
# =====================================================================
MODEL_PATH = "models/best.tflite"
LOG_PATH = "data/ecocycle_log.csv"
MESSAGES_PATH = "data/messages.csv"
CLASS_LABELS = {"39": "PET Plastic"}   # model class name -> display name
SERVO_PIN = 18                         # BCM pin, only used on a Raspberry Pi
SERVO_HOLD_SECONDS = 0.6
MISS_FRAMES_TO_RESET = 6
SPEED_TARGET_MS = 200

# Contact details shown on the Contact page. Empty values are hidden. Edit these.
CONTACT = {"person": "Tsungirirai Machingura", "email": "", "phone": "", "location": ""}

TEAL, TEAL_DARK, GREEN, ORANGE = "#0F766E", "#0B4F4A", "#15803D", "#EA580C"
INK, MUTED, EDGE, BG = "#0F1F2E", "#475467", "#DDE4EA", "#F3F6F8"

os.makedirs("data", exist_ok=True)
st.set_page_config(page_title="EcoCycle Energy Hub", page_icon=":material/recycling:", layout="wide",
                   initial_sidebar_state="collapsed")

# =====================================================================
# ICONS (inline SVG, no emojis)
# =====================================================================
ICON_PATHS = {
    "camera": '<path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z"/><circle cx="12" cy="13" r="3"/>',
    "cpu": '<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><path d="M9 1v3M15 1v3M9 20v3M15 20v3M20 9h3M20 14h3M1 9h3M1 14h3"/>',
    "chart": '<path d="M18 20V10M12 20V4M6 20v-6"/>',
    "leaf": '<path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.5 19 2c1 2 2 4.2 2 8 0 5.5-4.8 10-10 10z"/><path d="M2 21c0-3 1.9-5.4 5.1-6C9.5 14.5 12 13 13 12"/>',
    "users": '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.9M16 3.1a4 4 0 0 1 0 7.8"/>',
    "city": '<rect x="4" y="2" width="16" height="20" rx="2"/><path d="M9 22v-4h6v4M8 6h.01M12 6h.01M16 6h.01M8 10h.01M12 10h.01M16 10h.01M8 14h.01M12 14h.01M16 14h.01"/>',
    "factory": '<path d="M2 20h20M4 20V10l6 4v-4l6 4V4h4v16"/>',
    "shield": '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>',
    "mail": '<rect x="2" y="4" width="20" height="16" rx="2"/><path d="m22 7-10 6L2 7"/>',
    "phone": '<path d="M22 16.9v3a2 2 0 0 1-2.2 2 19.8 19.8 0 0 1-8.6-3.1 19.5 19.5 0 0 1-6-6A19.8 19.8 0 0 1 2.1 4.2 2 2 0 0 1 4.1 2h3a2 2 0 0 1 2 1.7c.1 1 .4 1.9.7 2.8a2 2 0 0 1-.5 2.1L8.1 9.9a16 16 0 0 0 6 6l1.3-1.3a2 2 0 0 1 2.1-.4c.9.3 1.8.6 2.8.7a2 2 0 0 1 1.7 2z"/>',
    "pin": '<path d="M20 10c0 6-8 12-8 12S4 16 4 10a8 8 0 0 1 16 0z"/><circle cx="12" cy="10" r="3"/>',
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "eye": '<path d="M2 12s4-8 10-8 10 8 10 8-4 8-10 8S2 12 2 12z"/><circle cx="12" cy="12" r="3"/>',
    "offline": '<path d="M1 1l22 22M16.7 11.1A11 11 0 0 1 19 12.5M5 12.5a11 11 0 0 1 5.2-2.4M8.5 16.1a6 6 0 0 1 7 0M12 20h.01"/>',
    "box": '<path d="M21 8l-9-5-9 5v8l9 5 9-5z"/><path d="M3 8l9 5 9-5M12 13v8"/>',
    "layers": '<path d="M12 2 2 7l10 5 10-5-10-5z"/><path d="m2 17 10 5 10-5M2 12l10 5 10-5"/>',
    "globe": '<circle cx="12" cy="12" r="10"/><path d="M2 12h20M12 2a15 15 0 0 1 0 20M12 2a15 15 0 0 0 0 20"/>',
    "recycle": '<path d="M7 19H4.8a1.8 1.8 0 0 1-1.6-2.7L4.3 14.5M11 19h6.5a1.8 1.8 0 0 0 1.6-2.7l-.6-1M14 16l-3 3 3 3M8.3 13.6 4.3 14.5l-.9-4M9.6 6.6l1.3-2.3a1.8 1.8 0 0 1 3.2 0l1.1 1.9M12.6 8 15.2 5.9l2.7 3.6M17.9 12.5l3 .1-.6 4"/>',
    "bolt": '<path d="M13 2 3 14h9l-1 8 10-12h-9l1-8z"/>',
    "send": '<path d="m22 2-7 20-4-9-9-4 20-7zM22 2 11 13"/>',
}


def icon(name, size=22):
    return (f'<svg class="ico" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            f'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{ICON_PATHS[name]}</svg>')


# =====================================================================
# STYLE
# =====================================================================
CSS = """
<style>
:root { color-scheme: light; }
html, body, .stApp { font-family: 'Segoe UI', system-ui, -apple-system, Roboto, 'Helvetica Neue', Arial, sans-serif; font-size:16px; }
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"], .main { background:%BG% !important; color:%INK% !important; }
.block-container { padding: 1rem 1.4rem 3rem 1.4rem; max-width: 1180px; }
header, [data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDeployButton"], #MainMenu, footer,
[data-testid="stDecoration"], [data-testid="stStatusWidget"], [data-testid="stSidebar"], [data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"] { display:none !important; visibility:hidden !important; }
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5 { color:%INK% !important; letter-spacing:-0.01em; font-weight:700; }
.stApp .stMarkdown, .stApp .stMarkdown p, .stApp .stMarkdown li, .stApp label, .stApp [data-testid="stWidgetLabel"] p,
.stApp [data-testid="stCaptionContainer"], .stApp [data-testid="stExpander"] summary p { color:%INK% !important; }
.stApp input, .stApp textarea, .stApp [data-baseweb="input"], .stApp [data-baseweb="base-input"], .stApp [data-baseweb="select"] > div
  { background:#fff !important; color:%INK% !important; }
[data-testid="stExpander"] { background:#fff; border:1px solid %EDGE%; border-radius:14px; }
[data-testid="stFileUploader"] * { color:%INK% !important; }
[data-testid="stFileUploader"] section { border-radius:14px; border:2px dashed #9FC8C2; background:#fff; }
[data-testid="stFileUploader"] button { background:#fff !important; border:1px solid %EDGE% !important; }
div[role="slider"] { background:%TEAL% !important; }
[data-testid="stTabs"] button { font-weight:700; }
[data-testid="stTabs"] button[aria-selected="true"] { color:%TEAL% !important; }
div[data-testid="stImage"] img { border-radius:14px; }
[data-testid="stForm"] { background:#fff; border:1px solid %EDGE%; border-radius:16px; padding:18px; }

/* buttons: orange = main action, white = everything else */
[data-testid^="stBaseButton-primary"], button[kind="primary"], button[kind="primaryFormSubmit"] {
  background:%ORANGE% !important; border:1px solid %ORANGE% !important; color:#fff !important; font-weight:700 !important;
  border-radius:10px !important; padding:.55rem 1.4rem !important; transition:filter .15s, transform .15s; }
[data-testid^="stBaseButton-primary"] p, button[kind="primary"] p { color:#fff !important; }
[data-testid^="stBaseButton-primary"]:hover, button[kind="primary"]:hover { filter:brightness(.92); transform:translateY(-1px); }
[data-testid^="stBaseButton-secondary"], button[kind="secondary"] {
  background:#fff !important; border:1px solid #B8C4CE !important; color:%INK% !important; font-weight:600 !important;
  border-radius:10px !important; padding:.55rem 1.3rem !important; transition:border-color .15s, background .15s; }
[data-testid^="stBaseButton-secondary"] p, button[kind="secondary"] p { color:%INK% !important; }
[data-testid^="stBaseButton-secondary"]:hover, button[kind="secondary"]:hover { border-color:%TEAL% !important; background:#F0F9F7 !important; }

/* top bar */
.topbar { display:flex; flex-wrap:wrap; align-items:center; justify-content:space-between; gap:10px; background:#fff;
          border:1px solid %EDGE%; border-radius:14px; padding:10px 16px; margin-bottom:8px; }
.brand { display:flex; align-items:center; gap:10px; font-weight:800; font-size:1.1rem; color:%INK%; }
.brand .logo { width:34px; height:34px; border-radius:10px; background:%TEAL%; color:#fff; display:flex; align-items:center; justify-content:center; }
.brand small { display:block; font-weight:600; font-size:.74rem; color:%MUTED%; margin-top:-2px; }
.chips { display:flex; flex-wrap:wrap; gap:8px; }
.chip { display:inline-flex; align-items:center; gap:7px; border-radius:999px; padding:5px 12px; font-size:.82rem; font-weight:600;
        background:#F0F9F7; border:1px solid #BFE3DD; color:%TEAL_DARK%; }
.chip .dot { width:9px; height:9px; border-radius:50%; background:#16A34A; box-shadow:0 0 0 3px rgba(22,163,74,.2); }
.chip.plain { background:#F3F4F6; border-color:%EDGE%; color:%MUTED%; }

/* navigation: underline tabs, sticky on desktop, bottom bar on phones */
.st-key-appnav { position:sticky; top:0; z-index:90; background:rgba(243,246,248,.95); backdrop-filter:blur(8px);
                 padding:2px 0 0 0; margin-bottom:14px; border-bottom:1px solid %EDGE%; }
.st-key-appnav [data-testid="stButtonGroup"] { gap:2px; flex-wrap:wrap; }
.st-key-appnav button { background:transparent !important; border:none !important; border-bottom:3px solid transparent !important;
        border-radius:0 !important; color:%MUTED% !important; font-weight:600 !important; padding:.6rem 1rem !important; transition:color .15s, background .15s; }
.st-key-appnav button p { color:inherit !important; font-size:.98rem !important; }
.st-key-appnav button:hover { color:%INK% !important; background:rgba(15,118,110,.07) !important; }
.st-key-appnav [data-testid="stBaseButton-pillsActive"], .st-key-appnav button[kind="pillsActive"], .st-key-appnav button[aria-checked="true"] {
        color:%TEAL% !important; border-bottom-color:%TEAL% !important; font-weight:700 !important; background:transparent !important; }

/* hero */
.hero { display:flex; flex-wrap:wrap; align-items:center; justify-content:space-between; gap:20px;
        background:linear-gradient(135deg,%TEAL_DARK% 0%,#0D5C57 55%,%TEAL% 100%); border-radius:20px; padding:34px 36px; margin:4px 0 14px 0; }
.hero .copy { flex:1 1 320px; min-width:0; }
.hero .hero-title { color:#fff !important; font-size:2.6rem; line-height:1.1; font-weight:800; margin:0 0 12px 0; letter-spacing:-0.02em; }
.hero p { color:#D3EFEA !important; font-size:1.12rem; margin:0; max-width:30em; line-height:1.5; }
.hero .art { flex:0 0 auto; margin:0 auto; }
.pagehead { margin:4px 0 14px 0; }
.pagehead .ph-title { font-size:1.8rem; font-weight:800; color:%INK%; letter-spacing:-0.01em; }
.pagehead p { color:%MUTED% !important; margin:2px 0 0 0; font-size:1.02rem; }
.sec { font-size:1.2rem; font-weight:700; color:%INK%; margin:22px 0 8px 0; }

/* grids and cards */
.grid { display:grid; gap:14px; margin:6px 0 12px 0; }
.grid.kpis  { grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); }
.grid.cards { grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); }
.grid.cards.wide { grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); }
.kpi, .card, .flow .step, .block { transition:transform .18s ease, box-shadow .18s ease, border-color .18s ease, background .18s ease; }
.kpi { background:#fff; border:1px solid %EDGE%; border-radius:14px; padding:16px 18px; box-shadow:0 1px 2px rgba(15,31,46,.04); }
.kpi .k-label { color:%MUTED%; font-size:.9rem; font-weight:600; }
.kpi .k-value { color:%INK%; font-size:2.2rem; font-weight:800; line-height:1.2; margin:2px 0; }
.kpi .k-sub { color:#52606D; font-size:.84rem; }
.kpi.green .k-value { color:%GREEN%; } .kpi.teal .k-value { color:%TEAL%; } .kpi.warn .k-value { color:%ORANGE%; }
.card { background:#fff; border:1px solid %EDGE%; border-radius:14px; padding:18px 20px; box-shadow:0 1px 2px rgba(15,31,46,.04); }
.card .ib { width:44px; height:44px; border-radius:12px; background:#E6F4F1; color:%TEAL%; display:flex; align-items:center; justify-content:center; margin-bottom:12px; transition:background .18s, color .18s; }
.card .c-title { margin:0 0 4px 0; font-size:1.05rem; font-weight:700; color:%INK%; }
.card p { margin:0; color:#344054 !important; font-size:.95rem; line-height:1.55; }
.card .tag { display:inline-block; font-size:.78rem; font-weight:700; border-radius:6px; padding:2px 9px; margin-bottom:8px; background:#E6F4F1; color:%TEAL_DARK%; }
.card .tag.plan { background:#EEF1F4; color:#475467; }
.card.planned { border-style:dashed; background:#FAFBFC; }
.card:hover, .kpi:hover, .flow .step:hover, .block:hover { transform:translateY(-3px); box-shadow:0 12px 26px rgba(15,31,46,.10); border-color:#8CCFC5; }
.card:hover .ib { background:%TEAL%; color:#fff; }
.sdg { display:flex; gap:14px; align-items:flex-start; }
.sdg .badge { flex:0 0 64px; height:64px; border-radius:12px; color:#fff; font-weight:800; font-size:.8rem; display:flex; flex-direction:column;
              align-items:center; justify-content:center; line-height:1.1; }
.sdg .badge b { font-size:1.6rem; }

/* expandable cards */
details.card { cursor:pointer; }
details.card summary { list-style:none; display:flex; align-items:center; gap:14px; }
details.card summary::-webkit-details-marker { display:none; }
details.card summary .ib { margin:0; flex:0 0 44px; }
details.card summary .st b { display:block; font-size:1.05rem; color:%INK%; }
details.card summary .st small { display:block; color:%MUTED%; font-size:.9rem; margin-top:1px; }
details.card summary::after { content:"+"; margin-left:auto; font-size:1.5rem; color:%TEAL%; font-weight:600; }
details.card[open] summary::after { content:"\\2212"; }
details.card .body { margin-top:12px; padding-top:12px; border-top:1px solid #EEF2F5; color:#344054; font-size:.95rem; line-height:1.55; }

/* decisions */
.decision { border-radius:14px; padding:16px 20px; margin:10px 0; font-weight:800; font-size:1.25rem; }
.decision small { display:block; font-weight:600; font-size:.92rem; margin-top:2px; }
.decision.recycle { background:#DCFCE7; border:1px solid #86EFAC; color:#14532D; }
.decision.none { background:#F3F4F6; border:1px solid %EDGE%; color:#344054; }
.decision.idle { background:#fff; border:1px solid %EDGE%; color:%MUTED%; font-weight:600; font-size:1rem; }

/* process flow */
.flow { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:14px; margin:8px 0 14px 0; }
.flow .step { position:relative; background:#fff; border:1px solid %EDGE%; border-radius:14px; padding:18px 14px; text-align:center; }
.flow .step .em { width:44px; height:44px; margin:0 auto 8px auto; border-radius:12px; background:#E6F4F1; color:%TEAL%; display:flex;
                  align-items:center; justify-content:center; font-weight:800; font-size:1.15rem; }
.flow .step b { display:block; color:%INK%; font-size:1rem; }
.flow .step span { display:block; color:%MUTED%; font-size:.88rem; margin-top:3px; }
.flow .step.on { border-color:%TEAL%; background:#F0F9F7; }
.flow .step.on .em { background:%TEAL%; color:#fff; }
.flow .step.planned { border-style:dashed; background:#FAFBFC; }
.flow .step:not(:last-child)::after { content:"\\203A"; position:absolute; right:-12px; top:50%; transform:translateY(-50%);
        color:%TEAL%; font-weight:800; font-size:1.5rem; z-index:2; }
.block { background:#fff; border:1px solid %EDGE%; border-top:3px solid %TEAL%; border-radius:12px; padding:14px; text-align:center; font-size:.92rem; color:%INK%; }
.block b { display:block; color:%TEAL_DARK%; margin-bottom:3px; font-size:1rem; }
.block.data { border-top-color:%GREEN%; }
.note { color:%MUTED% !important; font-size:.88rem; }

/* feed */
.feed { background:#fff; border:1px solid %EDGE%; border-radius:14px; overflow:hidden; }
.feed .row { display:flex; align-items:center; gap:12px; padding:12px 16px; border-bottom:1px solid #EEF2F5; transition:background .15s; }
.feed .row:hover { background:#F7FAFB; }
.feed .row:last-child { border-bottom:none; }
.feed .ic { flex:0 0 38px; height:38px; border-radius:10px; display:flex; align-items:center; justify-content:center; background:#E6F4F1; color:%TEAL%; }
.feed .tx { flex:1 1 auto; min-width:0; font-weight:600; color:%INK%; font-size:.96rem; }
.feed .tx small { display:block; color:%MUTED%; font-weight:500; font-size:.84rem; }
.feed .tm { color:%MUTED%; font-size:.84rem; white-space:nowrap; }
.contact-row { display:flex; align-items:center; gap:12px; padding:10px 0; color:%INK%; font-weight:600; }
.contact-row .ic { width:38px; height:38px; border-radius:10px; background:#E6F4F1; color:%TEAL%; display:flex; align-items:center; justify-content:center; flex:0 0 38px; }
.contact-row small { display:block; color:%MUTED%; font-weight:500; }
.appfoot { text-align:center; color:#667085; font-size:.84rem; margin-top:34px; padding-top:14px; border-top:1px solid %EDGE%; }

@media (max-width: 640px) {
  .block-container { padding: .6rem .8rem 6.5rem .8rem; }
  .topbar { padding:9px 12px; } .chips { display:none; }
  .hero { padding:24px 20px; border-radius:16px; } .hero .hero-title { font-size:1.9rem; } .hero p { font-size:1rem; }
  .pagehead .ph-title { font-size:1.5rem; }
  .kpi { padding:12px 14px; } .kpi .k-value { font-size:1.7rem; }
  .grid.kpis { grid-template-columns:repeat(2,1fr); }
  .flow { grid-template-columns:1fr; }
  .flow .step:not(:last-child)::after { content:"\\2193"; right:50%; top:auto; bottom:-18px; transform:translateX(50%); }
  .decision { font-size:1.05rem; }
  .st-key-appnav { position:fixed; left:0; right:0; bottom:0; top:auto; z-index:999; background:#fff; border-top:1px solid %EDGE%;
                   border-bottom:none; padding:4px 6px calc(4px + env(safe-area-inset-bottom)) 6px; margin:0; box-shadow:0 -6px 18px rgba(15,31,46,.08); }
  .st-key-appnav [data-testid="stButtonGroup"] { flex-wrap:nowrap; overflow-x:auto; gap:0; scrollbar-width:none; }
  .st-key-appnav button { padding:.5rem .8rem !important; white-space:nowrap; }
}
@media (prefers-reduced-motion: reduce) { * { transition:none !important; animation:none !important; } }
</style>
"""
for _k, _v in {"%INK%": INK, "%MUTED%": MUTED, "%EDGE%": EDGE, "%BG%": BG, "%TEAL_DARK%": TEAL_DARK, "%TEAL%": TEAL,
               "%GREEN%": GREEN, "%ORANGE%": ORANGE}.items():
    CSS = CSS.replace(_k, _v)
st.markdown(CSS, unsafe_allow_html=True)

BOTTLE_SVG = """<svg class="art" viewBox="0 0 130 235" width="120" height="217" role="img" aria-label="PET bottle with a detection box"><defs><linearGradient id="bg" x1="0" x2="1"><stop offset="0" stop-color="#CFFAF0"/><stop offset="1" stop-color="#7DE3D2"/></linearGradient></defs><rect x="49" y="8" width="32" height="16" rx="4" fill="#0B4F4A"/><path d="M51 24h28v22c0 14 26 22 26 52v112c0 10-8 16-18 16H43c-10 0-18-6-18-16V98c0-30 26-38 26-52z" fill="url(#bg)" stroke="#E6FFFA" stroke-width="3"/><rect x="25" y="118" width="80" height="52" fill="#15803D"/><text x="65" y="150" text-anchor="middle" font-size="22" font-weight="800" fill="#fff" font-family="Segoe UI,Arial">PET 1</text><path d="M37 104v92" stroke="#fff" stroke-opacity=".6" stroke-width="5" stroke-linecap="round"/><path d="M31 188h68M31 199h68" stroke="#0B4F4A" stroke-opacity=".3" stroke-width="2"/><rect x="9" y="34" width="112" height="196" rx="8" fill="none" stroke="#4ADE80" stroke-width="3" stroke-dasharray="9 6"/><rect x="9" y="12" width="44" height="20" rx="5" fill="#4ADE80"/><text x="31" y="27" text-anchor="middle" font-size="13" font-weight="700" fill="#052E16" font-family="Segoe UI,Arial">PET</text></svg>"""


# =====================================================================
# HTML HELPERS (single-line strings so Markdown never treats them as code)
# =====================================================================
def kpi(label, value, sub="", tone=""):
    return (f'<div class="kpi {tone}"><div class="k-label">{label}</div>'
            f'<div class="k-value">{value}</div><div class="k-sub">{sub}</div></div>')


def kpi_grid(items):
    return '<div class="grid kpis">' + "".join(kpi(*i) for i in items) + "</div>"


def card(title, body, ico="", tag="", plan=False):
    t = f'<span class="tag{" plan" if plan else ""}">{tag}</span><br>' if tag else ""
    i = f'<div class="ib">{icon(ico)}</div>' if ico else ""
    return f'<div class="card{" planned" if plan else ""}">{i}{t}<div class="c-title">{title}</div><p>{body}</p></div>'


def detail_card(title, line, body, ico, tag="", plan=False):
    t = f'<span class="tag{" plan" if plan else ""}">{tag}</span> ' if tag else ""
    return (f'<details class="card{" planned" if plan else ""}"><summary><span class="ib">{icon(ico)}</span>'
            f'<span class="st"><b>{title}</b><small>{line}</small></span></summary><div class="body">{t}{body}</div></details>')


def card_grid(cards, wide=False):
    return f'<div class="grid cards{" wide" if wide else ""}">' + "".join(cards) + "</div>"


def flow(steps, on=None):
    """steps: (icon_name | step_number, title, subtitle, state)."""
    out = '<div class="flow">'
    for i, (sym, t, sub, state) in enumerate(steps):
        cls = "step" + (" on" if on == "all" or on == i else "") + (" planned" if state == "planned" else "")
        mark = str(sym) if isinstance(sym, int) else icon(sym)
        out += f'<div class="{cls}"><div class="em">{mark}</div><b>{t}</b><span>{sub}</span></div>'
    return out + "</div>"


LIVE_STEPS = [("camera", "Camera", "Captures the item", ""), ("cpu", "AI detection", "Runs on the device", ""),
              ("recycle", "PET sorting", "Flipper diverts PET", ""), ("chart", "Data log", "Counts and weight", "")]


def head(title, sub=""):
    st.markdown(f'<div class="pagehead"><div class="ph-title">{title}</div>' + (f"<p>{sub}</p>" if sub else "") + "</div>",
                unsafe_allow_html=True)


def sec(text):
    st.markdown(f'<div class="sec">{text}</div>', unsafe_allow_html=True)


def show_image(target, img_bgr):
    try:
        target.image(img_bgr, channels="BGR", width="stretch")
    except TypeError:
        target.image(img_bgr, channels="BGR", use_container_width=True)


def show_chart(fig):
    try:
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def style_fig(fig, ytitle, height=300):
    axis = dict(gridcolor="#E4EAEF", linecolor="#98A2B3", tickfont=dict(size=13, color=INK), title_font=dict(size=13, color=INK))
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(color=INK, size=13), showlegend=False,
                      xaxis=dict(type="category", **axis), yaxis=dict(title=ytitle, rangemode="tozero", **axis))
    return fig


# =====================================================================
# MODEL, SERVO, LOG
# =====================================================================
@st.cache_resource(show_spinner="Loading the AI model...")
def load_model():
    return YOLO(MODEL_PATH, task="detect")


def label_of(model, cls_id):
    raw = str(model.names[int(cls_id)])
    return CLASS_LABELS.get(raw, raw)


class Servo:
    """Drives a real servo on a Raspberry Pi if gpiozero exists, otherwise simulates it."""
    def __init__(self):
        self.real = False
        try:
            from gpiozero import AngularServo
            self.dev = AngularServo(SERVO_PIN, min_angle=-45, max_angle=45)
            self.dev.angle = -45
            self.real = True
        except Exception:
            self.dev = None

    def flip(self):
        if self.real:
            self.dev.angle = 45
            time.sleep(SERVO_HOLD_SECONDS)
            self.dev.angle = -45


@st.cache_resource
def get_servo():
    return Servo()


LOG_COLS = ["timestamp", "source", "pet", "other", "confidence", "speed_ms", "est_grams"]


def load_log():
    if os.path.exists(LOG_PATH):
        try:
            df = pd.read_csv(LOG_PATH)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["other"] = pd.to_numeric(df["other"], errors="coerce")
            return df
        except Exception:
            pass
    return pd.DataFrame(columns=LOG_COLS)


def append_log(source, pet, other, confidence, speed_ms, grams_per_bottle):
    row = {"timestamp": datetime.now().isoformat(timespec="seconds"), "source": source, "pet": int(pet),
           "other": other if other is not None else "", "confidence": round(float(confidence), 3),
           "speed_ms": round(float(speed_ms)), "est_grams": int(pet) * int(grams_per_bottle)}
    pd.DataFrame([row]).to_csv(LOG_PATH, mode="a", index=False, header=not os.path.exists(LOG_PATH))
    return row


def stats(df):
    pet = int(df["pet"].sum()) if len(df) else 0
    sorted_ = int(df.loc[df["source"] == "camera", "pet"].sum()) if len(df) else 0
    known = df.dropna(subset=["other"]) if len(df) else df
    other = int(known["other"].sum()) if len(known) else 0
    denom = (int(known["pet"].sum()) + other) if len(known) else 0
    rate = (known["pet"].sum() / denom) if denom else None
    kg = (df["est_grams"].sum() / 1000) if len(df) else 0.0
    return dict(pet=pet, sorted=sorted_, other=other, rate=rate, kg=kg)


def pct(x):
    return f"{x:.0%}" if x is not None else "--"


# =====================================================================
# STATE
# =====================================================================
DEFAULTS = {"speeds": [], "recent": [], "present": False, "misses": 0, "cam_pet": 0, "scan": None, "warm": False}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v if not isinstance(v, list) else list(v))
ss = st.session_state
conf = ss.get("_conf", 0.5)
grams = ss.get("_grams", 20)
servo = get_servo()

# =====================================================================
# TOP BAR + NAVIGATION
# =====================================================================
st.markdown(
    f'<div class="topbar"><div class="brand"><div class="logo">{icon("leaf", 20)}</div>'
    '<div>EcoCycle Energy Hub<small>Smart PET Sorter</small></div></div><div class="chips">'
    '<span class="chip"><span class="dot"></span>AI engine: offline / local</span>'
    '<span class="chip plain">Prototype</span></div></div>', unsafe_allow_html=True)

PAGES = ["Home", "About", "Services", "How it works", "Scan", "Dashboard", "Contact"]
if ss.get("nav") not in PAGES:
    ss.nav = PAGES[0]


def goto(p):
    ss.nav = p


with st.container(key="appnav"):
    try:
        choice = st.pills("Menu", PAGES, selection_mode="single", key="nav", label_visibility="collapsed")
    except AttributeError:                       # older Streamlit without st.pills
        choice = st.radio("Menu", PAGES, key="nav", horizontal=True, label_visibility="collapsed")
page = choice or PAGES[0]

df_log = load_log()
S = stats(df_log)

# =====================================================================
# HOME
# =====================================================================
if page == "Home":
    st.markdown('<div class="hero"><div class="copy"><div class="hero-title">Turning PET waste into sustainable value</div>'
                '<p>Offline AI that finds PET bottles in mixed waste and separates them for recycling.</p></div>'
                + BOTTLE_SVG + '</div>', unsafe_allow_html=True)
    b1, b2, _ = st.columns([1, 1, 3])
    b1.button("Scan waste", type="primary", on_click=goto, args=("Scan",), key="cta1")
    b2.button("How it works", on_click=goto, args=("How it works",), key="cta2")

    st.markdown(kpi_grid([
        ("PET detected", S["pet"], "all scans and live sessions", "green"),
        ("Bottles sorted", S["sorted"], "by the flipper, live camera", "teal"),
        ("Recovery rate", pct(S["rate"]), "PET share of items in scanned photos", ""),
        ("Waste diverted", f"{S['kg']:.2f} kg", f"estimated, at {grams} g per bottle", "green")]), unsafe_allow_html=True)

    sec("Why Smart PET Sorter")
    st.markdown(card_grid([
        card("Works offline", "Detects PET on the device. No internet needed.", "offline"),
        card("Low-cost hardware", "Runs on a Raspberry Pi 4 or similar.", "cpu"),
        card("Built for communities", "Safer sorting, close to where waste is made.", "users")]), unsafe_allow_html=True)

    sec("The problem")
    st.markdown(kpi_grid([("Unsorted waste in Kigali", "~400 tonnes", "every day (Rwanda Green Taxonomy)", "warn")])
                + '<p class="note">Recyclables like PET are mixed with general waste, so valuable plastic ends up in landfill.</p>',
                unsafe_allow_html=True)

    sec("Recent activity")
    if df_log.empty:
        st.markdown('<div class="decision idle">No activity yet. Run a scan to see results here.</div>', unsafe_allow_html=True)
    else:
        rows = ""
        for _, r_ in df_log.sort_values("timestamp", ascending=False).head(5).iterrows():
            n_ = int(r_["pet"])
            photo = r_["source"] == "photo"
            msg = f'{n_} PET bottle{"s" if n_ != 1 else ""} {"found" if photo else "sorted"}'
            rows += (f'<div class="row"><div class="ic">{icon("camera" if photo else "cpu", 20)}</div>'
                     f'<div class="tx">{msg}<small>{"Photo scan" if photo else "Live camera"} | {float(r_["confidence"]):.0%} confidence | '
                     f'{int(r_["speed_ms"])} ms</small></div><div class="tm">{r_["timestamp"].strftime("%d %b, %H:%M")}</div></div>')
        st.markdown(f'<div class="feed">{rows}</div>', unsafe_allow_html=True)

# =====================================================================
# ABOUT
# =====================================================================
elif page == "About":
    head("About us")
    sec("Who we are")
    st.markdown(card_grid([card("EcoCycle Energy Hub",
        "A circular-economy venture that turns urban waste into valuable resources, using localised AI recovery technology "
        "for high-density communities.", "leaf")], wide=True), unsafe_allow_html=True)
    st.markdown(card_grid([
        card("Our mission", "Turn urban waste into sustainable livelihoods and clean resources through accessible technology.", "target"),
        card("Our vision", "A cleaner, more resilient Africa where waste is a resource, not a problem.", "eye")], wide=True),
        unsafe_allow_html=True)

    sec("Our core values")
    st.markdown(card_grid([
        card("Community first", "Technology that assists and protects waste collectors. It never replaces human judgment.", "users"),
        card("Circularity", "Waste is a resource. We recover it close to where it is made.", "recycle"),
        card("Accessible technology", "Low-cost, low-power and offline, so more communities can use it.", "globe"),
        card("Responsible data", "We collect operational waste data only. No personal data is stored.", "shield")]), unsafe_allow_html=True)

    sec("Sustainable Development Goals")
    st.markdown(
        '<div class="grid cards wide">'
        '<div class="card"><div class="sdg"><div class="badge" style="background:#FD9D24"><span>SDG</span><b>11</b></div>'
        '<div><div class="c-title">Sustainable Cities and Communities</div>'
        '<p>Cleaner, safer high-density neighbourhoods with less waste in streets, drains and dumpsites.</p></div></div></div>'
        '<div class="card"><div class="sdg"><div class="badge" style="background:#BF8B2E"><span>SDG</span><b>12</b></div>'
        '<div><div class="c-title">Responsible Consumption and Production</div>'
        '<p>PET is recovered and returned to the circular economy instead of landfill.</p></div></div></div></div>',
        unsafe_allow_html=True)

# =====================================================================
# SERVICES
# =====================================================================
elif page == "Services":
    head("Our services", "Tap a card for details.")
    st.markdown(card_grid([
        detail_card("Automated PET sorting", "Find and separate PET bottles at the source",
                    "A camera and on-device AI identify PET. A servo flipper diverts it into a recovery bin.", "recycle",
                    tag="V1 in development"),
        detail_card("Recovery data and reports", "Counts, weight and recovery rate",
                    "Every detection is logged and can be exported as a CSV report for partners.", "chart", tag="V1 in development"),
        detail_card("Offline AI deployment", "Works without internet",
                    "The model runs on low-cost hardware such as a Raspberry Pi 4. The design is solar-ready.", "offline",
                    tag="V1 in development"),
        detail_card("Multi-material sorting", "HDPE, aluminium and cardboard",
                    "The AI will be trained to recognise more recyclable materials.", "layers", tag="V2 planned", plan=True),
        detail_card("Circular energy", "Organic waste to biogas",
                    "Organic residues and community resources such as water treatment.", "bolt", tag="V3 vision", plan=True)],
        wide=True), unsafe_allow_html=True)

    sec("Who we serve")
    st.markdown(card_grid([
        card("Recycling facilities", "High-purity PET feedstock.", "factory"),
        card("Municipal authorities", "Less landfill waste and recovery data to report.", "city"),
        card("Landfill operators", "Fewer recyclables reaching the site.", "box"),
        card("Waste collectors", "Less contact with hazardous mixed waste.", "users"),
        card("Communities", "Cleaner neighbourhoods and green livelihoods.", "leaf")]), unsafe_allow_html=True)

# =====================================================================
# HOW IT WORKS
# =====================================================================
elif page == "How it works":
    head("How Smart PET Sorter works")
    st.markdown(flow([(1, "Waste enters", "Chute or small conveyor", ""),
                      (2, "Camera captures", "Overhead, high speed", ""),
                      (3, "AI detects", "On the device, offline", ""),
                      (4, "Flipper separates", "PET goes to its own bin", ""),
                      (5, "Data is logged", "Counts and weight", "")], on="all"), unsafe_allow_html=True)

    sec("After sorting")
    st.markdown(flow([("box", "Sorted PET", "High-purity feedstock", ""),
                      ("cpu", "Processing", "Planned", "planned"),
                      ("layers", "Recycled products", "Filament and construction pellets, planned", "planned")]), unsafe_allow_html=True)

    sec("System")
    st.markdown('<div class="grid cards">'
                '<div class="block"><b>Sensing</b>Camera module</div>'
                '<div class="block"><b>Processing</b>Raspberry Pi 4<br>YOLOv8-Nano (LiteRT)</div>'
                '<div class="block"><b>Logic</b>Python script<br>checks the PET class</div>'
                '<div class="block"><b>Actuation</b>Servo motor<br>diversion flap</div></div>'
                '<div class="block data"><b>Data layer</b>Item counting and recovery logging</div>', unsafe_allow_html=True)

# =====================================================================
# SCAN
# =====================================================================
elif page == "Scan":
    head("Scan", "Upload a photo or use the live camera.")
    if not os.path.exists(MODEL_PATH):
        st.error(f"AI model not found at `{MODEL_PATH}`. Copy `best.tflite` into the `models` folder and refresh.")
        st.stop()

    with st.expander("Scan settings"):
        conf = st.slider("Detection sensitivity", 0.2, 0.9, float(conf), 0.05,
                         help="Lower finds more bottles but may add false alarms.")
        grams = st.number_input("Estimated grams per bottle", 5, 100, int(grams))
        ss._conf, ss._grams = conf, grams
        st.caption("Servo: " + ("real (GPIO)" if servo.real else "simulated on this computer") +
                   "  |  AI model: YOLOv8-Nano (LiteRT), runs locally")
        if st.button("Reset live-camera counters"):
            for k_ in ("speeds", "recent"):
                ss[k_] = []
            ss.cam_pet, ss.present, ss.misses = 0, False, 0
            st.rerun()
    _msg = ss.pop("toast", None)
    if _msg:
        st.toast(_msg)

    tab_scan, tab_live = st.tabs(["Scan a photo", "Live camera"])

    # ---------------- Scan a photo ----------------
    with tab_scan:
        up = st.file_uploader("Upload a photo of waste", type=["jpg", "jpeg", "png"])
        with st.expander("Take a photo with your camera"):
            snap = st.camera_input("Camera", label_visibility="collapsed")
        other = st.number_input("Other (non-PET) items in the photo", 0, 50, 0,
                                help="Optional. The model finds PET bottles only, so enter other items yourself to get a recovery rate.")
        source = snap if snap is not None else up
        if st.button("Scan photo", type="primary", disabled=source is None):
            img = cv2.imdecode(np.frombuffer(source.getvalue(), np.uint8), cv2.IMREAD_COLOR)
            model = load_model()
            if not ss.warm:
                model.predict(img, conf=conf, verbose=False)
                ss.warm = True
            r = model.predict(img, conf=conf, verbose=False)[0]
            n = len(r.boxes)
            confs = [float(c) for c in r.boxes.conf] if n else []
            top = max(confs) if confs else 0.0
            append_log("photo", n, int(other), np.mean(confs) if confs else 0.0, r.speed["inference"], grams)
            ss.scan = dict(img=r.plot(), pet=n, other=int(other), top=top, ms=float(r.speed["inference"]),
                           label=label_of(model, r.boxes.cls[0]) if n else "None")
            ss.toast = "Scan saved to the dashboard"
            st.rerun()
        if source is None:
            st.markdown('<div class="decision idle">Add a photo, then press Scan photo.</div>', unsafe_allow_html=True)

        sc = ss.scan
        if sc:
            total = sc["pet"] + sc["other"]
            rate = (sc["pet"] / total) if total else None
            if sc["pet"] > 0:
                st.markdown(f'<div class="decision recycle">Decision: RECYCLE<small>{sc["pet"]} PET '
                            f'bottle{"s" if sc["pet"] != 1 else ""} found. Send to the PET recovery bin.</small></div>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<div class="decision none">No PET found<small>Try a clearer photo or lower the sensitivity.</small></div>',
                            unsafe_allow_html=True)
            a, b = st.columns([3, 2])
            with a:
                show_image(st, sc["img"])
            with b:
                st.markdown(kpi_grid([
                    ("Detected", sc["label"], "what the AI sees", "green" if sc["pet"] else ""),
                    ("Confidence", f"{sc['top']:.0%}" if sc["pet"] else "--", "highest detection", "teal"),
                    ("Classification", "PET" if sc["pet"] else "Non-PET", "material type", ""),
                    ("Speed", f"{sc['ms']:.0f} ms", f"target under {SPEED_TARGET_MS} ms",
                     "warn" if sc["ms"] > SPEED_TARGET_MS else "")]), unsafe_allow_html=True)
            st.markdown(kpi_grid([
                ("Objects counted", total, "PET plus the others you entered"),
                ("PET bottles", sc["pet"], "found by the AI", "green"),
                ("Non-PET items", sc["other"], "entered by you"),
                ("PET recovery", pct(rate), "PET share of all items", "teal")]), unsafe_allow_html=True)
            st.markdown('<p class="note">Saved to the dashboard. The model recognises PET bottles only.</p>', unsafe_allow_html=True)

    # ---------------- Live camera ----------------
    with tab_live:
        top_l, top_r = st.columns([3, 1])
        run = top_l.toggle("Start live camera", value=False)
        cam_index = top_r.number_input("Camera number", 0, 5, 0, help="Try 1 if the default camera does not open.")
        kpi_slot = st.empty()
        pipe_slot = st.empty()
        left, right = st.columns([2, 1])
        with left:
            frame_slot = st.empty()
            banner_slot = st.empty()
        with right:
            st.markdown("**Recent detections**")
            log_slot = st.empty()

        def avg_speed():
            r_ = ss.speeds[-30:]
            return sum(r_) / len(r_) if r_ else None

        def draw_kpis():
            a_ = avg_speed()
            kpi_slot.markdown(kpi_grid([
                ("PET detected", ss.cam_pet, "this live session", "green"),
                ("Flipper triggers", ss.cam_pet, "real servo" if servo.real else "simulated servo", "teal"),
                ("Average speed", f"{a_:.0f} ms" if a_ else "--", f"target under {SPEED_TARGET_MS} ms",
                 "warn" if a_ and a_ > SPEED_TARGET_MS else ""),
                ("Estimated weight", f"{ss.cam_pet * grams / 1000:.2f} kg", f"at {grams} g per bottle")]),
                unsafe_allow_html=True)

        def draw_recent():
            if ss.recent:
                log_slot.dataframe(pd.DataFrame(ss.recent[::-1]).head(10), hide_index=True)
            else:
                log_slot.info("Nothing detected yet.")

        draw_kpis()
        draw_recent()
        pipe_slot.markdown(flow(LIVE_STEPS), unsafe_allow_html=True)
        banner_slot.markdown('<div class="decision idle">Waiting for an item...</div>', unsafe_allow_html=True)

        if not run:
            frame_slot.info("Switch on Start live camera, then hold a plastic bottle in front of it.")
        else:
            model = load_model()
            cap = cv2.VideoCapture(int(cam_index), cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(int(cam_index))
            try:
                if not cap.isOpened():
                    frame_slot.error("The camera did not open. Close other apps that use it, or set the camera number to 1.")
                while run and cap.isOpened():
                    ok, frame = cap.read()
                    if not ok:
                        frame_slot.error("The camera stopped sending pictures.")
                        break
                    r = model.predict(frame, conf=conf, verbose=False)[0]
                    ms = float(r.speed["inference"])
                    ss.speeds.append(ms)
                    found = len(r.boxes) > 0
                    if found:
                        ss.misses = 0
                        if not ss.present:                       # a new bottle entered the frame
                            ss.present = True
                            ss.cam_pet += 1
                            best = max(r.boxes, key=lambda b_: float(b_.conf))
                            row = append_log("camera", 1, None, float(best.conf), ms, grams)
                            ss.recent.append({"time": row["timestamp"][11:], "item": label_of(model, best.cls),
                                              "confidence": f"{float(best.conf):.0%}", "ms": row["speed_ms"]})
                            servo.flip()
                    else:
                        ss.misses += 1
                        if ss.misses >= MISS_FRAMES_TO_RESET:
                            ss.present = False
                    show_image(frame_slot, r.plot())
                    pipe_slot.markdown(flow(LIVE_STEPS, on="all" if found else 0), unsafe_allow_html=True)
                    banner_slot.markdown(
                        '<div class="decision recycle">PET detected<small>Flipper triggered</small></div>' if found
                        else '<div class="decision idle">Waiting for an item...</div>', unsafe_allow_html=True)
                    draw_kpis()
                    draw_recent()
            finally:
                cap.release()

# =====================================================================
# DASHBOARD
# =====================================================================
elif page == "Dashboard":
    head("Dashboard", "Results from your scans and live sessions.")
    if df_log.empty:
        st.markdown('<div class="decision idle">No data yet. Run a scan and it will appear here.</div>', unsafe_allow_html=True)
        st.button("Go to Scan", type="primary", on_click=goto, args=("Scan",), key="dash_cta")
    else:
        st.markdown(kpi_grid([
            ("Total PET collected", S["pet"], "all scans and live sessions", "green"),
            ("Sorted by flipper", S["sorted"], "live camera", "teal"),
            ("Non-PET reported", S["other"], "entered on photo scans"),
            ("Recovery rate", pct(S["rate"]), "PET share on scanned photos", "teal"),
            ("Waste diverted", f"{S['kg']:.2f} kg", "estimated PET weight", "green")]), unsafe_allow_html=True)
        d = df_log.sort_values("timestamp").copy()
        c1, c2 = st.columns(2)
        with c1:
            sec("PET detected per day")
            daily = d.groupby(d["timestamp"].dt.date)["pet"].sum()
            fig = go.Figure(go.Bar(x=[x.strftime("%d %b") for x in daily.index], y=daily.values, marker_color=TEAL,
                                   text=daily.values, textposition="outside", textfont=dict(size=14, color=INK), width=0.45))
            show_chart(style_fig(fig, "PET bottles"))
        with c2:
            sec("Waste diverted over time")
            kg = (d["est_grams"].cumsum() / 1000).round(3)
            fig2 = go.Figure(go.Scatter(x=d["timestamp"].dt.strftime("%d %b %H:%M"), y=kg, mode="lines+markers",
                                        line=dict(color=GREEN, width=3), marker=dict(size=10, color=GREEN)))
            show_chart(style_fig(fig2, "Kilograms"))
        with st.expander("All records"):
            st.dataframe(d.sort_values("timestamp", ascending=False), hide_index=True)
        st.download_button("Download report (CSV)", df_log.to_csv(index=False), "ecocycle_report.csv", "text/csv")

# =====================================================================
# CONTACT
# =====================================================================
else:
    head("Contact us", "Questions, partnerships or a pilot? Send us a message.")
    left, right = st.columns([3, 2])
    with left:
        with st.form("contact_form", clear_on_submit=True):
            name = st.text_input("Full name")
            email = st.text_input("Email")
            org = st.text_input("Organisation (optional)")
            who = st.selectbox("I am a", ["Recycling facility", "Municipal authority", "Landfill operator",
                                          "Waste collector", "Partner or funder", "Other"])
            message = st.text_area("Message", height=130)
            sent = st.form_submit_button("Send message", type="primary")
        if sent:
            if not name.strip() or "@" not in email or not message.strip():
                st.error("Please enter your name, a valid email and a message.")
            else:
                pd.DataFrame([{"timestamp": datetime.now().isoformat(timespec="seconds"), "name": name.strip(),
                               "email": email.strip(), "organisation": org.strip(), "role": who,
                               "message": message.strip()}]).to_csv(
                    MESSAGES_PATH, mode="a", index=False, header=not os.path.exists(MESSAGES_PATH))
                st.success("Message recorded. Thank you.")
        st.markdown('<p class="note">In this prototype, messages are saved on this device in data/messages.csv.</p>',
                    unsafe_allow_html=True)
    with right:
        rows = ""
        for key, ico, label in [("person", "users", "Contact person"), ("email", "mail", "Email"),
                                ("phone", "phone", "Phone"), ("location", "pin", "Location")]:
            if CONTACT.get(key):
                rows += (f'<div class="contact-row"><div class="ic">{icon(ico, 20)}</div>'
                         f'<div><small>{label}</small>{CONTACT[key]}</div></div>')
        st.markdown(f'<div class="card"><div class="c-title">EcoCycle Energy Hub</div>{rows or "<p>Use the form to reach us.</p>"}</div>',
                    unsafe_allow_html=True)

st.markdown('<div class="appfoot">EcoCycle Energy Hub | Smart PET Sorter prototype | SDG 11 and SDG 12</div>', unsafe_allow_html=True)