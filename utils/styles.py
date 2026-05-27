"""
utils/styles.py  (extended — supports both dashboard helpers and uploaded page modules)
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# ── Brand palette ──────────────────────────────────────────────────────────────
PRIMARY   = "#6C63FF"
SECONDARY = "#FF6584"
SUCCESS   = "#43D9AD"
WARNING   = "#FFB547"
DARK      = "#0F0F1A"
CARD_BG   = "#1A1A2E"
CARD2     = "#16213E"
TEXT      = "#E8E8F0"
MUTED     = "#8888AA"

# Used by uploaded page modules directly
COLOR_SEQ = [
    "#6C63FF", "#43E97B", "#FF6584", "#38F9D7",
    "#FFA94D", "#B48EFF", "#FF8EE8", "#06B6D4",
]

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#E8E8F0", family="Space Grotesk"),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    margin=dict(l=10, r=10, t=40, b=10),
    xaxis=dict(showgrid=False, zeroline=False, color="#8888AA"),
    yaxis=dict(showgrid=False, zeroline=False, color="#8888AA"),
)


# ── CSS ────────────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Syne:wght@400;600;700;800&display=swap');

    :root {
        --primary:#6C63FF;--secondary:#FF6584;--success:#43D9AD;
        --warning:#FFB547;--dark:#0F0F1A;--card:#1A1A2E;--card2:#16213E;
        --text:#E8E8F0;--muted:#8888AA;--font-mono:'Fira Code',monospace;
    }
    html,body,[class*="css"]{font-family:'Space Grotesk',sans-serif;background:var(--dark)!important;color:var(--text)!important;}
    [data-testid="stSidebar"]{background:linear-gradient(180deg,#0F0F1A 0%,#1A1A2E 100%)!important;border-right:1px solid rgba(108,99,255,0.2);}
    [data-testid="stSidebar"] *{color:var(--text)!important;}
    [data-testid="stSidebar"] .stButton button{background:linear-gradient(135deg,var(--primary),#8B5CF6)!important;border:none!important;color:white!important;width:100%;border-radius:10px;font-weight:600;margin-top:4px;}
    .main .block-container{padding:2rem 2.5rem!important;}
    .stApp{background:var(--dark)!important;}

    /* welcome card */
    .welcome-card{background:linear-gradient(135deg,#1A1A2E 0%,#16213E 100%);border:1px solid rgba(108,99,255,0.25);border-radius:18px;padding:1.8rem 2rem;margin-bottom:1.5rem;box-shadow:0 8px 32px rgba(108,99,255,0.05);}
    .welcome-title{font-family:'Syne',sans-serif;font-size:1.8rem;font-weight:800;color:#E8E8F0;margin:0 0 .3rem;}
    .welcome-name{background:linear-gradient(135deg,#6C63FF,#FF6584);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
    .welcome-text{color:#8888AA;font-size:.95rem;margin:0;line-height:1.5;}

    /* ml-card — used by uploaded modules */
    .ml-card{background:linear-gradient(135deg,#1A1A2E,#16213E);border:1px solid rgba(108,99,255,0.2);border-radius:14px;padding:1.2rem 1.4rem;margin-bottom:0.8rem;transition:transform .2s,box-shadow .2s;}
    .ml-card:hover{transform:translateY(-3px);box-shadow:0 6px 24px rgba(108,99,255,0.2);}

    /* metric cards */
    .metric-card{background:linear-gradient(135deg,#1A1A2E,#16213E);border:1px solid rgba(108,99,255,0.25);border-radius:16px;padding:1.3rem 1.5rem;text-align:center;transition:transform .2s,box-shadow .2s;}
    .metric-card:hover{transform:translateY(-4px);box-shadow:0 8px 32px rgba(108,99,255,0.25);}
    .metric-value{font-family:'Syne',sans-serif;font-size:2.2rem;font-weight:800;background:linear-gradient(135deg,#6C63FF,#FF6584);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
    .metric-label{font-size:0.82rem;color:#8888AA;text-transform:uppercase;letter-spacing:1.5px;margin-top:.3rem;}

    /* progress bar */
    .progress-container{background:rgba(255,255,255,0.08);border-radius:20px;height:10px;overflow:hidden;margin:6px 0;}
    .progress-bar{height:100%;border-radius:20px;transition:width .4s;}

    /* boxes */
    .info-box{background:rgba(108,99,255,0.1);border-left:3px solid #6C63FF;border-radius:10px;padding:.9rem 1.2rem;margin:.8rem 0;font-size:.88rem;}
    .warning-box{background:rgba(255,181,71,0.1);border-left:3px solid #FFB547;border-radius:10px;padding:.9rem 1.2rem;margin:.8rem 0;font-size:.88rem;}
    .success-box{background:rgba(67,217,173,0.1);border-left:3px solid #43D9AD;border-radius:10px;padding:.9rem 1.2rem;margin:.8rem 0;font-size:.88rem;}
    .concept-box{background:linear-gradient(135deg,rgba(108,99,255,.12),rgba(139,92,246,.08));border:1px solid rgba(108,99,255,0.3);border-radius:14px;padding:1.2rem 1.5rem;margin:1rem 0;font-size:.9rem;}
    .concept-box h4{color:#6C63FF;font-family:'Syne',sans-serif;font-weight:700;margin:0 0 .6rem;}

    /* hero */
    .hero-block{background:linear-gradient(135deg,#1A1A2E 0%,#16213E 50%,#0F3460 100%);border:1px solid rgba(108,99,255,0.3);border-radius:20px;padding:2rem 2.5rem;margin-bottom:1.5rem;}
    .hero-block h1{font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;background:linear-gradient(135deg,#6C63FF,#FF6584);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0 0 .4rem;}
    .hero-block p{color:#8888AA;font-size:.95rem;margin:0;}

    /* section title */
    .section-title{font-family:'Syne',sans-serif;font-size:1.7rem;font-weight:800;background:linear-gradient(135deg,#6C63FF,#FF6584);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:.2rem;}
    .section-sub{color:#8888AA;font-size:.9rem;margin-bottom:1.5rem;}

    /* module card */
    .module-card{background:linear-gradient(135deg,#1A1A2E,#16213E);border:1px solid rgba(108,99,255,0.2);border-radius:16px;padding:1.4rem;margin-bottom:1rem;transition:all .2s;}
    .module-card:hover{border-color:#6C63FF;box-shadow:0 4px 24px rgba(108,99,255,0.2);}
    .module-card h4{font-family:'Syne',sans-serif;font-size:1.05rem;font-weight:700;color:#E8E8F0;margin:.5rem 0 .3rem;}
    .module-card p{color:#8888AA;font-size:.85rem;margin:0;}

    /* badge */
    .badge{display:inline-block;padding:3px 10px;border-radius:20px;font-size:.72rem;font-weight:600;text-transform:uppercase;letter-spacing:.5px;}
    .badge-primary{background:rgba(108,99,255,0.2);color:#6C63FF;}
    .badge-success{background:rgba(67,217,173,0.2);color:#43D9AD;}
    .badge-warning{background:rgba(255,181,71,0.2);color:#FFB547;}
    .badge-danger{background:rgba(255,101,132,0.2);color:#FF6584;}

    /* streamlit overrides */
    .stButton button{background:linear-gradient(135deg,#6C63FF,#8B5CF6)!important;border:none!important;color:white!important;border-radius:10px!important;font-weight:600!important;font-family:'Space Grotesk',sans-serif!important;padding:.5rem 1.4rem!important;transition:all .2s!important;}
    .stButton button:hover{opacity:.88!important;transform:translateY(-1px)!important;}
    .stTextInput input,.stSelectbox select,.stTextArea textarea{background:#16213E!important;border:1px solid rgba(108,99,255,0.3)!important;border-radius:10px!important;color:#E8E8F0!important;font-family:'Space Grotesk',sans-serif!important;}
    .stProgress .st-bo{background:#6C63FF!important;}
    .stTabs [data-baseweb="tab"]{color:#8888AA!important;font-weight:600;}
    .stTabs [aria-selected="true"]{color:#6C63FF!important;border-bottom-color:#6C63FF!important;}
    .stExpander{background:#1A1A2E!important;border:1px solid rgba(108,99,255,0.2)!important;border-radius:12px!important;}
    div[data-testid="stMetricValue"]{color:#6C63FF!important;font-family:'Syne',sans-serif!important;}
    .logo-text{font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:800;background:linear-gradient(135deg,#6C63FF,#FF6584);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
    </style>
    """, unsafe_allow_html=True)


# ── HTML component factories (for uploaded page modules) ───────────────────────

def welcome_message(name: str, custom_text: str = "Ready to continue your AI/ML acceleration journey? Pick a training track below.") -> str:
    return f"""
    <div class="welcome-card">
        <div class="welcome-title">Hey , <span class="welcome-name">{name}</span> ! 🙈</div>
        <p class="welcome-text">{custom_text}</p>
    </div>"""


def hero(title: str, subtitle: str, icon: str = "🤖") -> str:
    return f"""
    <div class="hero-block">
        <h1>{icon} {title}</h1>
        <p>{subtitle}</p>
    </div>"""


def concept_box(title: str, body: str, icon: str = "💡") -> str:
    return f"""
    <div class="concept-box">
        <h4>{icon} {title}</h4>
        <p style="color:#CCCCEE;margin:0;line-height:1.7;">{body}</p>
    </div>"""


def section_header(icon: str, title: str, subtitle: str = "") -> str:
    sub = f'<p style="color:#8888AA;font-size:.85rem;margin:2px 0 0 0;">{subtitle}</p>' if subtitle else ""
    return f"""
    <div style="margin:1.2rem 0 .8rem;">
        <span style="font-family:Syne,sans-serif;font-size:1.3rem;font-weight:800;
                     background:linear-gradient(135deg,#6C63FF,#FF6584);
                     -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            {icon} {title}
        </span>{sub}
    </div>"""


def info_box(text: str) -> str:
    return f'<div class="info-box">{text}</div>'


def warning_box(text: str) -> str:
    return f'<div class="warning-box">{text}</div>'


def success_box(text: str) -> str:
    return f'<div class="success-box">{text}</div>'


# ── Streamlit component helpers (used by dashboard/admin) ─────────────────────

def kpi_card(label: str, value, icon: str = "📊", color: str = PRIMARY):
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size:1.8rem">{icon}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>""", unsafe_allow_html=True)


def section_header_st(title: str, subtitle: str = ""):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-sub">{subtitle}</div>', unsafe_allow_html=True)

# Alias kept for backward compat with student/dashboard.py
def _section_header_compat(title, subtitle=""):
    section_header_st(title, subtitle)


def badge(text: str, kind: str = "primary") -> str:
    return f'<span class="badge badge-{kind}">{text}</span>'


def alert(text: str, kind: str = "info"):
    mapping = {"info": "info-box", "warning": "warning-box", "success": "success-box"}
    st.markdown(f'<div class="{mapping.get(kind, "info-box")}">{text}</div>', unsafe_allow_html=True)


def progress_chart(data: dict, title: str = "Module Progress"):
    df = pd.DataFrame(list(data.items()), columns=["Module", "Progress"])
    fig = px.bar(df, x="Progress", y="Module", orientation="h",
                 color="Progress", color_continuous_scale=["#1A1A2E", PRIMARY, SUCCESS],
                 range_x=[0, 100])
                 
    custom_layout = {**PLOTLY_LAYOUT}
    custom_layout["margin"] = dict(l=0, r=10, t=40, b=10)
    
    fig.update_layout(
        **custom_layout,
        title=dict(text=title, font=dict(family="Syne", size=16, color=TEXT)),
        coloraxis_showscale=False, height=300
    )
    fig.update_traces(marker_line_width=0)
    return fig


def score_trend_chart(attempts):
    if not attempts:
        return None
    df = pd.DataFrame(attempts)
    df["pct"] = (df["score"] / df["total"] * 100).round(1)
    df["date"] = pd.to_datetime(df["attempted_at"]).dt.strftime("%d %b")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["pct"], mode="lines+markers",
        line=dict(color=PRIMARY, width=3), marker=dict(size=8, color=SECONDARY),
        fill="tozeroy", fillcolor="rgba(108,99,255,0.1)"
    ))
    
    custom_layout = {**PLOTLY_LAYOUT}
    custom_layout["margin"] = dict(l=0, r=0, t=10, b=10)
    
    fig.update_layout(
        **custom_layout,
        height=220
    )

    fig.update_yaxes(
        range=[0, 105],
        showgrid=False,
        zeroline=False,
        color=MUTED
    )

    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        color=MUTED
    )
    return fig