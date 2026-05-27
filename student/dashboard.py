"""Student Dashboard — overview, progress, announcements."""

import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from database.db import get_progress, get_attempts, get_certificates, get_announcements, platform_stats
from utils.styles import kpi_card, section_header, alert, progress_chart, score_trend_chart, badge
import plotly.express as px
import pandas as pd


MODULES = [
    "AI Introduction", "Machine Learning", "Deep Learning",
    "NLP", "Computer Vision", "Generative AI", "Python"
]


def show_dashboard(user):
    uid = user["id"]
    from utils.styles import welcome_message
    st.markdown(
    welcome_message(
        user['name'].split()[0],
        "Here's your learning snapshot for today."
    ),
    unsafe_allow_html=True
)

    # ── KPI row ────────────────────────────────────────────────────────────
    progress_rows = get_progress(uid)
    attempts      = get_attempts(uid)
    certs         = get_certificates(uid)

    completed_mods = sum(1 for p in progress_rows if p["progress_pct"] >= 100)
    avg_score      = (sum(a["score"]*100//a["total"] for a in attempts) // len(attempts)
                      if attempts else 0)

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("Modules Completed", f"{completed_mods}/{len(MODULES)}", "📚")
    with c2: kpi_card("Quiz Attempts",    len(attempts),  "📝")
    with c3: kpi_card("Avg Quiz Score",   f"{avg_score}%", "🏆")
    with c4: kpi_card("Certificates",     len(certs),     "🎓")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Two-column: progress + score trend ────────────────────────────────
    left, right = st.columns([1.2, 1])

    with left:
        st.markdown('<p style="font-family:Syne,sans-serif;font-weight:700;font-size:1.1rem;color:#E8E8F0">📊 Module Progress</p>', unsafe_allow_html=True)
        prog_dict = {p["module"]: p["progress_pct"] for p in progress_rows}
        # fill in 0 for unvisited modules
        for m in MODULES:
            prog_dict.setdefault(m, 0)
        fig = progress_chart(prog_dict)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown('<p style="font-family:Syne,sans-serif;font-weight:700;font-size:1.1rem;color:#E8E8F0">📈 Quiz Score Trend</p>', unsafe_allow_html=True)
        trend = score_trend_chart(attempts[-10:] if attempts else [])
        if trend:
            st.plotly_chart(trend, use_container_width=True)
        else:
            alert("No quiz attempts yet. Head to the Quiz section to start! 🎯", "info")

    # ── Announcements ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p style="font-family:Syne,sans-serif;font-weight:700;font-size:1.1rem;color:#E8E8F0">📢 Announcements</p>', unsafe_allow_html=True)
    announcements = get_announcements()
    if announcements:
        for ann in announcements[:3]:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#1A1A2E,#16213E);
                        border:1px solid rgba(108,99,255,0.2);border-radius:12px;
                        padding:1rem 1.2rem;margin-bottom:0.7rem;">
                <span style="font-weight:700;color:#E8E8F0">{ann['title']}</span>
                <span style="float:right;color:#8888AA;font-size:0.8rem">{ann['created_at'][:10]}</span>
                <p style="color:#AAAACC;font-size:0.88rem;margin-top:0.4rem;margin-bottom:0">{ann['body']}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        alert("No announcements yet. Check back later!", "info")

    # ── Recent quiz scores ─────────────────────────────────────────────────
    if attempts:
        st.markdown("---")
        st.markdown('<p style="font-family:Syne,sans-serif;font-weight:700;font-size:1.1rem;color:#E8E8F0">🕐 Recent Quiz Activity</p>', unsafe_allow_html=True)
        recent = attempts[:5]
        for a in recent:
            pct = a["score"] * 100 // a["total"]
            color = "#43D9AD" if pct >= 70 else "#FFB547" if pct >= 50 else "#FF6584"
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:1rem;
                        background:#1A1A2E;border-radius:10px;padding:0.7rem 1rem;
                        margin-bottom:0.5rem;border:1px solid rgba(108,99,255,0.15)">
                <span style="font-size:1.2rem">📝</span>
                <span style="flex:1;color:#E8E8F0;font-weight:600">{a['topic']}</span>
                <span style="color:{color};font-weight:700">{pct}%</span>
                <span style="color:#8888AA;font-size:0.8rem">{a['attempted_at'][:10]}</span>
            </div>
            """, unsafe_allow_html=True)
