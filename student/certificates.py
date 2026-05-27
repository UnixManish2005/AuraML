"""Certificate System — issue and display certificates."""

import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from database.db import get_certificates, issue_certificate, get_progress, get_attempts
from utils.styles import section_header, alert

CERTIFIABLE_COURSES = {
    "AI Introduction":   ("Complete AI Introduction module at 100%", "progress"),
    "Machine Learning":  ("Complete Machine Learning module at 100%", "progress"),
    "Deep Learning":     ("Complete Deep Learning module at 100%", "progress"),
    "NLP":               ("Complete NLP module at 100%", "progress"),
    "Quiz Champion — ML":("Score ≥ 80% in Machine Learning quiz", "quiz"),
    "Quiz Champion — DL":("Score ≥ 80% in Deep Learning quiz", "quiz"),
}


def show_certificates(user):
    uid = user["id"]
    section_header("🎓 My Certificates", "Earn certificates by completing modules and quizzes.")

    # Check eligibility
    progress_rows = {p["module"]: p["progress_pct"] for p in get_progress(uid)}
    attempts      = get_attempts(uid)

    quiz_best = {}
    for a in attempts:
        pct = a["score"] * 100 // a["total"]
        topic = a["topic"]
        quiz_best[topic] = max(quiz_best.get(topic, 0), pct)

    st.markdown("#### 🏅 Certificates You Can Earn")
    cols = st.columns(2)
    for i, (course, (requirement, kind)) in enumerate(CERTIFIABLE_COURSES.items()):
        with cols[i % 2]:
            # Check eligibility
            eligible = False
            if kind == "progress":
                eligible = progress_rows.get(course, 0) >= 100
            elif kind == "quiz":
                topic = course.replace("Quiz Champion — ", "")
                eligible = quiz_best.get(topic, 0) >= 80

            status_color = "#43D9AD" if eligible else "#8888AA"
            status_text  = "✅ Eligible" if eligible else "🔒 Not yet"

            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#1A1A2E,#16213E);
                        border:1px solid {'rgba(67,217,173,0.3)' if eligible else 'rgba(108,99,255,0.15)'};
                        border-radius:14px;padding:1rem 1.2rem;margin-bottom:0.8rem">
                <h4 style="color:#E8E8F0;font-family:Syne,sans-serif;font-weight:700;margin:0">{course}</h4>
                <p style="color:#8888AA;font-size:0.82rem;margin:0.3rem 0">{requirement}</p>
                <span style="color:{status_color};font-weight:600;font-size:0.85rem">{status_text}</span>
            </div>
            """, unsafe_allow_html=True)
            if eligible:
                if st.button(f"🎓 Claim Certificate", key=f"cert_{i}"):
                    cert_id = issue_certificate(uid, course)
                    st.success(f"🎉 Certificate issued! ID: **{cert_id}**")
                    st.rerun()

    # Display existing certificates
    certs = get_certificates(uid)
    if certs:
        st.markdown("---")
        st.markdown("#### 🎖️ Your Certificates")
        for cert in certs:
            _render_certificate(user["name"], cert["course"], cert["cert_id"], cert["issued_at"])
    else:
        alert("No certificates yet. Complete modules and quizzes to earn them!", "info")


def _render_certificate(name, course, cert_id, issued_at):
    date = issued_at[:10] if issued_at else "2025"
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #1A1A2E 0%, #16213E 50%, #0F3460 100%);
        border: 2px solid #6C63FF;
        border-radius: 20px;
        padding: 2.5rem 3rem;
        margin: 1rem 0;
        text-align: center;
        position: relative;
        overflow: hidden;
    ">
        <div style="position:absolute;top:20px;left:30px;font-size:3rem;opacity:0.06">🌟</div>
        <div style="position:absolute;bottom:20px;right:30px;font-size:3rem;opacity:0.06">🏆</div>

        <div style="font-family:Syne,sans-serif;font-size:0.9rem;color:#8888AA;
                    letter-spacing:3px;text-transform:uppercase;margin-bottom:0.5rem">
            Certificate of Completion
        </div>

        <div style="font-family:Syne,sans-serif;font-size:2rem;font-weight:800;
                    background:linear-gradient(135deg,#6C63FF,#FF6584);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                    margin:0.5rem 0">
            AILearn Pro
        </div>

        <p style="color:#8888AA;font-size:0.88rem;margin:0">This is to certify that</p>

        <div style="font-family:Syne,sans-serif;font-size:1.6rem;font-weight:700;
                    color:#E8E8F0;margin:0.5rem 0 0.3rem">
            {name}
        </div>

        <p style="color:#8888AA;font-size:0.88rem;margin:0">has successfully completed</p>

        <div style="font-family:Syne,sans-serif;font-size:1.2rem;font-weight:700;
                    color:#43D9AD;margin:0.5rem 0 1rem">
            {course}
        </div>

        <div style="display:flex;justify-content:center;gap:3rem;margin-top:1rem">
            <div style="text-align:center">
                <div style="color:#8888AA;font-size:0.75rem;letter-spacing:1px">DATE ISSUED</div>
                <div style="color:#E8E8F0;font-weight:600">{date}</div>
            </div>
            <div style="text-align:center">
                <div style="color:#8888AA;font-size:0.75rem;letter-spacing:1px">CERTIFICATE ID</div>
                <div style="color:#6C63FF;font-weight:600;font-family:monospace">{cert_id}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
