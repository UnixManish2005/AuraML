"""Quiz Section — timed MCQ, instant feedback, leaderboard."""

import streamlit as st
import time
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from database.db import get_questions, save_attempt, get_attempts, get_leaderboard
from utils.styles import section_header, alert, kpi_card
import plotly.graph_objects as go
from datetime import datetime


def fmt_dt(value, fmt="%d %b %Y %H:%M"):
    """Safely format datetime — handles both Python datetime objects and strings."""
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.strftime(fmt)
    return str(value)[:16]


TOPICS = ["Statistics","Machine Learning", "Deep Learning", "NLP", "Python", "Gen AI"]
DIFFICULTY_ICON = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}


def show_quiz(user):
    uid = user["id"]
    section_header("📝 Quiz Arena", "Test your knowledge. Climb the leaderboard!")

    tab1, tab2, tab3 = st.tabs(["🎯 Take Quiz", "📊 My Scores", "🏆 Leaderboard"])

    # ── Tab 1: Take Quiz ───────────────────────────────────────────────────
    with tab1:
        if "quiz_active" not in st.session_state:
            st.session_state["quiz_active"] = False

        if not st.session_state["quiz_active"]:
            _quiz_lobby(uid)
        else:
            _quiz_session(uid)

    # ── Tab 2: My Scores ───────────────────────────────────────────────────
    with tab2:
        attempts = get_attempts(uid)
        if not attempts:
            alert("No quiz attempts yet. Take your first quiz!", "info")
            return

        # Stat cards
        best  = max(a["score"]*100//a["total"] for a in attempts)
        avg   = sum(a["score"]*100//a["total"] for a in attempts) // len(attempts)
        total = len(attempts)
        c1, c2, c3 = st.columns(3)
        with c1: kpi_card("Total Attempts", total, "📝")
        with c2: kpi_card("Best Score",  f"{best}%",  "🥇")
        with c3: kpi_card("Avg Score",   f"{avg}%",   "📊")

        st.markdown("#### 📋 Attempt History")
        for a in attempts[:15]:
            pct = a["score"] * 100 // a["total"]
            color = "#43D9AD" if pct >= 70 else "#FFB547" if pct >= 50 else "#FF6584"
            grade = "Excellent!" if pct >= 80 else "Good" if pct >= 60 else "Keep Practicing"
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:1rem;
                        background:#1A1A2E;border-radius:10px;padding:0.8rem 1rem;
                        margin-bottom:0.5rem;border:1px solid rgba(108,99,255,0.15)">
                <div style="flex:1">
                    <span style="font-weight:700;color:#E8E8F0">{a['topic']}</span>
                    <span style="color:#8888AA;font-size:0.78rem;margin-left:0.5rem">{fmt_dt(a['attempted_at'])}</span>
                </div>
                <div style="text-align:right">
                    <span style="color:{color};font-size:1.2rem;font-weight:800">{pct}%</span>
                    <span style="color:#8888AA;font-size:0.78rem;margin-left:0.5rem">{a['score']}/{a['total']} · {grade}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Tab 3: Leaderboard ─────────────────────────────────────────────────
    with tab3:
        rows = get_leaderboard()
        if not rows:
            alert("Leaderboard is empty. Be the first!", "info")
            return
        st.markdown("#### 🏆 Global Leaderboard")
        medals = ["🥇","🥈","🥉"] + ["🎖️"]*20
        for i, r in enumerate(rows):
            avg_pct = round(r.get("avg_pct", 0))
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:1rem;
                        background:{'linear-gradient(135deg,#1A1A2E,#16213E)' if i<3 else '#1A1A2E'};
                        border:1px solid {'rgba(108,99,255,0.4)' if i<3 else 'rgba(108,99,255,0.15)'};
                        border-radius:12px;padding:0.8rem 1.2rem;margin-bottom:0.5rem">
                <span style="font-size:1.5rem">{medals[i]}</span>
                <span style="flex:1;font-weight:700;color:#E8E8F0">{r['name']}</span>
                <span style="color:#6C63FF;font-weight:800;font-size:1.1rem">{avg_pct}%</span>
                <span style="color:#8888AA;font-size:0.8rem">{r['attempts']} attempts</span>
            </div>
            """, unsafe_allow_html=True)


def _quiz_lobby(uid):
    c1, c2 = st.columns(2)
    with c1:
        topic = st.selectbox("📚 Select Topic", TOPICS, key="quiz_topic_sel")

        # Count available questions for the selected topic
        available = get_questions(topic)
        n_available = len(available)

        # Let admin/student choose how many to attempt (up to all available)
        max_q = max(n_available, 1)
        # --- Replace the old st.slider section with this logic ---

    if max_q < 5:
        # If there are fewer than 5 questions, fall back gracefully
        st.info(f"No Questions Available for this Module")
        num_q = max_q
    else:
        # Only render the slider if max_value is genuinely greater than or equal to min_value
        num_q = st.slider(
            "Number of questions", 
            min_value=5,
            max_value=max_q, 
            value=min(max_q, 20),
            step=5, 
            key="quiz_num_q"
        )

        st.markdown(f"""
        <div style="background:#1A1A2E;border-radius:12px;padding:1rem;
                    border:1px solid rgba(108,99,255,0.2);margin-top:0.5rem">
            <p style="color:#8888AA;margin:0;font-size:0.88rem">
            ⏱ 30 seconds per question &nbsp;|&nbsp;
            <b style="color:#6C63FF">{n_available} questions available</b>
            &nbsp;|&nbsp; Auto-evaluated
            </p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 Start Quiz!", use_container_width=True):
            questions = get_questions(topic, limit=num_q)
            if not questions:
                alert(f"No questions available for **{topic}** yet.", "warning")
            else:
                st.session_state.update({
                    "quiz_active": True,
                    "quiz_questions": questions,
                    "quiz_idx": 0,
                    "quiz_answers": {},
                    "quiz_start": time.time(),
                    "quiz_topic": topic,
                    "quiz_done": False,
                })
                st.rerun()


def _quiz_session(uid):
    qs     = st.session_state["quiz_questions"]
    idx    = st.session_state["quiz_idx"]
    done   = st.session_state.get("quiz_done", False)

    if done:
        _quiz_results(uid)
        return

    # Progress bar
    progress = idx / len(qs)
    st.progress(progress)
    st.markdown(f"<p style='color:#8888AA;font-size:0.85rem'>Question {idx+1} of {len(qs)}</p>",
                unsafe_allow_html=True)

    q = qs[idx]
    diff_icon = DIFFICULTY_ICON.get(q.get("difficulty","medium"), "🟡")

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#1A1A2E,#16213E);
                border:1px solid rgba(108,99,255,0.3);border-radius:16px;
                padding:1.5rem 2rem;margin:1rem 0 1.5rem">
        <div style="display:flex;justify-content:space-between;margin-bottom:0.8rem">
            <span style="color:#8888AA;font-size:0.82rem">
                {diff_icon} {q.get('difficulty','medium').upper()} &nbsp;·&nbsp; {q.get('topic','')}
            </span>
        </div>
        <h3 style="color:#E8E8F0;font-family:Syne,sans-serif;font-weight:700;
                   margin:0;font-size:1.15rem">{q['question']}</h3>
    </div>
    """, unsafe_allow_html=True)

    opts = {}
    for k, label in [("A","option_a"),("B","option_b"),("C","option_c"),("D","option_d")]:
        if q.get(label):
            opts[k] = q[label]

    choice = st.radio("Choose your answer:",
                      [f"{k}. {v}" for k, v in opts.items()],
                      key=f"q_{idx}", index=None)

    col_next, col_skip, col_quit = st.columns([2, 1, 1])
    with col_next:
        if st.button("Next →", use_container_width=True):
            if choice:
                chosen_key = choice.split(".")[0].strip()
                st.session_state["quiz_answers"][idx] = chosen_key
            _advance_quiz()
    with col_skip:
        if st.button("Skip", use_container_width=True):
            _advance_quiz()
    with col_quit:
        if st.button("Quit ✕", use_container_width=True):
            _finish_quiz(uid)


def _advance_quiz():
    idx = st.session_state["quiz_idx"]
    total = len(st.session_state["quiz_questions"])
    if idx + 1 >= total:
        st.session_state["quiz_done"] = True
    else:
        st.session_state["quiz_idx"] = idx + 1
    st.rerun()


def _finish_quiz(uid):
    qs      = st.session_state["quiz_questions"]
    answers = st.session_state["quiz_answers"]
    topic   = st.session_state["quiz_topic"]
    elapsed = int(time.time() - st.session_state["quiz_start"])

    score = sum(1 for i, q in enumerate(qs)
                if answers.get(i) == q["answer"])
    save_attempt(uid, topic, score, len(qs), elapsed)
    st.session_state["quiz_done"] = True
    st.rerun()


def _quiz_results(uid):
    qs      = st.session_state["quiz_questions"]
    answers = st.session_state["quiz_answers"]
    topic   = st.session_state["quiz_topic"]
    elapsed = int(time.time() - st.session_state.get("quiz_start", time.time()))

    score = sum(1 for i, q in enumerate(qs)
                if answers.get(i) == q["answer"])
    pct = score * 100 // len(qs)

    # Save if not already saved
    if not st.session_state.get("quiz_saved"):
        save_attempt(uid, topic, score, len(qs), elapsed)
        st.session_state["quiz_saved"] = True

    color = "#43D9AD" if pct >= 70 else "#FFB547" if pct >= 50 else "#FF6584"
    msg   = "🎉 Excellent!" if pct >= 80 else "👍 Good Job!" if pct >= 60 else "💪 Keep Practicing!"

    st.markdown(f"""
    <div style="text-align:center;padding:2rem;
                background:linear-gradient(135deg,#1A1A2E,#16213E);
                border:1px solid rgba(108,99,255,0.3);border-radius:20px;margin-bottom:1.5rem">
        <div style="font-size:3.5rem;font-family:Syne,sans-serif;font-weight:800;color:{color}">{pct}%</div>
        <div style="font-size:1.3rem;color:#E8E8F0;font-weight:600">{msg}</div>
        <div style="color:#8888AA;margin-top:0.5rem">{score}/{len(qs)} correct &nbsp;·&nbsp; {elapsed}s</div>
    </div>
    """, unsafe_allow_html=True)

    # Answer review
    st.markdown("#### 📋 Answer Review")
    for i, q in enumerate(qs):
        user_ans = answers.get(i, "—")
        correct  = q["answer"]
        is_right = user_ans == correct
        icon     = "✅" if is_right else "❌"
        bg_color = "rgba(67,217,173,0.08)" if is_right else "rgba(255,101,132,0.08)"

        opts = {k: q.get(f"option_{k.lower()}", "") for k in ["A","B","C","D"]}
        answer_text = opts.get(correct, correct)
        user_text   = opts.get(user_ans, user_ans)

        st.markdown(f"""
        <div style="background:{bg_color};border-radius:10px;
                    padding:0.8rem 1rem;margin-bottom:0.6rem;
                    border:1px solid {'rgba(67,217,173,0.2)' if is_right else 'rgba(255,101,132,0.2)'}">
            <p style="color:#E8E8F0;font-weight:600;margin:0 0 0.3rem">{icon} Q{i+1}. {q['question']}</p>
            {'<p style="color:#43D9AD;margin:0;font-size:0.85rem">Your answer: ' + user_text + '</p>' if is_right else
             '<p style="color:#FF6584;margin:0;font-size:0.85rem">Your answer: ' + user_text + '</p><p style="color:#43D9AD;margin:0;font-size:0.85rem">Correct: ' + answer_text + '</p>'}
        </div>
        """, unsafe_allow_html=True)

    if st.button("🔄 Retake Quiz", use_container_width=True):
        for k in ["quiz_active","quiz_questions","quiz_idx","quiz_answers",
                  "quiz_start","quiz_topic","quiz_done","quiz_saved"]:
            st.session_state.pop(k, None)
        st.rerun()