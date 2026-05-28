"""
Admin Panel — Dashboard, Student Mgmt, Quiz Mgmt, Announcements.
Compatible with the @contextmanager get_connection() pattern (PostgreSQL/Supabase).
Never calls conn.close() — the context manager handles that automatically.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import datetime
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from database.db import (platform_stats, get_all_students, toggle_student,
                          get_leaderboard, add_question, add_announcement,
                          get_announcements, get_connection)
from utils.styles import section_header, kpi_card, alert


def fmt_dt(value, fmt="%d %b %Y %H:%M"):
    """Safely format datetime — handles both Python datetime objects and strings."""
    if value is None:
        return ""
    if isinstance(value, datetime.datetime):
        return value.strftime(fmt)
    return str(value)[:16]


# ── Safe query helper ─────────────────────────────────────────────────────────
# Uses "with get_connection() as conn:" so it works with @contextmanager db.py

def _query_df(sql: str, params=None) -> pd.DataFrame:
    """Run a SELECT and return a DataFrame. Works with psycopg2 RealDictCursor."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params or ())
                rows = cur.fetchall()
                if not rows:
                    return pd.DataFrame()
                # RealDictCursor rows → list of plain dicts → DataFrame
                return pd.DataFrame([dict(r) for r in rows])
    except Exception as e:
        st.error(f"Query error: {e}")
        return pd.DataFrame()


def delete_question(question_id: int):
    """Delete a quiz question by ID."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM quiz_questions WHERE id = %s;", (question_id,))


# ── Main admin page ───────────────────────────────────────────────────────────

def show_admin(user):
    section_header("⚙️ Admin Control Panel", "Manage the entire AILearn Pro platform.")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📊 Dashboard", "👥 Students", "📝 Quiz Manager", "📢 Announcements", "📈 Analytics"])

    # ── Tab 1: Dashboard ───────────────────────────────────────────────────
    with tab1:
        try:
            stats = platform_stats()
        except Exception as e:
            st.error(f"Could not load stats: {e}")
            stats = {"students": 0, "attempts": 0, "certs": 0, "questions": 0}

        c1, c2, c3, c4 = st.columns(4)
        with c1: kpi_card("Total Students", stats["students"], "👥")
        with c2: kpi_card("Quiz Attempts",  stats["attempts"], "📝")
        with c3: kpi_card("Certificates",   stats["certs"],    "🎓")
        with c4: kpi_card("Questions",       stats["questions"],"❓")

        st.markdown("<br>", unsafe_allow_html=True)

        c_left, c_right = st.columns(2)
        with c_left:
            st.markdown("#### 🏆 Top Performers")
            try:
                board = get_leaderboard()
            except Exception:
                board = []
            if board:
                df_board = pd.DataFrame(board)
                # Rename columns safely
                col_map = {}
                for col in df_board.columns:
                    if col.lower() in ("avg_pct", "best_pct"):
                        col_map[col] = "Avg Score (%)"
                    elif col.lower() == "name":
                        col_map[col] = "Name"
                    elif col.lower() == "attempts":
                        col_map[col] = "Attempts"
                df_board = df_board.rename(columns=col_map)
                if "Avg Score (%)" in df_board.columns:
                    df_board["Avg Score (%)"] = df_board["Avg Score (%)"].astype(float).round(1)
                st.dataframe(df_board.head(10), use_container_width=True, hide_index=True)
            else:
                alert("No quiz data yet.", "info")

        with c_right:
            st.markdown("#### 📊 Platform Activity (Last 7 Days)")
            days = [
                (datetime.date.today() - datetime.timedelta(days=i)).strftime("%d %b")
                for i in range(6, -1, -1)
            ]
            activity = [12, 18, 9, 22, 15, 28, 20]
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=days, y=activity,
                marker_color=["#6C63FF","#8B5CF6","#FF6584",
                               "#6C63FF","#43D9AD","#FFB547","#6C63FF"]
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#E8E8F0"), height=280,
                margin=dict(l=0, r=0, t=10, b=10),
                xaxis=dict(showgrid=False, color="#8888AA"),
                yaxis=dict(showgrid=False, color="#8888AA")
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Tab 2: Students ────────────────────────────────────────────────────
    with tab2:
        st.markdown("#### 👥 All Students")
        try:
            students = get_all_students()
        except Exception as e:
            st.error(f"Could not load students: {e}")
            students = []

        if not students:
            alert("No students registered yet.", "info")
        else:
            search = st.text_input("🔍 Search by name or email", key="admin_search")
            df_stu = pd.DataFrame(students)
            if search:
                mask = (
                    df_stu["name"].str.contains(search, case=False, na=False) |
                    df_stu["email"].str.contains(search, case=False, na=False)
                )
                df_stu = df_stu[mask]
            df_stu["Status"] = df_stu["is_active"].apply(
                lambda x: "🟢 Active" if int(x) == 1 else "🔴 Blocked"
            )
            st.dataframe(
                df_stu[["id","name","email","Status","created_at"]],
                use_container_width=True, hide_index=True
            )

            st.markdown("#### 🔧 Block / Unblock Student")
            c1, c2, c3 = st.columns(3)
            with c1:
                sid = st.number_input("Student ID", min_value=1, step=1, key="block_id")
            with c2:
                action = st.selectbox("Action", ["Unblock (Activate)", "Block (Deactivate)"])
            with c3:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Apply", use_container_width=True):
                    try:
                        toggle_student(int(sid), action.startswith("Unblock"))
                        st.success(f"✅ Student {sid} updated.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")

    # ── Tab 3: Quiz Manager ────────────────────────────────────────────────
    with tab3:
        add_tab, manage_tab, stats_tab = st.tabs(
            ["➕ Add Question", "🗑️ Manage Questions", "📊 Quiz Stats"])

        # ── Add ───────────────────────────────────────────────────────────
        with add_tab:
            c1, c2 = st.columns(2)
            with c1:
                q_topic = st.selectbox("Topic", [
                    "Machine Learning", "Deep Learning", "NLP",
                    "Python", "Computer Vision"])
                q_diff = st.selectbox("Difficulty", ["easy", "medium", "hard"])
                q_type = st.selectbox("Type", ["mcq", "true_false"])
            with c2:
                q_text = st.text_area("Question text", key="q_text", height=100)
                q_ans  = st.selectbox("Correct answer", ["A", "B", "C", "D"])

            q_a = st.text_input("Option A", key="qa")
            q_b = st.text_input("Option B", key="qb")
            q_c = st.text_input("Option C", key="qc")
            q_d = st.text_input("Option D", key="qd")

            if st.button("✅ Add Question", use_container_width=True):
                if q_text and q_a and q_b:
                    try:
                        add_question(q_topic, q_text, q_a, q_b, q_c, q_d,
                                     q_ans, q_diff, q_type, user["id"])
                        st.success("✅ Question added!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error adding question: {e}")
                else:
                    st.warning("Fill in question text and at least options A & B.")

        # ── Manage / Delete ───────────────────────────────────────────────
        with manage_tab:
            df_q = _query_df(
                "SELECT id, topic, question, answer, difficulty "
                "FROM quiz_questions ORDER BY id DESC;"
            )

            if df_q.empty:
                alert("No questions in the database yet.", "info")
            else:
                topics_available = ["All"] + sorted(df_q["topic"].unique().tolist())
                filter_topic = st.selectbox("Filter by topic", topics_available, key="del_filter")
                df_view = df_q if filter_topic == "All" else df_q[df_q["topic"] == filter_topic]

                st.markdown(f"Showing **{len(df_view)}** question(s)")
                st.dataframe(df_view, use_container_width=True, hide_index=True)

                st.markdown("---")
                st.markdown("#### 🗑️ Delete a Question")
                alert("⚠️ Deletion is permanent and cannot be undone.", "warning")

                c1, c2 = st.columns([2, 1])
                with c1:
                    del_id = st.number_input(
                        "Enter Question ID to delete",
                        min_value=1, step=1, key="del_q_id",
                        help="Copy the ID from the table above"
                    )
                with c2:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("🗑️ Delete Question", use_container_width=True):
                        st.session_state["confirm_delete_id"] = int(del_id)

                # Two-step confirmation
                if st.session_state.get("confirm_delete_id"):
                    cid = st.session_state["confirm_delete_id"]
                    match = df_q[df_q["id"] == cid]
                    if match.empty:
                        st.error(f"❌ No question found with ID {cid}.")
                        st.session_state.pop("confirm_delete_id", None)
                    else:
                        q_preview = match.iloc[0]["question"]
                        st.markdown(f"""
                        <div style="background:rgba(255,101,132,0.1);
                                    border:1px solid rgba(255,101,132,0.4);
                                    border-radius:12px;padding:1rem 1.2rem;margin:.5rem 0">
                            <p style="color:#FF6584;font-weight:700;margin:0 0 .4rem">
                                ⚠️ Confirm deletion of Question ID {cid}
                            </p>
                            <p style="color:#CCCCEE;font-size:.88rem;margin:0">"{q_preview}"</p>
                        </div>
                        """, unsafe_allow_html=True)
                        col_yes, col_no = st.columns(2)
                        with col_yes:
                            if st.button("✅ Yes, delete it", use_container_width=True,
                                         key="confirm_yes"):
                                try:
                                    delete_question(cid)
                                    st.session_state.pop("confirm_delete_id", None)
                                    st.success(f"🗑️ Question ID {cid} deleted.")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Delete failed: {e}")
                        with col_no:
                            if st.button("❌ Cancel", use_container_width=True,
                                         key="confirm_no"):
                                st.session_state.pop("confirm_delete_id", None)
                                st.rerun()

        # ── Stats ─────────────────────────────────────────────────────────
        with stats_tab:
            df_a = _query_df("""
                SELECT topic,
                       COUNT(*)                           AS attempts,
                       ROUND(AVG(score * 100.0 / total), 1) AS avg_pct
                FROM quiz_attempts
                GROUP BY topic
                ORDER BY attempts DESC;
            """)
            if not df_a.empty:
                df_a["attempts"] = df_a["attempts"].astype(int)
                df_a["avg_pct"]  = df_a["avg_pct"].astype(float)
                fig = px.bar(
                    df_a, x="topic", y="avg_pct", color="attempts",
                    color_continuous_scale=["#1A1A2E", "#6C63FF"],
                    labels={"avg_pct": "Avg Score (%)", "topic": "Topic"}
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#E8E8F0"), height=300,
                    margin=dict(l=0, r=0, t=10, b=10),
                    coloraxis_showscale=False,
                    xaxis=dict(showgrid=False, color="#E8E8F0"),
                    yaxis=dict(showgrid=False, color="#8888AA")
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                alert("No quiz attempts recorded yet.", "info")

    # ── Tab 4: Announcements ───────────────────────────────────────────────
    with tab4:
        st.markdown("#### 📢 Post Announcement")
        ann_title = st.text_input("Title", key="ann_title")
        ann_body  = st.text_area("Message", key="ann_body", height=120)
        if st.button("📢 Post Announcement", use_container_width=True):
            if ann_title and ann_body:
                try:
                    add_announcement(ann_title, ann_body, user["id"])
                    st.success("✅ Announcement posted!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.warning("Fill in title and message.")

        st.markdown("---")
        st.markdown("#### 📋 Recent Announcements")
        try:
            anns = get_announcements()
        except Exception:
            anns = []
        if anns:
            for ann in anns:
                st.markdown(f"""
                <div style="background:#1A1A2E;border-radius:10px;padding:.8rem 1rem;
                            border:1px solid rgba(108,99,255,0.2);margin-bottom:.5rem">
                    <strong style="color:#E8E8F0">{ann['title']}</strong>
                    <span style="float:right;color:#8888AA;font-size:.8rem">
                        {fmt_dt(ann['created_at'])}
                    </span>
                    <p style="color:#AAAACC;font-size:.85rem;margin:.3rem 0 0">{ann['body']}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            alert("No announcements yet.", "info")

    # ── Tab 5: Analytics ───────────────────────────────────────────────────
    with tab5:
        st.markdown("#### 📈 Platform Analytics")

        df_prog = _query_df("""
            SELECT module,
                   COUNT(*)         AS learners,
                   AVG(progress_pct) AS avg_pct
            FROM learning_progress
            GROUP BY module
            ORDER BY learners DESC;
        """)
        df_certs = _query_df("""
            SELECT course, COUNT(*) AS count
            FROM certificates
            GROUP BY course;
        """)

        c1, c2 = st.columns(2)
        with c1:
            if not df_prog.empty:
                st.markdown("**Most Popular Modules**")
                df_prog["learners"] = df_prog["learners"].astype(int)
                fig = px.bar(df_prog, x="module", y="learners",
                             color_discrete_sequence=["#6C63FF"])
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#E8E8F0"), height=280,
                    margin=dict(l=0, r=0, t=10, b=10),
                    xaxis=dict(showgrid=False, color="#E8E8F0", tickangle=30),
                    yaxis=dict(showgrid=False, color="#8888AA")
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                alert("No module activity yet.", "info")

        with c2:
            if not df_certs.empty:
                st.markdown("**Certificates Issued by Course**")
                df_certs["count"] = df_certs["count"].astype(int)
                fig2 = px.pie(df_certs, names="course", values="count",
                              color_discrete_sequence=px.colors.qualitative.Vivid,
                              hole=0.45)
                fig2.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#E8E8F0"), height=280,
                    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
                    margin=dict(l=0, r=0, t=10, b=10)
                )
                st.plotly_chart(fig2, use_container_width=True)
            else:
                alert("No certificates issued yet.", "info")