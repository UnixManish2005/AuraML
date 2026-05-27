"""
AILearn Pro — Main entry point.
Run: streamlit run app.py
"""

import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from database.db import init_db
init_db()

from utils.styles import inject_css

st.set_page_config(
    page_title="AuraML",
    page_icon="🐬",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()


# ── Auth pages ──────────────────────────────────────────────────────────────
def show_auth():
    from auth.auth_ui import show_login, show_register
    st.markdown("""
    <div style="text-align:center;padding:1.5rem 0 0.5rem">
        <span style="font-family:Syne,sans-serif;font-size:3rem;font-weight:800;
                     background:linear-gradient(135deg,#6C63FF,#FF6584);
                     -webkit-background-clip:text;-webkit-text-fill-color:transparent">
            🐬 AuraML
        </span>
        <p style="color:#8888AA;font-size:1rem;margin:.3rem 0 1.5rem">
            The All-in-One AI/ML Learning Platform
        </p>
    </div>""", unsafe_allow_html=True)

    t1, t2 = st.tabs(["🔑 Sign In", "🚀 Sign Up"])
    with t1: show_login()
    with t2: show_register()


# ── Navigation map ─────────────────────────────────────────────────────────
STUDENT_NAV = {
    # ── Core ────────────────────────────────────────────
    "🏠 Dashboard":            "dashboard",
    # ── Learning ────────────────────────────────────────
    "📚 Learning Modules":     "modules",
    "🧪 Deep Learning Labs":   "deep_learning",
    # ── Interactive ML ──────────────────────────────────
    "📈 Linear Regression":    "regression",
    "🎯 Classification":       "classification",
    "🌳 Decision Tree":        "decision_tree",
    "🔍 KNN":                  "knn",
    "🧠 Neural Network":       "neural_network",
    "🔵 Clustering (K-Means)": "clustering",
    # ── Data & Training ─────────────────────────────────
    "📊 Dataset Explorer":     "dataset_viz",
    "🔬 Model Training Lab":   "training_lab",
    # ── Projects ────────────────────────────────────────
    "🌍 Live ML Projects":     "projects_live",
    "🏗️ Project Builder":     "projects",
    # ── Tools ───────────────────────────────────────────
    "📝 Quiz Arena":           "quiz",
    "📄 Resume Builder":       "resume",
    "📖 Notes & Resources":    "notes",
    "🎓 Certificates":         "certificates",
    "🤖 AI Tutor":             "chatbot",
}

ADMIN_EXTRA = {"⚙️ Admin Panel": "admin"}

# Group labels for sidebar sections
SIDEBAR_SECTIONS = [
    ("Core",        ["🏠 Dashboard"]),
    ("Learning",    ["📚 Learning Modules", "🧪 Deep Learning Labs"]),
    ("Interactive ML", [
        "📈 Linear Regression", "🎯 Classification", "🌳 Decision Tree",
        "🔍 KNN", "🧠 Neural Network", "🔵 Clustering (K-Means)",
    ]),
    ("Data & Training", ["📊 Dataset Explorer", "🔬 Model Training Lab"]),
    ("Projects",    ["🌍 Live ML Projects", "🏗️ Project Builder"]),
    ("Tools",       [
        "📝 Quiz Arena", "📄 Resume Builder", "📖 Notes & Resources",
        "🎓 Certificates", "🤖 AI Tutor",
    ]),
]


# ── Sidebar ─────────────────────────────────────────────────────────────────
def show_sidebar(user):
    with st.sidebar:
        # Logo
        st.markdown("""
        <div style="padding:1rem 0 1.2rem;text-align:center;
                    border-bottom:1px solid rgba(108,99,255,0.2);margin-bottom:1rem">
            <span style="font-family:Syne,sans-serif;font-size:1.5rem;font-weight:800;
                         background:linear-gradient(135deg,#6C63FF,#FF6584);
                         -webkit-background-clip:text;-webkit-text-fill-color:transparent">
                🐬 AuraML 🐬
            </span>
        </div>""", unsafe_allow_html=True)

        # User chip
        role_badge = "👑 Admin" if user["role"] == "admin" else "🎓 Student"
        st.markdown(f"""
        <div style="background:#1A1A2E;border-radius:12px;padding:.7rem 1rem;
                    border:1px solid rgba(108,99,255,0.2);margin-bottom:1rem;text-align:center">
            <div style="font-size:1.6rem">👤</div>
            <div style="font-weight:700;color:#E8E8F0;font-size:.9rem">{user['name']}</div>
            <div style="color:#8888AA;font-size:.75rem">{user['email']}</div>
            <span style="background:rgba(108,99,255,.2);color:#6C63FF;
                         padding:2px 10px;border-radius:20px;font-size:.72rem;font-weight:600">
                {role_badge}
            </span>
        </div>""", unsafe_allow_html=True)

        current = st.session_state.get("page", "dashboard")

        # Admin section
        if user["role"] == "admin":
            st.markdown('<p style="color:#8888AA;font-size:.7rem;text-transform:uppercase;letter-spacing:1.5px;margin:4px 0 4px 4px">Admin</p>', unsafe_allow_html=True)
            if st.button("⚙️ Admin Panel", key="nav_admin", use_container_width=True):
                _go("admin")

        # Student sections
        for section_label, labels in SIDEBAR_SECTIONS:
            st.markdown(f'<p style="color:#8888AA;font-size:.7rem;text-transform:uppercase;letter-spacing:1.5px;margin:8px 0 4px 4px">{section_label}</p>', unsafe_allow_html=True)
            for label in labels:
                page_key = STUDENT_NAV[label]
                is_active = current == page_key
                btn_label = f"› {label}" if is_active else label
                if st.button(btn_label, key=f"nav_{page_key}", use_container_width=True):
                    _go(page_key)

        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()


def _go(page_key: str):
    # Clear quiz state on navigation
    if page_key != "quiz":
        for k in ["quiz_active","quiz_questions","quiz_idx","quiz_answers",
                   "quiz_start","quiz_topic","quiz_done","quiz_saved"]:
            st.session_state.pop(k, None)
    st.session_state["page"] = page_key
    st.rerun()


# ── Router ──────────────────────────────────────────────────────────────────
def route(user):
    page = st.session_state.get("page", "dashboard")

    # ── Admin ──────────────────────────────────────────
    if page == "admin" and user["role"] == "admin":
        from admin.admin_ui import show_admin
        show_admin(user); return

    # ── Core ───────────────────────────────────────────
    if page == "dashboard":
        from student.dashboard import show_dashboard
        show_dashboard(user); return

    # ── Original learning modules ───────────────────────
    #if page == "modules":
        #from student.modules import show_modules
       # show_modules(user); return

    # ── Deep Learning Labs ──────────────────────────────
    if page == "deep_learning":
        from modules.deep_learning import render
        _page_header("🧪 Deep Learning Labs",
                     "CNNs, RNNs, GANs, Transformers — hands-on experiments.")
        render(); return

    # ── Interactive ML pages ────────────────────────────
    if page == "regression":
        from modules.regression import render
        render(); return

    if page == "classification":
        from modules.classification import render
        render(); return

    if page == "decision_tree":
        from modules.decision_tree import render
        render(); return

    if page == "knn":
        from modules.knn import render
        render(); return

    if page == "neural_network":
        from modules.neural_network import render
        render(); return

    if page == "clustering":
        from modules.clustering import render
        render(); return

    # ── Data & Training ─────────────────────────────────
    if page == "dataset_viz":
        from modules.dataset_viz import render
        render(); return

    if page == "training_lab":
        from modules.training_lab import render
        render(); return

    # ── Projects ────────────────────────────────────────
    if page == "projects_live":
        from modules.projects_live import render
        render(); return

    if page == "projects":
        from student.projects import show_projects
        show_projects(user); return

    # ── Tools ───────────────────────────────────────────
    if page == "quiz":
        from quizzes.quiz_ui import show_quiz
        show_quiz(user); return

    #if page == "resume":
     #  from resume_builder.resume_ui import show_resume_builder
       # show_resume_builder(user); return

    if page == "notes":
        from student.projects import show_notes
        show_notes(user); return

    #if page == "certificates":
        #from student.certificates import show_certificates
       # show_certificates(user); return

    #if page == "chatbot":
       # from ai_modules.chatbot import show_ai_tutor
       # show_ai_tutor(user); return

    # Fallback
    from student.dashboard import show_dashboard
    show_dashboard(user)


def _page_header(title: str, subtitle: str):
    st.markdown(f"""
    <div style="font-family:Syne,sans-serif;font-size:1.7rem;font-weight:800;
                background:linear-gradient(135deg,#6C63FF,#FF6584);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                margin-bottom:.2rem">{title}</div>
    <div style="color:#8888AA;font-size:.9rem;margin-bottom:1.2rem">{subtitle}</div>
    """, unsafe_allow_html=True)


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    user = st.session_state.get("user")
    if not user:
        show_auth()
    else:
        show_sidebar(user)
        route(user)


if __name__ == "__main__":
    main()
