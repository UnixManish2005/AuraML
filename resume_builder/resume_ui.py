"""ATS Resume Builder — AI-powered, PDF export, keyword optimisation."""

import streamlit as st
import json
import re
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from database.db import save_resume, get_resume
from utils.styles import section_header, alert, kpi_card

# ── AI project description templates ─────────────────────────────────────────

PROJECT_TEMPLATES = {
    "Fake News Detection": {
        "desc": "Built an NLP-based binary classification pipeline to detect misinformation in news articles using TF-IDF vectorisation and a Logistic Regression / LSTM hybrid model achieving 94.2% accuracy.",
        "tech": "Python, NLTK, Scikit-learn, TensorFlow, Pandas, Streamlit",
        "bullets": [
            "Preprocessed 20,000+ news articles — tokenisation, stop-word removal, lemmatisation",
            "Engineered TF-IDF features (bigrams, 15K vocab) and fine-tuned LSTM on GloVe embeddings",
            "Deployed interactive Streamlit app with real-time prediction and confidence scores",
            "Achieved 94.2% accuracy, 93.8% F1-score on held-out test set",
        ],
    },
    "House Price Prediction": {
        "desc": "Developed an end-to-end regression pipeline predicting residential property prices using ensemble methods (XGBoost + LightGBM stacking) with feature engineering on 79 variables, achieving RMSE of ₹2.1L.",
        "tech": "Python, Pandas, NumPy, Scikit-learn, XGBoost, LightGBM, SHAP, FastAPI, Streamlit",
        "bullets": [
            "Engineered 30+ features including interaction terms, log transforms, and neighbourhood aggregations",
            "Applied K-Fold cross-validation with stacking ensemble — XGBoost + LightGBM + Ridge meta-learner",
            "Used SHAP values to explain individual predictions — improved stakeholder trust by 40%",
            "Deployed REST API via FastAPI with Streamlit frontend; containerised with Docker",
        ],
    },
    "Sentiment Analysis": {
        "desc": "Built a multi-class sentiment classifier (Positive/Negative/Neutral) for product reviews using fine-tuned BERT achieving 91.5% accuracy on the Amazon review dataset.",
        "tech": "Python, HuggingFace Transformers, PyTorch, BERT, Pandas, FastAPI",
        "bullets": [
            "Fine-tuned bert-base-uncased on 50K labelled Amazon reviews using AdamW with warm-up scheduler",
            "Implemented data augmentation (back-translation, synonym replacement) to handle class imbalance",
            "Built real-time inference API with FastAPI — <200ms p95 latency",
            "Visualised attention weights to provide explainable predictions",
        ],
    },
    "Student Performance Predictor": {
        "desc": "Designed a ML system to predict at-risk students in online courses using behavioural features (login frequency, assignment completion, forum activity), enabling early intervention with 88% recall.",
        "tech": "Python, Pandas, Scikit-learn, XGBoost, SHAP, Streamlit, SQLite",
        "bullets": [
            "Collected and processed LMS data — 15 behavioural features across 3,000+ student records",
            "Applied SMOTE to handle 1:4 class imbalance; trained Random Forest, XGBoost, SVM (best: XGBoost)",
            "Achieved 88% recall on at-risk class — critical for early intervention effectiveness",
            "Built Streamlit dashboard for educators to view risk scores and SHAP feature explanations",
        ],
    },
    "Image Classifier (CNN)": {
        "desc": "Trained a custom CNN and fine-tuned ResNet-50 for 10-class image classification on CIFAR-10, reaching 93.1% top-1 accuracy using data augmentation and learning-rate scheduling.",
        "tech": "Python, TensorFlow/Keras, OpenCV, NumPy, Matplotlib, Streamlit",
        "bullets": [
            "Designed 5-layer custom CNN; then fine-tuned ResNet-50 with frozen base layers — 93.1% accuracy",
            "Applied augmentation pipeline: random crop, horizontal flip, colour jitter, mixup",
            "Reduced training time by 35% with learning-rate warm-up + cosine decay schedule",
            "Deployed Streamlit app with drag-and-drop image upload and real-time class probability bars",
        ],
    },
    "Chatbot (NLP)": {
        "desc": "Developed a domain-specific FAQ chatbot for an e-commerce platform using intent classification (98% accuracy) and retrieval-augmented generation (RAG) with a vector database.",
        "tech": "Python, HuggingFace, LangChain, FAISS, FastAPI, Streamlit, Docker",
        "bullets": [
            "Built intent classifier (fine-tuned DistilBERT) on 2,000 labelled queries — 98% accuracy",
            "Implemented RAG pipeline: embedded 500 FAQs into FAISS vector store, retrieved top-3 passages",
            "Reduced support ticket volume by 35% in A/B test with 1,200 users",
            "Containerised with Docker; deployed on AWS EC2 with auto-scaling",
        ],
    },
}

SKILL_CATEGORIES = {
    "Programming Languages": ["Python", "R", "SQL", "Scala", "Java"],
    "ML/DL Frameworks":      ["Scikit-learn", "TensorFlow", "PyTorch", "Keras", "XGBoost", "LightGBM"],
    "Data Tools":            ["Pandas", "NumPy", "Matplotlib", "Seaborn", "Plotly"],
    "Cloud & DevOps":        ["AWS", "GCP", "Azure", "Docker", "Kubernetes", "Terraform", "GitHub Actions"],
    "Databases":             ["PostgreSQL", "MySQL", "MongoDB", "SQLite", "Redis", "Pinecone", "FAISS"],
    "APIs & Frameworks":     ["FastAPI", "Flask", "Streamlit", "LangChain", "HuggingFace"],
}

ATS_KEYWORDS = [
    "machine learning","deep learning","nlp","neural network","data science",
    "python","sql","tensorflow","pytorch","scikit-learn","pandas","numpy",
    "data analysis","feature engineering","model deployment","docker","aws",
    "fastapi","streamlit","xgboost","transformer","bert","llm","rag",
    "computer vision","classification","regression","clustering",
]


def compute_ats_score(data: dict) -> int:
    text = json.dumps(data).lower()
    hits = sum(1 for kw in ATS_KEYWORDS if kw in text)
    base = min(hits * 4, 60)          # keywords: max 60
    # bonus for completeness
    bonus = 0
    if data.get("experience"):     bonus += 10
    if data.get("projects"):       bonus += 10
    if len(data.get("skills",[])) >= 5: bonus += 10
    if data.get("certifications"): bonus += 5
    if data.get("summary"):        bonus += 5
    return min(base + bonus, 100)


def show_resume_builder(user):
    uid = user["id"]
    section_header("📄 ATS Resume Builder",
                   "Build an AI-optimised resume that passes Applicant Tracking Systems.")

    # Load existing data
    existing = get_resume(uid)
    resume   = json.loads(existing["data_json"]) if existing else {}

    tab1, tab2, tab3, tab4 = st.tabs(
        ["👤 Personal & Education", "💼 Skills & Projects", "✨ AI Tools", "📊 ATS Score & Export"])

    # ── Tab 1 ──────────────────────────────────────────────────────────────
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 👤 Personal Details")
            resume["name"]     = st.text_input("Full Name *", resume.get("name",""), key="r_name")
            resume["email"]    = st.text_input("Email *",     resume.get("email", user.get("email","")), key="r_email")
            resume["phone"]    = st.text_input("Phone",       resume.get("phone",""), key="r_phone")
            resume["linkedin"] = st.text_input("LinkedIn URL", resume.get("linkedin",""), key="r_li")
            resume["github"]   = st.text_input("GitHub URL",  resume.get("github",""), key="r_gh")
            resume["location"] = st.text_input("Location",    resume.get("location",""), key="r_loc")
        with c2:
            st.markdown("#### 🎓 Education")
            resume["edu_degree"]  = st.text_input("Degree", resume.get("edu_degree",""), key="r_deg",
                                                   placeholder="MCA / B.Tech / M.Sc")
            resume["edu_college"] = st.text_input("College", resume.get("edu_college",""), key="r_col")
            resume["edu_year"]    = st.text_input("Year", resume.get("edu_year",""), key="r_yr",
                                                   placeholder="2023–2025")
            resume["edu_cgpa"]    = st.text_input("CGPA / %", resume.get("edu_cgpa",""), key="r_cgpa")
            st.markdown("#### 💡 Professional Summary")
            resume["summary"] = st.text_area("Summary (2–3 lines)",
                                              resume.get("summary",""), key="r_sum", height=100)

    # ── Tab 2 ──────────────────────────────────────────────────────────────
    with tab2:
        st.markdown("#### 🔧 Technical Skills")
        selected_skills = []
        for cat, skills in SKILL_CATEGORIES.items():
            with st.expander(cat):
                chosen = st.multiselect(f"{cat}", skills,
                                        default=[s for s in skills if s in resume.get("skills",[])],
                                        key=f"sk_{cat}")
                selected_skills.extend(chosen)
        resume["skills"] = selected_skills

        st.markdown("#### 🚀 Projects")
        n_projects = st.number_input("Number of projects", 1, 6, max(1, len(resume.get("projects",[]))), key="np")
        projects = []
        for i in range(int(n_projects)):
            ex_proj = resume.get("projects", [{}]*6)
            ex = ex_proj[i] if i < len(ex_proj) else {}
            with st.expander(f"Project {i+1}", expanded=(i==0)):
                title = st.text_input("Title", ex.get("title",""), key=f"pt_{i}")
                tech  = st.text_input("Technologies", ex.get("tech",""), key=f"ptech_{i}")
                desc  = st.text_area("Description", ex.get("desc",""), key=f"pd_{i}", height=80)
                link  = st.text_input("GitHub/Live link", ex.get("link",""), key=f"pl_{i}")
                projects.append({"title":title,"tech":tech,"desc":desc,"link":link})
        resume["projects"] = projects

        st.markdown("#### 📜 Certifications & Experience")
        resume["certifications"] = st.text_area("Certifications (one per line)",
                                                 resume.get("certifications",""), key="r_certs", height=80)
        resume["experience"] = st.text_area("Work Experience / Internships",
                                            resume.get("experience",""), key="r_exp", height=100)

    # ── Tab 3: AI Tools ────────────────────────────────────────────────────
    with tab3:
        st.markdown("#### 🤖 AI Project Description Generator")
        proj_choice = st.selectbox("Select a project type", list(PROJECT_TEMPLATES.keys()))
        if st.button("✨ Generate Description"):
            tmpl = PROJECT_TEMPLATES[proj_choice]
            st.markdown(f"""
            <div style="background:#1A1A2E;border:1px solid rgba(108,99,255,0.3);
                        border-radius:14px;padding:1.2rem 1.4rem;margin-top:0.5rem">
                <h4 style="color:#6C63FF;font-family:Syne,sans-serif;font-weight:700">{proj_choice}</h4>
                <p style="color:#E8E8F0;font-size:0.9rem">{tmpl['desc']}</p>
                <p style="color:#8888AA;font-size:0.83rem"><strong style="color:#43D9AD">Tech:</strong> {tmpl['tech']}</p>
                <ul style="color:#CCCCEE;font-size:0.85rem">
                    {''.join(f"<li>{b}</li>" for b in tmpl['bullets'])}
                </ul>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### 🔗 LinkedIn Summary Generator")
        if resume.get("name") and resume.get("skills"):
            if st.button("Generate LinkedIn Summary"):
                skills_str = ", ".join(resume["skills"][:8])
                summary = f"""🚀 Aspiring AI/ML Engineer | {resume.get('edu_degree','Data Science Graduate')}

Passionate about building intelligent systems that solve real-world problems. Skilled in {skills_str}. {('Currently pursuing ' + resume.get('edu_degree','') + ' at ' + resume.get('edu_college','')) if resume.get('edu_degree') else ''}

I love turning raw data into actionable insights and deploying end-to-end ML pipelines. Always learning, always building. Open to opportunities in ML Engineering, Data Science, and AI Research.

#MachineLearning #DataScience #Python #AI #OpenToWork"""
                st.text_area("Your LinkedIn Summary (copy & paste!)", summary, height=200)

        st.markdown("---")
        st.markdown("#### 📝 GitHub README Generator")
        sel_proj = st.selectbox("Select project for README", list(PROJECT_TEMPLATES.keys()), key="readme_proj")
        if st.button("Generate README"):
            tmpl = PROJECT_TEMPLATES[sel_proj]
            readme = f"""# {sel_proj}

> {tmpl['desc']}

## 🛠️ Tech Stack
`{tmpl['tech'].replace(', ', '` `')}`

## ✨ Key Features
{chr(10).join('- ' + b for b in tmpl['bullets'])}

## 🚀 Quick Start
```bash
git clone https://github.com/yourusername/{sel_proj.lower().replace(' ','-')}
cd {sel_proj.lower().replace(' ','-')}
pip install -r requirements.txt
streamlit run app.py
```

## 📊 Results
See the results and demo in the [live app](#).

## 📄 License
MIT
"""
            st.code(readme, language="markdown")

    # ── Tab 4: ATS Score & Export ──────────────────────────────────────────
    with tab4:
        # Save resume
        ats = compute_ats_score(resume)
        if st.button("💾 Save Resume & Calculate ATS Score", use_container_width=True):
            save_resume(uid, json.dumps(resume), ats)
            st.success(f"✅ Resume saved! ATS Score: **{ats}/100**")

        # Display ATS score
        color = "#43D9AD" if ats >= 75 else "#FFB547" if ats >= 50 else "#FF6584"
        grade = "Excellent" if ats >= 75 else "Good" if ats >= 50 else "Needs Work"
        st.markdown(f"""
        <div style="text-align:center;padding:2rem;
                    background:linear-gradient(135deg,#1A1A2E,#16213E);
                    border:1px solid rgba(108,99,255,0.3);border-radius:20px;margin:1rem 0">
            <div style="font-size:3.5rem;font-family:Syne,sans-serif;font-weight:800;color:{color}">{ats}</div>
            <div style="color:#E8E8F0;font-size:1.1rem;font-weight:600">ATS Score — {grade}</div>
            <div style="color:#8888AA;font-size:0.85rem;margin-top:0.5rem">
                Based on keyword density, completeness, and section coverage
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Tips
        st.markdown("#### 💡 Optimisation Tips")
        tips = []
        if ats < 75:
            if not resume.get("summary"):   tips.append("Add a professional summary with ML keywords")
            if not resume.get("experience"):tips.append("Add internship or work experience")
            if len(resume.get("skills",[])) < 8:
                tips.append("Add more technical skills (target 10+)")
            if not resume.get("certifications"):
                tips.append("Add certifications (e.g., AWS, Azure AI, NPTEL)")
        if not tips:
            tips = ["Your resume is well-optimised! Keep adding quantified achievements."]
        for tip in tips:
            alert(f"💡 {tip}", "warning")

        # Text preview
        st.markdown("#### 📋 Resume Preview (Text)")
        if resume.get("name"):
            preview = _build_text_resume(resume)
            st.text_area("Resume text (copy to Word/PDF)", preview, height=500)
            st.download_button("⬇️ Download as .txt", preview,
                               file_name=f"{resume.get('name','resume').replace(' ','_')}_resume.txt",
                               mime="text/plain", use_container_width=True)


def _build_text_resume(r):
    lines = []
    lines.append(r.get("name","").upper())
    contact = " | ".join(filter(None, [r.get("email"), r.get("phone"),
                                        r.get("linkedin"), r.get("github"), r.get("location")]))
    lines.append(contact)
    lines.append("="*70)

    if r.get("summary"):
        lines += ["", "PROFESSIONAL SUMMARY", "-"*30, r["summary"]]

    if r.get("edu_degree"):
        lines += ["", "EDUCATION", "-"*30]
        lines.append(f"{r.get('edu_degree','')} | {r.get('edu_college','')} | {r.get('edu_year','')}")
        if r.get("edu_cgpa"): lines.append(f"CGPA/Percentage: {r['edu_cgpa']}")

    if r.get("skills"):
        lines += ["", "TECHNICAL SKILLS", "-"*30]
        lines.append(", ".join(r["skills"]))

    if r.get("projects"):
        lines += ["", "PROJECTS", "-"*30]
        for p in r["projects"]:
            if p.get("title"):
                lines.append(f"\n{p['title']}")
                if p.get("tech"):  lines.append(f"Technologies: {p['tech']}")
                if p.get("desc"):  lines.append(p["desc"])
                if p.get("link"):  lines.append(f"Link: {p['link']}")

    if r.get("certifications"):
        lines += ["", "CERTIFICATIONS", "-"*30, r["certifications"]]

    if r.get("experience"):
        lines += ["", "EXPERIENCE", "-"*30, r["experience"]]

    return "\n".join(lines)
