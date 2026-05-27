# 🧠 AILearn Pro — All-in-One AI/ML EdTech Platform

> A production-ready Streamlit-based EdTech SaaS platform for learning AI/ML — built like Coursera + DataCamp + Kaggle in one app.

---

## ✨ Features

| Module | Description |
|--------|-------------|
| 🔐 Auth System | Register / Login / Role-based access (Admin & Student) |
| 🏠 Student Dashboard | Progress bars, quiz scores, announcements, activity feed |
| 📚 Learning Modules | 7 visual AI/ML modules with interactive demos & charts |
| 🔬 ML Playground | Upload CSV, pick algorithm, train & visualise live results |
| 📝 Quiz Arena | Timed MCQ quizzes, leaderboard, instant answer review |
| 📄 Resume Builder | ATS scorer, AI project descriptions, LinkedIn & README generators |
| 🏗️ Project Builder | 4 full project templates with code + deployment guides |
| 📖 Notes & Resources | Cheat sheets, interview Q&A, downloadable markdown files |
| 🎓 Certificates | Auto-issue certificates for module & quiz completion |
| 🤖 AI Tutor | Claude-powered chatbot for AI/ML Q&A |
| ⚙️ Admin Panel | Student management, quiz management, announcements, analytics |

---

## 🚀 Quick Start

### 1. Clone / Extract the project
```bash
cd ailearn
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 🔑 Default Credentials

| Role | Email | Password |
|------|-------|----------|
| Admin | admin@ailearn.com | admin123 |
| Student | demo@ailearn.com | demo123 |

> ⚠️ Change these in production by updating the database directly.

---

## 🗂️ Project Structure

```
ailearn/
├── app.py                    # Main entry point & router
│
├── database/
│   └── db.py                 # SQLite DB — all models & helper functions
│
├── utils/
│   └── styles.py             # CSS injection, charts, shared UI components
│
├── auth/
│   └── auth_ui.py            # Login & Register pages
│
├── student/
│   ├── dashboard.py          # Student home dashboard
│   ├── modules.py            # AI/ML learning modules with interactive demos
│   ├── projects.py           # Project builder + Notes & resources
│   └── certificates.py      # Certificate display & issuance
│
├── ml_labs/
│   └── playground.py         # Interactive ML training playground
│
├── quizzes/
│   └── quiz_ui.py            # Quiz system: timer, MCQ, leaderboard
│
├── resume_builder/
│   └── resume_ui.py          # ATS Resume Builder with AI tools
│
├── ai_modules/
│   └── chatbot.py            # AI Tutor chatbot (Anthropic Claude API)
│
├── admin/
│   └── admin_ui.py           # Admin panel: dashboard, student mgmt, analytics
│
├── requirements.txt
└── README.md
```

---

## 🔬 ML Playground — Supported Algorithms

| Algorithm | Task | Key Metric |
|-----------|------|-----------|
| Linear Regression | Regression | RMSE, R² |
| Logistic Regression | Classification | Accuracy, F1 |
| Decision Tree | Both | Accuracy, Feature Importance |
| Random Forest | Both | Accuracy, Feature Importance |
| K-Nearest Neighbors | Classification | Accuracy |
| SVM | Classification | Accuracy |
| K-Means | Clustering | Inertia |

---

## 🤖 AI Tutor Setup

The AI Tutor uses the **Anthropic Claude API**. To enable it:

1. The API key is automatically read from environment or Streamlit secrets.
2. For Streamlit Cloud, add to `.streamlit/secrets.toml`:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```
3. For local dev: `export ANTHROPIC_API_KEY=sk-ant-...`

---

## ☁️ Deployment

### Streamlit Cloud (Recommended — Free)
1. Push this folder to a GitHub repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. New app → select repo → entry point: `app.py`
4. Add secrets if using AI Tutor
5. Deploy! ✅

### Render / Railway
1. Add `Procfile`: `web: streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
2. Push to GitHub → connect to Render/Railway

### Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
```

```bash
docker build -t ailearn .
docker run -p 8501:8501 ailearn
```

### AWS EC2
```bash
git clone <your-repo>
cd ailearn
pip install -r requirements.txt
nohup streamlit run app.py --server.port 8501 &
```
Configure security group to allow port 8501.

---

## 📦 Dependencies

```
streamlit      — UI framework
pandas         — Data manipulation
numpy          — Numerical computing
plotly         — Interactive charts
scikit-learn   — ML algorithms
requests       — API calls (AI Tutor)
python-docx    — Document generation
```

---

## 🔮 Future Enhancements

- [ ] Firebase / JWT authentication
- [ ] Video lesson integration
- [ ] Coding sandbox (Jupyter-lite embed)
- [ ] Assignment submission system
- [ ] Peer review system
- [ ] Stripe payment integration
- [ ] Email notifications
- [ ] Mobile-responsive PWA

---

## 📄 License
MIT — Free to use, modify, and deploy.

Built with ❤️ using Python + Streamlit.
