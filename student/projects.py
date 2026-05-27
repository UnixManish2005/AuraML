"""Project Builder — generate project structure & starter code."""

import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.styles import section_header, alert

PROJECTS = {
    "🏠 House Price Prediction": {
        "difficulty": "Beginner",
        "time": "2–3 hours",
        "dataset": "Kaggle: House Prices — Advanced Regression Techniques",
        "dataset_link": "https://www.kaggle.com/c/house-prices-advanced-regression-techniques",
        "tech": ["Python", "Pandas", "Scikit-learn", "XGBoost", "Streamlit"],
        "structure": """
house_price_prediction/
├── app.py              # Streamlit UI
├── model.py            # Training pipeline
├── preprocess.py       # Feature engineering
├── data/
│   ├── train.csv
│   └── test.csv
├── models/
│   └── xgb_model.pkl
└── requirements.txt
""",
        "code": '''# model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import joblib

def train():
    df = pd.read_csv("data/train.csv")
    # Feature engineering
    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    features = ["OverallQual","GrLivArea","GarageCars","TotalSF","YearBuilt","FullBath"]
    X = df[features].fillna(0)
    y = np.log1p(df["SalePrice"])  # log-transform target
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6)
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], early_stopping_rounds=50, verbose=False)
    preds = np.expm1(model.predict(X_te))
    rmse = np.sqrt(mean_squared_error(np.expm1(y_te), preds))
    print(f"RMSE: {rmse:.2f}")
    joblib.dump(model, "models/xgb_model.pkl")
    return model

if __name__ == "__main__":
    train()
''',
        "deployment": """
**Deploy to Streamlit Cloud:**
1. Push code to GitHub
2. Go to share.streamlit.io → New app
3. Select your repo & `app.py`
4. Add `kaggle.json` to Streamlit secrets if needed
5. Deploy! 🚀
""",
    },
    "💬 Sentiment Analyser": {
        "difficulty": "Intermediate",
        "time": "3–5 hours",
        "dataset": "Kaggle: IMDB Movie Reviews",
        "dataset_link": "https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews",
        "tech": ["Python", "NLTK", "Scikit-learn", "Streamlit", "Transformers"],
        "structure": """
sentiment_analyser/
├── app.py
├── train.py
├── predict.py
├── data/
│   └── imdb.csv
├── models/
│   └── sentiment_model.pkl
└── requirements.txt
""",
        "code": '''# train.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib, re, nltk
nltk.download("stopwords")
from nltk.corpus import stopwords

STOP = set(stopwords.words("english"))

def clean(text):
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    return " ".join(w for w in text.lower().split() if w not in STOP)

df = pd.read_csv("data/imdb.csv")
df["clean"] = df["review"].apply(clean)
X_tr, X_te, y_tr, y_te = train_test_split(df["clean"], df["sentiment"], test_size=0.2)

tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
X_tr_v = tfidf.fit_transform(X_tr)
X_te_v = tfidf.transform(X_te)

model = LogisticRegression(max_iter=1000, C=5)
model.fit(X_tr_v, y_tr)
print(classification_report(y_te, model.predict(X_te_v)))
joblib.dump((tfidf, model), "models/sentiment_model.pkl")
''',
        "deployment": "Deploy on Streamlit Cloud or Render. Add `transformers` to requirements.txt for BERT upgrade.",
    },
    "🎓 Student Performance Predictor": {
        "difficulty": "Intermediate",
        "time": "4–6 hours",
        "dataset": "UCI: Student Performance Dataset",
        "dataset_link": "https://archive.ics.uci.edu/ml/datasets/Student+Performance",
        "tech": ["Python", "Pandas", "Scikit-learn", "XGBoost", "SHAP", "Streamlit"],
        "structure": """
student_predictor/
├── app.py
├── train.py
├── explain.py         # SHAP explanations
├── data/student.csv
├── models/model.pkl
└── requirements.txt
""",
        "code": '''# train.py
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

df = pd.read_csv("data/student.csv", sep=";")
df["pass"] = (df["G3"] >= 10).astype(int)
features = ["studytime","failures","absences","G1","G2","Medu","Fedu","goout","freetime"]
X = df[features]
y = df["pass"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
model = GradientBoostingClassifier(n_estimators=200, max_depth=4)
model.fit(X_tr, y_tr)
print(classification_report(y_te, model.predict(X_te)))
joblib.dump(model, "models/model.pkl")
''',
        "deployment": "Perfect for Streamlit Cloud. Add SHAP + Plotly for interactive explanations.",
    },
    "🤖 RAG Chatbot": {
        "difficulty": "Advanced",
        "time": "6–10 hours",
        "dataset": "Your own PDF documents",
        "dataset_link": "",
        "tech": ["Python", "LangChain", "FAISS", "HuggingFace", "FastAPI", "Streamlit"],
        "structure": """
rag_chatbot/
├── app.py
├── ingest.py          # PDF → vector store
├── chain.py           # RAG chain
├── docs/              # Your PDFs
├── vectorstore/       # FAISS index
└── requirements.txt
""",
        "code": '''# chain.py — RAG pipeline
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("vectorstore", embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# LLM
pipe = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=256)
llm = HuggingFacePipeline(pipeline=pipe)

# Chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

def ask(question: str) -> str:
    return qa_chain.run(question)
''',
        "deployment": "Deploy on Render or AWS EC2. Consider Pinecone for production vector storage.",
    },
}

NOTES = {
    "📋 ML Cheat Sheet": {
        "content": """
## Machine Learning Quick Reference

### Supervised Algorithms
| Algorithm | Type | Key Params |
|-----------|------|-----------|
| Linear Regression | Regression | fit_intercept, normalize |
| Logistic Regression | Classification | C, max_iter, solver |
| Decision Tree | Both | max_depth, min_samples_split |
| Random Forest | Both | n_estimators, max_features |
| XGBoost | Both | n_estimators, learning_rate, max_depth |
| SVM | Both | C, kernel, gamma |
| KNN | Both | n_neighbors, metric |

### Model Evaluation
| Metric | Formula | Use When |
|--------|---------|---------|
| Accuracy | (TP+TN)/Total | Balanced classes |
| Precision | TP/(TP+FP) | Minimise false positives |
| Recall | TP/(TP+FN) | Minimise false negatives |
| F1 Score | 2×(P×R)/(P+R) | Imbalanced classes |
| RMSE | √(Σ(y-ŷ)²/n) | Regression |
| R² | 1 - SS_res/SS_tot | Regression |

### Bias–Variance Trade-off
- **High Bias (Underfitting)** → model too simple → more features, complex model
- **High Variance (Overfitting)** → model too complex → regularisation, more data, dropout
""",
    },
    "🔤 NLP Cheat Sheet": {
        "content": """
## NLP Quick Reference

### Text Preprocessing Pipeline
```
Raw Text → Lowercase → Remove HTML/Punct → Tokenise → Remove Stop Words → Stem/Lemmatise → Vectorise
```

### Vectorisation Methods
| Method | Description | When |
|--------|-------------|------|
| Bag of Words | Count of each word | Simple baselines |
| TF-IDF | Term frequency × Inverse doc freq | Classic NLP |
| Word2Vec | Dense 300-dim word vectors | Semantic similarity |
| GloVe | Pre-trained global vectors | Transfer learning |
| BERT | Contextual embeddings | State-of-the-art |

### Key Formulas
- **TF(t,d)** = count(t in d) / total words in d
- **IDF(t)** = log(N / df(t))
- **TF-IDF** = TF × IDF
""",
    },
    "❓ Top 50 ML Interview Q&A": {
        "content": """
## Top ML Interview Questions

**Q1. What is the difference between supervised and unsupervised learning?**
Supervised: labelled data → learn mapping X→y. Unsupervised: unlabelled data → find structure (clusters, embeddings).

**Q2. Explain the bias–variance trade-off.**
Bias = error from wrong assumptions (underfitting). Variance = error from sensitivity to training data (overfitting). Goal: balance both.

**Q3. What is regularisation? Types?**
Penalty on model complexity. L1 (Lasso) = absolute weights → feature selection. L2 (Ridge) = squared weights → small weights. ElasticNet = both.

**Q4. What is cross-validation?**
Split data into K folds; train on K-1, validate on 1; repeat K times; average scores. Reduces variance of evaluation.

**Q5. Explain precision vs recall.**
Precision: of all predicted positives, how many are truly positive. Recall: of all actual positives, how many did we catch.

**Q6. What is gradient descent?**
Iterative optimisation algorithm. Updates parameters opposite to gradient: θ = θ - α·∇J(θ). Learning rate α controls step size.

**Q7. Explain overfitting and how to prevent it.**
Model memorises training data, fails on new data. Solutions: more data, regularisation, dropout, early stopping, k-fold CV.

**Q8. What is feature engineering?**
Creating new features from raw data to improve model performance. Includes log transforms, interaction terms, binning, encoding.

**Q9. What is a confusion matrix?**
Table comparing actual vs predicted labels. Contains TP, TN, FP, FN — basis for all classification metrics.

**Q10. Explain Random Forest.**
Ensemble of decision trees trained on bootstrap samples (bagging) with random feature subsets. Reduces variance via averaging.
""",
    },
}


def show_projects(user):
    section_header("🏗️ Project Builder",
                   "Choose a project template, get structure, code, and deployment guide.")

    cols = st.columns(2)
    for i, (name, proj) in enumerate(PROJECTS.items()):
        with cols[i % 2]:
            diff_color = {"Beginner":"#43D9AD","Intermediate":"#FFB547","Advanced":"#FF6584"}.get(proj["difficulty"],"#6C63FF")
            st.markdown(f"""
            <div class="module-card">
                <h4>{name}</h4>
                <p style="margin:0.2rem 0">
                    <span style="color:{diff_color};font-weight:600">{proj['difficulty']}</span>
                    &nbsp;·&nbsp;
                    <span style="color:#8888AA">⏱ {proj['time']}</span>
                </p>
                <p style="color:#8888AA;font-size:0.82rem;margin-top:0.3rem">
                    Tech: {' · '.join(proj['tech'][:3])}...
                </p>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"Explore →", key=f"proj_{i}", use_container_width=True):
                st.session_state["selected_project"] = name
                st.rerun()

    selected = st.session_state.get("selected_project")
    if selected and selected in PROJECTS:
        proj = PROJECTS[selected]
        st.markdown("---")
        st.markdown(f"### {selected}")

        t1, t2, t3, t4 = st.tabs(["📁 Structure", "💻 Starter Code", "📊 Dataset", "🚀 Deploy"])
        with t1:
            st.code(proj["structure"], language="")
        with t2:
            st.code(proj["code"], language="python")
        with t3:
            st.markdown(f"""
            **Recommended Dataset:** {proj['dataset']}
            {"" if not proj.get('dataset_link') else f"[Open on Kaggle/UCI]({proj['dataset_link']})"}
            """)
        with t4:
            st.markdown(proj["deployment"])


def show_notes(user):
    section_header("📖 Notes & Resources",
                   "Cheat sheets, interview Q&A, formula references — all in one place.")

    for title, note in NOTES.items():
        with st.expander(title, expanded=False):
            st.markdown(note["content"])
            st.download_button(
                f"⬇️ Download {title}",
                data=note["content"],
                file_name=f"{title.replace(' ','_').replace('/','_')}.md",
                mime="text/markdown",
                key=f"dl_{title}",
            )
