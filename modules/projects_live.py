"""
pages/projects.py
Real-World ML Mini Projects Demo Page
"""

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from utils.styles import hero, concept_box, section_header, success_box, warning_box, info_box
from utils.data_helpers import (get_house_price_data, get_spam_data,
                                  get_loan_data, get_student_data)


def render():
    st.markdown(hero(
        "Real-World AI Projects 🌍",
        "6 live mini-projects powered by real ML models. Enter inputs and get instant predictions!",
        "🌍"
    ), unsafe_allow_html=True)

    proj = st.selectbox("🚀 Choose a Project:", [
        "🏠 House Price Predictor",
        "📧 Email Spam Detector",
        "🎓 Student Performance Predictor",
        "💳 Loan Approval Predictor",
        "💬 Sentiment Analyser",
        "📰 Fake News Detector (Demo)",
    ])

    st.markdown("---")

    if proj == "🏠 House Price Predictor":
        _house_price()
    elif proj == "📧 Email Spam Detector":
        _spam_detector()
    elif proj == "🎓 Student Performance Predictor":
        _student_perf()
    elif proj == "💳 Loan Approval Predictor":
        _loan_approval()
    elif proj == "💬 Sentiment Analyser":
        _sentiment()
    else:
        _fake_news_demo()


# ─────────────────────────────────────────────
# PROJECT 1: House Price
# ─────────────────────────────────────────────
def _house_price():
    st.markdown(section_header("🏠", "House Price Predictor",
                "Linear Regression — Real Estate"), unsafe_allow_html=True)

    df = get_house_price_data(500)
    X  = df[["Size_sqft", "Rooms", "Age_years"]].values
    y  = df["Price_USD"].values
    model = LinearRegression()
    model.fit(X, y)

    col_in, col_out = st.columns(2)
    with col_in:
        size  = st.slider("📐 House Size (sq ft)", 500, 5000, 1500, 50)
        rooms = st.slider("🛏️ Number of Rooms",      1,   10,    3)
        age   = st.slider("🏚️ House Age (years)",     1,   50,   10)

    pred = model.predict([[size, rooms, age]])[0]
    conf_low  = pred * 0.9
    conf_high = pred * 1.1

    with col_out:
        st.markdown(f"""
        <div class="ml-card" style="text-align:center;padding:36px;">
            <div style="font-size:3rem;margin-bottom:12px;">🏠</div>
            <div style="color:#8892A4;font-size:0.9rem;">Estimated Price</div>
            <div style="font-size:2.2rem;font-weight:700;color:#43E97B;">
                ${pred:,.0f}
            </div>
            <div style="color:#8892A4;font-size:0.8rem;margin-top:8px;">
                Range: ${conf_low:,.0f} – ${conf_high:,.0f}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(info_box(
        f"🔢 <b>How the model works:</b> "
        f"Price = {model.coef_[0]:.1f}×size + {model.coef_[1]:.0f}×rooms "
        f"+ {model.coef_[2]:.0f}×age + {model.intercept_:,.0f}"
    ), unsafe_allow_html=True)
    _business_use("Real estate platforms (Magicbricks, 99acres) use similar models to show instant price estimates.")


# ─────────────────────────────────────────────
# PROJECT 2: Spam Detector
# ─────────────────────────────────────────────
def _spam_detector():
    st.markdown(section_header("📧", "Email Spam Detector",
                "Logistic Regression — NLP Features"), unsafe_allow_html=True)

    df = get_spam_data(500)
    X  = df[["Word_Count", "Links", "Capital_Ratio"]].values
    y  = df["Is_Spam"].values
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    model = LogisticRegression(max_iter=500)
    model.fit(Xs, y)

    col_in, col_out = st.columns(2)
    with col_in:
        wc   = st.slider("📝 Word Count",        50, 500, 150)
        lnk  = st.slider("🔗 Number of Links",    0,  20,   2)
        cap  = st.slider("🔠 Capital Letter Ratio", 0.0, 1.0, 0.15, 0.05)

    inp  = sc.transform([[wc, lnk, cap]])
    prob = model.predict_proba(inp)[0][1]
    pred = int(prob > 0.5)

    with col_out:
        color = "#FF6584" if pred else "#43E97B"
        label = "⚠️ SPAM!" if pred else "✅ Legit Email"
        st.markdown(f"""
        <div class="ml-card" style="text-align:center;padding:36px;border-color:{color};">
            <div style="font-size:3rem;">{'🚫' if pred else '✉️'}</div>
            <div style="font-size:1.8rem;font-weight:700;color:{color};margin:12px 0;">{label}</div>
            <div style="color:#8892A4;">Confidence: <b style="color:{color};">{max(prob,1-prob)*100:.1f}%</b></div>
        </div>
        """, unsafe_allow_html=True)

    _business_use("Gmail, Outlook, and Yahoo use ML-based spam filters processing billions of emails daily.")


# ─────────────────────────────────────────────
# PROJECT 3: Student Performance
# ─────────────────────────────────────────────
def _student_perf():
    st.markdown(section_header("🎓", "Student Performance Predictor",
                "Linear Regression + Pass/Fail Classifier"), unsafe_allow_html=True)

    df = get_student_data(400)
    X  = df[["Study_Hours", "Sleep_Hours", "Attendance_%"]].values
    y_score  = df["Score"].values
    y_pass   = df["Passed"].values

    reg = LinearRegression()
    reg.fit(X, y_score)
    clf = LogisticRegression(max_iter=500)
    clf.fit(X, y_pass)

    col_in, col_out = st.columns(2)
    with col_in:
        study  = st.slider("📚 Study Hours / day",    0, 10, 5)
        sleep  = st.slider("😴 Sleep Hours / night",  4, 10, 7)
        attend = st.slider("🏫 Attendance %",         50, 100, 80)

    pred_score = reg.predict([[study, sleep, attend]])[0]
    pred_pass  = clf.predict([[study, sleep, attend]])[0]
    pred_prob  = clf.predict_proba([[study, sleep, attend]])[0][1]
    pred_score = np.clip(pred_score, 0, 100)

    with col_out:
        grade = "A+" if pred_score>=90 else "A" if pred_score>=80 else "B" if pred_score>=70 else "C" if pred_score>=60 else "D" if pred_score>=40 else "F"
        color = "#43E97B" if pred_pass else "#FF6584"
        st.markdown(f"""
        <div class="ml-card" style="text-align:center;padding:24px;">
            <div style="font-size:3rem;margin-bottom:8px;">🎓</div>
            <div style="display:flex;justify-content:space-around;margin-top:16px;">
                <div>
                    <div style="color:#8892A4;font-size:0.8rem;">Predicted Score</div>
                    <div style="font-size:2rem;font-weight:700;color:#6C63FF;">{pred_score:.1f}</div>
                    <div style="font-size:1.2rem;color:#B48EFF;">{grade}</div>
                </div>
                <div>
                    <div style="color:#8892A4;font-size:0.8rem;">Result</div>
                    <div style="font-size:1.6rem;font-weight:700;color:{color};">
                        {'✅ PASS' if pred_pass else '❌ FAIL'}
                    </div>
                    <div style="color:{color};font-size:0.85rem;">{pred_prob*100:.0f}% chance</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    _business_use("EdTech platforms use this to identify at-risk students early and provide personalised support.")


# ─────────────────────────────────────────────
# PROJECT 4: Loan Approval
# ─────────────────────────────────────────────
def _loan_approval():
    st.markdown(section_header("💳", "Loan Approval Predictor",
                "Random Forest — Banking"), unsafe_allow_html=True)

    df = get_loan_data(600)
    X  = df[["Income", "Credit_Score", "Debt_Ratio"]].values
    y  = df["Approved"].values
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(Xs, y)

    col_in, col_out = st.columns(2)
    with col_in:
        income = st.number_input("💰 Annual Income ($)", 20_000, 300_000, 65_000, 5_000)
        credit = st.slider("📊 Credit Score",   300, 850, 680)
        debt   = st.slider("💸 Debt Ratio",     0.0,  0.9, 0.3, 0.05)

    inp  = sc.transform([[income, credit, debt]])
    prob = model.predict_proba(inp)[0][1]
    pred = int(prob >= 0.5)

    with col_out:
        color = "#43E97B" if pred else "#FF6584"
        label = "✅ APPROVED" if pred else "❌ REJECTED"
        bar_w = int(prob * 100)
        st.markdown(f"""
        <div class="ml-card" style="text-align:center;padding:36px;border-color:{color};">
            <div style="font-size:3rem;">{'🏦' if pred else '🚫'}</div>
            <div style="font-size:1.8rem;font-weight:700;color:{color};margin:12px 0;">{label}</div>
            <div style="color:#8892A4;margin-bottom:8px;">Approval Probability</div>
            <div class="progress-container">
                <div class="progress-bar" style="width:{bar_w}%;background:{'linear-gradient(90deg,#43E97B,#38F9D7)' if pred else 'linear-gradient(90deg,#FF6584,#FF8EE8)'};"></div>
            </div>
            <div style="color:{color};font-weight:700;font-size:1.4rem;margin-top:8px;">{prob*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    fi = model.feature_importances_
    feat_names = ["Income", "Credit Score", "Debt Ratio"]
    for name, importance in zip(feat_names, fi):
        st.markdown(f"**{name}** — Impact: {importance*100:.0f}%")
        st.progress(float(importance))

    _business_use("HDFC, SBI, and global banks use Random Forest / XGBoost for instant loan decisions.")


# ─────────────────────────────────────────────
# PROJECT 5: Sentiment
# ─────────────────────────────────────────────
def _sentiment():
    st.markdown(section_header("💬", "Sentiment Analyser",
                "Rule-based NLP Demo"), unsafe_allow_html=True)

    positive_words = {"great", "excellent", "amazing", "good", "fantastic", "love",
                      "wonderful", "best", "happy", "outstanding", "perfect", "brilliant",
                      "superb", "impressive", "awesome", "delighted", "pleased"}
    negative_words = {"bad", "terrible", "awful", "hate", "worst", "poor",
                      "disappointing", "horrible", "disgusting", "mediocre",
                      "boring", "useless", "dreadful", "pathetic", "frustrating"}

    text = st.text_area("✍️ Enter a review:", "This product is absolutely amazing! I love the quality and service.", height=120)

    if st.button("🔍 Analyse Sentiment"):
        words = text.lower().split()
        pos_c = sum(1 for w in words if w.strip(".,!?") in positive_words)
        neg_c = sum(1 for w in words if w.strip(".,!?") in negative_words)
        total = pos_c + neg_c

        score = pos_c / total if total > 0 else 0.5
        if score > 0.6:
            label, color, emoji = "POSITIVE 😊", "#43E97B", "😊"
        elif score < 0.4:
            label, color, emoji = "NEGATIVE 😞", "#FF6584", "😞"
        else:
            label, color, emoji = "NEUTRAL 😐", "#FFA94D", "😐"

        st.markdown(f"""
        <div class="ml-card" style="text-align:center;border-color:{color};padding:32px;">
            <div style="font-size:3rem;">{emoji}</div>
            <div style="font-size:1.8rem;font-weight:700;color:{color};margin:12px 0;">{label}</div>
            <div style="color:#8892A4;">
                Positive words: <b style="color:#43E97B;">{pos_c}</b> &nbsp;|&nbsp;
                Negative words: <b style="color:#FF6584;">{neg_c}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

        found_pos = [w for w in words if w.strip(".,!?") in positive_words]
        found_neg = [w for w in words if w.strip(".,!?") in negative_words]
        if found_pos:
            st.success(f"✅ Positive words: {', '.join(found_pos)}")
        if found_neg:
            st.error(f"❌ Negative words: {', '.join(found_neg)}")

    _business_use("Amazon, Flipkart, and Google use sentiment analysis on millions of reviews to improve products and rankings.")


# ─────────────────────────────────────────────
# PROJECT 6: Fake News Demo
# ─────────────────────────────────────────────
def _fake_news_demo():
    st.markdown(section_header("📰", "Fake News Detector (Demo)",
                "NLP + Feature Engineering"), unsafe_allow_html=True)

    st.markdown(concept_box(
        "How Fake News Detection Works",
        "Real systems use NLP to extract features like: "
        "writing style, source credibility, factual claims, emotional language, "
        "and compare against verified fact databases. "
        "This demo uses a rule-based approach for illustration.",
        "📰"
    ), unsafe_allow_html=True)

    headline = st.text_input("📰 Enter a news headline:",
                              "Scientists discover water on Mars using NASA satellite data")

    clickbait_words = {"shocking", "unbelievable", "you won't believe", "secret", "banned",
                       "miracle", "cure", "explosive", "bombshell", "revealed", "hoax"}
    credible_words  = {"scientists", "researchers", "nasa", "who", "study", "published",
                       "journal", "university", "data", "evidence", "analysis"}

    if st.button("🔍 Check Headline"):
        words   = headline.lower().split()
        cb_hits = sum(1 for w in words if w in clickbait_words)
        cr_hits = sum(1 for w in words if w in credible_words)
        risk    = cb_hits * 0.35 - cr_hits * 0.2
        risk    = np.clip(risk + 0.3, 0, 1)

        label = "⚠️ POSSIBLY FAKE" if risk > 0.5 else "✅ LIKELY REAL"
        color = "#FF6584" if risk > 0.5 else "#43E97B"

        st.markdown(f"""
        <div class="ml-card" style="text-align:center;border-color:{color};padding:28px;">
            <div style="font-size:1.6rem;font-weight:700;color:{color};">{label}</div>
            <div style="color:#8892A4;margin-top:8px;">
                Risk Score: <b style="color:{color};">{risk*100:.0f}%</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.warning("⚠️ This is a simplified demo. Real fake news detection uses deep NLP models and fact databases.")

    _business_use("Twitter, Facebook, and WhatsApp use AI-based fake news detectors to flag misinformation at scale.")


def _business_use(text: str):
    st.markdown(f"""
    <div class="info-box" style="margin-top:16px;">
        <b>💼 Real-World Business Use:</b><br>
        <span style="color:#8892A4;font-size:0.88rem;">{text}</span>
    </div>
    """, unsafe_allow_html=True)
