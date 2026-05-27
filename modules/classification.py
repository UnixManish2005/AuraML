"""
pages/classification.py
Logistic Regression + Classification Visualizer
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, confusion_matrix)
from utils.styles import hero, concept_box, section_header, success_box, info_box
from utils.data_helpers import get_classification_2d, get_spam_data, PLOTLY_LAYOUT, COLOR_SEQ


def render():
    st.markdown(hero(
        "Classification 🎯",
        "Draw decision boundaries! See how Logistic Regression separates classes — spam vs not-spam, pass vs fail.",
        "🎯"
    ), unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(
        ["🔵 Decision Boundary", "📧 Spam Detector", "💡 Concepts"]
    )

    # ─────────────────────────────────────────────
    # TAB 1: Decision Boundary
    # ─────────────────────────────────────────────
    with tab1:
        st.markdown(section_header("🔵", "Decision Boundary Playground"), unsafe_allow_html=True)

        col_ctrl, col_chart = st.columns([1, 3])
        with col_ctrl:
            n_pts     = st.slider("Data Points", 100, 500, 200, key="cls_n")
            test_size = st.slider("Test %", 10, 40, 20, key="cls_test")
            threshold = st.slider("Decision Threshold", 0.1, 0.9, 0.5, 0.05, key="cls_thr")
            n_centers = st.radio("Classes", [2, 3], horizontal=True)

        X, y = get_classification_2d(n_pts, n_centers)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size/100, random_state=42)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_tr, y_tr)

        y_prob = model.predict_proba(X_te)[:, 1] if n_centers == 2 else None
        y_pred = (y_prob >= threshold).astype(int) if y_prob is not None else model.predict(X_te)

        acc  = accuracy_score(y_te, y_pred)
        prec = precision_score(y_te, y_pred, average='weighted', zero_division=0)
        rec  = recall_score(y_te, y_pred, average='weighted', zero_division=0)
        f1   = f1_score(y_te, y_pred, average='weighted', zero_division=0)

        with col_chart:
            fig = _plot_decision_boundary(model, X, y, X_te, y_te)
            st.plotly_chart(fig, use_container_width=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy",  f"{acc*100:.1f}%")
        m2.metric("Precision", f"{prec*100:.1f}%")
        m3.metric("Recall",    f"{rec*100:.1f}%")
        m4.metric("F1-Score",  f"{f1*100:.1f}%")

        # Confusion Matrix
        cm = confusion_matrix(y_te, y_pred)
        fig_cm = px.imshow(cm, text_auto=True, aspect="auto",
                           color_continuous_scale="Purples",
                           title="Confusion Matrix",
                           labels=dict(x="Predicted", y="Actual"))
        fig_cm.update_layout(**PLOTLY_LAYOUT, height=350)
        st.plotly_chart(fig_cm, use_container_width=True)

        # Sigmoid curve (binary only)
        if n_centers == 2 and y_prob is not None:
            st.markdown(section_header("〽️", "Sigmoid Curve"), unsafe_allow_html=True)
            z = np.linspace(-8, 8, 300)
            sig = 1 / (1 + np.exp(-z))
            fig_sig = go.Figure()
            fig_sig.add_trace(go.Scatter(x=z, y=sig,
                                          mode='lines',
                                          line=dict(color=COLOR_SEQ[0], width=3),
                                          name='σ(z)'))
            fig_sig.add_hline(y=threshold, line_dash="dash",
                               line_color=COLOR_SEQ[2],
                               annotation_text=f"Threshold = {threshold}")
            fig_sig.add_vline(x=0, line_dash="dot", line_color="#8892A4")
            fig_sig.update_layout(**PLOTLY_LAYOUT,
                                   title="Sigmoid Function: Maps score → Probability (0–1)",
                                   xaxis_title="z (linear combination of features)",
                                   yaxis_title="Probability", height=300)
            st.plotly_chart(fig_sig, use_container_width=True)

    # ─────────────────────────────────────────────
    # TAB 2: Spam Detector
    # ─────────────────────────────────────────────
    with tab2:
        st.markdown(section_header("📧", "Email Spam Detector",
                    "Adjust features and see spam probability"), unsafe_allow_html=True)

        df_spam = get_spam_data(400)
        X_s = df_spam[["Word_Count", "Links", "Capital_Ratio"]].values
        y_s = df_spam["Is_Spam"].values
        spam_model = LogisticRegression(max_iter=500)
        spam_model.fit(X_s, y_s)

        col_in, col_out = st.columns(2)
        with col_in:
            wc   = st.slider("Word Count",       50, 500, 200)
            lnk  = st.slider("# Links in Email",  0,  20,   3)
            cap  = st.slider("CAPS Ratio",       0.0,  1.0, 0.2, 0.05)

        prob = spam_model.predict_proba([[wc, lnk, cap]])[0]
        spam_p = prob[1]

        with col_out:
            color = "#FF6584" if spam_p > 0.5 else "#43E97B"
            label = "⚠️ SPAM" if spam_p > 0.5 else "✅ NOT SPAM"
            st.markdown(f"""
            <div class="ml-card" style="text-align:center;padding:40px;border-color:{color};">
                <div style="font-size:3rem;margin-bottom:12px;">{'🚫' if spam_p>0.5 else '✉️'}</div>
                <div style="font-size:1.6rem;font-weight:700;color:{color};">{label}</div>
                <div style="margin-top:16px;">
                    <div style="color:#8892A4;font-size:0.85rem;">Spam Probability</div>
                    <div style="font-size:2rem;font-weight:700;color:{color};font-family:var(--font-mono);">
                        {spam_p*100:.1f}%
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Feature importance bar
        coef = spam_model.coef_[0]
        feat_names = ["Word Count", "Links", "Capital Ratio"]
        fig_imp = px.bar(x=feat_names, y=np.abs(coef),
                         color=coef, color_continuous_scale="RdBu",
                         title="Feature Importance (Logistic Regression Coefficients)",
                         labels={"x": "Feature", "y": "|Coefficient|"})
        fig_imp.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig_imp, use_container_width=True)

    # ─────────────────────────────────────────────
    # TAB 3: Concepts
    # ─────────────────────────────────────────────
    with tab3:
        st.markdown(section_header("💡", "Classification Concepts"), unsafe_allow_html=True)

        items = [
            ("🎯", "Decision Boundary", "#6C63FF",
             "The line (or curve) that separates classes. Points on one side = Class A, other side = Class B."),
            ("〽️", "Sigmoid Function", "#43E97B",
             "σ(z) = 1 / (1 + e⁻ᶻ). Squashes any number into 0–1 range. Perfect for probabilities!"),
            ("✅", "True Positive (TP)", "#38F9D7",
             "Model predicted Positive and it was actually Positive. Correct spam detection!"),
            ("❌", "False Positive (FP)", "#FFA94D",
             "Model predicted Positive but it was actually Negative. Legitimate email marked as spam."),
            ("📊", "Precision", "#B48EFF",
             "Of all predicted positives, how many were actually positive? TP / (TP + FP)"),
            ("🔁", "Recall (Sensitivity)", "#FF8EE8",
             "Of all actual positives, how many did we catch? TP / (TP + FN)"),
        ]

        rows = [items[:3], items[3:]]
        for row in rows:
            cols = st.columns(3)
            for col, (icon, title, color, desc) in zip(cols, row):
                col.markdown(f"""
                <div class="ml-card" style="border-top:3px solid {color};min-height:130px;">
                    <div style="font-size:1.8rem;margin-bottom:8px;">{icon}</div>
                    <h4 style="color:{color};margin:0 0 8px 0;font-size:0.92rem;">{title}</h4>
                    <p style="color:#8892A4;font-size:0.82rem;margin:0;line-height:1.5;">{desc}</p>
                </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _plot_decision_boundary(model, X, y, X_te, y_te):
    h = 0.05
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    colors = [COLOR_SEQ[i % len(COLOR_SEQ)] for i in y]
    te_colors = [COLOR_SEQ[i % len(COLOR_SEQ)] for i in y_te]

    fig = go.Figure()
    fig.add_trace(go.Contour(
        x=np.arange(x_min, x_max, h),
        y=np.arange(y_min, y_max, h),
        z=Z, showscale=False, opacity=0.3,
        colorscale=[[0, "rgba(108,99,255,0.3)"], [0.5, "rgba(255,101,132,0.3)"],
                    [1, "rgba(67,233,123,0.3)"]],
        contours_coloring='fill',
        name='Decision Region'
    ))
    fig.add_trace(go.Scatter(
        x=X[:, 0], y=X[:, 1], mode='markers',
        marker=dict(color=colors, size=6, opacity=0.5,
                    line=dict(color='white', width=0.3)),
        name='Training Data'
    ))
    fig.add_trace(go.Scatter(
        x=X_te[:, 0], y=X_te[:, 1], mode='markers',
        marker=dict(color=te_colors, size=9, opacity=1,
                    symbol='star', line=dict(color='white', width=1)),
        name='Test Data'
    ))
    fig.update_layout(**PLOTLY_LAYOUT, title="Decision Boundary",
                       xaxis_title="Feature 1", yaxis_title="Feature 2",
                       height=420)
    return fig
