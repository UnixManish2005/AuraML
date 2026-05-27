"""
pages/training_lab.py
Model Training Lab — Upload, Select, Train, Evaluate
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io, joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, mean_squared_error,
                              r2_score, confusion_matrix,
                              classification_report)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from utils.styles import hero, concept_box, section_header, info_box, warning_box
from utils.data_helpers import SAMPLE_DATASETS, load_sample, PLOTLY_LAYOUT, COLOR_SEQ


CLASSIFIERS = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree":       DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN":                 KNeighborsClassifier(n_neighbors=5),
    "SVM":                 SVC(probability=True, random_state=42),
}
REGRESSORS = {
    "Linear Regression":   LinearRegression(),
    "Decision Tree":       DecisionTreeRegressor(max_depth=5, random_state=42),
    "Random Forest":       RandomForestRegressor(n_estimators=100, random_state=42),
}


def render():
    st.markdown(hero(
        "Model Training Lab 🔬",
        "Upload your dataset, pick a target and algorithm, then train and evaluate your model live!",
        "🔬"
    ), unsafe_allow_html=True)

    # ── Step 1: Load Data ─────────────────────
    st.markdown(section_header("1️⃣", "Load Dataset"), unsafe_allow_html=True)
    source = st.radio("Data source:", ["Sample Dataset", "Upload CSV"], horizontal=True)

    df = None
    if source == "Sample Dataset":
        name = st.selectbox("Choose sample:", list(SAMPLE_DATASETS.keys()))
        df   = load_sample(name)
    else:
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)

    if df is None:
        st.markdown(warning_box("📂 Load a dataset to begin training."), unsafe_allow_html=True)
        return

    st.dataframe(df.head(5), use_container_width=True)

    # ── Step 2: Configure ─────────────────────
    st.markdown(section_header("2️⃣", "Configure Training"), unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        target = st.selectbox("🎯 Target Column:", df.columns.tolist())
    with col2:
        task_type = st.radio("Task Type:", ["Classification", "Regression"])
    with col3:
        test_pct  = st.slider("Test Split %", 10, 40, 20)

    algo_options = list(CLASSIFIERS.keys()) if task_type == "Classification" else list(REGRESSORS.keys())
    algorithm    = st.selectbox("⚙️ Algorithm:", algo_options)

    feature_cols = [c for c in df.select_dtypes(include=np.number).columns if c != target]
    if not feature_cols:
        st.error("No numeric feature columns found!")
        return

    feat_sel = st.multiselect("📊 Feature Columns:", feature_cols, default=feature_cols)
    if not feat_sel:
        st.warning("Select at least one feature.")
        return

    # ── Step 3: Train ─────────────────────────
    st.markdown(section_header("3️⃣", "Train Model"), unsafe_allow_html=True)

    if st.button("🚀 Train Model!", use_container_width=True):
        with st.spinner("Training in progress..."):
            X = df[feat_sel].fillna(df[feat_sel].median()).values
            y_raw = df[target].fillna(df[target].mode()[0])

            # Encode target if classification
            le = LabelEncoder()
            if task_type == "Classification":
                y = le.fit_transform(y_raw)
            else:
                y = y_raw.astype(float).values

            sc = StandardScaler()
            X  = sc.fit_transform(X)
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_pct/100, random_state=42)

            models_dict = CLASSIFIERS if task_type == "Classification" else REGRESSORS
            model = models_dict[algorithm]
            model.fit(X_tr, y_tr)

            st.session_state["lab_model"]   = model
            st.session_state["lab_scaler"]  = sc
            st.session_state["lab_le"]      = le if task_type == "Classification" else None
            st.session_state["lab_feats"]   = feat_sel

        # ── Step 4: Results ───────────────────
        st.markdown(section_header("4️⃣", "Results"), unsafe_allow_html=True)
        y_pred = model.predict(X_te)

        if task_type == "Classification":
            acc  = accuracy_score(y_te, y_pred)
            cv   = cross_val_score(model, X, y, cv=5).mean()

            m1, m2, m3 = st.columns(3)
            m1.metric("Test Accuracy",   f"{acc*100:.1f}%")
            m2.metric("CV Accuracy",     f"{cv*100:.1f}%")
            m3.metric("Training Samples", str(len(y_tr)))

            # Confusion matrix
            # Confusion matrix
            cm = confusion_matrix(y_te, y_pred)

            # Safe labels
            if le is not None:
                lbls = [str(c) for c in le.classes_]
            else:
                lbls = [str(i) for i in range(cm.shape[0])]

            # Ensure labels length matches matrix size
            if len(lbls) != cm.shape[0]:
                lbls = [str(i) for i in range(cm.shape[0])]

            fig_cm = px.imshow(
                cm,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="Purples",
                x=lbls,
                y=lbls,
                title="Confusion Matrix",
                labels=dict(
                    x="Predicted",
                    y="Actual",
                    color="Count"
                )
            )

            fig_cm.update_layout(
                **PLOTLY_LAYOUT,
                height=350
            )

            st.plotly_chart(
                fig_cm,
                use_container_width=True,
                key="training_lab_confusion_matrix"
            )

            # Classification report
            report = classification_report(y_te, y_pred,
                                            target_names=lbls, output_dict=True)
            rdf = pd.DataFrame(report).transpose()
            st.dataframe(rdf.round(3), use_container_width=True)

        else:  # Regression
            rmse = np.sqrt(mean_squared_error(y_te, y_pred))
            r2   = r2_score(y_te, y_pred)

            m1, m2, m3 = st.columns(3)
            m1.metric("RMSE",  f"{rmse:.2f}")
            m2.metric("R² Score", f"{r2:.3f}")
            m3.metric("Training Samples", str(len(y_tr)))

            # Actual vs Predicted
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(x=y_te, y=y_pred, mode='markers',
                                           marker=dict(color=COLOR_SEQ[0], size=6, opacity=0.6),
                                           name='Predictions'))
            min_v = min(y_te.min(), y_pred.min())
            max_v = max(y_te.max(), y_pred.max())
            fig_pred.add_trace(go.Scatter(x=[min_v, max_v], y=[min_v, max_v],
                                           mode='lines', line=dict(color=COLOR_SEQ[1], dash='dash'),
                                           name='Perfect Fit'))
            fig_pred.update_layout(**PLOTLY_LAYOUT, title="Actual vs Predicted",
                                    xaxis_title="Actual", yaxis_title="Predicted", height=380)
            st.plotly_chart(fig_pred, use_container_width=True)

        # Feature Importance
        if hasattr(model, "feature_importances_"):
            fi = model.feature_importances_
            fig_fi = px.bar(x=feat_sel, y=fi,
                            title="Feature Importance",
                            labels={"x": "Feature", "y": "Importance"},
                            color=fi, color_continuous_scale="Purples")
            fig_fi.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig_fi, use_container_width=True)

        # Download model
        buf = io.BytesIO()
        joblib.dump(model, buf)
        buf.seek(0)
        st.download_button(
            label="⬇️ Download Trained Model (.pkl)",
            data=buf,
            file_name=f"{algorithm.replace(' ', '_')}_model.pkl",
            mime="application/octet-stream"
        )

        st.markdown(info_box(
            f"✅ <b>{algorithm}</b> trained successfully! "
            f"Download the .pkl file and load it with <code>joblib.load('model.pkl')</code> in Python."
        ), unsafe_allow_html=True)
