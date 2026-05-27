"""ML Playground — upload datasets, train models, view results interactively."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.styles import section_header, alert

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, confusion_matrix, mean_squared_error,
                              r2_score, classification_report)

SAMPLE_DATASETS = {
    "Iris (Classification)": "iris",
    "Diabetes (Regression)":  "diabetes",
    "Breast Cancer (Binary)": "breast_cancer",
}

CLASSIFIERS = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree":       DecisionTreeClassifier(max_depth=5),
    "Random Forest":       RandomForestClassifier(n_estimators=100),
    "KNN":                 KNeighborsClassifier(),
    "SVM":                 SVC(probability=True),
}


def _load_sample(name):
    from sklearn import datasets
    loaders = {
        "iris":         datasets.load_iris,
        "diabetes":     datasets.load_diabetes,
        "breast_cancer":datasets.load_breast_cancer,
    }
    data = loaders[name]()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    if hasattr(data, "target"):
        df["target"] = data.target
    return df


def show_playground(user):
    section_header("🔬 ML Playground",
                   "Upload data, pick an algorithm, train and visualise — in seconds.")

    tab1, tab2, tab3 = st.tabs(["📁 Data", "🤖 Train Model", "📊 Visualise"])

    # ── Tab 1: Load Data ───────────────────────────────────────────────────
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 📂 Load a Dataset")
            sample = st.selectbox("Choose sample dataset", list(SAMPLE_DATASETS.keys()))
            if st.button("Load Sample Dataset"):
                key = SAMPLE_DATASETS[sample]
                df = _load_sample(key)
                st.session_state["playground_df"] = df
                st.success(f"✅ Loaded {sample} — {df.shape[0]} rows × {df.shape[1]} cols")
        with c2:
            st.markdown("#### 📤 Or Upload Your Own")
            uploaded = st.file_uploader("Upload CSV", type=["csv"])
            if uploaded:
                df = pd.read_csv(uploaded)
                st.session_state["playground_df"] = df
                st.success(f"✅ Uploaded — {df.shape[0]} rows × {df.shape[1]} cols")

        if "playground_df" in st.session_state:
            df = st.session_state["playground_df"]
            st.markdown("#### 🔍 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Rows",    df.shape[0])
            c2.metric("Columns", df.shape[1])
            c3.metric("Missing", int(df.isnull().sum().sum()))

            st.markdown("#### 📊 Feature Distributions")
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            if num_cols:
                sel = st.selectbox("Feature to plot", num_cols, key="dist_col")
                fig = px.histogram(df, x=sel, nbins=30,
                                   color_discrete_sequence=["#6C63FF"])
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  font=dict(color="#E8E8F0"), height=280,
                                  margin=dict(l=0,r=0,t=10,b=10),
                                  xaxis=dict(showgrid=False,color="#8888AA"),
                                  yaxis=dict(showgrid=False,color="#8888AA"))
                st.plotly_chart(fig, use_container_width=True)

    # ── Tab 2: Train ───────────────────────────────────────────────────────
    with tab2:
        if "playground_df" not in st.session_state:
            alert("Load a dataset in the **Data** tab first.", "warning")
            return

        df = st.session_state["playground_df"]
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        all_cols = df.columns.tolist()

        st.markdown("#### ⚙️ Configure Training")
        c1, c2 = st.columns(2)
        with c1:
            target = st.selectbox("🎯 Target column", all_cols, index=len(all_cols)-1)
            task = st.selectbox("📋 Task type", ["Classification", "Regression", "Clustering"])
        with c2:
            features = st.multiselect("📐 Feature columns",
                                      [c for c in num_cols if c != target],
                                      default=[c for c in num_cols[:4] if c != target])
            test_size = st.slider("Test set size (%)", 10, 40, 20)

        if task == "Classification":
            algo = st.selectbox("🤖 Algorithm", list(CLASSIFIERS.keys()))
        elif task == "Regression":
            algo = "Linear Regression"
            st.info("Using **Linear Regression** for regression tasks.")
        else:
            n_clusters = st.slider("Number of clusters (K)", 2, 10, 3)
            algo = "K-Means"

        if st.button("🚀 Train Model", use_container_width=True):
            if not features:
                alert("Select at least one feature column.", "warning")
                return
            _train_model(df, features, target, task, algo,
                         test_size/100,
                         n_clusters if task == "Clustering" else 3)

    # ── Tab 3: Visualise ───────────────────────────────────────────────────
    with tab3:
        if "playground_df" not in st.session_state:
            alert("Load a dataset first.", "warning")
            return
        df = st.session_state["playground_df"]
        num_cols = df.select_dtypes(include=np.number).columns.tolist()

        st.markdown("#### 📈 Scatter Plot Explorer")
        c1, c2, c3 = st.columns(3)
        with c1: x_col = st.selectbox("X axis", num_cols, key="scat_x")
        with c2: y_col = st.selectbox("Y axis", num_cols, index=min(1, len(num_cols)-1), key="scat_y")
        with c3:
            color_col = st.selectbox("Colour by", ["None"] + df.columns.tolist(), key="scat_c")

        fig = px.scatter(df, x=x_col, y=y_col,
                         color=None if color_col == "None" else color_col,
                         color_discrete_sequence=px.colors.qualitative.Vivid,
                         opacity=0.75)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font=dict(color="#E8E8F0"), height=380,
                          margin=dict(l=0,r=0,t=10,b=10),
                          xaxis=dict(showgrid=False,color="#8888AA"),
                          yaxis=dict(showgrid=False,color="#8888AA"),
                          legend=dict(bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig, use_container_width=True)

        # Correlation heatmap
        st.markdown("#### 🔥 Correlation Heatmap")
        corr = df[num_cols].corr()
        fig2 = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                         zmin=-1, zmax=1)
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font=dict(color="#E8E8F0"), height=420,
                           margin=dict(l=0,r=0,t=10,b=10))
        st.plotly_chart(fig2, use_container_width=True)


def _train_model(df, features, target, task, algo, test_size, n_clusters):
    with st.spinner("Training model..."):
        try:
            X = df[features].fillna(df[features].median())
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            if task == "Clustering":
                km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = km.fit_predict(X_scaled)
                st.success(f"✅ K-Means trained! Inertia: {km.inertia_:.2f}")
                if len(features) >= 2:
                    fig = px.scatter(x=X_scaled[:, 0], y=X_scaled[:, 1],
                                     color=labels.astype(str),
                                     labels={"x": features[0], "y": features[1]},
                                     color_discrete_sequence=px.colors.qualitative.Vivid,
                                     title="Cluster Assignments")
                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                      plot_bgcolor="rgba(0,0,0,0)",
                                      font=dict(color="#E8E8F0"), height=380,
                                      legend=dict(bgcolor="rgba(0,0,0,0)"),
                                      xaxis=dict(showgrid=False,color="#8888AA"),
                                      yaxis=dict(showgrid=False,color="#8888AA"))
                    st.plotly_chart(fig, use_container_width=True)
                return

            y_raw = df[target]
            if task == "Classification":
                le = LabelEncoder()
                y = le.fit_transform(y_raw.astype(str))
            else:
                y = y_raw.values

            X_tr, X_te, y_tr, y_te = train_test_split(
                X_scaled, y, test_size=test_size, random_state=42)

            if task == "Classification":
                model = CLASSIFIERS[algo]
                model.fit(X_tr, y_tr)
                preds = model.predict(X_te)
                acc = accuracy_score(y_te, preds)

                st.success(f"✅ **{algo}** trained!")
                c1, c2, c3 = st.columns(3)
                c1.metric("Accuracy",   f"{acc:.2%}")
                c2.metric("Train size", len(X_tr))
                c3.metric("Test size",  len(X_te))

                # Confusion matrix
                cm = confusion_matrix(y_te, preds)
                labels_ = [str(l) for l in le.classes_[:cm.shape[0]]]
                fig = px.imshow(cm, text_auto=True,
                                color_continuous_scale=["#0F0F1A","#6C63FF"],
                                x=labels_, y=labels_,
                                labels=dict(x="Predicted", y="Actual"),
                                title="Confusion Matrix")
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                  plot_bgcolor="rgba(0,0,0,0)",
                                  font=dict(color="#E8E8F0"), height=360,
                                  margin=dict(l=0,r=0,t=40,b=10))
                st.plotly_chart(fig, use_container_width=True)

                # Feature importance for tree-based models
                if hasattr(model, "feature_importances_"):
                    imp = pd.Series(model.feature_importances_, index=features).sort_values()
                    fig2 = px.bar(imp, orientation="h",
                                  color=imp.values,
                                  color_continuous_scale=["#1A1A2E","#6C63FF","#FF6584"],
                                  title="Feature Importances")
                    fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                       plot_bgcolor="rgba(0,0,0,0)",
                                       font=dict(color="#E8E8F0"), height=280,
                                       coloraxis_showscale=False,
                                       margin=dict(l=0,r=0,t=40,b=10),
                                       xaxis=dict(showgrid=False,color="#8888AA"),
                                       yaxis=dict(showgrid=False,color="#E8E8F0"))
                    st.plotly_chart(fig2, use_container_width=True)

            else:
                model = LinearRegression()
                model.fit(X_tr, y_tr)
                preds = model.predict(X_te)
                rmse = np.sqrt(mean_squared_error(y_te, preds))
                r2   = r2_score(y_te, preds)

                st.success("✅ **Linear Regression** trained!")
                c1, c2 = st.columns(2)
                c1.metric("RMSE", f"{rmse:.4f}")
                c2.metric("R²",   f"{r2:.4f}")

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=y_te, y=preds, mode="markers",
                                         marker=dict(color="#6C63FF", size=7, opacity=0.7),
                                         name="Predicted vs Actual"))
                line_r = [min(y_te), max(y_te)]
                fig.add_trace(go.Scatter(x=line_r, y=line_r, mode="lines",
                                         line=dict(color="#FF6584", width=2, dash="dash"),
                                         name="Perfect fit"))
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                  plot_bgcolor="rgba(0,0,0,0)",
                                  font=dict(color="#E8E8F0"), height=360,
                                  xaxis_title="Actual", yaxis_title="Predicted",
                                  legend=dict(bgcolor="rgba(0,0,0,0)"),
                                  xaxis=dict(showgrid=False,color="#8888AA"),
                                  yaxis=dict(showgrid=False,color="#8888AA"),
                                  margin=dict(l=0,r=0,t=10,b=10))
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Training failed: {e}")
