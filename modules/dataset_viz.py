"""
pages/dataset_viz.py
Dataset Visualization & Exploration Page
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.styles import hero, concept_box, section_header, info_box, warning_box
from utils.data_helpers import SAMPLE_DATASETS, load_sample, PLOTLY_LAYOUT, COLOR_SEQ


def render():
    st.markdown(hero(
        "Dataset Explorer 📊",
        "Upload your own dataset or try a sample. Learn what features, targets, and data distributions mean.",
        "📊"
    ), unsafe_allow_html=True)

    # ── Load Dataset ──────────────────────────────
    st.markdown(section_header("📂", "Load Dataset"), unsafe_allow_html=True)

    source = st.radio("Data source:", ["Sample Dataset", "Upload CSV"], horizontal=True)

    df = None
    if source == "Sample Dataset":
        name = st.selectbox("Choose a sample:", list(SAMPLE_DATASETS.keys()))
        df = load_sample(name)
        st.success(f"✅ Loaded **{name}** — {df.shape[0]} rows × {df.shape[1]} columns")
    else:
        uploaded = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
            st.success(f"✅ Uploaded — {df.shape[0]} rows × {df.shape[1]} columns")
        else:
            st.markdown(warning_box("📂 Upload a CSV file or switch to 'Sample Dataset' to begin."),
                        unsafe_allow_html=True)

    if df is None:
        return

    # ── Overview Tabs ─────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📋 Preview", "📈 Distributions", "🔥 Correlations", "📦 Box Plots", "💡 Insights"]
    )

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    # ── Tab 1: Preview ────────────────────────────
    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows",    df.shape[0])
        c2.metric("Columns", df.shape[1])
        c3.metric("Numeric", len(num_cols))
        c4.metric("Missing", int(df.isnull().sum().sum()))

        st.markdown("#### 📋 First 10 Rows")
        st.dataframe(df.head(10), use_container_width=True)

        st.markdown("#### 📊 Statistical Summary")
        st.dataframe(df.describe().round(2), use_container_width=True)

        miss = df.isnull().sum()
        miss = miss[miss > 0]
        if len(miss):
            st.markdown("#### ⚠️ Missing Values")
            fig = px.bar(x=miss.index, y=miss.values,
                         labels={"x": "Column", "y": "Missing Count"},
                         title="Missing Values per Column",
                         color=miss.values, color_continuous_scale="Reds")
            fig.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No missing values — dataset is clean!")

    # ── Tab 2: Distributions ──────────────────────
    with tab2:
        if not num_cols:
            st.info("No numeric columns found.")
        else:
            col_sel = st.selectbox("Select column:", num_cols, key="dist_col")

            c_hist, c_stat = st.columns([3, 1])
            with c_hist:
                fig = px.histogram(df, x=col_sel, nbins=30,
                                   title=f"Distribution of {col_sel}",
                                   color_discrete_sequence=[COLOR_SEQ[0]])
                fig.update_layout(**PLOTLY_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)
            with c_stat:
                s = df[col_sel].describe()
                for k, v in s.items():
                    st.metric(k, f"{v:.2f}")

            if cat_cols:
                cat_sel = st.selectbox("Color by (optional):", ["None"] + cat_cols)
                if cat_sel != "None":
                    fig2 = px.histogram(df, x=col_sel, color=cat_sel, nbins=25,
                                        barmode="overlay", opacity=0.7,
                                        title=f"{col_sel} by {cat_sel}",
                                        color_discrete_sequence=COLOR_SEQ)
                    fig2.update_layout(**PLOTLY_LAYOUT)
                    st.plotly_chart(fig2, use_container_width=True)

    # ── Tab 3: Correlations ───────────────────────
    with tab3:
        if len(num_cols) < 2:
            st.info("Need at least 2 numeric columns for correlation.")
        else:
            corr = df[num_cols].corr()
            fig = px.imshow(corr, text_auto=True, aspect="auto",
                            color_continuous_scale="RdBu_r",
                            title="Correlation Heatmap")
            fig.update_layout(**PLOTLY_LAYOUT, height=500)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(concept_box(
                "What is Correlation?",
                "Correlation measures how two features move together. "
                "<b>+1</b> = perfectly positive, <b>-1</b> = perfectly negative, <b>0</b> = no relationship. "
                "High correlation between a feature and the target is a good sign!",
                "🔗"
            ), unsafe_allow_html=True)

            # Scatter matrix
            if len(num_cols) <= 6:
                st.markdown("#### 🔵 Scatter Matrix (Pair Plot)")
                color_col = cat_cols[0] if cat_cols else None
                fig2 = px.scatter_matrix(df, dimensions=num_cols,
                                         color=color_col,
                                         color_discrete_sequence=COLOR_SEQ,
                                         title="All Features vs Each Other")
                fig2.update_layout(**PLOTLY_LAYOUT, height=600)
                fig2.update_traces(diagonal_visible=False, marker=dict(size=3, opacity=0.6))
                st.plotly_chart(fig2, use_container_width=True)

    # ── Tab 4: Box Plots ──────────────────────────
    with tab4:
        if not num_cols:
            st.info("No numeric columns.")
        else:
            cols_sel = st.multiselect("Select columns:", num_cols, default=num_cols[:4])
            if cols_sel:
                fig = go.Figure()
                for i, c in enumerate(cols_sel):
                    fig.add_trace(go.Box(y=df[c], name=c,
                                         marker_color=COLOR_SEQ[i % len(COLOR_SEQ)],
                                         boxpoints="outliers"))
                fig.update_layout(**PLOTLY_LAYOUT, title="Box Plots — Spotting Outliers")
                st.plotly_chart(fig, use_container_width=True)

                st.markdown(concept_box(
                    "Reading a Box Plot",
                    "The <b>box</b> = middle 50% of data (IQR). "
                    "The <b>line</b> inside = median. "
                    "The <b>whiskers</b> = normal range. "
                    "Dots outside = <b>outliers</b> (unusual values)!",
                    "📦"
                ), unsafe_allow_html=True)

    # ── Tab 5: Insights ───────────────────────────
    with tab5:
        st.markdown("#### 🤖 Auto Insights")

        insights = []
        for col in num_cols:
            skew = df[col].skew()
            if abs(skew) > 1:
                direction = "right (positive)" if skew > 0 else "left (negative)"
                insights.append(f"📊 **{col}** is heavily skewed {direction} — consider log transform.")

        miss_pct = (df.isnull().sum() / len(df) * 100)
        for col, pct in miss_pct[miss_pct > 0].items():
            insights.append(f"⚠️ **{col}** has {pct:.1f}% missing values — needs imputation.")

        if len(num_cols) >= 2:
            corr = df[num_cols].corr()
            for i in range(len(corr)):
                for j in range(i+1, len(corr)):
                    if abs(corr.iloc[i, j]) > 0.85:
                        insights.append(
                            f"🔗 **{corr.index[i]}** and **{corr.columns[j]}** are highly correlated "
                            f"({corr.iloc[i,j]:.2f}) — consider removing one."
                        )

        if not insights:
            st.success("✅ Dataset looks clean and well-distributed!")
        else:
            for ins in insights:
                st.markdown(f"> {ins}")

        st.markdown(concept_box(
            "Key ML Vocabulary",
            "<b>Feature</b>: An input column (e.g., age, income). "
            "<b>Target</b>: What we predict (e.g., price, spam/not spam). "
            "<b>Preprocessing</b>: Cleaning data before training — critical for good models!",
            "📖"
        ), unsafe_allow_html=True)
