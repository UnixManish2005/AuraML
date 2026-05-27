"""
pages/clustering.py
K-Means Clustering Interactive Visualizer
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from utils.styles import hero, concept_box, section_header, info_box
from utils.data_helpers import get_customer_data, PLOTLY_LAYOUT, COLOR_SEQ


def render():
    st.markdown(hero(
        "K-Means Clustering 🔵",
        "Group similar customers together. Watch centroids move step-by-step as clusters form!",
        "🔵"
    ), unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(
        ["👥 Customer Segmentation", "📈 Elbow Method", "💡 Concepts"]
    )

    # ─────────────────────────────────────────────
    # TAB 1: Customer Segmentation
    # ─────────────────────────────────────────────
    with tab1:
        col_ctrl, col_chart = st.columns([1, 3])

        with col_ctrl:
            k        = st.slider("K (Clusters)", 2, 8, 3)
            feature1 = st.selectbox("X-Axis Feature", ["Age", "Annual_Income", "Spending_Score"], index=1)
            feature2 = st.selectbox("Y-Axis Feature", ["Age", "Annual_Income", "Spending_Score"], index=2)
            n_init   = st.slider("Restarts", 1, 20, 10)
            animate  = st.checkbox("Step-by-step view", False)

        df = get_customer_data(300)
        X  = df[[feature1, feature2]].values

        model = KMeans(n_clusters=k, n_init=n_init, random_state=42)
        model.fit(X)
        labels    = model.labels_
        centroids = model.cluster_centers_

        with col_chart:
            fig = go.Figure()
            for i in range(k):
                mask = labels == i
                fig.add_trace(go.Scatter(
                    x=X[mask, 0], y=X[mask, 1],
                    mode='markers',
                    marker=dict(color=COLOR_SEQ[i % len(COLOR_SEQ)], size=7, opacity=0.7),
                    name=f'Cluster {i+1}'
                ))
            fig.add_trace(go.Scatter(
                x=centroids[:, 0], y=centroids[:, 1],
                mode='markers',
                marker=dict(color='white', size=16, symbol='x',
                            line=dict(color='black', width=2)),
                name='Centroids'
            ))
            fig.update_layout(**PLOTLY_LAYOUT,
                               title=f"Customer Segmentation — {k} Clusters",
                               xaxis_title=feature1,
                               yaxis_title=feature2,
                               height=420)
            st.plotly_chart(fig, use_container_width=True)

        if animate:
            _show_kmeans_steps(X, k)

        # Cluster stats
        st.markdown(section_header("📊", "Cluster Profiles"), unsafe_allow_html=True)
        df["Cluster"] = labels + 1
        cluster_stats = df.groupby("Cluster").agg({
            "Age": "mean", "Annual_Income": "mean", "Spending_Score": "mean"
        }).round(1)
        cluster_stats.columns = ["Avg Age", "Avg Income ($)", "Avg Spend Score"]

        cols_st = st.columns(k)
        for i, col in enumerate(cols_st[:k]):
            row = cluster_stats.iloc[i]
            col.markdown(f"""
            <div class="ml-card" style="border-top:3px solid {COLOR_SEQ[i % len(COLOR_SEQ)]};">
                <h4 style="color:{COLOR_SEQ[i % len(COLOR_SEQ)]};margin:0 0 12px 0;">
                    Cluster {i+1}
                </h4>
                {''.join([f'<div style="display:flex;justify-content:space-between;margin:6px 0;"><span style="color:#8892A4;font-size:0.82rem;">{c}</span><b style="font-size:0.82rem;">{v:.0f}</b></div>' for c, v in zip(row.index, row.values)])}
            </div>""", unsafe_allow_html=True)

        sil = silhouette_score(X, labels) if k > 1 else 1.0
        st.metric("Silhouette Score (closer to 1 = better clusters)", f"{sil:.3f}")

    # ─────────────────────────────────────────────
    # TAB 2: Elbow Method
    # ─────────────────────────────────────────────
    with tab2:
        st.markdown(section_header("📈", "Elbow Method — Find Optimal K"), unsafe_allow_html=True)

        df2 = get_customer_data(300)
        X2  = df2[["Annual_Income", "Spending_Score"]].values

        k_range = range(2, 11)
        inertias = []
        sil_scores = []
        for kv in k_range:
            m = KMeans(n_clusters=kv, n_init=10, random_state=42)
            m.fit(X2)
            inertias.append(m.inertia_)
            sil_scores.append(silhouette_score(X2, m.labels_))

        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(
            x=list(k_range), y=inertias, mode='lines+markers',
            line=dict(color=COLOR_SEQ[0], width=2),
            marker=dict(size=9), name='Inertia (WCSS)'
        ))
        fig_elbow.update_layout(**PLOTLY_LAYOUT,
                                  title="Elbow Method: Inertia vs K",
                                  xaxis_title="K (Number of Clusters)",
                                  yaxis_title="Inertia (Within-Cluster Sum of Squares)",
                                  height=320)
        st.plotly_chart(fig_elbow, use_container_width=True)

        fig_sil = go.Figure()
        fig_sil.add_trace(go.Bar(
            x=list(k_range), y=sil_scores,
            marker_color=COLOR_SEQ, name='Silhouette'
        ))
        fig_sil.update_layout(**PLOTLY_LAYOUT,
                                title="Silhouette Score vs K (Higher = Better)",
                                xaxis_title="K",
                                yaxis_title="Silhouette Score",
                                height=280)
        st.plotly_chart(fig_sil, use_container_width=True)

        best_k = list(k_range)[np.argmax(sil_scores)]
        st.markdown(info_box(
            f"📌 <b>Optimal K ≈ {best_k}</b> based on silhouette score. "
            "Look for the 'elbow' in the inertia plot — where adding more clusters "
            "stops providing significant benefit."
        ), unsafe_allow_html=True)

    # ─────────────────────────────────────────────
    # TAB 3: Concepts
    # ─────────────────────────────────────────────
    with tab3:
        st.markdown(section_header("💡", "K-Means Concepts"), unsafe_allow_html=True)

        st.markdown(concept_box(
            "K-Means Algorithm Steps",
            "1. <b>Initialise</b>: Randomly place K centroids.<br>"
            "2. <b>Assign</b>: Each point joins the nearest centroid.<br>"
            "3. <b>Update</b>: Move each centroid to the mean of its cluster.<br>"
            "4. <b>Repeat</b> steps 2–3 until centroids stop moving!",
            "⚙️"
        ), unsafe_allow_html=True)

        items = [
            ("📍", "Centroid", "#6C63FF",
             "The centre point of a cluster. K-Means iteratively moves centroids to minimize distance."),
            ("📏", "Inertia (WCSS)", "#43E97B",
             "Within-Cluster Sum of Squares — total distance of all points from their centroid. Lower is better."),
            ("📐", "Silhouette Score", "#FF6584",
             "Measures how well each point fits its cluster vs the next-best cluster. Range: -1 to 1."),
            ("🎲", "K++ Initialization", "#38F9D7",
             "Smart initialisation: choose centroids spread out across the data for better convergence."),
        ]
        cols = st.columns(4)
        for col, (icon, title, color, desc) in zip(cols, items):
            col.markdown(f"""
            <div class="ml-card" style="border-top:3px solid {color};min-height:150px;">
                <div style="font-size:1.8rem;margin-bottom:8px;">{icon}</div>
                <h4 style="color:{color};margin:0 0 8px 0;font-size:0.9rem;">{title}</h4>
                <p style="color:#8892A4;font-size:0.8rem;margin:0;line-height:1.5;">{desc}</p>
            </div>""", unsafe_allow_html=True)


def _show_kmeans_steps(X, k, max_iters=5):
    """Show first few K-Means iterations."""
    st.markdown("#### ⚙️ Step-by-Step K-Means")
    rng = np.random.RandomState(0)
    centroids = X[rng.choice(len(X), k, replace=False)]

    for step in range(max_iters):
        dists  = np.sqrt(((X[:, None] - centroids[None])**2).sum(axis=2))
        labels = dists.argmin(axis=1)
        new_c  = np.array([X[labels == i].mean(axis=0) if (labels==i).any()
                           else centroids[i] for i in range(k)])

        fig = go.Figure()
        for i in range(k):
            mask = labels == i
            fig.add_trace(go.Scatter(x=X[mask,0], y=X[mask,1], mode='markers',
                                      marker=dict(color=COLOR_SEQ[i%len(COLOR_SEQ)], size=6, opacity=0.6),
                                      name=f'C{i+1}'))
        fig.add_trace(go.Scatter(x=new_c[:,0], y=new_c[:,1], mode='markers',
                                  marker=dict(color='white', size=14, symbol='x',
                                              line=dict(color='black', width=2)),
                                  name='Centroids'))
        fig.update_layout(**PLOTLY_LAYOUT, title=f"Iteration {step+1}", height=280)
        st.plotly_chart(fig, use_container_width=True)
        centroids = new_c
