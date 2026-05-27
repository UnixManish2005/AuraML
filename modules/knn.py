"""
pages/knn.py
K-Nearest Neighbours Interactive Visualizer
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.neighbors import KNeighborsClassifier
from utils.styles import hero, concept_box, section_header, info_box
from utils.data_helpers import PLOTLY_LAYOUT, COLOR_SEQ


def render():
    st.markdown(hero(
        "K-Nearest Neighbours 🔍",
        "Classify by proximity! Add points, change K, and watch how the neighbourhood decision boundary changes.",
        "🔍"
    ), unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🎮 KNN Playground", "💡 Concepts"])

    # ─────────────────────────────────────────────
    # TAB 1: Playground
    # ─────────────────────────────────────────────
    with tab1:
        col_ctrl, col_chart = st.columns([1, 3])

        with col_ctrl:
            k = st.slider("K (Neighbours)", 1, 15, 3)
            metric = st.radio("Distance Metric", ["euclidean", "manhattan"])
            n_pts  = st.slider("Training Points", 30, 200, 80)
            n_cls  = st.radio("Classes", [2, 3])

            st.markdown("---")
            st.markdown("**🎯 Test a New Point**")
            test_x = st.slider("X coordinate", -4.0, 4.0, 0.0, 0.1)
            test_y = st.slider("Y coordinate", -4.0, 4.0, 0.0, 0.1)

        # Generate data
        rng = np.random.RandomState(42)
        centres = [[-1.5, -1.5], [1.5, 1.5], [0, 2.5]][:n_cls]
        X_parts, y_parts = [], []
        per = n_pts // n_cls
        for i, c in enumerate(centres):
            X_parts.append(rng.randn(per, 2) * 0.9 + c)
            y_parts.append(np.full(per, i))
        X_train = np.vstack(X_parts)
        y_train = np.concatenate(y_parts)

        model = KNeighborsClassifier(n_neighbors=k, metric=metric)
        model.fit(X_train, y_train)

        test_point = np.array([[test_x, test_y]])
        pred_class = model.predict(test_point)[0]
        probs = model.predict_proba(test_point)[0]

        with col_chart:
            fig = _plot_knn_boundary(model, X_train, y_train, test_point, pred_class, k)
            st.plotly_chart(fig, use_container_width=True)

        # K-distance bar
        dists, indices = model.kneighbors(test_point)
        st.markdown(f"#### 🔵 {k} Nearest Neighbours to Test Point")
        cols_dist = st.columns(min(k, 5))
        for i, (d, idx) in enumerate(zip(dists[0][:5], indices[0][:5])):
            neighbour_class = y_train[idx]
            cols_dist[i].markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="font-size:1.2rem;">{d:.2f}</div>
                <div class="metric-label">Neighbour {i+1}</div>
                <div style="margin-top:6px;">
                    <span class="badge badge-primary">Class {neighbour_class}</span>
                </div>
            </div>""", unsafe_allow_html=True)

        # Result
        class_colors = [COLOR_SEQ[0], COLOR_SEQ[1], COLOR_SEQ[2]]
        c = class_colors[pred_class % len(class_colors)]
        st.markdown(f"""
        <div class="ml-card" style="text-align:center;border-color:{c};padding:24px;margin-top:16px;">
            <div style="font-size:1.4rem;font-weight:700;color:{c};">
                🎯 Test Point Classified as: <b>Class {pred_class}</b>
            </div>
            <div style="color:#8892A4;margin-top:8px;">
                Probabilities: {' | '.join([f'Class {i}: {p*100:.1f}%' for i, p in enumerate(probs)])}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # K vs Accuracy chart
        st.markdown(section_header("📈", "How K Affects Accuracy"), unsafe_allow_html=True)
        from sklearn.model_selection import cross_val_score
        k_vals = list(range(1, min(21, len(X_train)//2)))
        cv_scores = []
        for kv in k_vals:
            scores = cross_val_score(KNeighborsClassifier(n_neighbors=kv, metric=metric),
                                      X_train, y_train, cv=3)
            cv_scores.append(scores.mean())

        fig_k = go.Figure()
        fig_k.add_trace(go.Scatter(x=k_vals, y=cv_scores, mode='lines+markers',
                                    line=dict(color=COLOR_SEQ[0], width=2),
                                    marker=dict(size=8, color=cv_scores,
                                                colorscale='Viridis'),
                                    name='CV Accuracy'))
        fig_k.add_vline(x=k, line_dash="dash", line_color=COLOR_SEQ[2],
                         annotation_text=f"Current K={k}")
        fig_k.update_layout(**PLOTLY_LAYOUT,
                              title="K Value vs Cross-Validation Accuracy",
                              xaxis_title="K",
                              yaxis_title="Accuracy",
                              height=280)
        st.plotly_chart(fig_k, use_container_width=True)

        st.markdown(info_box(
            "💡 <b>Small K</b> = complex boundary, sensitive to noise. "
            "<b>Large K</b> = smooth boundary, but may miss local patterns. "
            "Choose K using cross-validation!"
        ), unsafe_allow_html=True)

    # ─────────────────────────────────────────────
    # TAB 2: Concepts
    # ─────────────────────────────────────────────
    with tab2:
        st.markdown(section_header("💡", "KNN Concepts"), unsafe_allow_html=True)

        st.markdown(concept_box(
            "How KNN Works (Step by Step)",
            "1. Store all training examples.<br>"
            "2. For a new point, calculate distance to ALL training points.<br>"
            "3. Pick the K closest neighbours.<br>"
            "4. Take a <b>majority vote</b> among those K neighbours.<br>"
            "5. Assign the winning class to the new point!",
            "🔍"
        ), unsafe_allow_html=True)

        items = [
            ("📏", "Euclidean Distance", "#6C63FF",
             "Straight-line distance: √((x₁-x₂)² + (y₁-y₂)²). Most common metric."),
            ("🗺️", "Manhattan Distance", "#43E97B",
             "Grid-path distance: |x₁-x₂| + |y₁-y₂|. Like driving city blocks."),
            ("🔢", "K Value", "#FF6584",
             "Number of neighbours to vote. Small K = overfit, Large K = underfit. Use CV to tune."),
            ("⚖️", "Curse of Dimensionality", "#38F9D7",
             "KNN struggles with many features — distances become meaningless in high dimensions."),
            ("📊", "No Training Phase", "#FFA94D",
             "KNN is a 'lazy learner' — it just stores data. All computation happens at prediction time!"),
            ("🌍", "Real-world Use", "#B48EFF",
             "Recommendation systems (find similar users), medical diagnosis, anomaly detection."),
        ]

        for i in range(0, len(items), 3):
            cols = st.columns(3)
            for col, (icon, title, color, desc) in zip(cols, items[i:i+3]):
                col.markdown(f"""
                <div class="ml-card" style="border-top:3px solid {color};min-height:130px;">
                    <div style="font-size:1.8rem;margin-bottom:8px;">{icon}</div>
                    <h4 style="color:{color};margin:0 0 8px 0;font-size:0.92rem;">{title}</h4>
                    <p style="color:#8892A4;font-size:0.82rem;margin:0;line-height:1.5;">{desc}</p>
                </div>""", unsafe_allow_html=True)


def _plot_knn_boundary(model, X, y, test_point, pred_class, k):
    h = 0.1
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    class_colors = [COLOR_SEQ[0], COLOR_SEQ[1], COLOR_SEQ[2]]
    point_colors = [class_colors[int(ci) % len(class_colors)] for ci in y]

    fig = go.Figure()
    fig.add_trace(go.Contour(
        x=np.arange(x_min, x_max, h),
        y=np.arange(y_min, y_max, h),
        z=Z.astype(float), showscale=False, opacity=0.25,
        colorscale=[[0, "rgba(108,99,255,0.4)"], [0.5, "rgba(255,101,132,0.4)"],
                    [1, "rgba(67,233,123,0.4)"]],
        contours_coloring='fill'
    ))
    fig.add_trace(go.Scatter(
        x=X[:, 0], y=X[:, 1], mode='markers',
        marker=dict(color=point_colors, size=7, opacity=0.75,
                    line=dict(color='white', width=0.5)),
        name='Training Points'
    ))

    # Draw lines to K neighbours
    from sklearn.neighbors import KNeighborsClassifier
    dists, idxs = model.kneighbors(test_point)
    for idx in idxs[0]:
        fig.add_shape(type="line",
            x0=test_point[0, 0], y0=test_point[0, 1],
            x1=X[idx, 0], y1=X[idx, 1],
            line=dict(color="rgba(255,255,255,0.4)", dash="dot", width=1.5))

    pred_c = class_colors[int(pred_class) % len(class_colors)]
    fig.add_trace(go.Scatter(
        x=[test_point[0, 0]], y=[test_point[0, 1]],
        mode='markers',
        marker=dict(color=pred_c, size=16, symbol='star',
                    line=dict(color='white', width=2)),
        name='Test Point'
    ))

    fig.update_layout(**PLOTLY_LAYOUT,
                       title=f"KNN Decision Boundary (K={k})",
                       xaxis_title="Feature 1",
                       yaxis_title="Feature 2",
                       height=420)
    return fig
