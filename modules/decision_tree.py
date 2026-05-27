"""
pages/decision_tree.py
Decision Tree Interactive Visualizer
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from utils.styles import hero, concept_box, section_header, info_box
from utils.data_helpers import get_loan_data, PLOTLY_LAYOUT, COLOR_SEQ


def render():
    st.markdown(hero(
        "Decision Tree 🌳",
        "Watch how a tree splits data at each node. Perfect for loan approval, disease diagnosis, and more.",
        "🌳"
    ), unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(
        ["💳 Loan Approval Tree", "🔢 Tree Stats", "💡 Concepts"]
    )

    df = get_loan_data(500)
    features = ["Income", "Credit_Score", "Debt_Ratio"]
    X = df[features].values
    y = df["Approved"].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

# ─────────────────────────────────────────────
    # TAB 1: Tree Visualization
    # ─────────────────────────────────────────────
    with tab1:

        col_ctrl, col_chart = st.columns([1, 3])

        # -------------------------------
        # LEFT SIDE CONTROLS
        # -------------------------------
        with col_ctrl:

            with st.form("prediction_form"):

                max_depth = st.slider(
                    "Max Depth",
                    1,
                    8,
                    3
                )

                criterion = st.radio(
                    "Criterion",
                    ["gini", "entropy"]
                )

                min_leaf = st.slider(
                    "Min Samples per Leaf",
                    1,
                    30,
                    5
                )

                st.markdown("---")

                inp_income = st.number_input(
                    "💰 Income ($)",
                    20_000,
                    200_000,
                    60_000,
                    5_000
                )

                inp_credit = st.number_input(
                    "📊 Credit Score",
                    300,
                    850,
                    650,
                    10
                )

                inp_debt = st.number_input(
                    "💳 Debt Ratio",
                    0.0,
                    0.9,
                    0.3,
                    0.05
                )

                predict_btn = st.form_submit_button(
                    "🔮 Predict Approval",
                    use_container_width=True
                )

        # -------------------------------
        # MODEL TRAINING
        # -------------------------------
        model = DecisionTreeClassifier(
            max_depth=max_depth,
            criterion=criterion,
            min_samples_leaf=min_leaf,
            random_state=42
        )

        model.fit(X_tr, y_tr)

        acc = accuracy_score(
            y_te,
            model.predict(X_te)
        )

        # -------------------------------
        # TREE VISUALIZATION
        # -------------------------------
        with col_chart:

            fig = _draw_tree(
                model,
                features,
                max_depth
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                key="decision_tree_chart"
            )

        # -------------------------------
        # PREDICTION
        # -------------------------------
        if predict_btn:

            pred = model.predict([
                [inp_income, inp_credit, inp_debt]
            ])[0]

            prob = model.predict_proba([
                [inp_income, inp_credit, inp_debt]
            ])[0][1]

        else:

            pred = 0
            prob = 0.0

        result_color = "#43E97B" if pred == 1 else "#FF6584"

        result_label = (
            "✅ APPROVED"
            if pred == 1
            else "❌ REJECTED"
        )

        # -------------------------------
        # METRICS
        # -------------------------------
        col_res1, col_res2, col_res3 = st.columns(3)

        with col_res1:
            st.metric(
                "Model Accuracy",
                f"{acc*100:.1f}%"
            )

        with col_res2:
            st.metric(
                "Tree Depth",
                str(model.get_depth())
            )

        with col_res3:
            st.metric(
                "Leaf Nodes",
                str(model.get_n_leaves())
            )

        # -------------------------------
        # RESULT CARD
        st.markdown(f"""
        <div class="ml-card"
            style="
            text-align:center;
            border-color:{result_color};
            padding:28px;
            ">

            <div style="
                font-size:2rem;
                font-weight:700;
                color:{result_color};
            ">
                {result_label}
            </div>

            <div style="
                color:#8892A4;
                margin-top:8px;
            ">
                Approval Probability:

                <b style="color:{result_color};">
                    {prob*100:.1f}%
                </b>
            </div>

        </div>
        """, unsafe_allow_html=True)

        st.progress(float(prob))

        # -------------------------------
        # TEXT TREE
        # -------------------------------
        with st.expander("📋 View Text Tree"):

            tree_text = export_text(
                model,
                feature_names=features
            )

            st.code(
                tree_text,
                language="text"
            )
    # ─────────────────────────────────────────────
    # TAB 2: Tree Stats
    # ─────────────────────────────────────────────
    with tab2:
        st.markdown(section_header("📊", "Depth vs Accuracy Analysis"), unsafe_allow_html=True)

        depths  = list(range(1, 15))
        tr_accs = []
        te_accs = []
        for d in depths:
            m = DecisionTreeClassifier(max_depth=d, random_state=42)
            m.fit(X_tr, y_tr)
            tr_accs.append(accuracy_score(y_tr, m.predict(X_tr)))
            te_accs.append(accuracy_score(y_te, m.predict(X_te)))

        fig_depth = go.Figure()
        fig_depth.add_trace(go.Scatter(x=depths, y=tr_accs, mode='lines+markers',
                                        name='Train Accuracy',
                                        line=dict(color=COLOR_SEQ[0], width=2),
                                        marker=dict(size=8)))
        fig_depth.add_trace(go.Scatter(x=depths, y=te_accs, mode='lines+markers',
                                        name='Test Accuracy',
                                        line=dict(color=COLOR_SEQ[1], width=2),
                                        marker=dict(size=8)))
        fig_depth.update_layout(**PLOTLY_LAYOUT,
                                 title="Depth vs Accuracy: Spot Overfitting!",
                                 xaxis_title="Max Depth",
                                 yaxis_title="Accuracy",
                                 )
        fig_depth.update_yaxes(range=[0.5, 1.05])
        st.plotly_chart(fig_depth, use_container_width=True)

        st.markdown(info_box(
            "📌 Notice: Train accuracy keeps rising with depth, but Test accuracy peaks and then drops — "
            "that's <b>overfitting</b>! The tree memorises training data but fails on new data."
        ), unsafe_allow_html=True)

        # Feature importance
        m_fi = DecisionTreeClassifier(max_depth=4, random_state=42)
        m_fi.fit(X_tr, y_tr)
        fi = m_fi.feature_importances_
        fig_fi = px.bar(x=features, y=fi,
                        labels={"x": "Feature", "y": "Importance"},
                        title="Feature Importance — Which feature matters most?",
                        color=fi, color_continuous_scale="Viridis")
        fig_fi.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig_fi, use_container_width=True)

    # ─────────────────────────────────────────────
    # TAB 3: Concepts
    # ─────────────────────────────────────────────
    with tab3:
        st.markdown(section_header("💡", "Decision Tree Concepts"), unsafe_allow_html=True)

        items = [
            ("🌳", "Root Node", "#6C63FF",
             "The very first split — uses the most important feature. All data starts here."),
            ("🍃", "Leaf Node", "#43E97B",
             "Terminal nodes. Once you reach a leaf, the tree gives a final class prediction."),
            ("↗️", "Gini Impurity", "#FF6584",
             "Measures how mixed the classes are at a node. 0 = pure (one class only). We want lower Gini!"),
            ("🔢", "Entropy", "#38F9D7",
             "Information theory measure of disorder. Higher entropy = more mixed classes at a node."),
            ("✂️", "Splitting", "#FFA94D",
             "At each node, the tree finds the best feature+threshold to split data into two groups."),
            ("✂️", "Pruning", "#B48EFF",
             "Cutting back tree branches to prevent overfitting. max_depth and min_samples_leaf do this."),
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


# ─────────────────────────────────────────────
# TREE DRAWING (Plotly)
# ─────────────────────────────────────────────

def _draw_tree(model, feature_names, max_depth):
    """Draw a visual representation of the tree using Plotly."""

    from sklearn.tree import _tree

    tree_ = model.tree_
    colors = COLOR_SEQ

    node_x, node_y = [], []
    node_text, node_color = [], []

    edge_x, edge_y = [], []

    def recurse(node, x, y, dx, depth):

        if depth > max_depth or node == _tree.TREE_LEAF:
            return

        feature = tree_.feature[node]
        threshold = tree_.threshold[node]
        values = tree_.value[node][0]

        cls = np.argmax(values)

        if feature != _tree.TREE_UNDEFINED:

            feat_str = (
                feature_names[feature]
                if feature < len(feature_names)
                else f"F{feature}"
            )

            text = (
                f"{feat_str} ≤ {threshold:.1f}"
                f"<br>samples = {int(tree_.n_node_samples[node])}"
            )

            color = colors[depth % len(colors)]

        else:

            label = "✅ Yes" if cls == 1 else "❌ No"

            text = (
                f"{label}"
                f"<br>samples = {int(tree_.n_node_samples[node])}"
            )

            color = "#43E97B" if cls == 1 else "#FF6584"

        node_x.append(x)
        node_y.append(y)

        node_text.append(text)
        node_color.append(color)

        left = tree_.children_left[node]
        right = tree_.children_right[node]

        if left != _tree.TREE_LEAF:

            nx_l = x - dx
            nx_r = x + dx

            ny = y - 1

            edge_x.extend([x, nx_l, None])
            edge_y.extend([y, ny, None])

            edge_x.extend([x, nx_r, None])
            edge_y.extend([y, ny, None])

            recurse(left, nx_l, ny, dx / 2, depth + 1)
            recurse(right, nx_r, ny, dx / 2, depth + 1)

    recurse(0, 0, 0, 4, 0)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode='lines',
            line=dict(
                color='rgba(136,146,164,0.4)',
                width=1.5
            ),
            hoverinfo='none',
            showlegend=False
        )
    )

    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="bottom center",
            marker=dict(
                size=28,
                color=node_color,
                opacity=0.9,
                line=dict(color='white', width=1.5)
            ),
            hoverinfo='text',
            showlegend=False,
            textfont=dict(size=8, color='white')
        )
    )

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Decision Tree — Loan Approval",
        height=450
    )

    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        showticklabels=False
    )

    fig.update_yaxes(
        showgrid=False,
        zeroline=False,
        showticklabels=False
    )

    return fig