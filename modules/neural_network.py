"""
pages/neural_network.py
Neural Network Playground — beginner-friendly visualizer
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils.styles import hero, concept_box, section_header, info_box
from utils.data_helpers import PLOTLY_LAYOUT, COLOR_SEQ


def render():
    st.markdown(hero(
        "Neural Network Playground 🧠",
        "Build your own neural network! Add layers, change neurons, train and see the magic happen.",
        "🧠"
    ), unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(
        ["🏗️ Build & Train", "📉 Training Curves", "💡 Concepts"]
    )

    # ─────────────────────────────────────────────
    # TAB 1: Build & Train
    # ─────────────────────────────────────────────
    with tab1:
        col_arch, col_data, col_train = st.columns(3)

        with col_arch:
            st.markdown("**🏗️ Architecture**")
            n_hidden = st.slider("Hidden Layers", 1, 4, 2)
            neurons  = st.slider("Neurons per Layer", 4, 64, 16)
            activation = st.selectbox("Activation", ["relu", "tanh", "logistic"])

        with col_data:
            st.markdown("**📊 Dataset**")
            dataset  = st.selectbox("Problem", ["Moons", "Circles", "Blobs"])
            n_pts    = st.slider("Data Points", 100, 500, 200)
            noise    = st.slider("Noise", 0.0, 0.4, 0.1, 0.02)

        with col_train:
            st.markdown("**⚙️ Training**")
            lr       = st.select_slider("Learning Rate", [0.0001, 0.001, 0.01, 0.1], value=0.001)
            epochs   = st.slider("Epochs", 10, 500, 100, 10)
            run_btn  = st.button("🚀 Train Network!", use_container_width=True)

        # Visualize network architecture
        arch_layers = [2] + [neurons]*n_hidden + [2]
        fig_arch = _draw_nn(arch_layers, activation)
        st.plotly_chart(fig_arch, use_container_width=True)

        if run_btn:
            X, y = _get_data(dataset, n_pts, noise)
            scaler = StandardScaler()
            X_sc = scaler.fit_transform(X)
            X_tr, X_te, y_tr, y_te = train_test_split(X_sc, y, test_size=0.2, random_state=42)

            hidden = tuple([neurons]*n_hidden)
            model = MLPClassifier(hidden_layer_sizes=hidden,
                                   activation=activation,
                                   learning_rate_init=lr,
                                   max_iter=1,
                                   warm_start=True,
                                   random_state=42)

            losses, train_accs, test_accs = [], [], []
            progress = st.progress(0)
            checkpoint_epochs = list(range(1, epochs+1, max(1, epochs//10))) + [epochs]

            for ep in checkpoint_epochs:
                model.max_iter = ep
                model.fit(X_tr, y_tr)
                losses.append(model.loss_)
                train_accs.append(model.score(X_tr, y_tr))
                test_accs.append(model.score(X_te, y_te))
                progress.progress(ep / epochs)

            # Decision boundary
            fig_db = _plot_boundary(model, X_sc, y)
            st.plotly_chart(fig_db, use_container_width=True)

            # Store in session for Tab 2
            st.session_state["nn_losses"]     = losses
            st.session_state["nn_train_accs"] = train_accs
            st.session_state["nn_test_accs"]  = test_accs
            st.session_state["nn_epochs"]     = checkpoint_epochs

            m1, m2, m3 = st.columns(3)
            m1.metric("Final Loss",       f"{losses[-1]:.4f}")
            m2.metric("Train Accuracy",   f"{train_accs[-1]*100:.1f}%")
            m3.metric("Test Accuracy",    f"{test_accs[-1]*100:.1f}%")

    # ─────────────────────────────────────────────
    # TAB 2: Training Curves
    # ─────────────────────────────────────────────
    with tab2:
        if "nn_losses" not in st.session_state:
            st.info("👆 Train a network in the first tab to see curves here.")
        else:
            eps    = st.session_state["nn_epochs"]
            losses = st.session_state["nn_losses"]
            tr_a   = st.session_state["nn_train_accs"]
            te_a   = st.session_state["nn_test_accs"]

            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(x=eps, y=losses, mode='lines+markers',
                                           line=dict(color=COLOR_SEQ[2], width=2),
                                           name='Loss'))
            fig_loss.update_layout(**PLOTLY_LAYOUT, title="Training Loss",
                                    xaxis_title="Epoch", yaxis_title="Loss", height=280)
            st.plotly_chart(fig_loss, use_container_width=True)

            fig_acc = go.Figure()
            fig_acc.add_trace(go.Scatter(x=eps, y=tr_a, mode='lines+markers',
                                          line=dict(color=COLOR_SEQ[0], width=2),
                                          name='Train Accuracy'))
            fig_acc.add_trace(go.Scatter(x=eps, y=te_a, mode='lines+markers',
                                          line=dict(color=COLOR_SEQ[1], width=2),
                                          name='Test Accuracy'))
            fig_acc.update_layout(**PLOTLY_LAYOUT, title="Train vs Test Accuracy",
                                   xaxis_title="Epoch", yaxis_title="Accuracy", height=280)
            st.plotly_chart(fig_acc, use_container_width=True)

            if te_a[-1] < tr_a[-1] - 0.1:
                st.markdown(info_box(
                    "⚠️ <b>Possible Overfitting!</b> Training accuracy is much higher than test. "
                    "Try: fewer layers, fewer neurons, or more data."
                ), unsafe_allow_html=True)
            else:
                from utils.styles import success_box
                st.markdown(success_box(
                    f"✅ Good generalisation! Train={tr_a[-1]*100:.1f}%, Test={te_a[-1]*100:.1f}%"
                ), unsafe_allow_html=True)

    # ─────────────────────────────────────────────
    # TAB 3: Concepts
    # ─────────────────────────────────────────────
    with tab3:
        st.markdown(section_header("💡", "Neural Network Concepts"), unsafe_allow_html=True)

        st.markdown(concept_box(
            "What is a Neural Network?",
            "Inspired by the human brain! Layers of 'neurons' learn to transform inputs into outputs. "
            "Input layer → Hidden layers → Output layer. "
            "Each connection has a <b>weight</b> that is learned during training.",
            "🧠"
        ), unsafe_allow_html=True)

        items = [
            ("⚡", "Activation Function", "#6C63FF",
             "Decides if a neuron 'fires'. ReLU: max(0,x). Sigmoid: 0–1. Tanh: -1 to 1."),
            ("⬇️", "Backpropagation", "#43E97B",
             "The algorithm that flows errors backward to update weights. The heart of neural network training."),
            ("📉", "Loss Function", "#FF6584",
             "Measures how wrong the model is. We minimise this using gradient descent."),
            ("🏋️", "Epoch", "#38F9D7",
             "One full pass through the entire training dataset. More epochs = more training."),
            ("📦", "Batch Size", "#FFA94D",
             "How many examples to process before updating weights. Smaller = noisier but faster updates."),
            ("⚖️", "Weights & Biases", "#B48EFF",
             "Parameters that the network learns. Weights control connection strength, bias shifts the output."),
        ]
        for i in range(0, len(items), 3):
            cols = st.columns(3)
            for col, (icon, title, color, desc) in zip(cols, items[i:i+3]):
                col.markdown(f"""
                <div class="ml-card" style="border-top:3px solid {color};min-height:130px;">
                    <div style="font-size:1.8rem;margin-bottom:8px;">{icon}</div>
                    <h4 style="color:{color};margin:0 0 8px 0;font-size:0.9rem;">{title}</h4>
                    <p style="color:#8892A4;font-size:0.8rem;margin:0;line-height:1.5;">{desc}</p>
                </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _get_data(name, n, noise):
    if name == "Moons":
        return make_moons(n_samples=n, noise=noise, random_state=42)
    elif name == "Circles":
        return make_circles(n_samples=n, noise=noise, factor=0.5, random_state=42)
    else:
        from sklearn.datasets import make_blobs
        X, y = make_blobs(n_samples=n, centers=2, cluster_std=noise*5+0.5, random_state=42)
        return X, y


def _plot_boundary(model, X, y):
    h = 0.05
    x_min, x_max = X[:, 0].min()-1, X[:, 0].max()+1
    y_min, y_max = X[:, 1].min()-1, X[:, 1].max()+1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    colors = [COLOR_SEQ[int(ci)] for ci in y]

    fig = go.Figure()
    fig.add_trace(go.Contour(
        x=np.arange(x_min, x_max, h),
        y=np.arange(y_min, y_max, h),
        z=Z.astype(float), showscale=False, opacity=0.3,
        colorscale=[[0, "rgba(108,99,255,0.4)"], [1, "rgba(67,233,123,0.4)"]],
        contours_coloring='fill'
    ))
    fig.add_trace(go.Scatter(
        x=X[:,0], y=X[:,1], mode='markers',
        marker=dict(color=colors, size=6, opacity=0.8,
                    line=dict(color='white', width=0.5)),
        name='Data'
    ))
    fig.update_layout(**PLOTLY_LAYOUT, title="Neural Network Decision Boundary", height=380)
    return fig


def _draw_nn(layer_sizes, activation):
    """Draw a simple neural network diagram using Plotly."""
    max_n = max(layer_sizes)
    node_x, node_y, node_text, node_color = [], [], [], []
    edge_x, edge_y = [], []
    n_layers = len(layer_sizes)
    layer_labels = (["Input"] + [f"Hidden {i}" for i in range(1, n_layers-1)] + ["Output"])

    for li, (n_nodes, label) in enumerate(zip(layer_sizes, layer_labels)):
        x = li / (n_layers - 1) if n_layers > 1 else 0.5
        for ni in range(n_nodes):
            y = (ni - (n_nodes-1)/2) / max(max_n, 1)
            node_x.append(x)
            node_y.append(y)
            node_text.append(label)
            if label == "Input":
                node_color.append(COLOR_SEQ[3])
            elif label == "Output":
                node_color.append(COLOR_SEQ[1])
            else:
                node_color.append(COLOR_SEQ[0])

    # Edges (only for small networks)
    offset = 0
    layer_offsets = []
    for n in layer_sizes:
        layer_offsets.append(offset)
        offset += n

    for li in range(n_layers - 1):
        if layer_sizes[li] * layer_sizes[li+1] <= 200:
            for i in range(layer_sizes[li]):
                for j in range(layer_sizes[li+1]):
                    xi = node_x[layer_offsets[li] + i]
                    yi = node_y[layer_offsets[li] + i]
                    xj = node_x[layer_offsets[li+1] + j]
                    yj = node_y[layer_offsets[li+1] + j]
                    edge_x += [xi, xj, None]
                    edge_y += [yi, yj, None]

    fig = go.Figure()
    if edge_x:
        fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines',
                                  line=dict(color='rgba(136,146,164,0.15)', width=1),
                                  hoverinfo='none', showlegend=False))
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode='markers',
        marker=dict(size=18, color=node_color, opacity=0.9,
                    line=dict(color='white', width=1.5)),
        text=node_text, hoverinfo='text',
        showlegend=False
    ))

    labels_x = [i / (n_layers-1) if n_layers > 1 else 0.5 for i in range(n_layers)]
    labels_y = [max_n / max(max_n, 1) + 0.2] * n_layers
    for lx, ly, lbl in zip(labels_x, labels_y, layer_labels):
        fig.add_annotation(x=lx, y=ly, text=lbl,
                            showarrow=False, font=dict(color='#8892A4', size=11))

    fig.update_layout(
    **PLOTLY_LAYOUT,
    height=420
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
