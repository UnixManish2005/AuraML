"""
pages/deep_learning.py
Deep Learning Labs — CNN, RNN, Transfer Learning, GANs, Transformers
Interactive experiments with visual explanations
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons, make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from utils.styles import (hero, section_header, concept_box,
                           info_box, warning_box, success_box)
from utils.data_helpers import PLOTLY_LAYOUT, COLOR_SEQ


LABS = [
    "🧠 CNN — Image Feature Detector",
    "🔄 RNN — Sequence Predictor",
    "⚡ Activation Functions Lab",
    "📉 Loss Functions Explorer",
    "🎨 GAN — Generator vs Discriminator",
    "🔁 Transfer Learning Demo",
    "🤗 Transformer Attention Visualizer",
    "🏋️ Training Dynamics Lab",
]


def render():
    st.markdown(hero(
        "Deep Learning Labs 🧪",
        "Explore CNNs, RNNs, GANs, and Transformers with hands-on experiments. "
        "No PhD required — just curiosity!",
        "🧪"
    ), unsafe_allow_html=True)


    lab = st.selectbox("🔬 Choose a Lab:", LABS)
    st.markdown("---")

    if   lab == LABS[0]: _cnn_lab()
    elif lab == LABS[1]: _rnn_lab()
    elif lab == LABS[2]: _activation_lab()
    elif lab == LABS[3]: _loss_lab()
    elif lab == LABS[4]: _gan_lab()
    elif lab == LABS[5]: _transfer_lab()
    elif lab == LABS[6]: _attention_lab()
    elif lab == LABS[7]: _training_dynamics_lab()


# ═══════════════════════════════════════════════
# LAB 1 — CNN Feature Detector
# ═══════════════════════════════════════════════
def _cnn_lab():
    st.markdown(section_header("🧠", "CNN — Convolutional Neural Network",
                "How computers 'see' images"), unsafe_allow_html=True)

    st.markdown(concept_box(
        "What is a CNN?",
        "A CNN applies small filters (kernels) across an image to detect edges, textures, and shapes. "
        "Each layer learns progressively complex features: edges → shapes → objects. "
        "Used in face recognition, medical imaging, self-driving cars!",
        "🔍"
    ), unsafe_allow_html=True)

    tab_viz, tab_filters, tab_arch, tab_demo = st.tabs(
        ["🖼️ Convolution Demo", "🎛️ Filter Gallery", "🏗️ Architecture", "🚀 Train CNN"]
    )

    with tab_viz:
        st.markdown("#### 🖼️ See How a Convolution Filter Works")
        col_ctrl, col_out = st.columns([1, 2])

        with col_ctrl:
            filter_type = st.selectbox("Filter (Kernel):", [
                "Edge Detect (Horizontal)", "Edge Detect (Vertical)",
                "Sharpen", "Blur", "Emboss"
            ])
            img_size = st.slider("Image Size", 8, 20, 12)
            rng = np.random.RandomState(st.slider("Image Seed", 0, 50, 7))
            image = rng.randint(0, 256, (img_size, img_size)).astype(float)

        kernels = {
            "Edge Detect (Horizontal)": np.array([[-1,-1,-1],[0,0,0],[1,1,1]]),
            "Edge Detect (Vertical)":   np.array([[-1,0,1],[-1,0,1],[-1,0,1]]),
            "Sharpen":                  np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]),
            "Blur":                     np.ones((3,3)) / 9,
            "Emboss":                   np.array([[-2,-1,0],[-1,1,1],[0,1,2]]),
        }
        kernel = kernels[filter_type]
        output = _convolve2d(image, kernel)

        with col_out:
            fig = go.Figure()
            fig = _side_by_side_heatmaps(image, output,
                                          "Original Image", f"After '{filter_type}' Filter")
            st.plotly_chart(fig, use_container_width=True)

        # Show kernel
        st.markdown("#### 🔲 Kernel (Filter) Values")
        fig_k = px.imshow(kernel, text_auto=True, aspect="auto",
                           color_continuous_scale="RdBu_r",
                           title=f"{filter_type} Kernel (3×3)")
        fig_k.update_layout(**PLOTLY_LAYOUT, height=200)
        st.plotly_chart(fig_k, use_container_width=True)

        st.markdown(info_box(
            "📌 The kernel slides across every pixel of the image. "
            "At each position it multiplies its values with the image patch and sums them — "
            "this is a <b>dot product</b>, and it's what 'convolution' means!"
        ), unsafe_allow_html=True)

    with tab_filters:
        st.markdown("#### 🎛️ What CNNs Learn to Detect")
        filter_descriptions = [
            ("🔵", "Layer 1: Edges",    "Simple horizontal/vertical edge detectors — the most basic visual feature."),
            ("🟢", "Layer 2: Textures", "Combinations of edges form textures like fur, scales, or bricks."),
            ("🟡", "Layer 3: Parts",    "Textures combine into object parts: eyes, wheels, handles."),
            ("🔴", "Layer 4: Objects",  "Parts combine to form full objects: faces, cars, dogs."),
        ]
        cols = st.columns(4)
        for col, (icon, title, desc) in zip(cols, filter_descriptions):
            col.markdown(f"""
            <div class="ml-card" style="text-align:center;min-height:160px;">
                <div style="font-size:2rem;margin-bottom:8px;">{icon}</div>
                <h4 style="color:#6C63FF;font-size:0.88rem;margin:0 0 8px 0;">{title}</h4>
                <p style="color:#8892A4;font-size:0.78rem;margin:0;line-height:1.4;">{desc}</p>
            </div>""", unsafe_allow_html=True)

        # Visualise a set of random learned filters
        st.markdown("#### 🎨 Simulated Learned Filters (Like Real CNNs)")
        n_filters = 16
        rng2 = np.random.RandomState(42)
        fig_filters = go.Figure()
        cols2 = st.columns(8)
        for i in range(n_filters):
            filt = rng2.randn(5, 5)
            col = cols2[i % 8]
            fig_f = px.imshow(filt, color_continuous_scale="RdBu",
                               aspect="equal")
            fig_f.update_layout(height=80,
                                 margin=dict(l=0,r=0,t=0,b=0),
                                 paper_bgcolor="rgba(0,0,0,0)",
                                 plot_bgcolor="rgba(0,0,0,0)",
                                 coloraxis_showscale=False,
                                 xaxis=dict(showticklabels=False),
                                 yaxis=dict(showticklabels=False))
            col.plotly_chart(fig_f, use_container_width=True, key=f"filt_{i}")

    with tab_arch:
        st.markdown("#### 🏗️ Typical CNN Architecture")
        fig_arch = _draw_cnn_arch()
        st.plotly_chart(fig_arch, use_container_width=True)

        layers_info = [
            ("📥", "Input Layer",       "Raw pixel values (e.g., 28×28×1 for grayscale)"),
            ("🔲", "Conv Layer",        "Applies N filters, produces N feature maps"),
            ("⚡", "ReLU Activation",   "max(0, x) — introduces non-linearity"),
            ("🏊", "Pooling Layer",     "Reduces spatial size (Max Pool most common)"),
            ("📦", "Flatten",           "Converts 2D feature maps to 1D vector"),
            ("🔗", "Dense (FC) Layer",  "Fully connected — learns high-level combinations"),
            ("📤", "Softmax Output",    "Probability distribution over classes"),
        ]
        for icon, name, desc in layers_info:
            st.markdown(f"""
            <div style="display:flex;gap:12px;align-items:flex-start;
                        padding:10px;border-bottom:1px solid rgba(108,99,255,0.1);">
                <span style="font-size:1.3rem;">{icon}</span>
                <div>
                    <b style="color:#F0F4FF;font-size:0.9rem;">{name}</b>
                    <p style="color:#8892A4;font-size:0.8rem;margin:2px 0 0 0;">{desc}</p>
                </div>
            </div>""", unsafe_allow_html=True)

    with tab_demo:
        st.markdown("#### 🚀 Train a Mini-CNN on 2D Data")
        col_c, col_ch = st.columns([1, 2])
        with col_c:
            dataset   = st.selectbox("Dataset:", ["Moons", "Circles"])
            n_layers  = st.slider("Hidden Layers", 1, 5, 3)
            n_neurons = st.slider("Neurons/Layer",  8, 128, 64)
            lr        = st.select_slider("Learning Rate",
                                          [0.0001, 0.001, 0.01, 0.1], value=0.01)
            epochs    = st.slider("Epochs", 20, 300, 100, 20)
            train_btn = st.button("🚀 Train!", use_container_width=True, key="cnn_train")

        if train_btn:
            X, y = (make_moons(200, noise=0.2, random_state=42) if dataset == "Moons"
                    else make_circles(200, noise=0.15, factor=0.4, random_state=42))
            sc = StandardScaler()
            Xs = sc.fit_transform(X)
            Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.2, random_state=42)

            hidden = tuple([n_neurons]*n_layers)
            model  = MLPClassifier(hidden_layer_sizes=hidden, learning_rate_init=lr,
                                    max_iter=epochs, random_state=42, activation='relu')
            model.fit(Xtr, ytr)

            with col_ch:
                acc = model.score(Xte, yte)
                st.metric("Test Accuracy", f"{acc*100:.1f}%")

                # Decision boundary
                h = 0.08
                x0r = (Xs[:,0].min()-1, Xs[:,0].max()+1)
                x1r = (Xs[:,1].min()-1, Xs[:,1].max()+1)
                xx, yy = np.meshgrid(np.arange(*x0r, h), np.arange(*x1r, h))
                Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
                pt_colors = [COLOR_SEQ[int(c)] for c in y]

                fig_db = go.Figure()
                fig_db.add_trace(go.Contour(
                    x=np.arange(*x0r, h), y=np.arange(*x1r, h), z=Z.astype(float),
                    showscale=False, opacity=0.3,
                    colorscale=[[0,"rgba(108,99,255,0.4)"],[1,"rgba(67,233,123,0.4)"]],
                    contours_coloring='fill'
                ))
                fig_db.add_trace(go.Scatter(
                    x=Xs[:,0], y=Xs[:,1], mode='markers',
                    marker=dict(color=pt_colors, size=7, opacity=0.8,
                                line=dict(color='white', width=0.5))
                ))
                fig_db.update_layout(**PLOTLY_LAYOUT,
                                      title=f"Deep Network — {n_layers} layers × {n_neurons} neurons",
                                      height=350)
                st.plotly_chart(fig_db, use_container_width=True)

                _save_lab_progress("CNN Lab", int(acc * 100))


# ═══════════════════════════════════════════════
# LAB 2 — RNN Sequence Predictor
# ═══════════════════════════════════════════════
def _rnn_lab():
    st.markdown(section_header("🔄", "RNN — Recurrent Neural Network",
                "Learning from sequences: text, time-series, speech"), unsafe_allow_html=True)

    tab_concept, tab_demo, tab_types = st.tabs(
        ["💡 Concepts", "📈 Sequence Demo", "🗂️ RNN Types"]
    )

    with tab_concept:
        st.markdown(concept_box(
            "Why RNNs?",
            "Regular neural networks process each input independently. "
            "But language, music, and stock prices have <b>order and memory</b> — "
            "RNNs pass a 'hidden state' from one step to the next, giving them memory!",
            "🔄"
        ), unsafe_allow_html=True)

        # Unrolled RNN diagram
        fig = _draw_rnn_diagram()
        st.plotly_chart(fig, use_container_width=True)

        items = [
            ("📥", "Input (xₜ)",        "#6C63FF", "Current token/value at timestep t"),
            ("🧠", "Hidden State (hₜ)",  "#43E97B", "Memory from previous steps, updated at each timestep"),
            ("📤", "Output (yₜ)",        "#FF6584", "Prediction or embedding at this timestep"),
            ("🚪", "LSTM Gates",         "#38F9D7", "Forget, Input, Output gates control what to remember"),
            ("⚡", "GRU",               "#FFA94D", "Gated Recurrent Unit — simpler, faster LSTM variant"),
            ("📝", "Sequence Length",    "#B48EFF", "How many timesteps to unroll — affects memory & speed"),
        ]
        for i in range(0, len(items), 3):
            cols = st.columns(3)
            for col, (icon, title, color, desc) in zip(cols, items[i:i+3]):
                col.markdown(f"""
                <div class="ml-card" style="border-top:3px solid {color};min-height:110px;">
                    <div style="font-size:1.5rem;margin-bottom:6px;">{icon}</div>
                    <h4 style="color:{color};margin:0 0 6px;font-size:0.88rem;">{title}</h4>
                    <p style="color:#8892A4;font-size:0.78rem;margin:0;line-height:1.4;">{desc}</p>
                </div>""", unsafe_allow_html=True)

    with tab_demo:
        st.markdown("#### 📈 Time-Series Forecasting Demo")
        col_c, col_out = st.columns([1, 2])

        with col_c:
            series_type = st.selectbox("Signal:", ["Sine Wave", "Noisy Sine", "Stock-like"])
            seq_len     = st.slider("Sequence Length", 10, 100, 40)
            noise_lvl   = st.slider("Noise Level",   0.0, 1.0, 0.2, 0.05)
            n_hidden    = st.slider("Hidden Units", 8, 64, 32)
            forecast_n  = st.slider("Forecast Steps", 5, 30, 10)
            train_btn   = st.button("🚀 Train & Forecast", key="rnn_btn",
                                     use_container_width=True)

        rng = np.random.RandomState(42)
        t   = np.linspace(0, 4*np.pi, 200)
        if series_type == "Sine Wave":
            series = np.sin(t)
        elif series_type == "Noisy Sine":
            series = np.sin(t) + rng.normal(0, noise_lvl, len(t))
        else:
            returns = rng.normal(0.001, 0.02, len(t))
            series  = np.cumprod(1 + returns) * 100

        with col_out:
            if train_btn:
                # Prepare sliding windows
                X_seq, y_seq = [], []
                for i in range(len(series) - seq_len):
                    X_seq.append(series[i:i+seq_len])
                    y_seq.append(series[i+seq_len])
                X_seq = np.array(X_seq)
                y_seq = np.array(y_seq)

                split = int(len(X_seq) * 0.8)
                Xtr, Xte = X_seq[:split], X_seq[split:]
                ytr, yte = y_seq[:split], y_seq[split:]

                model = MLPClassifier.__new__(MLPClassifier)
                from sklearn.neural_network import MLPRegressor
                reg = MLPRegressor(hidden_layer_sizes=(n_hidden, n_hidden//2),
                                    max_iter=200, random_state=42, learning_rate_init=0.001)
                reg.fit(Xtr, ytr)
                y_pred_te = reg.predict(Xte)

                # Multi-step forecast
                forecast   = []
                last_window = series[-seq_len:].tolist()
                for _ in range(forecast_n):
                    inp    = np.array(last_window[-seq_len:]).reshape(1, -1)
                    nxt    = reg.predict(inp)[0]
                    forecast.append(nxt)
                    last_window.append(nxt)

                t_all  = list(range(len(series)))
                t_fore = list(range(len(series), len(series) + forecast_n))

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=t_all, y=series, mode='lines',
                                          line=dict(color=COLOR_SEQ[0], width=2),
                                          name='Observed'))
                fig.add_trace(go.Scatter(
                    x=list(range(split + seq_len, len(series))),
                    y=y_pred_te, mode='lines',
                    line=dict(color=COLOR_SEQ[1], width=2, dash='dot'),
                    name='Model Fit'
                ))
                fig.add_trace(go.Scatter(x=t_fore, y=forecast, mode='lines+markers',
                                          line=dict(color=COLOR_SEQ[2], width=2.5),
                                          marker=dict(size=6),
                                          name=f'{forecast_n}-step Forecast'))
                fig.update_layout(**PLOTLY_LAYOUT,
                                   title="RNN-Style Sequence Forecasting",
                                   xaxis_title="Timestep",
                                   yaxis_title="Value", height=380)
                st.plotly_chart(fig, use_container_width=True)

                from sklearn.metrics import mean_squared_error
                rmse = np.sqrt(mean_squared_error(yte, y_pred_te))
                st.metric("Test RMSE", f"{rmse:.4f}")
                _save_lab_progress("RNN Lab", 80)
            else:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=list(range(len(series))), y=series,
                                          mode='lines', line=dict(color=COLOR_SEQ[0], width=2),
                                          name='Signal'))
                fig.update_layout(**PLOTLY_LAYOUT, title="Input Signal", height=340)
                st.plotly_chart(fig, use_container_width=True)

    with tab_types:
        types = [
            ("🔁", "Vanilla RNN",   "#6C63FF",
             "Basic recurrent cell. Suffers from vanishing gradient for long sequences."),
            ("🚪", "LSTM",          "#43E97B",
             "Long Short-Term Memory. Gates control what to remember/forget. Best for long sequences."),
            ("⚡", "GRU",           "#FF6584",
             "Gated Recurrent Unit. Simpler than LSTM, often similar performance, trains faster."),
            ("↔️", "Bi-RNN",        "#38F9D7",
             "Processes sequence both forwards and backwards — richer context for NLP."),
            ("📚", "Seq2Seq",       "#FFA94D",
             "Encoder-Decoder architecture for translation, summarisation, Q&A."),
            ("🤗", "Transformer",   "#B48EFF",
             "Replaces RNN with self-attention. Basis of GPT, BERT, and all modern LLMs."),
        ]
        for i in range(0, len(types), 3):
            cols = st.columns(3)
            for col, (icon, name, color, desc) in zip(cols, types[i:i+3]):
                col.markdown(f"""
                <div class="ml-card" style="border-top:3px solid {color};">
                    <div style="font-size:1.8rem;margin-bottom:8px;">{icon}</div>
                    <h4 style="color:{color};margin:0 0 8px;font-size:0.92rem;">{name}</h4>
                    <p style="color:#8892A4;font-size:0.8rem;margin:0;line-height:1.4;">{desc}</p>
                </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# LAB 3 — Activation Functions
# ═══════════════════════════════════════════════
def _activation_lab():
    st.markdown(section_header("⚡", "Activation Functions Lab",
                "The non-linearity that makes neural networks powerful"), unsafe_allow_html=True)

    activations = {
        "Sigmoid":     (lambda x: 1/(1+np.exp(-np.clip(x,-500,500))),
                        "σ(x) = 1/(1+e⁻ˣ)", "Output: 0–1. Used in binary classification output."),
        "Tanh":        (lambda x: np.tanh(x),
                        "tanh(x) = (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ)", "Output: -1 to 1. Better than Sigmoid for hidden layers."),
        "ReLU":        (lambda x: np.maximum(0, x),
                        "ReLU(x) = max(0,x)", "Output: 0 to ∞. Most popular. Fast, avoids vanishing gradient."),
        "Leaky ReLU":  (lambda x: np.where(x>0, x, 0.01*x),
                        "LReLU(x) = x if x>0 else 0.01x", "Fixes 'dying ReLU' problem with small gradient for x<0."),
        "ELU":         (lambda x: np.where(x>0, x, 1.0*(np.exp(x)-1)),
                        "ELU(x) = x if x>0 else α(eˣ-1)", "Smooth version of ReLU. Negative values have gradient."),
        "Softmax":     (None,
                        "softmax(x)ᵢ = eˣⁱ / Σeˣʲ", "Converts logits to probabilities. Always used in output for multi-class."),
        "GELU":        (lambda x: x * 0.5 * (1 + np.tanh(np.sqrt(2/np.pi)*(x+0.044715*x**3))),
                        "GELU(x) ≈ x·Φ(x)", "Used in BERT, GPT. Smooth approximation combining ReLU + Dropout."),
        "Swish":       (lambda x: x * (1/(1+np.exp(-np.clip(x,-500,500)))),
                        "Swish(x) = x·σ(x)", "Google's self-gated activation. Outperforms ReLU on deep networks."),
    }

    col_sel, col_chart = st.columns([1, 2])
    with col_sel:
        selected = st.multiselect(
            "Select activations to compare:",
            list(activations.keys()),
            default=["ReLU", "Sigmoid", "Tanh", "Leaky ReLU"]
        )
        x_range = st.slider("X Range", 2, 10, 5)
        show_deriv = st.checkbox("Show Derivatives", False)

    x = np.linspace(-x_range, x_range, 400)

    with col_chart:
        fig = go.Figure()
        for i, name in enumerate(selected):
            fn = activations[name][0]
            if fn is None:
                continue
            y = fn(x)
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines',
                                      line=dict(color=COLOR_SEQ[i % len(COLOR_SEQ)], width=2.5),
                                      name=name))
            if show_deriv:
                dy = np.gradient(y, x)
                fig.add_trace(go.Scatter(x=x, y=dy, mode='lines',
                                          line=dict(color=COLOR_SEQ[i % len(COLOR_SEQ)],
                                                    width=1.5, dash='dot'),
                                          name=f"{name}′", opacity=0.6))

        fig.add_hline(y=0, line_color="rgba(136,146,164,0.3)", line_width=1)
        fig.add_vline(x=0, line_color="rgba(136,146,164,0.3)", line_width=1)
        fig.update_layout(**PLOTLY_LAYOUT,
                           title="Activation Function Comparison",
                           xaxis_title="x (pre-activation)", yaxis_title="f(x)",
                           yaxis=dict(range=[-2, 3]), height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Detail cards
    if selected:
        cols = st.columns(min(len(selected), 4))
        for col, name in zip(cols, selected[:4]):
            _, formula, desc = activations[name]
            col.markdown(f"""
            <div class="ml-card" style="min-height:120px;">
                <b style="color:#6C63FF;font-size:0.88rem;">{name}</b>
                <div style="font-family:monospace;color:#43E97B;font-size:0.75rem;
                            margin:6px 0;background:rgba(67,233,123,0.05);
                            padding:6px;border-radius:6px;">{formula}</div>
                <p style="color:#8892A4;font-size:0.78rem;margin:0;line-height:1.4;">{desc}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown(info_box(
        "🏆 <b>Rule of thumb:</b> Use <b>ReLU</b> (or GELU/Swish) in hidden layers. "
        "Use <b>Sigmoid</b> for binary output. Use <b>Softmax</b> for multi-class output. "
        "Never use Sigmoid/Tanh in very deep networks — vanishing gradient kills learning!"
    ), unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# LAB 4 — Loss Functions
# ═══════════════════════════════════════════════
def _loss_lab():
    st.markdown(section_header("📉", "Loss Functions Explorer",
                "How the model measures its own mistakes"), unsafe_allow_html=True)

    tab_reg, tab_cls = st.tabs(["📊 Regression Losses", "🎯 Classification Losses"])

    with tab_reg:
        col_c, col_ch = st.columns([1, 2])
        with col_c:
            noise  = st.slider("Outlier Magnitude", 1, 10, 3, key="loss_noise")
            n_out  = st.slider("Number of Outliers", 0, 20, 5, key="loss_nout")

        rng = np.random.RandomState(42)
        n   = 80
        y_true = rng.randn(n) * 2
        y_pred = y_true + rng.randn(n) * 0.8
        if n_out > 0:
            out_idx = rng.choice(n, n_out, replace=False)
            y_true[out_idx] += rng.randn(n_out) * noise * 3

        resid = y_true - y_pred
        mse   = np.mean(resid**2)
        rmse  = np.sqrt(mse)
        mae   = np.mean(np.abs(resid))
        huber_d = 1.0
        huber = np.mean(np.where(np.abs(resid) <= huber_d,
                                   0.5*resid**2,
                                   huber_d*(np.abs(resid)-0.5*huber_d)))

        with col_ch:
            fig = go.Figure()
            r_range = np.linspace(-6, 6, 300)
            for name, fn, color in [
                ("MSE",          lambda r: r**2,                               COLOR_SEQ[0]),
                ("MAE",          lambda r: np.abs(r),                          COLOR_SEQ[1]),
                ("Huber (δ=1)",  lambda r: np.where(np.abs(r)<=1, 0.5*r**2, np.abs(r)-0.5), COLOR_SEQ[2]),
            ]:
                fig.add_trace(go.Scatter(x=r_range, y=fn(r_range), mode='lines',
                                          line=dict(color=color, width=2.5), name=name))
            fig.update_layout(**PLOTLY_LAYOUT, title="Loss vs Residual",
                               xaxis_title="Residual (y - ŷ)", yaxis_title="Loss",
                               yaxis=dict(range=[0, 12]), height=320)
            st.plotly_chart(fig, use_container_width=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("MSE",   f"{mse:.3f}")
        m2.metric("RMSE",  f"{rmse:.3f}")
        m3.metric("MAE",   f"{mae:.3f}")
        m4.metric("Huber", f"{huber:.3f}")

        st.markdown(info_box(
            "⚠️ Notice: When outliers increase, <b>MSE explodes</b> (squares errors) "
            "but <b>MAE</b> stays more stable. <b>Huber loss</b> gives the best of both!"
        ), unsafe_allow_html=True)

    with tab_cls:
        st.markdown("#### 🎯 Binary Cross-Entropy Visualizer")
        y_prob = np.linspace(0.01, 0.99, 300)
        bce_1  = -np.log(y_prob)        # when true label = 1
        bce_0  = -np.log(1 - y_prob)   # when true label = 0

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_prob, y=bce_1, mode='lines',
                                  line=dict(color=COLOR_SEQ[0], width=2.5),
                                  name='BCE when y=1 (penalise low probability)'))
        fig.add_trace(go.Scatter(x=y_prob, y=bce_0, mode='lines',
                                  line=dict(color=COLOR_SEQ[2], width=2.5),
                                  name='BCE when y=0 (penalise high probability)'))
        fig.update_layout(**PLOTLY_LAYOUT,
                           title="Binary Cross-Entropy Loss",
                           xaxis_title="Predicted Probability", yaxis_title="Loss",
                           yaxis=dict(range=[0, 5]), height=320)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(concept_box(
            "Cross-Entropy Formula",
            "BCE = -[y·log(p) + (1-y)·log(1-p)]<br><br>"
            "If true label y=1 and model predicts p=0.9 → low loss ✅<br>"
            "If true label y=1 and model predicts p=0.1 → huge loss ❌<br>"
            "The model is punished heavily for confident wrong predictions!",
            "📐"
        ), unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# LAB 5 — GAN
# ═══════════════════════════════════════════════
def _gan_lab():
    st.markdown(section_header("🎨", "GAN — Generative Adversarial Network",
                "Two networks battle it out to create realistic data"), unsafe_allow_html=True)

    st.markdown(concept_box(
        "What is a GAN?",
        "A GAN has two networks competing:<br>"
        "🎨 <b>Generator</b>: Creates fake data from random noise, trying to fool the discriminator.<br>"
        "🔍 <b>Discriminator</b>: Tries to tell real data from fake.<br>"
        "They train together — the generator improves until its fakes are indistinguishable from real!",
        "⚔️"
    ), unsafe_allow_html=True)

    tab_viz, tab_sim, tab_apps = st.tabs(["🏗️ Architecture", "🎮 Simulation", "🌍 Applications"])

    with tab_viz:
        fig = _draw_gan_diagram()
        st.plotly_chart(fig, use_container_width=True)

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("""
            <div class="ml-card" style="border-top:3px solid #6C63FF;">
                <h4 style="color:#6C63FF;">🎨 Generator Network</h4>
                <ul style="color:#8892A4;font-size:0.82rem;padding-left:16px;line-height:1.8;">
                    <li>Input: Random noise vector (latent space z)</li>
                    <li>Output: Fake image / data</li>
                    <li>Goal: Fool the discriminator (D(G(z)) → 1)</li>
                    <li>Loss: -log(D(G(z)))</li>
                </ul>
            </div>""", unsafe_allow_html=True)
        with col_r:
            st.markdown("""
            <div class="ml-card" style="border-top:3px solid #43E97B;">
                <h4 style="color:#43E97B;">🔍 Discriminator Network</h4>
                <ul style="color:#8892A4;font-size:0.82rem;padding-left:16px;line-height:1.8;">
                    <li>Input: Real or fake data</li>
                    <li>Output: Probability of being real (0–1)</li>
                    <li>Goal: Correctly classify real vs fake</li>
                    <li>Loss: -[log(D(x)) + log(1-D(G(z)))]</li>
                </ul>
            </div>""", unsafe_allow_html=True)

    with tab_sim:
        st.markdown("#### 🎮 Simulate GAN Training on 1D Distribution")
        col_ctrl, col_out = st.columns([1, 2])
        with col_ctrl:
            target_dist   = st.selectbox("Real Data Distribution:", ["Gaussian", "Bimodal", "Uniform"])
            training_iters = st.slider("Training Iterations", 10, 200, 50)
            noise_dim      = st.slider("Noise Dimension",       2,  32,  8)
            start_btn      = st.button("▶ Simulate", use_container_width=True, key="gan_sim")

        if start_btn:
            rng = np.random.RandomState(42)
            if target_dist == "Gaussian":
                real = rng.normal(3, 1, 500)
            elif target_dist == "Bimodal":
                real = np.concatenate([rng.normal(-2, 0.5, 250), rng.normal(4, 0.7, 250)])
            else:
                real = rng.uniform(-3, 6, 500)

            # Simulate generator improving over training
            with col_out:
                fake_history = []
                for it in [1, training_iters//4, training_iters//2, training_iters]:
                    progress = it / training_iters
                    gen_mean = 0 + progress * np.mean(real)
                    gen_std  = 3 - progress * (3 - np.std(real))
                    fake = rng.normal(gen_mean, max(gen_std, 0.3), 500)
                    fake_history.append((it, fake))

                fig = go.Figure()
                alphas = [0.2, 0.4, 0.7, 1.0]
                for (it, fake), alpha in zip(fake_history, alphas):
                    fig.add_trace(go.Histogram(
                        x=fake, name=f"Generator iter {it}",
                        opacity=alpha, histnorm='probability density',
                        nbinsx=40,
                        marker_color=f"rgba(108,99,255,{alpha})"
                    ))
                fig.add_trace(go.Histogram(
                    x=real, name="Real Data",
                    opacity=0.5, histnorm='probability density',
                    nbinsx=40, marker_color="rgba(67,233,123,0.6)"
                ))
                fig.update_layout(**PLOTLY_LAYOUT, barmode='overlay',
                                   title="Generator Learning Real Distribution",
                                   xaxis_title="Value", yaxis_title="Density", height=380)
                st.plotly_chart(fig, use_container_width=True)

            st.markdown(success_box(
                "✅ Notice how the generator distribution (purple) gets closer to the "
                "real data distribution (green) as training progresses!"
            ), unsafe_allow_html=True)
            _save_lab_progress("GAN Lab", 75)

    with tab_apps:
        apps = [
            ("🖼️", "Image Synthesis",   "DALL-E 2, Stable Diffusion, Midjourney — all use GAN/Diffusion ideas"),
            ("🎭", "Face Generation",   "ThisPersonDoesNotExist.com — realistic faces of non-existent people"),
            ("🎵", "Music Generation",  "MuseGAN generates multi-track music in various styles"),
            ("🧬", "Drug Discovery",    "GANs generate candidate drug molecules for disease treatment"),
            ("🎮", "Game Assets",       "Auto-generating textures, environments, and characters for games"),
            ("🔍", "Data Augmentation", "Generate extra training samples to improve model performance"),
            ("🎬", "DeepFakes",         "Video face-swapping — raises ethical concerns about authenticity"),
            ("👗", "Fashion Design",    "AI-generated clothing designs for rapid prototyping"),
        ]
        cols = st.columns(4)
        for i, (icon, title, desc) in enumerate(apps):
            cols[i % 4].markdown(f"""
            <div class="app-card" style="margin-bottom:12px;">
                <span class="icon">{icon}</span>
                <h4>{title}</h4>
                <p>{desc}</p>
            </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# LAB 6 — Transfer Learning
# ═══════════════════════════════════════════════
def _transfer_lab():
    st.markdown(section_header("🔁", "Transfer Learning",
                "Don't start from scratch — reuse knowledge!"), unsafe_allow_html=True)

    st.markdown(concept_box(
        "What is Transfer Learning?",
        "Instead of training from scratch, we take a model pre-trained on a large dataset "
        "(like ImageNet with 14M images) and <b>fine-tune</b> it on our small dataset. "
        "The model already knows edges, textures, and shapes — we just teach it our specific classes!",
        "🔁"
    ), unsafe_allow_html=True)

    tab_concept, tab_strategy, tab_demo = st.tabs(
        ["💡 Why It Works", "🗺️ Strategies", "🧪 Accuracy Comparison"]
    )

    with tab_concept:
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("""
            <div class="ml-card">
                <h4 style="color:#FF6584;">❌ Training from Scratch</h4>
                <ul style="color:#8892A4;font-size:0.82rem;padding-left:16px;line-height:1.8;">
                    <li>Needs millions of labelled examples</li>
                    <li>Takes days/weeks to train</li>
                    <li>Requires expensive GPUs</li>
                    <li>Poor performance with small datasets</li>
                </ul>
            </div>""", unsafe_allow_html=True)
        with col_r:
            st.markdown("""
            <div class="ml-card">
                <h4 style="color:#43E97B;">✅ Transfer Learning</h4>
                <ul style="color:#8892A4;font-size:0.82rem;padding-left:16px;line-height:1.8;">
                    <li>Works with just 100–1000 examples</li>
                    <li>Trains in minutes on a laptop</li>
                    <li>State-of-the-art results immediately</li>
                    <li>The standard approach in industry</li>
                </ul>
            </div>""", unsafe_allow_html=True)

        famous_models = [
            ("🖼️", "VGG16/19",    "Oxford, 2014", "Image classification backbone"),
            ("🔍", "ResNet-50",   "Microsoft, 2015", "Skip connections solve vanishing gradient"),
            ("📱", "MobileNet",   "Google, 2017",    "Efficient CNN for mobile devices"),
            ("⚡", "EfficientNet","Google, 2019",    "Best accuracy/compute tradeoff"),
            ("🤗", "BERT",        "Google, 2018",    "Pre-trained transformer for NLP"),
            ("🚀", "GPT-4",       "OpenAI, 2023",    "Large language model for text tasks"),
        ]
        st.markdown("#### 🌟 Popular Pre-trained Models")
        cols = st.columns(3)
        for i, (icon, name, org, desc) in enumerate(famous_models):
            cols[i % 3].markdown(f"""
            <div class="ml-card" style="margin:6px 0;">
                <div style="display:flex;gap:10px;align-items:center;">
                    <span style="font-size:1.5rem;">{icon}</span>
                    <div>
                        <b style="color:#F0F4FF;font-size:0.88rem;">{name}</b>
                        <div style="color:#6C63FF;font-size:0.72rem;">{org}</div>
                        <div style="color:#8892A4;font-size:0.75rem;">{desc}</div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

    with tab_strategy:
        strategies = [
            ("❄️ Freeze All", "Use pretrained model as fixed feature extractor. Only train final layer. Best when: very small dataset, similar domain."),
            ("🌡️ Partial Fine-tune", "Freeze early layers, fine-tune later layers. Good balance for medium datasets."),
            ("🔥 Full Fine-tune", "Unfreeze all layers and train end-to-end with small LR. Best for large datasets, different domain."),
            ("🔀 Domain Adaptation", "Specifically adapt the model to handle distribution shift between source and target domains."),
        ]
        for icon_n, (title, desc) in zip(["❄️","🌡️","🔥","🔀"], strategies):
            st.markdown(f"""
            <div class="info-box">
                <b style="color:#F0F4FF;">{title}</b>
                <p style="color:#8892A4;font-size:0.85rem;margin:4px 0 0 0;">{desc}</p>
            </div>""", unsafe_allow_html=True)

    with tab_demo:
        st.markdown("#### 📊 Transfer Learning vs From Scratch")
        dataset_sizes = [50, 100, 500, 1000, 5000, 10000]
        scratch_acc   = [0.38, 0.45, 0.60, 0.70, 0.82, 0.88]
        transfer_acc  = [0.72, 0.80, 0.88, 0.91, 0.94, 0.96]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dataset_sizes, y=scratch_acc, mode='lines+markers',
                                  name='From Scratch', line=dict(color=COLOR_SEQ[2], width=2.5)))
        fig.add_trace(go.Scatter(x=dataset_sizes, y=transfer_acc, mode='lines+markers',
                                  name='Transfer Learning', line=dict(color=COLOR_SEQ[1], width=2.5)))
        fig.update_layout(**PLOTLY_LAYOUT,
                           title="Dataset Size vs Accuracy: Transfer vs From Scratch",
                           xaxis_title="Training Dataset Size", yaxis_title="Accuracy",
                           yaxis=dict(range=[0.3, 1.0]),
                           xaxis_type='log', height=360)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(success_box(
            "✅ Transfer learning gives <b>far better accuracy</b> especially with small datasets. "
            "With only 100 examples, transfer learning achieves 80% vs 45% from scratch!"
        ), unsafe_allow_html=True)
        _save_lab_progress("Transfer Learning", 100)


# ═══════════════════════════════════════════════
# LAB 7 — Attention Visualizer
# ═══════════════════════════════════════════════
def _attention_lab():
    st.markdown(section_header("🤗", "Transformer Attention Visualizer",
                "How BERT and GPT 'pay attention' to words"), unsafe_allow_html=True)

    st.markdown(concept_box(
        "What is Self-Attention?",
        "Instead of processing words one-by-one like RNNs, Transformers look at "
        "<b>all words simultaneously</b> and learn which words to 'attend to' for each prediction. "
        "This is why GPT can understand long-range dependencies — 'The cat that sat on the mat <b>was</b> happy' "
        "— 'was' attends to 'cat' not 'mat'!",
        "👀"
    ), unsafe_allow_html=True)

    tab_viz, tab_formula, tab_apps = st.tabs(
        ["👁️ Attention Heatmap", "📐 Math Behind It", "🌍 Real Models"]
    )

    with tab_viz:
        example_sentences = {
            "The bank by the river was steep.":
                "bank river was steep The by the",
            "She gave him the book because he asked.":
                "She gave him book because he asked",
            "The dog didn't chase the cat because it was tired.":
                "dog cat tired because it didn't chase",
            "I ate the pizza although it was cold.":
                "pizza cold although ate it I was",
        }

        sentence = st.selectbox("📝 Example sentence:", list(example_sentences.keys()))
        n_heads  = st.slider("Attention Heads", 1, 8, 4)
        layer    = st.slider("Transformer Layer", 1, 12, 3)

        words = sentence.split()
        n_w   = len(words)

        fig = go.Figure()
        rng  = np.random.RandomState(hash(sentence + str(n_heads)) % 10000)

        for head in range(n_heads):
            attn = rng.dirichlet(np.ones(n_w), n_w)
            # Make attention meaningful (focus on semantically similar words)
            for i in range(n_w):
                attn[i, i] += 0.3
                if i > 0:       attn[i, i-1] += 0.15
                if i < n_w - 1: attn[i, i+1] += 0.15
                attn[i] /= attn[i].sum()

        # Show one combined head
        combined = np.zeros((n_w, n_w))
        for head in range(n_heads):
            rng2 = np.random.RandomState(hash(sentence + str(head)) % 10000)
            a    = rng2.dirichlet(np.ones(n_w), n_w)
            combined += a
        combined /= n_heads

        fig = px.imshow(combined, x=words, y=words,
                         color_continuous_scale="Viridis",
                         title=f"Attention Heatmap — Layer {layer}, {n_heads} Heads Combined",
                         labels=dict(x="Key (attended to)", y="Query (attending from)",
                                     color="Attention Weight"))
        fig.update_layout(**PLOTLY_LAYOUT, height=420)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(info_box(
            "🔍 <b>Reading the heatmap:</b> Each row is a word (query). "
            "Brighter cells = stronger attention to that column word (key). "
            "Diagonal = attending to itself. Off-diagonal = cross-word relationships!"
        ), unsafe_allow_html=True)

    with tab_formula:
        st.markdown("#### 📐 Attention Formula")
        st.markdown("""
        <div class="ml-card" style="padding:28px;">
            <h4 style="color:#43E97B;margin:0 0 16px 0;">Scaled Dot-Product Attention</h4>
            <div style="background:rgba(108,99,255,0.1);padding:20px;border-radius:12px;
                        text-align:center;font-family:monospace;font-size:1.1rem;
                        color:#B48EFF;letter-spacing:1px;">
                Attention(Q, K, V) = softmax( QKᵀ / √d_k ) · V
            </div>
            <div style="margin-top:20px;display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;">
                <div>
                    <b style="color:#6C63FF;">Q (Query)</b>
                    <p style="color:#8892A4;font-size:0.82rem;margin:4px 0;">
                        "What am I looking for?" — current word's question
                    </p>
                </div>
                <div>
                    <b style="color:#43E97B;">K (Key)</b>
                    <p style="color:#8892A4;font-size:0.82rem;margin:4px 0;">
                        "What do I contain?" — other words' labels
                    </p>
                </div>
                <div>
                    <b style="color:#FF6584;">V (Value)</b>
                    <p style="color:#8892A4;font-size:0.82rem;margin:4px 0;">
                        "What do I give?" — actual content to aggregate
                    </p>
                </div>
            </div>
            <div style="margin-top:16px;color:#8892A4;font-size:0.82rem;">
                <b style="color:#FFA94D;">√d_k scaling:</b> Prevents dot products from getting too large
                when embedding dimension d_k is big — keeps gradients stable.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with tab_apps:
        models = [
            ("🤗", "BERT",         "Bidirectional encoder. Used for classification, NER, Q&A. Base: 110M params."),
            ("🚀", "GPT-4",        "Autoregressive decoder. Generates text. Powers ChatGPT. ~1.8T params."),
            ("🔤", "T5",           "Text-to-Text Transfer Transformer. Converts every NLP task to text generation."),
            ("🌐", "mBERT",        "Multilingual BERT. Supports 104 languages including Hindi."),
            ("👁️", "ViT",          "Vision Transformer — applies transformers to image patches instead of pixels."),
            ("🎵", "Music Transformer", "Generates long-range musical structure using relative attention."),
            ("💊", "AlphaFold 2",  "Uses transformers to predict 3D protein structures — Nobel Prize level."),
            ("🧬", "ESM-2",        "Protein language model — understands amino acid sequences like text."),
        ]
        cols = st.columns(4)
        for i, (icon, name, desc) in enumerate(models):
            cols[i % 4].markdown(f"""
            <div class="app-card" style="margin-bottom:12px;">
                <span class="icon">{icon}</span>
                <h4>{name}</h4>
                <p>{desc}</p>
            </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# LAB 8 — Training Dynamics
# ═══════════════════════════════════════════════
def _training_dynamics_lab():
    st.markdown(section_header("🏋️", "Training Dynamics Lab",
                "Understand what happens inside training: LR, momentum, batch size"), unsafe_allow_html=True)

    tab_lr, tab_batch, tab_reg = st.tabs(
        ["📉 Learning Rate", "📦 Batch Size", "🛡️ Regularisation"]
    )

    with tab_lr:
        st.markdown("#### 📉 Learning Rate Schedules")
        col_c, col_ch = st.columns([1, 2])
        with col_c:
            init_lr   = st.select_slider("Initial LR", [0.1, 0.01, 0.001], value=0.1)
            schedule  = st.selectbox("Schedule:", [
                "Constant", "Step Decay", "Exponential Decay",
                "Cosine Annealing", "Warmup + Cosine"
            ])
            epochs_lr = st.slider("Epochs", 20, 200, 100)

        t = np.arange(epochs_lr)
        if schedule == "Constant":
            lr = np.full(epochs_lr, init_lr)
        elif schedule == "Step Decay":
            lr = init_lr * (0.1 ** (t // (epochs_lr // 3)))
        elif schedule == "Exponential Decay":
            lr = init_lr * np.exp(-0.05 * t / epochs_lr * 10)
        elif schedule == "Cosine Annealing":
            lr = init_lr * 0.5 * (1 + np.cos(np.pi * t / epochs_lr))
        else:  # Warmup + Cosine
            warmup = epochs_lr // 10
            lr_warmup = np.linspace(0, init_lr, warmup)
            lr_cos    = init_lr * 0.5 * (1 + np.cos(np.pi * np.arange(epochs_lr - warmup) / (epochs_lr - warmup)))
            lr = np.concatenate([lr_warmup, lr_cos])

        with col_ch:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t, y=lr, mode='lines',
                                      line=dict(color=COLOR_SEQ[0], width=2.5), name='LR'))
            fig.update_layout(**PLOTLY_LAYOUT, title=f"{schedule} Schedule",
                               xaxis_title="Epoch", yaxis_title="Learning Rate", height=300)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown(info_box(
            "💡 <b>Cosine Annealing</b> and <b>Warmup</b> are the most popular modern schedules. "
            "Warmup prevents early instability; cosine decay improves final convergence."
        ), unsafe_allow_html=True)

    with tab_batch:
        st.markdown("#### 📦 Batch Size Trade-offs")
        batch_sizes = [1, 8, 32, 128, 512, 2048]
        noise  = [1.0, 0.65, 0.35, 0.20, 0.12, 0.08]
        speed  = [0.05, 0.25, 0.65, 0.85, 0.95, 1.0]
        memory = [0.02, 0.08, 0.20, 0.45, 0.80, 1.0]
        gen    = [0.92, 0.88, 0.85, 0.78, 0.70, 0.62]

        fig = go.Figure()
        for vals, name, color in [
            (noise,  "Gradient Noise (lower=stable)", COLOR_SEQ[0]),
            (speed,  "Training Speed",                COLOR_SEQ[1]),
            (gen,    "Generalisation",                COLOR_SEQ[2]),
            (memory, "Memory Usage",                  COLOR_SEQ[3]),
        ]:
            fig.add_trace(go.Scatter(x=[str(b) for b in batch_sizes], y=vals,
                                      mode='lines+markers', name=name,
                                      line=dict(color=color, width=2)))
        fig.update_layout(**PLOTLY_LAYOUT,
                           title="Batch Size Trade-offs (Relative Scale)",
                           xaxis_title="Batch Size", yaxis_title="Relative Value",
                           yaxis=dict(range=[0, 1.1]), height=340)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(warning_box(
            "⚠️ Large batches train faster but <b>generalise worse</b> (sharp minima). "
            "Small batches are noisy but find <b>flatter minima</b> that transfer better. "
            "Common sweet spot: <b>32–256</b>."
        ), unsafe_allow_html=True)

    with tab_reg:
        st.markdown("#### 🛡️ Regularisation Techniques")
        x_plot = np.linspace(-3, 3, 300)
        lam = st.slider("Regularisation Strength (λ)", 0.0, 2.0, 0.5, 0.1)
        weights = np.linspace(-3, 3, 300)
        loss_base = weights**2 * 0.5

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=weights, y=loss_base,
                                  name='Loss (no reg)', line=dict(color=COLOR_SEQ[0], width=2)))
        fig.add_trace(go.Scatter(x=weights, y=loss_base + lam * weights**2,
                                  name=f'L2 (Ridge) λ={lam}',
                                  line=dict(color=COLOR_SEQ[1], width=2)))
        fig.add_trace(go.Scatter(x=weights, y=loss_base + lam * np.abs(weights),
                                  name=f'L1 (Lasso) λ={lam}',
                                  line=dict(color=COLOR_SEQ[2], width=2, dash='dot')))
        fig.update_layout(**PLOTLY_LAYOUT, title="Effect of Regularisation on Loss",
                           xaxis_title="Weight Value", yaxis_title="Loss",
                           yaxis=dict(range=[0, 10]), height=300)
        st.plotly_chart(fig, use_container_width=True)

        reg_types = [
            ("L2 (Ridge)",   "Penalises large weights: +λΣw². Keeps weights small. Never zero."),
            ("L1 (Lasso)",   "Penalises absolute weights: +λΣ|w|. Can zero out weights → feature selection!"),
            ("Dropout",      "Randomly zero out neurons during training. Forces redundant representations."),
            ("Batch Norm",   "Normalises activations per batch. Stabilises training, allows higher LR."),
            ("Early Stop",   "Stop training when validation loss stops improving. Simplest regulariser."),
            ("Data Aug.",    "Artificially expand dataset with flips, crops, rotations — reduces overfitting."),
        ]
        cols = st.columns(3)
        for i, (name, desc) in enumerate(reg_types):
            cols[i % 3].markdown(f"""
            <div class="ml-card" style="min-height:100px;padding:16px;">
                <b style="color:#6C63FF;font-size:0.88rem;">{name}</b>
                <p style="color:#8892A4;font-size:0.78rem;margin:6px 0 0 0;line-height:1.4;">{desc}</p>
            </div>""", unsafe_allow_html=True)
        _save_lab_progress("Training Dynamics", 100)


# ═══════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════

def _convolve2d(image, kernel):
    kh, kw = kernel.shape
    oh = image.shape[0] - kh + 1
    ow = image.shape[1] - kw + 1
    output = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            output[i, j] = (image[i:i+kh, j:j+kw] * kernel).sum()
    return output


def _side_by_side_heatmaps(img1, img2, title1, title2):
    import plotly.subplots as sp
    fig = sp.make_subplots(rows=1, cols=2,
                            subplot_titles=[title1, title2])
    fig.add_trace(go.Heatmap(z=img1, colorscale='Gray', showscale=False), row=1, col=1)
    fig.add_trace(go.Heatmap(z=img2, colorscale='RdBu', showscale=False), row=1, col=2)
    custom_layout = {**PLOTLY_LAYOUT}

    custom_layout["xaxis"] = dict(
        showticklabels=False,
        showgrid=False,
        zeroline=False
    )

    custom_layout["yaxis"] = dict(
        showticklabels=False,
        showgrid=False,
        zeroline=False
    )

    custom_layout["xaxis2"] = dict(
        showticklabels=False,
        showgrid=False,
        zeroline=False
    )

    custom_layout["yaxis2"] = dict(
        showticklabels=False,
        showgrid=False,
        zeroline=False
    )

    fig.update_layout(
        **custom_layout,
        height=300
    )
    return fig


def _draw_cnn_arch():
    layer_names = ["Input\n28×28", "Conv\n26×26×32", "Pool\n13×13×32",
                   "Conv\n11×11×64", "Pool\n5×5×64", "Flatten\n1600", "Dense\n128", "Softmax\n10"]
    colors = [COLOR_SEQ[3], COLOR_SEQ[0], COLOR_SEQ[0], COLOR_SEQ[1], COLOR_SEQ[1],
              COLOR_SEQ[4], COLOR_SEQ[2], COLOR_SEQ[1]]

    fig = go.Figure()
    for i, (name, color) in enumerate(zip(layer_names, colors)):
        fig.add_trace(go.Bar(x=[i], y=[1], marker_color=color, width=0.7,
                              text=name.replace('\n', '<br>'), textposition='inside',
                              textfont=dict(size=9, color='white'),
                              showlegend=False))
        if i < len(layer_names) - 1:
            fig.add_annotation(x=i + 0.5, y=0.5, text="→",
                                showarrow=False, font=dict(size=18, color='#8892A4'))
    custom_layout = {**PLOTLY_LAYOUT}

    custom_layout["xaxis"] = dict(
        showticklabels=False,
        showgrid=False,
        zeroline=False
    )

    custom_layout["yaxis"] = dict(
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        range=[0, 1.4]
    )

    fig.update_layout(
        **custom_layout,
        title="CNN Architecture",
        height=180,
        barmode='group'
    )
    return fig


def _draw_rnn_diagram():
    steps  = 5
    labels = ["x₁", "x₂", "x₃", "x₄", "x₅"]
    hiddens = ["h₁", "h₂", "h₃", "h₄", "h₅"]
    outputs = ["y₁", "y₂", "y₃", "y₄", "y₅"]

    fig = go.Figure()
    xs = list(range(steps))

    # Edges: input → hidden
    for x in xs:
        fig.add_shape(type='line', x0=x, y0=0, x1=x, y1=1,
                       line=dict(color='rgba(108,99,255,0.5)', width=1.5))
        fig.add_shape(type='line', x0=x, y0=1, x1=x, y1=2,
                       line=dict(color='rgba(67,233,123,0.5)', width=1.5))
    # Recurrent edges
    for x in range(steps - 1):
        fig.add_shape(type='line', x0=x, y0=1, x1=x+1, y1=1,
                       line=dict(color='rgba(255,101,132,0.7)', width=2, dash='dot'))

    # Nodes
    for x, (lbl, hid, out) in enumerate(zip(labels, hiddens, outputs)):
        fig.add_trace(go.Scatter(x=[x], y=[0], mode='markers+text',
                                  marker=dict(size=28, color=COLOR_SEQ[3]),
                                  text=[lbl], textposition='middle center',
                                  showlegend=False, textfont=dict(size=10)))
        fig.add_trace(go.Scatter(x=[x], y=[1], mode='markers+text',
                                  marker=dict(size=36, color=COLOR_SEQ[0]),
                                  text=[hid], textposition='middle center',
                                  showlegend=False, textfont=dict(size=10)))
        fig.add_trace(go.Scatter(x=[x], y=[2], mode='markers+text',
                                  marker=dict(size=28, color=COLOR_SEQ[1]),
                                  text=[out], textposition='middle center',
                                  showlegend=False, textfont=dict(size=10)))

    fig.update_layout(**PLOTLY_LAYOUT, height=280, title="Unrolled RNN",
                       xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
                       yaxis=dict(showgrid=False, showticklabels=False, zeroline=False,
                                  range=[-0.5, 2.5]),
                       annotations=[
                           dict(x=-0.5, y=0, text="Input", showarrow=False,
                                font=dict(color='#8892A4', size=10)),
                           dict(x=-0.5, y=1, text="Hidden", showarrow=False,
                                font=dict(color='#8892A4', size=10)),
                           dict(x=-0.5, y=2, text="Output", showarrow=False,
                                font=dict(color='#8892A4', size=10)),
                       ])
    return fig


def _draw_gan_diagram():
    fig = go.Figure()

    # Boxes: Noise → Generator → Fake Data → Discriminator → Real/Fake
    boxes = [
        (0.0, "🎲 Random\nNoise", "#8892A4"),
        (0.2, "🎨 Generator\n(G)", "#6C63FF"),
        (0.4, "🖼️ Fake\nData", "#B48EFF"),
        (0.6, "🔍 Discriminator\n(D)", "#43E97B"),
        (0.8, "❓ Real or\nFake?", "#FF6584"),
    ]
    # Real data branch
    real_boxes = [
        (0.4, "📷 Real\nData", "#43E97B"),
    ]

    for x, label, color in boxes:
        fig.add_shape(type='rect', x0=x-0.08, y0=0.35, x1=x+0.08, y1=0.65,
                       fillcolor=color, opacity=0.8,
                       line=dict(color='white', width=1))
        fig.add_annotation(x=x, y=0.5, text=label.replace('\n','<br>'),
                            showarrow=False, font=dict(color='white', size=9),
                            bgcolor='rgba(0,0,0,0)')

    # Real data box (below)
    fig.add_shape(type='rect', x0=0.32, y0=0.05, x1=0.48, y1=0.28,
                   fillcolor='#43E97B', opacity=0.7,
                   line=dict(color='white', width=1))
    fig.add_annotation(x=0.4, y=0.165, text="📷 Real<br>Data",
                        showarrow=False, font=dict(color='white', size=9))

    # Arrows
    for x in [0.12, 0.32, 0.52, 0.72]:
        fig.add_annotation(x=x, y=0.5, text="→", showarrow=False,
                            font=dict(size=20, color='#8892A4'))
    fig.add_annotation(x=0.6, y=0.3, text="↑", showarrow=False,
                        font=dict(size=20, color='#43E97B'))

    # Feedback arrow (discriminator → generator)
    fig.add_annotation(x=0.4, y=0.85, text="🔴 Loss Gradient flows back to Generator",
                        showarrow=False, font=dict(color='#FF6584', size=10))
    fig.add_shape(type='path',
                   path='M 0.8,0.65 L 0.8,0.9 L 0.2,0.9 L 0.2,0.65',
                   line=dict(color='rgba(255,101,132,0.6)', width=1.5, dash='dot'))

    fig.update_layout(**PLOTLY_LAYOUT, title="GAN Architecture",
                       height=320,
                       xaxis=dict(range=[-0.1,1.0], showgrid=False,
                                  showticklabels=False, zeroline=False),
                       yaxis=dict(range=[-0.05,1.0], showgrid=False,
                                  showticklabels=False, zeroline=False))
    return fig


def _save_lab_progress(lab_name: str, pct: int):
    pass  # progress saved via main app
