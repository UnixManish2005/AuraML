"""
pages/regression.py
Linear Regression Interactive Visualizer
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from utils.styles import hero, concept_box, section_header, info_box, success_box
from utils.data_helpers import get_house_price_data, PLOTLY_LAYOUT, COLOR_SEQ


def render():
    st.markdown(hero(
        "Linear Regression 📈",
        "Draw the best-fit line through data. Predict house prices visually — understand slope, intercept, and error.",
        "📈"
    ), unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(
        ["🏠 House Price Predictor", "🎛️ Experiment Lab", "💡 Concepts"]
    )

    # ─────────────────────────────────────────────
    # TAB 1: House Price Predictor
    # ─────────────────────────────────────────────
    with tab1:
        st.markdown(section_header("🏠", "House Price Prediction",
                    "See how size predicts price"), unsafe_allow_html=True)

        col_ctrl, col_chart = st.columns([1, 3])

        with col_ctrl:
            #noise  = st.slider("Data Noise", 10, 100, 40, key="reg_noise")
            noise = st.slider("Noise level", 0.1, 3.0, 1.0, key="mod_reg_noise")
            n_pts  = st.slider("# Data Points", 30, 300, 100, key="reg_n")
            show_errors = st.checkbox("Show Error Lines", True)
            show_ci     = st.checkbox("Show Confidence Band", True)

        X_raw, y_raw = _gen_house_data(n_pts, noise)

        # Fit OLS manually for transparency
        slope, intercept = _ols(X_raw, y_raw)
        y_pred = slope * X_raw + intercept
        residuals = y_raw - y_pred
        rmse  = np.sqrt(np.mean(residuals**2))
        r2    = 1 - np.sum(residuals**2) / np.sum((y_raw - y_raw.mean())**2)

        with col_chart:
            fig = go.Figure()

            # Confidence band
            if show_ci:
                x_sorted = np.sort(X_raw)
                y_fit    = slope * x_sorted + intercept
                se       = rmse * np.sqrt(1/len(X_raw) + (x_sorted - X_raw.mean())**2 /
                                          np.sum((X_raw - X_raw.mean())**2))
                fig.add_trace(go.Scatter(
                    x=np.concatenate([x_sorted, x_sorted[::-1]]),
                    y=np.concatenate([y_fit + 2*se, (y_fit - 2*se)[::-1]]),
                    fill='toself',
                    fillcolor='rgba(108,99,255,0.1)',
                    line=dict(color='rgba(0,0,0,0)'),
                    name='95% CI', showlegend=True
                ))

            # Error lines
            if show_errors:
                for xi, yi, yp in zip(X_raw[:40], y_raw[:40], y_pred[:40]):
                    fig.add_shape(type="line",
                        x0=xi, y0=yi, x1=xi, y1=yp,
                        line=dict(color="rgba(255,101,132,0.35)", width=1))

            # Data points
            fig.add_trace(go.Scatter(
                x=X_raw, y=y_raw, mode='markers',
                marker=dict(color=COLOR_SEQ[0], size=6, opacity=0.7),
                name='Training Data'
            ))

            # Regression line
            x_line = np.linspace(X_raw.min(), X_raw.max(), 200)
            fig.add_trace(go.Scatter(
                x=x_line, y=slope * x_line + intercept,
                mode='lines',
                line=dict(color=COLOR_SEQ[1], width=3),
                name=f'Best Fit: y = {slope:.1f}x + {intercept:.0f}'
            ))

            fig.update_layout(**PLOTLY_LAYOUT,
                              title="House Size vs Price",
                              xaxis_title="Size (sq ft)",
                              yaxis_title="Price ($)",
                              height=420)
            st.plotly_chart(fig, use_container_width=True)

        # Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Slope",     f"{slope:.2f}")
        m2.metric("Intercept", f"${intercept:,.0f}")
        m3.metric("RMSE",      f"${rmse:,.0f}")
        m4.metric("R² Score",  f"{r2:.3f}")

        st.markdown(concept_box(
            "What does R² mean?",
            f"R² = <b>{r2:.3f}</b> means the model explains "
            f"<b>{r2*100:.1f}%</b> of the variation in house prices. "
            "R² closer to 1.0 means a better fit!",
            "📏"
        ), unsafe_allow_html=True)

        # Predictor
        st.markdown("#### 🔮 Make a Prediction")
        inp_size = st.number_input("Enter house size (sq ft):", 500, 5000, 1500, 50)
        pred_price = slope * inp_size + intercept
        st.markdown(success_box(
            f"🏠 A <b>{inp_size} sq ft</b> house is predicted to cost <b>${pred_price:,.0f}</b>"
        ), unsafe_allow_html=True)

    # ─────────────────────────────────────────────
    # TAB 2: Experiment Lab
    # ─────────────────────────────────────────────
    with tab2:
        st.markdown(section_header("🎛️", "Gradient Descent Lab",
                    "Watch the algorithm find the best line"), unsafe_allow_html=True)

        col_l, col_r = st.columns([1, 2])
        with col_l:
            lr     = st.select_slider("Learning Rate", [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5], value=0.01)
            epochs = st.slider("Epochs", 10, 500, 100, 10)
            m_init = st.slider("Initial Slope", -5.0, 10.0, 0.0, 0.5)
            b_init = st.slider("Initial Intercept", -50.0, 200.0, 0.0, 5.0)
            run_gd = st.button("▶ Run Gradient Descent", use_container_width=True)

        X_gd, y_gd = _gen_house_data(80, 30, scale=True)

        if run_gd:
            history = _gradient_descent(X_gd, y_gd, m_init, b_init, lr, epochs)
            with col_r:
                # Loss curve
                fig_loss = px.line(x=list(range(len(history["loss"]))),
                                   y=history["loss"],
                                   labels={"x": "Epoch", "y": "MSE Loss"},
                                   title="Loss Reduction over Epochs",
                                   color_discrete_sequence=[COLOR_SEQ[2]])
                fig_loss.update_layout(**PLOTLY_LAYOUT, height=280)
                st.plotly_chart(fig_loss, use_container_width=True)

                # Final line
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=X_gd, y=y_gd, mode='markers',
                                          marker=dict(color=COLOR_SEQ[0], size=6, opacity=0.7),
                                          name='Data'))
                m_f, b_f = history["m"][-1], history["b"][-1]
                x_l = np.linspace(X_gd.min(), X_gd.max(), 100)
                fig2.add_trace(go.Scatter(x=x_l, y=m_f*x_l+b_f,
                                          line=dict(color=COLOR_SEQ[1], width=3),
                                          name=f'Final: y={m_f:.2f}x+{b_f:.2f}'))
                fig2.update_layout(**PLOTLY_LAYOUT, title="Final Fitted Line", height=280,
                                    xaxis_title="X (normalised)", yaxis_title="y")
                st.plotly_chart(fig2, use_container_width=True)

            st.markdown(info_box(
                f"🏁 After <b>{epochs} epochs</b>: slope={history['m'][-1]:.3f}, "
                f"intercept={history['b'][-1]:.3f}, final loss={history['loss'][-1]:.4f}"
            ), unsafe_allow_html=True)
        else:
            with col_r:
                st.markdown("""
                <div class="ml-card" style="text-align:center;padding:60px;">
                    <div style="font-size:3rem;margin-bottom:16px;">⚙️</div>
                    <p style="color:#8892A4;">Adjust parameters and click <b>Run Gradient Descent</b> to see the algorithm in action!</p>
                </div>""", unsafe_allow_html=True)

    # ─────────────────────────────────────────────
    # TAB 3: Concepts
    # ─────────────────────────────────────────────
    with tab3:
        st.markdown(section_header("💡", "Linear Regression Concepts"), unsafe_allow_html=True)

        concepts = [
            ("📐", "Slope (m)", "#6C63FF",
             "How much y changes for every 1-unit increase in x. "
             "E.g., slope = 150 means: every extra sq ft adds $150 to price."),
            ("↕️", "Intercept (b)", "#43E97B",
             "The predicted y value when x = 0. "
             "It's the 'base value' of the prediction."),
            ("📉", "MSE Loss", "#FF6584",
             "Mean Squared Error = average squared difference between actual and predicted values. "
             "Lower MSE = better model."),
            ("🎯", "R² Score", "#38F9D7",
             "Explains what % of variance the model captures. "
             "R² = 0.85 means model explains 85% of the data — great!"),
            ("⬇️", "Gradient Descent", "#FFA94D",
             "An optimisation algorithm that adjusts slope and intercept step-by-step "
             "to minimise the loss — like rolling a ball downhill to find the lowest point."),
            ("🔊", "Residuals", "#B48EFF",
             "Residual = Actual − Predicted. Small residuals = accurate model. "
             "Plot residuals to check if the model is biased."),
        ]
        rows = [concepts[:3], concepts[3:]]
        for row in rows:
            cols = st.columns(3)
            for col, (icon, title, color, desc) in zip(cols, row):
                col.markdown(f"""
                <div class="ml-card" style="border-top:3px solid {color};min-height:140px;">
                    <div style="font-size:1.8rem;margin-bottom:8px;">{icon}</div>
                    <h4 style="color:{color};margin:0 0 8px 0;font-size:0.95rem;">{title}</h4>
                    <p style="color:#8892A4;font-size:0.82rem;margin:0;line-height:1.5;">{desc}</p>
                </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def _gen_house_data(n=100, noise=40, scale=False, seed=42):
    rng = np.random.RandomState(seed)
    X   = rng.randint(500, 5000, n).astype(float)
    y   = 150 * X + 50_000 + rng.normal(0, noise * 500, n)
    if scale:
        X = (X - X.mean()) / X.std()
        y = (y - y.mean()) / y.std()
    return X, y


def _ols(X, y):
    """Ordinary Least Squares closed-form solution."""
    x_m, y_m = X.mean(), y.mean()
    slope = np.sum((X - x_m) * (y - y_m)) / np.sum((X - x_m)**2)
    intercept = y_m - slope * x_m
    return slope, intercept


def _gradient_descent(X, y, m, b, lr, epochs):
    n = len(X)
    history = {"m": [m], "b": [b], "loss": []}
    for _ in range(epochs):
        y_pred = m * X + b
        loss   = np.mean((y - y_pred)**2)
        dm = (-2/n) * np.sum(X * (y - y_pred))
        db = (-2/n) * np.sum(y - y_pred)
        m -= lr * dm
        b -= lr * db
        history["m"].append(m)
        history["b"].append(b)
        history["loss"].append(loss)
    return history
