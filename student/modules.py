"""AI/ML Learning Modules — visual, interactive, beginner-friendly."""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from database.db import update_progress
from utils.styles import section_header, alert, badge


MODULES = {
    "AI Introduction": {
        "icon": "🤖", "color": "#6C63FF",
        "tagline": "What is AI and why does it matter?",
        "sections": ["What is AI?", "History of AI", "Types of AI", "AI vs ML vs DL", "Real-world Applications"],
    },
    "Machine Learning": {
        "icon": "📊", "color": "#FF6584",
        "tagline": "Learn to build models that learn from data.",
        "sections": ["What is ML?", "Supervised Learning", "Unsupervised Learning", "Model Evaluation", "Feature Engineering"],
    },
    "Deep Learning": {
        "icon": "🧠", "color": "#43D9AD",
        "tagline": "Neural networks from perceptron to transformers.",
        "sections": ["Neural Networks", "Backpropagation", "CNNs", "RNNs & LSTMs", "Transfer Learning"],
    },
    "NLP": {
        "icon": "💬", "color": "#FFB547",
        "tagline": "Teaching computers to understand human language.",
        "sections": ["Text Preprocessing", "TF-IDF & Embeddings", "Sentiment Analysis", "Named Entity Recognition", "Transformers & BERT"],
    },
    "Computer Vision": {
        "icon": "👁️", "color": "#8B5CF6",
        "tagline": "Giving machines the ability to see.",
        "sections": ["Image Basics", "Convolutions", "Object Detection", "Image Segmentation", "Face Recognition"],
    },
    "Generative AI": {
        "icon": "✨", "color": "#EC4899",
        "tagline": "Create images, text, and code with AI.",
        "sections": ["GANs", "VAEs", "Diffusion Models", "LLMs", "Prompt Engineering"],
    },
    "Python": {
        "icon": "🐍", "color": "#06B6D4",
        "tagline": "Python essentials for data science & ML.",
        "sections": ["Python Basics", "NumPy", "Pandas", "Matplotlib & Seaborn", "Scikit-learn Basics"],
    },
}

CONTENT = {
    "What is AI?": """
**Artificial Intelligence (AI)** is the simulation of human intelligence processes by machines — especially computer systems.

### 🔑 Core Concepts
- **Perception** — Recognising patterns in data (images, text, audio)
- **Reasoning** — Making decisions based on rules or learned knowledge
- **Learning** — Improving performance through experience
- **Action** — Performing tasks autonomously

### 🌍 Real-World Examples
| Domain | AI Application |
|--------|---------------|
| Healthcare | Disease diagnosis from X-rays |
| Finance | Fraud detection |
| Transport | Self-driving cars |
| Retail | Recommendation engines |
| Language | ChatGPT, Google Translate |
""",
    "What is ML?": """
**Machine Learning (ML)** is a subset of AI that enables systems to learn from data without being explicitly programmed.

### 🔄 The ML Workflow
1. **Collect Data** → 2. **Preprocess** → 3. **Train Model** → 4. **Evaluate** → 5. **Deploy**

### 📚 Types of Machine Learning

| Type | Description | Example |
|------|-------------|---------|
| Supervised | Learn from labelled data | Spam detection |
| Unsupervised | Find patterns in unlabelled data | Customer segmentation |
| Reinforcement | Learn via rewards & penalties | Game-playing AI |
""",
    "Neural Networks": """
**Neural Networks** are computational models inspired by the human brain.

### 🧠 Structure
- **Input Layer** — receives raw data
- **Hidden Layers** — extracts features and patterns  
- **Output Layer** — produces predictions

### ⚙️ How it works
Each neuron computes: **y = activation(W·x + b)**

Where:
- **W** = weights (learned from data)
- **b** = bias
- **activation** = e.g., ReLU, Sigmoid
""",
    "Text Preprocessing": """
**Text Preprocessing** converts raw text into a format models can understand.

### 🔧 Steps
1. **Tokenisation** — Split text into words/tokens
2. **Lowercasing** — Standardise case
3. **Stop-word removal** — Remove "the", "a", "is", etc.
4. **Stemming / Lemmatisation** — Reduce to root form
5. **Vectorisation** — Convert to numbers (TF-IDF, Word2Vec)

### 💡 Example
```
Raw:    "The cats are running quickly"
After:  ["cat", "run", "quick"]
```
""",
    "Image Basics": """
**Digital Images** are 2D arrays of pixel values (0–255 per channel).

### 📐 Colour Channels
- **Grayscale** — 1 channel
- **RGB** — 3 channels (Red, Green, Blue)
- **RGBA** — 4 channels (+ Alpha/transparency)

### 🔢 Representation
A 28×28 greyscale image = **784 numbers**
A 224×224 RGB image = **150,528 numbers**

### 🛠️ Common Operations
- Resize, crop, flip (data augmentation)
- Normalise pixel values to [0, 1]
- Convert BGR ↔ RGB (OpenCV uses BGR)
""",
}


def show_modules(user):
    uid = user["id"]

    section_header("📚 AI/ML Learning Modules",
                   "Visual, interactive lessons — learn by doing.")

    # ── Module grid ────────────────────────────────────────────────────────
    cols = st.columns(3)
    selected = st.session_state.get("selected_module")

    for i, (name, meta) in enumerate(MODULES.items()):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="module-card" style="border-color:rgba({_hex_to_rgb(meta['color'])},0.3)">
                <div style="font-size:2rem">{meta['icon']}</div>
                <h4>{name}</h4>
                <p>{meta['tagline']}</p>
                <div style="margin-top:0.6rem">
                    <span style="font-size:0.75rem;color:#8888AA">{len(meta['sections'])} sections</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"Open {meta['icon']}", key=f"mod_{name}", use_container_width=True):
                st.session_state["selected_module"] = name
                st.rerun()
        

    # ── Module detail ──────────────────────────────────────────────────────
    if selected and selected in MODULES:
        st.markdown("---")
        meta = MODULES[selected]
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:1rem;margin-bottom:1rem">
            <span style="font-size:2.5rem">{meta['icon']}</span>
            <div>
                <h2 style="font-family:Syne,sans-serif;font-weight:800;color:#E8E8F0;margin:0">{selected}</h2>
                <p style="color:#8888AA;margin:0">{meta['tagline']}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Tabs for each section
# Tabs for each section
        tabs = st.tabs(meta["sections"])
        for j, (tab, section) in enumerate(zip(tabs, meta["sections"])):
            with tab:
                _render_section(selected, section, j, uid)
                
                # ── MOVE THE DEMO IN HERE ───────────────────────────────────
                # This ensures the interactive demo logic only processes 
                # contextually for relevant tab sections.
                if selected == "Machine Learning" and section == "Supervised Learning":
                    st.markdown("---")
                    _interactive_demo(selected)
                elif selected == "Deep Learning" and section == "Neural Networks":
                    st.markdown("---")
                    _interactive_demo(selected)
                elif selected == "NLP" and section == "TF-IDF & Embeddings":
                    st.markdown("---")
                    _interactive_demo(selected)
                # ────────────────────────────────────────────────────────────

                # Update progress
                pct = int((j + 1) / len(meta["sections"]) * 100)
                update_progress(uid, selected, pct)

        # ── REMOVE OR COMMENT OUT THESE OLD BOTTOM LINES ────────────────────
        # st.markdown("---")
        # _interactive_demo(selected)


def _render_section(module, section, idx, uid):
    content = CONTENT.get(section)
    if content:
        st.markdown(content)
    else:
        # Generic placeholder content
        st.markdown(f"""
        ### 📖 {section}

        This section covers **{section}** within the context of **{module}**.

        **Key Learning Objectives:**
        - Understand core concepts of {section}
        - Apply knowledge to real-world problems
        - Build hands-on intuition through practice

        > 💡 **Pro Tip:** The best way to learn {module} is to build projects.
        > Head to the **ML Playground** to experiment!
        """)
    _render_visual(module, section)


def _render_visual(module, section):
    """Render a contextually relevant chart or demo."""
    if module == "Machine Learning" and "Supervised" in section:
        _demo_regression("section")
    elif module == "Machine Learning" and "Evaluation" in section:
        _demo_confusion_matrix()
    elif module == "Deep Learning" and "Neural" in section:
        _demo_neural_viz("neural")
    elif module == "NLP" and "TF-IDF" in section:
        _demo_tfidf()
    elif module == "Python" and "NumPy" in section:
        _demo_numpy()


def _demo_regression(prefix="main"):
    st.markdown("#### 📉 Live: Linear Regression Demo")

    noise = st.slider(
        "Noise level",
        0.1,
        3.0,
        1.0,
        key=f"{prefix}_core_view_reg_noise"
    )

    np.random.seed(42)

    x = np.linspace(0, 10, 50)
    y = 2 * x + 3 + np.random.randn(50) * noise

    m, b = np.polyfit(x, y, 1)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=dict(color="#6C63FF", size=7),
            name="Data"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=m * x + b,
            mode="lines",
            line=dict(color="#FF6584", width=3),
            name=f"y={m:.2f}x+{b:.2f}"
        )
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E8E8F0"),
        height=300,
        margin=dict(l=0, r=0, t=10, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)")
    )

    fig.update_xaxes(showgrid=False, color="#8888AA")
    fig.update_yaxes(showgrid=False, color="#8888AA")

    st.plotly_chart(
        fig,
        use_container_width=True,
        key=f"{prefix}_regression_chart"
    )


def _demo_confusion_matrix():
    st.markdown("#### 🔲 Confusion Matrix Explained")
    cm = np.array([[45, 5], [8, 42]])
    fig = px.imshow(cm, text_auto=True, color_continuous_scale=["#0F0F1A", "#6C63FF"],
                    labels=dict(x="Predicted", y="Actual"),
                    x=["Negative","Positive"], y=["Negative","Positive"])
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#E8E8F0"), height=280,
                      margin=dict(l=0,r=0,t=10,b=10))
    st.plotly_chart(fig, use_container_width=True)
    acc = (cm[0,0]+cm[1,1])/cm.sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy",  f"{acc:.1%}")
    c2.metric("Precision", f"{cm[1,1]/(cm[0,1]+cm[1,1]):.1%}")
    c3.metric("Recall",    f"{cm[1,1]/(cm[1,0]+cm[1,1]):.1%}")


def _demo_neural_viz(chart_key="neural_chart"):
    st.markdown("#### 🧠 Neural Network Architecture")
    layers = [3, 4, 4, 2]
    fig = go.Figure()
    max_n = max(layers)
    for li, n in enumerate(layers):
        y_positions = np.linspace(0, max_n - 1, n)
        for yi in y_positions:
            fig.add_trace(go.Scatter(
                x=[li], y=[yi], mode="markers",
                marker=dict(size=22, color="#6C63FF", line=dict(color="#8B5CF6", width=2)),
                showlegend=False, hoverinfo="skip"
            ))
        # Draw connections
        if li > 0:
            prev_y = np.linspace(0, max_n - 1, layers[li - 1])
            for py in prev_y:
                for cy in y_positions:
                    fig.add_trace(go.Scatter(
                        x=[li-1, li], y=[py, cy], mode="lines",
                        line=dict(color="rgba(108,99,255,0.2)", width=1),
                        showlegend=False, hoverinfo="skip"
                    ))
    names = ["Input\n(3)", "Hidden\n(4)", "Hidden\n(4)", "Output\n(2)"]
    for i, n in enumerate(names):
        fig.add_annotation(x=i, y=-0.8, text=n, showarrow=False,
                           font=dict(color="#8888AA", size=11))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      height=320, margin=dict(l=0,r=0,t=10,b=50),
                      xaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
                      yaxis=dict(showgrid=False,zeroline=False,showticklabels=False))
    st.plotly_chart(
    fig,
    use_container_width=True,
    key=chart_key
)


def _demo_tfidf():
    st.markdown("#### 📊 TF-IDF Scores Visualised")
    words = ["machine", "learning", "data", "model", "neural", "python", "train"]
    scores = [0.82, 0.75, 0.68, 0.71, 0.65, 0.55, 0.60]
    fig = go.Figure(go.Bar(x=words, y=scores,
                           marker_color=["#6C63FF","#FF6584","#43D9AD","#FFB547",
                                        "#8B5CF6","#EC4899","#06B6D4"]))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#E8E8F0"), height=260,
                      margin=dict(l=0,r=0,t=10,b=10),
                      yaxis=dict(showgrid=False,color="#8888AA"),
                      xaxis=dict(showgrid=False,color="#E8E8F0"))
    st.plotly_chart(fig, use_container_width=True)


def _demo_numpy():
    st.markdown("#### 🔢 NumPy Array Operations — Live")
    size = st.slider("Array size", 5, 20, 10, key="np_size")
    arr = np.random.randint(1, 100, size)
    st.code(f"arr = {arr}\nmean = {arr.mean():.2f}  |  std = {arr.std():.2f}  |  sum = {arr.sum()}")
    fig = go.Figure(go.Bar(y=arr, marker_color="#6C63FF"))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      height=220, margin=dict(l=0,r=0,t=10,b=10),
                      xaxis=dict(showgrid=False), yaxis=dict(showgrid=False,color="#8888AA"))
    st.plotly_chart(fig, use_container_width=True)


def _interactive_demo(module):
    """Module-level interactive demo at the bottom."""
    if module == "Machine Learning":
        _demo_regression("interactive")
    elif module == "Deep Learning":
        _demo_neural_viz("neural_viz_section")
    elif module == "NLP":
        _demo_tfidf()


def _hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f"{r},{g},{b}"
