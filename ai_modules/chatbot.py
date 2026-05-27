"""AI Tutor Chatbot — powered by Anthropic Claude."""

import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.styles import section_header, alert

SYSTEM_PROMPT = """You are an expert AI/ML tutor for a learning platform called AILearn Pro.
Your role is to:
- Explain AI, Machine Learning, Deep Learning, NLP, Computer Vision concepts clearly
- Use simple analogies for beginners
- Provide Python code examples when relevant
- Guide students on projects
- Answer interview preparation questions
- Be encouraging and patient

Keep answers concise but thorough. Use markdown for formatting.
Always end with a follow-up question to deepen learning."""


def show_ai_tutor(user):
    section_header("🤖 AI Tutor", "Ask anything about AI, ML, Python, or your projects!")

    # Init conversation
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Suggested prompts
    st.markdown("#### 💡 Quick Questions")
    suggestions = [
        "Explain overfitting vs underfitting",
        "What is the difference between CNN and RNN?",
        "How does gradient descent work?",
        "Explain attention mechanism in transformers",
        "What is RAG and how does it work?",
        "Give me Python code for a simple neural network",
    ]
    cols = st.columns(3)
    for i, sug in enumerate(suggestions):
        with cols[i % 3]:
            if st.button(sug, key=f"sug_{i}", use_container_width=True):
                st.session_state["chat_history"].append({"role":"user","content":sug})
                _get_ai_response(sug)

    st.markdown("---")

    # Chat history display
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state["chat_history"]:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                st.markdown(f"""
                <div style="display:flex;justify-content:flex-end;margin-bottom:0.8rem">
                    <div style="background:linear-gradient(135deg,#6C63FF,#8B5CF6);
                                color:white;border-radius:16px 16px 4px 16px;
                                padding:0.8rem 1.2rem;max-width:75%;font-size:0.92rem">
                        {content}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="display:flex;justify-content:flex-start;margin-bottom:0.8rem">
                    <div style="font-size:1.5rem;margin-right:0.5rem;align-self:flex-start">🤖</div>
                    <div style="background:#1A1A2E;border:1px solid rgba(108,99,255,0.25);
                                color:#E8E8F0;border-radius:4px 16px 16px 16px;
                                padding:0.8rem 1.2rem;max-width:80%;font-size:0.92rem">
                """, unsafe_allow_html=True)
                st.markdown(content)
                st.markdown("</div></div>", unsafe_allow_html=True)

    # Input
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns([5, 1])
    with c1:
        user_input = st.text_input("Ask your AI tutor anything...", key="chat_input",
                                    placeholder="e.g. Explain how BERT works in simple terms")
    with c2:
        st.markdown("<br>", unsafe_allow_html=True)
        send = st.button("Send →", use_container_width=True)

    if send and user_input:
        st.session_state["chat_history"].append({"role":"user","content":user_input})
        _get_ai_response(user_input)
        st.rerun()

    # Clear button
    if st.session_state.get("chat_history"):
        if st.button("🗑️ Clear Chat", use_container_width=False):
            st.session_state["chat_history"] = []
            st.rerun()


def _get_ai_response(user_msg: str):
    """Call Anthropic API and append response."""
    import requests

    messages = []
    for msg in st.session_state["chat_history"]:
        if msg["role"] in ("user", "assistant"):
            messages.append({"role": msg["role"], "content": msg["content"]})

    # Ensure last message is user
    if not messages or messages[-1]["role"] != "user":
        messages.append({"role": "user", "content": user_msg})

    with st.spinner("AI Tutor is thinking..."):
        try:
            resp = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={"Content-Type": "application/json"},
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 1000,
                    "system": SYSTEM_PROMPT,
                    "messages": messages,
                },
                timeout=30,
            )
            if resp.status_code == 200:
                data = resp.json()
                reply = data["content"][0]["text"]
            else:
                reply = f"⚠️ API error {resp.status_code}. Please try again."
        except Exception as e:
            reply = f"⚠️ Connection error: {e}. Make sure the app is running with API access."

    st.session_state["chat_history"].append({"role":"assistant","content":reply})
