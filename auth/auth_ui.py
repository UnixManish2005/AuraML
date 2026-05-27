"""Authentication UI — Login / Register pages."""

import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from database.db import login_user, register_user


def show_login():
    

    col1, col2, col3 = st.columns([1, 1.4, 1])
    with col2:
        st.markdown("""
        <div style="background:linear-gradient(135deg,#1A1A2E,#16213E);
                    border:1px solid rgba(108,99,255,0.3);
                    border-radius:20px; padding:2rem 2rem 1.5rem;">
        <h3 style="font-family:'Syne',sans-serif;font-weight:700;
                   color:#E8E8F0;margin-bottom:1.5rem;text-align:center;">
         🫡 Welcome 🫡
        </h3>
        """, unsafe_allow_html=True)

        email = st.text_input("📧 Email", placeholder="you@gmail.com", key="login_email")
        password = st.text_input("🔑 Password", type="password", placeholder="••••••••", key="login_pw")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Sign In →", use_container_width=True):
            if email and password:
                user = login_user(email.strip(), password)
                if user:
                    st.session_state["user"] = user
                    st.session_state["page"] = "dashboard"
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials or account disabled.")
            else:
                st.warning("Please fill in all fields.")

        st.markdown("""
        <p style="text-align:center;color:#8888AA;font-size:0.82rem;margin-top:1rem;">
            Don't have an Account ? Sign Up Now !!!
        </p>
        </div>
        """, unsafe_allow_html=True)


def show_register():
    col1, col2, col3 = st.columns([1, 1.4, 1])
    with col2:
        st.markdown("""
        <div style="background:linear-gradient(135deg,#1A1A2E,#16213E);
                    border:1px solid rgba(108,99,255,0.3);
                    border-radius:20px;padding:2rem 2rem 1.5rem;">
        <h3 style="font-family:'Syne',sans-serif;font-weight:700;
                   color:#E8E8F0;margin-bottom:1.5rem;text-align:center;">
            Create an Account 🫰🏻
        </h3>
        """, unsafe_allow_html=True)

        name = st.text_input("👤 Full Name", placeholder="Enter your Name", key="reg_name")
        email = st.text_input("📧 Email", placeholder="you@gmail.com", key="reg_email")
        pw1 = st.text_input("🔑 Password", type="password", placeholder="min 6 chars", key="reg_pw1")
        pw2 = st.text_input("🔑 Confirm Password", type="password", placeholder="repeat password", key="reg_pw2")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Create Account →", use_container_width=True):
            if not all([name, email, pw1, pw2]):
                st.warning("All fields are required.")
            elif len(pw1) < 6:
                st.warning("Password must be at least 6 characters.")
            elif pw1 != pw2:
                st.error("Passwords do not match.")
            elif "@" not in email:
                st.error("Invalid email address.")
            else:
                ok, msg = register_user(name.strip(), email.strip(), pw1)
                if ok:
                    st.success(f"✅ {msg} Please login.")
                else:
                    st.error(f"❌ {msg}")

        st.markdown("</div>", unsafe_allow_html=True)
