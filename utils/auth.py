# utils/auth.py

import streamlit as st

# Dummy user database: {role: {username: password}}
USERS = {
    "Admin": {"admin": "admin123"},
    "Doctor": {"doctor": "docpass"},
    "Hospital": {"hospital": "hospass"},
}

def login():
    """Simple multi-role login system via sidebar."""
    with st.sidebar:
        st.subheader("🔐 User Login")

        role = st.selectbox("Login as", ["Admin", "Doctor", "Hospital"])
        username = st.text_input(f"{role} Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            valid_users = USERS.get(role, {})
            if valid_users.get(username) == password:
                st.session_state["authenticated"] = True
                st.session_state["user_role"] = role
                st.success(f"✅ Welcome {role} {username}!")
            else:
                st.session_state["authenticated"] = False
                st.error("❌ Invalid credentials")

    return st.session_state.get("authenticated", False)
