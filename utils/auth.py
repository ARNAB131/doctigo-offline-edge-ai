# utils/auth.py
from __future__ import annotations
import os
import time
from datetime import datetime, timedelta
from typing import Dict, Tuple
import secrets
import hashlib
import yaml
import streamlit as st

# ------------------------------------------------------------------
# Settings / storage locations (works with our OFFLINE repo layout)
# ------------------------------------------------------------------
USERS_FILE = os.path.join("config", "users.yaml")
DEFAULT_USERS = {
    "Admin":    {"admin":   "admin123"},
    "Doctor":   {"doctor":  "docpass"},
    "Hospital": {"hospital":"hospass"},
}
DEFAULT_SESSION_TIMEOUT_MIN = 20  # used if config not yet read here

# ------------------------------------------------------------------
# Password hashing helpers (PBKDF2-HMAC-SHA256, offline-safe)
# ------------------------------------------------------------------
def _hash_password(password: str, salt: bytes | None = None) -> Tuple[str, str]:
    """
    Returns (salt_hex, hash_hex).
    - Uses PBKDF2-HMAC-SHA256 with 200k iterations.
    """
    if salt is None:
        salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
    return salt.hex(), dk.hex()

def _verify_password(password: str, salt_hex: str, hash_hex: str) -> bool:
    try:
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(hash_hex)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
        # constant-time compare
        return secrets.compare_digest(dk, expected)
    except Exception:
        return False

# ------------------------------------------------------------------
# Users file handling (YAML). We store only salted hashes.
# ------------------------------------------------------------------
def _ensure_users_file_exists() -> None:
    """Create config/users.yaml with hashed defaults if missing."""
    if os.path.exists(USERS_FILE):
        return
    os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)

    users_doc = {"users": []}
    for role, users in DEFAULT_USERS.items():
        for username, plaintext in users.items():
            salt_hex, hash_hex = _hash_password(plaintext)
            users_doc["users"].append({
                "role": role,
                "username": username,
                "salt": salt_hex,
                "password_hash": hash_hex,
            })
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        yaml.safe_dump(users_doc, f, sort_keys=False)

def _load_users() -> Dict[str, Dict[str, Dict[str, str]]]:
    """
    Loads users.yaml -> { role: { username: {"salt":..., "password_hash":...} } }
    """
    _ensure_users_file_exists()
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}
    out: Dict[str, Dict[str, Dict[str, str]]] = {}
    for entry in doc.get("users", []):
        role = str(entry.get("role", "")).strip()
        username = str(entry.get("username", "")).strip()
        salt = str(entry.get("salt", "")).strip()
        phash = str(entry.get("password_hash", "")).strip()
        if not role or not username or not salt or not phash:
            continue
        out.setdefault(role, {})[username] = {"salt": salt, "password_hash": phash}
    return out

# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------
def logout():
    """Clear auth info from session."""
    for k in ("authenticated", "user_role", "username", "auth_time"):
        st.session_state.pop(k, None)

def login(session_timeout_min: int | None = None) -> bool:
    """
    Offline, role-based login with hashed passwords and session timeout.
    - Users stored in config/users.yaml (created on first run with defaults).
    - Returns True if the current session is authenticated.
    """
    users = _load_users()
    roles = sorted(users.keys()) or ["Admin", "Doctor", "Hospital"]

    # Session timeout
    timeout_minutes = session_timeout_min or st.session_state.get("session_timeout_min") or DEFAULT_SESSION_TIMEOUT_MIN
    if st.session_state.get("authenticated") and st.session_state.get("auth_time"):
        started = datetime.fromtimestamp(st.session_state["auth_time"])
        if datetime.now() - started > timedelta(minutes=timeout_minutes):
            st.warning("🔒 Session expired. Please log in again.")
            logout()

    with st.sidebar:
        st.subheader("🔐 User Login")

        # If already authenticated, show status + logout button
        if st.session_state.get("authenticated"):
            role = st.session_state.get("user_role", "Unknown")
            username = st.session_state.get("username", "user")
            st.success(f"✅ Logged in as {role} ({username})")
            if st.button("Logout"):
                logout()
                st.experimental_rerun()
            return True

        # Not authenticated — prompt for creds
        role = st.selectbox("Login as", roles, index=0)
        username = st.text_input(f"{role} Username")
        password = st.text_input("Password", type="password")
        col1, col2 = st.columns([1,1])
        with col1:
            login_clicked = st.button("Login")
        with col2:
            st.caption(f"Session timeout: {timeout_minutes} min")

        if login_clicked:
            role_users = users.get(role, {})
            user_rec = role_users.get(username)
            if user_rec and _verify_password(password, user_rec["salt"], user_rec["password_hash"]):
                st.session_state["authenticated"] = True
                st.session_state["user_role"] = role
                st.session_state["username"] = username
                st.session_state["auth_time"] = time.time()
                st.success(f"✅ Welcome {role} {username}!")
                st.experimental_rerun()
            else:
                st.session_state["authenticated"] = False
                st.error("❌ Invalid credentials")

    return st.session_state.get("authenticated", False)

# ------------------------------------------------------------------
# (Optional) helper to add/update a user via code (not exposed in UI)
# Usage example:
#   add_or_update_user("Admin", "newadmin", "newpass123")
# ------------------------------------------------------------------
def add_or_update_user(role: str, username: str, plaintext_password: str) -> None:
    _ensure_users_file_exists()
    doc = {"users": []}
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {"users": []}

    # index existing entries
    idx = {(u["role"], u["username"]): i for i, u in enumerate(doc.get("users", [])) if "role" in u and "username" in u}
    salt_hex, hash_hex = _hash_password(plaintext_password)
    new_entry = {"role": role, "username": username, "salt": salt_hex, "password_hash": hash_hex}

    if (role, username) in idx:
        doc["users"][idx[(role, username)]] = new_entry
    else:
        doc.setdefault("users", []).append(new_entry)

    with open(USERS_FILE, "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, sort_keys=False)
