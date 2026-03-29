"""
auth.py
-------
Handles everything authentication related:
  - Input validation (username, email, password strength)
  - Login and registration logic
  - Session management via st.session_state
  - Login / Register UI forms
  - Sidebar user widget and logout button
"""

import re
import streamlit as st
from database import create_user, verify_login, get_profile


# ── Session keys ───────────────────────────────────────────────────────────
_USER_KEY = "mediscan_user"


# ── Validation ─────────────────────────────────────────────────────────────

def _valid_email(email):
    return bool(re.match(r"^[\w\.-]+@[\w\.-]+\.\w{2,}$", email.strip()))

def _valid_username(username):
    return bool(re.match(r"^[a-zA-Z0-9_]{3,20}$", username.strip()))

def _strong_password(password):
    if len(password) < 8:        return False, "Password must be at least 8 characters."
    if not any(c.isupper() for c in password): return False, "Must contain an uppercase letter."
    if not any(c.islower() for c in password): return False, "Must contain a lowercase letter."
    if not any(c.isdigit() for c in password): return False, "Must contain a number."
    return True, "Strong."


# ── Session helpers ────────────────────────────────────────────────────────

def is_logged_in():
    return st.session_state.get(_USER_KEY) is not None

def get_current_user():
    return st.session_state.get(_USER_KEY)

def logout():
    keys_to_clear = [_USER_KEY, "last_prediction", "medicine_result",
                     "diet_plan_result", "routine_result",
                     "health_history", "symptom_history", "patient_data"]
    for k in keys_to_clear:
        st.session_state.pop(k, None)


# ── Auth logic ─────────────────────────────────────────────────────────────

def handle_login(identifier, password):
    if not identifier or not password:
        return False, "Please fill in all fields."
    success, message, user = verify_login(identifier, password)
    if success:
        st.session_state[_USER_KEY] = user
    return success, message


def handle_register(username, email, password, confirm, full_name):
    if not all([username, email, password, confirm]):
        return False, "All fields are required."
    if not _valid_username(username):
        return False, "Username: 3–20 chars, letters/numbers/underscores only."
    if not _valid_email(email):
        return False, "Please enter a valid email address."
    if password != confirm:
        return False, "Passwords do not match."
    strong, msg = _strong_password(password)
    if not strong:
        return False, msg
    return create_user(username, email, password, full_name)


# ── Shared CSS ─────────────────────────────────────────────────────────────

AUTH_CSS = """
<style>
.auth-wrap{max-width:440px;margin:1.5rem auto;background:#1C1C1C;
           border:1px solid rgba(255,255,255,.08);border-radius:20px;
           padding:2.5rem;box-shadow:0 12px 40px rgba(0,0,0,.6);}
.auth-logo{text-align:center;margin-bottom:1.5rem;}
.auth-logo .icon{font-size:2.8rem;display:block;}
.auth-logo .name{font-size:1.5rem;font-weight:700;color:#F0F0F0;}
.auth-logo .sub{font-size:.75rem;color:#505050;text-transform:uppercase;letter-spacing:.08em;}
.auth-err{background:rgba(248,113,113,.08);border-left:4px solid #F87171;
          border-radius:8px;padding:.75rem 1rem;font-size:.85rem;color:#F87171;margin:.75rem 0;}
.auth-ok{background:rgba(74,222,128,.08);border-left:4px solid #4ADE80;
         border-radius:8px;padding:.75rem 1rem;font-size:.85rem;color:#4ADE80;margin:.75rem 0;}
.pw-hint{font-size:.74rem;color:#505050;margin-top:.3rem;line-height:1.5;}
</style>
"""


# ── Auth gate UI (login + register tabs) ───────────────────────────────────

def render_auth_gate():
    """Show login/register page. Stops the app until the user signs in."""
    st.markdown(AUTH_CSS, unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;padding:1.5rem 0 .5rem;">
        <span style="font-size:3rem;">⚕️</span>
        <h1 style="font-weight:400;font-size:2rem;margin:.3rem 0 .2rem;color:#F0F0F0;">CareSync</h1>
        <p style="color:#505050;font-size:.8rem;text-transform:uppercase;letter-spacing:.08em;">
            Healthcare Diagnosis &amp; Recommendation System · Sign in to continue
        </p>
    </div>
    """, unsafe_allow_html=True)

    _, col, _ = st.columns([1, 2, 1])
    with col:
        tab_login, tab_reg = st.tabs(["🔐 Sign In", "🚀 Register"])

        with tab_login:
            with st.form("login_form"):
                identifier = st.text_input("Username or Email", placeholder="your username or email")
                password   = st.text_input("Password", type="password", placeholder="your password")
                submitted  = st.form_submit_button("Sign In", use_container_width=True)
            if submitted:
                ok, msg = handle_login(identifier, password)
                if ok:
                    st.markdown(f'<div class="auth-ok">✅ {msg}</div>', unsafe_allow_html=True)
                    st.rerun()
                else:
                    st.markdown(f'<div class="auth-err">❌ {msg}</div>', unsafe_allow_html=True)

        with tab_reg:
            with st.form("register_form"):
                full_name = st.text_input("Full Name",      placeholder="e.g. Shubham Raj Sinha")
                username  = st.text_input("Username",       placeholder="3–20 chars, letters/numbers/underscores")
                email     = st.text_input("Email",          placeholder="your@email.com")
                c1, c2    = st.columns(2)
                password  = c1.text_input("Password",          type="password", placeholder="Min 8 chars")
                confirm   = c2.text_input("Confirm Password",  type="password", placeholder="Repeat password")
                st.markdown('<div class="pw-hint">🔒 8+ chars with uppercase, lowercase & a number</div>', unsafe_allow_html=True)
                submitted = st.form_submit_button("Create Account", use_container_width=True)
            if submitted:
                ok, msg = handle_register(username, email, password, confirm, full_name)
                if ok:
                    st.markdown(f'<div class="auth-ok">✅ {msg} Switch to Sign In tab.</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="auth-err">❌ {msg}</div>', unsafe_allow_html=True)


# ── Sidebar user widget ────────────────────────────────────────────────────

def render_sidebar_user():
    """Show logged-in user card + Sign Out button in the sidebar."""
    user = get_current_user()
    if not user:
        return

    name     = user.get("full_name") or user["username"]
    initials = "".join(w[0].upper() for w in name.split()[:2])
    joined   = str(user.get("created_at", ""))[:10]
    profile  = get_profile(user["id"])
    blood    = (profile or {}).get("blood_group") or ""
    age_val  = (profile or {}).get("age") or ""

    # Build health info line only when profile data exists
    health_html = ""
    if blood and str(blood).strip() and age_val and str(age_val).strip():
        health_html = f'<span style="color:rgba(255,255,255,.35);">⚕️ {blood} &middot; {age_val} yrs</span>'

    html = (
        '<div style="background:linear-gradient(135deg,#0F2044,#1A3A6E);'
        'border:1px solid rgba(74,158,255,.2);border-radius:14px;'
        'padding:1.2rem;margin-bottom:1rem;">'
        '<div style="display:flex;align-items:center;gap:.75rem;margin-bottom:.75rem;">'
        '<div style="width:40px;height:40px;background:linear-gradient(135deg,#4A9EFF,#2DD4BF);'
        'border-radius:50%;display:flex;align-items:center;justify-content:center;'
        'font-weight:700;color:#0A1628;font-size:.9rem;flex-shrink:0;">' + initials + '</div>'
        '<div>'
        '<div style="font-size:.9rem;font-weight:600;color:#F0F0F0;">' + name + '</div>'
        '<div style="font-size:.72rem;color:rgba(255,255,255,.4);">@' + user["username"] + '</div>'
        '</div>'
        '</div>'
        '<div style="font-size:.7rem;color:rgba(255,255,255,.35);'
        'border-top:1px solid rgba(255,255,255,.08);'
        'padding-top:.5rem;display:flex;justify-content:space-between;">'
        '<span>🗓️ Joined ' + joined + '</span>'
        + health_html +
        '</div>'
        '</div>'
    )
    st.sidebar.markdown(html, unsafe_allow_html=True)

    if st.sidebar.button("🚪 Sign Out", use_container_width=True):
        logout()
        st.rerun()
