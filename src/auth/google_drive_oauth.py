from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from threading import RLock
from typing import Any

import streamlit as st
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow


DRIVE_SCOPES = [
    "https://www.googleapis.com/auth/drive.readonly",
]

SESSION_KEY = "google_drive_credentials"
PENDING_TTL_SECONDS = 600


# --------------------------------------------------
# OAuth configuration
# --------------------------------------------------

def _client_config() -> dict[str, Any]:
    """Load OAuth client settings from Streamlit secrets."""
    config = st.secrets["gdrive"]

    redirect_uri = str(config["redirect_uri"]).strip()

    return {
        "web": {
            "client_id": config["client_id"],
            "client_secret": config["client_secret"],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [redirect_uri],
        }
    }


def _new_flow(
    state: str | None = None,
    code_verifier: str | None = None,
) -> Flow:
    """Build a Flow, optionally restoring the original PKCE verifier."""
    config = _client_config()

    return Flow.from_client_config(
        config,
        scopes=DRIVE_SCOPES,
        state=state,
        redirect_uri=config["web"]["redirect_uris"][0],
        code_verifier=code_verifier,
    )


# --------------------------------------------------
# OAuth state validation
# --------------------------------------------------

def _make_oauth_state() -> str:
    """Create a signed state value with a timestamp and random nonce."""
    client_secret = st.secrets["gdrive"]["client_secret"]

    timestamp = str(int(time.time()))
    nonce = secrets.token_urlsafe(24)
    payload = f"{timestamp}.{nonce}"

    signature = hmac.new(
        client_secret.encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    return f"{payload}.{signature}"


def _validate_oauth_state(state: str | None) -> bool:
    """Check the state signature and its expiry."""
    if not state:
        return False

    try:
        timestamp, nonce, signature = state.split(".", 2)
        payload = f"{timestamp}.{nonce}"

        age = int(time.time()) - int(timestamp)

        if age < 0 or age > PENDING_TTL_SECONDS:
            return False

        client_secret = st.secrets["gdrive"]["client_secret"]

        expected_signature = hmac.new(
            client_secret.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        return hmac.compare_digest(signature, expected_signature)

    except (ValueError, TypeError, KeyError):
        return False


# --------------------------------------------------
# Pending OAuth transactions
# --------------------------------------------------

@st.cache_resource
def _pending_logins() -> dict[str, dict[str, Any]]:
    """
    Store pending logins by state.

    This survives reruns in the same process, but is not a shared,
    durable store across different workers or process restarts.
    """
    return {}


@st.cache_resource
def _pending_lock() -> RLock:
    return RLock()


def _purge_expired_logins() -> None:
    """Remove pending logins that are older than the allowed time."""
    now = time.time()
    store = _pending_logins()

    with _pending_lock():
        expired_states = [
            state
            for state, details in store.items()
            if now - details["created"] > PENDING_TTL_SECONDS
        ]

        for state in expired_states:
            store.pop(state, None)


# --------------------------------------------------
# Start OAuth
# --------------------------------------------------

def start_google_oauth() -> str:
    """Create Google's consent URL and save the PKCE verifier."""
    state = _make_oauth_state()
    code_verifier = secrets.token_urlsafe(64)

    flow = _new_flow(
        state=state,
        code_verifier=code_verifier,
    )

    # Do NOT pass code_verifier here. The authorization request should
    # contain the derived code_challenge, not the verifier itself.
    auth_url, _ = flow.authorization_url(
        state=state,
        code_challenge_method="S256",
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )

    _purge_expired_logins()

    with _pending_lock():
        _pending_logins()[state] = {
            "code_verifier": code_verifier,
            "created": time.time(),
        }

    return auth_url


# --------------------------------------------------
# Process Google's callback
# --------------------------------------------------

def process_google_oauth_callback() -> None:
    """Exchange the returned authorization code for credentials."""
    params = st.query_params

    if "error" in params:
        error = params.get("error", "unknown_error")
        st.query_params.clear()
        raise RuntimeError(f"Google returned an error: {error}")

    code = params.get("code")
    returned_state = params.get("state")

    if not code and not returned_state:
        return

    # Clear the one-time authorization code from the browser URL.
    st.query_params.clear()

    if not code or not returned_state:
        raise RuntimeError(
            "Google's callback is missing the authorization code or state. "
            "Please start sign-in again."
        )

    if not _validate_oauth_state(returned_state):
        raise RuntimeError(
            "OAuth state validation failed or expired. "
            "Please start a new Google sign-in."
        )

    _purge_expired_logins()

    # Pop makes this pending transaction single-use in this process.
    with _pending_lock():
        pending = _pending_logins().pop(returned_state, None)

    if pending is None:
        raise RuntimeError(
            "The pending Google sign-in could not be found. "
            "The app may have restarted or the callback may have reached "
            "a different worker. Please try signing in again."
        )

    code_verifier = pending.get("code_verifier")

    if not code_verifier:
        raise RuntimeError(
            "The PKCE verifier is missing. Please start Google sign-in again."
        )

    flow = _new_flow(
        state=returned_state,
        code_verifier=code_verifier,
    )

    # Supply the verifier for the token exchange, not the authorization URL.
    flow.fetch_token(
        code=code,
        code_verifier=code_verifier,
    )

    st.session_state[SESSION_KEY] = flow.credentials
    st.session_state.pop("google_drive_oauth_state", None)
    st.session_state.pop("google_drive_code_verifier", None)


# --------------------------------------------------
# Get / refresh credentials
# --------------------------------------------------

def get_google_drive_credentials() -> Credentials | None:
    """Return valid Google Drive credentials, refreshing when possible."""
    credentials = st.session_state.get(SESSION_KEY)

    if not credentials:
        return None

    if credentials.valid:
        return credentials

    if credentials.expired and credentials.refresh_token:
        try:
            credentials.refresh(Request())
            st.session_state[SESSION_KEY] = credentials
            return credentials
        except Exception:
            st.session_state.pop(SESSION_KEY, None)
            return None

    st.session_state.pop(SESSION_KEY, None)
    return None