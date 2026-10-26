import streamlit as st
import hashlib
import hmac
import secrets
import time
from google_auth_oauthlib.flow import Flow

DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]





def _client_config():
    config = st.secrets["gdrive"]
    redirect_uri = config["redirect_uri"]
    return {
        "web": {
            "client_id": config["client_id"],
            "client_secret": config["client_secret"],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [redirect_uri],
        }
    }


def _new_flow(state=None):
    config = _client_config()
    redirect_uri = config["web"]["redirect_uris"][0]
    return Flow.from_client_config(
        config,
        scopes=DRIVE_SCOPES,
        state=state,
        redirect_uri=redirect_uri,
    )

def _make_oauth_state():
    client_secret = st.secrets["gdrive"]["client_secret"]
    timestamp = str(int(time.time()))
    nonce = secrets.token_urlsafe(24)
    payload = f"{timestamp}.{nonce}"

    signature = hmac.new(
        client_secret.encode(),
        payload.encode(),
        hashlib.sha256,
    ).hexdigest()

    return f"{payload}.{signature}"


def _validate_oauth_state(state):
    if not state:
        return False

    try:
        timestamp, nonce, signature = state.split(".", 2)
        payload = f"{timestamp}.{nonce}"

        age = int(time.time()) - int(timestamp)
        if age < 0 or age > 600:
            return False

        client_secret = st.secrets["gdrive"]["client_secret"]
        expected_signature = hmac.new(
            client_secret.encode(),
            payload.encode(),
            hashlib.sha256,
        ).hexdigest()

        return hmac.compare_digest(signature, expected_signature)
    except (ValueError, TypeError):
        return False

    

def start_google_oauth():
    """Build the Google consent URL with a signed OAuth state."""
    state = _make_oauth_state()
    flow = _new_flow(state=state)

    auth_url, _ = flow.authorization_url(
        state=state,
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )

    return auth_url



def process_google_oauth_callback():
    code = st.query_params.get("code")
    if not code:
        return

    returned_state = st.query_params.get("state")

    if not _validate_oauth_state(returned_state):
        st.query_params.clear()
        raise ValueError(
            "OAuth state validation failed or expired. "
            "Please sign in again."
        )

    code_verifier = st.session_state.get("google_drive_code_verifier")
    if not code_verifier:
        st.query_params.clear()
        raise ValueError(
            "Google sign-in session expired. "
            "Please click 'Sign in with Google Drive' and try again."
        )

    flow = _new_flow(state=returned_state)
    flow.code_verifier = code_verifier
    flow.fetch_token(code=code)

    st.session_state["google_drive_credentials"] = flow.credentials
    st.session_state.pop("google_drive_code_verifier", None)
    st.query_params.clear()


def get_google_drive_credentials():
    return st.session_state.get("google_drive_credentials")
