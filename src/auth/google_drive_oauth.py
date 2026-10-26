import streamlit as st
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


def start_google_oauth():
    """Build the Google consent URL and remember the OAuth state for callback validation."""
    flow = _new_flow()
    auth_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )
    st.session_state["google_drive_oauth_state"] = state
    return auth_url


def process_google_oauth_callback():
    """Exchange Google's authorization code for a Drive API credential."""
    code = st.query_params.get("code")
    if not code:
        return

    returned_state = st.query_params.get("state")
    expected_state = st.session_state.get("google_drive_oauth_state")
    if not expected_state or not returned_state or returned_state != expected_state:
        st.query_params.clear()
        raise ValueError(
            "OAuth state validation failed. Please click 'Sign in with Google Drive' and try again."
        )

    flow = _new_flow(state=expected_state)
    flow.fetch_token(code=code)
    st.session_state["google_drive_credentials"] = flow.credentials
    st.session_state.pop("google_drive_oauth_state", None)
    st.query_params.clear()


def get_google_drive_credentials():
    return st.session_state.get("google_drive_credentials")
