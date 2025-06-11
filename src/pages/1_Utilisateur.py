import streamlit as st
from auth_helpers import ensure_authenticated, logout
from display_texts import dt

st.set_page_config(page_title=dt.LOGGED_AS, page_icon=":material/account_circle:")

if dt.LOGO:
    custom_css = """
    <style>
        div[data-testid="stSidebarHeader"] > img, div[data-testid="collapsedControl"] > img {
            height: 4rem;
            width: auto;
        }
        div[data-testid="stSidebarHeader"], div[data-testid="stSidebarHeader"] > *,
        div[data-testid="collapsedControl"], div[data-testid="collapsedControl"] > * {
            display: flex;
            align-items: center;
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)
    st.logo(image=dt.LOGO, size="large")

if not ensure_authenticated():
    st.stop()
    
st.title(dt.LOGGED_AS)

if "current_user_email" in st.session_state:
    st.markdown(f"**Email:** {st.session_state.current_user_email}")
else:
    st.warning(dt.USER_PROFILE_NO_EMAIL_WARNING)

if st.button("Logout", key="logout_button_profile_page", type='primary'):
    logout()