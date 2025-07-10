import streamlit as st
from auth_helpers import ensure_authenticated, logout
from display_texts import dt

if not ensure_authenticated():
    st.stop()
    

if "current_user_email" in st.session_state:
    st.markdown(f":material/account_circle: {dt.LOGGED_AS}: {st.session_state.current_user_email}")
else:
    st.warning(dt.USER_PROFILE_NO_EMAIL_WARNING)

if st.button(dt.LOGOUT_BUTTON, key="logout_button_profile_page", type='primary'):
    logout()