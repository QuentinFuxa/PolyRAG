import os
import streamlit as st
from display_texts import dt
from auth_helpers import login_ui

# st.markdown("<div style='position: absolute; top: 0px; left: 0px; font-weight: bold;'>xxxxx</div>", unsafe_allow_html=True)

if dt.LOGO:
    if dt.BIG_LOGO:
        custom_css = """
        <style>
            div[data-testid="stSidebarHeader"] > img, div[data-testid="collapsedControl"] > img {
                height: 3rem;
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

NO_AUTH = os.getenv("NO_AUTH", False)
NO_AUTH = False
if NO_AUTH:
    st.session_state.current_user_id = st.session_state.get('current_user_id', '00000000-0000-0000-0000-000000000001')
    st.session_state.current_user_email = st.session_state.get('current_user_email', "admin@admin")

logout_page = st.Page('frontend/user.py', title=dt.LOGOUT, icon=":material/logout:")
comments = st.Page("frontend/feedback.py", title=dt.FEEDBACK, icon=":material/feedback:")
help = st.Page(
    "frontend/help.py", title="Aide", icon=":material/lightbulb:")
changelog = st.Page(
    "frontend/changelog.py", title="Nouveautés v0.1.8 - 17/07/2025", icon=":material/source_notes:")
chatbot = st.Page("frontend/chat.py", title='Assistant', icon=":material/chat:", default=True)

if os.getenv('LANGUAGE', 'english') == "french":
    if "current_user_id" in st.session_state:
        pg = st.navigation(
        {
            "": [chatbot],
            "Utilisateur": [comments, logout_page],
            "PoC": [help, changelog],
        },
        )
    else:
        pg = st.navigation(pages=[
            st.Page(login_ui, title="Log in", icon=":material/login:")
        ])

else:
    if "current_user_id" in st.session_state:
        pg = st.navigation(
        {
            "": [chatbot],
            "User": [logout_page, comments],
        },
        )
    else:
       

        pg = st.navigation(pages=[
            st.Page(login_ui, title="Log in", icon=":material/login:")
        ])
        
pg.run()
