import os
import streamlit as st
from display_texts import dt
from auth_helpers import login_ui


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


logout_page = st.Page('frontend-pages/user.py', title="Déconnexion", icon=":material/logout:")
comments = st.Page("frontend-pages/feedback.py", title=dt.FEEDBACK, icon=":material/comment:")
help = st.Page(
    "frontend-pages/help.py", title="Aide", icon=":material/lightbulb:")
changelog = st.Page(
    "frontend-pages/changelog.py", title="Nouveautés v0.1 - 16/06/2025", icon=":material/source_notes:")
chatbot = st.Page("frontend-pages/chat.py", title='Assistant', icon=":material/chat:", default=True)

NO_AUTH = os.getenv("NO_AUTH", False)
if NO_AUTH:
    st.session_state.current_user_id = st.session_state.get('current_user_id', '00000000-0000-0000-0000-000000000001')
    st.session_state.current_user_email = st.session_state.get('current_user_email', "user@test.test")

if "current_user_id" in st.session_state:
    pg = st.navigation(
    {
        "Compte": [logout_page],
        "Commentaires et aide": [comments, help, changelog],
        "Assistant": [chatbot]
    },
    )
else:
    
    pg = st.navigation(pages=[
        st.Page(login_ui, title="Log in", icon=":material/login:")
    ])
pg.run()

