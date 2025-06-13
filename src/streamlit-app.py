import streamlit as st
from display_texts import dt
from auth_helpers import ensure_authenticated


if dt.LOGO:
    if dt.BIG_LOGO:
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


logout_page = st.Page('frontend-pages/user.py', title="Déconnexion", icon=":material/logout:")
comments = st.Page("frontend-pages/feedback.py", title=dt.FEEDBACK, icon=":material/comment:")
help = st.Page(
    "frontend-pages/help.py", title="Aide", icon=":material/lightbulb:")
changelog = st.Page(
    "frontend-pages/changelog.py", title="Nouveautés v0.1 - 12/06/2025", icon=":material/source_notes:")
chatbot = st.Page("frontend-pages/chat.py", title='Assistant', icon=":material/chat:", default=True)


if "current_user_id" in st.session_state:
    pg = st.navigation(
    {
        "Compte": [logout_page],
        "Commentaires et aide": [comments, help, changelog],
        "Assistant": [chatbot]
    },
    )
else:
    ensure_authenticated()

pg.run()