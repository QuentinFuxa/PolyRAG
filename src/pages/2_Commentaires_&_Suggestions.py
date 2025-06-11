import streamlit as st
from auth_helpers import ensure_authenticated
from display_texts import dt # Assuming dt contains necessary text constants

# Page configuration
st.set_page_config(page_title=dt.SUGGESTIONS_GALLERY_TITLE, page_icon=":material/lightbulb:")

if dt.LOGO:
    st.logo(image=dt.LOGO, size="large")
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


# Ensure user is authenticated before rendering the page
if not ensure_authenticated():
    st.stop()

st.title(dt.FEEDBACK_BUTTON)

feedback_text = st.text_area(dt.FEEDBACK_DIALOG, key="feedback_text_area")
if st.button(label="", icon=":material/send:", key="submit_feedback_button"):
    if feedback_text:

        try:
            # Send feedback through the agent client
            st.session_state.agent_client.create_feedback(
                user_id=st.session_state.current_user_id,
                feedback=feedback_text,
            )
            
            st.session_state["show_user_modal"] = False # Keeping as is, though it might refer to a different modal.
            st.rerun() # Often used to refresh state and close dialogs
        except Exception as e:
            st.error(f"An error occurred while submitting your feedback: {e}")
