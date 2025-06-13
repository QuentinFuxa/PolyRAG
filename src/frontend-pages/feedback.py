import streamlit as st
from auth_helpers import ensure_authenticated
from display_texts import dt

if not ensure_authenticated():
    st.stop()

st.title(dt.FEEDBACK)

feedback_text = st.text_area(dt.FEEDBACK_DIALOG, key="feedback_text_area")
if st.button(label="", icon=":material/send:", key="submit_feedback_button"):
    if feedback_text:

        try:
            st.session_state.agent_client.submit_user_feedback(
                user_id=st.session_state.current_user_id,
                feedback_content=feedback_text,
            )
            
            st.toast(dt.FEEDBACK_SUBMITTED_TOAST, icon=dt.FEEDBACK_STARS_ICON)
        except Exception as e:
            st.error(f"An error occurred while submitting your feedback: {e}")
