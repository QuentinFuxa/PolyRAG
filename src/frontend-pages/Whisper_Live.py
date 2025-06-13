import streamlit.components.v1 as components
from display_texts import dt
import streamlit as st

# Page configuration
# st.set_page_config(
#     page_title="Live Audio Transcription", 
#     page_icon="🎤",
#     layout="wide"
# )

if dt.LOGO:
    st.logo(image=dt.LOGO, size="large")

with open("src/pages_None/live_transcription.html", "r") as f:
    html_content = f.read()


with st.sidebar:
    st.markdown(
        """
        ## Live Audio Transcription
        You can try the live audio transcription feature here, with a time limit of 30 seconds per session.
        This feature uses the Whisper model to transcribe audio in real-time.
        Make sure to allow microphone access when prompted.
        
        - Model used: tiny.en
        """
    )


components.html(
    html_content, height=600,
)
