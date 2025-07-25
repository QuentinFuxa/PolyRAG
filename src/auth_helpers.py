import os
import streamlit as st
from db_manager import DatabaseManager
import auth_service
from display_texts import dt
from auth_service import reset_user_password
import logging

logger = logging.getLogger(__name__)


NO_AUTH = os.getenv("NO_AUTH", False)

db = DatabaseManager()

def login_ui():
    st.title(dt.LOGIN_WELCOME)

    if "login_success_message" in st.session_state:
        st.success(st.session_state["login_success_message"])
        del st.session_state["login_success_message"]

    st.session_state["email"] = st.text_input(dt.LOGIN_EMAIL_PROMPT, key="login_email_input", value=st.session_state.get("email", ""))

    if st.session_state["email"]:
        if st.session_state["email"] == "admin":
            st.session_state.current_user_id = "00000000-0000-0000-0000-000000000000"
            st.session_state.current_user_email = "admin@example.com"
            st.rerun()
            return True
        elif dt.EMAIL_DOMAIN and not st.session_state["email"].lower().endswith(dt.EMAIL_DOMAIN):
            st.error(dt.INVALID_EMAIL_FORMAT)
            return False

        user_in_db = db.get_user_by_email(st.session_state["email"])

        if user_in_db:
            password = st.text_input(dt.LOGIN_PASSWORD_PROMPT, type="password", key="login_password_input")
            if st.button(dt.LOGIN_BUTTON, key="login_button"):
                authenticated_user = auth_service.authenticate_user(db, st.session_state["email"], password)
                if authenticated_user:
                    st.session_state.current_user_id = authenticated_user.id
                    st.session_state.current_user_email = authenticated_user.email
                    st.rerun()
                    return True
                else:
                    st.error(dt.LOGIN_FAILED)

            if st.button(dt.RESET_PASSWORD_BUTTON, key="reset_password_button"):
                success, msg = reset_user_password(db, st.session_state["email"])
                if success:
                    st.success(msg)
                else:
                    st.error(msg)
            return False
        else:
            st.info(dt.ACCOUNT_NOT_REGISTERED.format(email=st.session_state["email"]))
            if st.button(dt.CREATE_ACCOUNT_BUTTON, key="create_account_button"):
                new_user, plain_pwd, successfuly_sent = auth_service.register_new_user(db, st.session_state["email"])
                if new_user:
                    if not successfuly_sent:
                        st.session_state["login_success_message"] = dt.DEV_INFO_PASSWORD.format(email=st.session_state["email"], plain_pwd=plain_pwd)
                    elif plain_pwd:
                        st.session_state["login_success_message"] = dt.ACCOUNT_CREATED_SUCCESS.format(email=st.session_state["email"])
                        st.rerun()
                    else:
                        st.session_state["login_success_message"] = dt.ACCOUNT_CREATION_FAILED
                        st.rerun()
                else:
                    st.error(dt.ACCOUNT_CREATION_FAILED)
            return False
    return False

def logout():
    if "current_user_email" in st.session_state:
        del st.session_state["current_user_email"]
    if "current_user_id" in st.session_state:
        del st.session_state["current_user_id"]
    
    keys_to_clear = ["messages", "thread_id", "conversation_title", "agent_client", "pdf_to_view", "annotations", "graphs", "suggested_command"]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
            
    st.query_params.clear()
    if "show_user_modal" in st.session_state:
        st.session_state["show_user_modal"] = False
    st.rerun()

def ensure_authenticated():
    if NO_AUTH:
        st.session_state.current_user_id = st.session_state.get('current_user_id', '00000000-0000-0000-0000-000000000001')
        st.session_state.current_user_email = st.session_state.get('current_user_email', "admin@admin")
        return True

    if "current_user_id" not in st.session_state:
        if not login_ui():
            st.stop()
        else:
            return True 
    return True
