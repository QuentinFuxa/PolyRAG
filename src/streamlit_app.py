import asyncio
import os
import urllib.parse
import uuid
from collections.abc import AsyncGenerator
from typing import List, Union, Dict, Set, Optional
from uuid import uuid4, UUID as UUID_TYPE
import streamlit as st
import pandas as pd
from io import StringIO
from dotenv import load_dotenv
from pydantic import ValidationError

from client import AgentClient, AgentClientError
from schema import ChatHistory, ChatMessage
from schema.task_data import TaskData, TaskDataStatus
import json
from frontend.pdf_viewer_with_annotations import display_pdf

from db_manager import DatabaseManager
import auth_service

class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)
        for k, v in self.items():
            if isinstance(v, dict):
                self[k] = DotDict(v)
            elif isinstance(v, list):
                self[k] = [DotDict(i) if isinstance(i, dict) else i for i in v]

display_texts_json_path = os.getenv("DISPLAY_TEXTS_JSON_PATH", "display_texts.json") # Added default

dt_data = {}
try:
    with open(display_texts_json_path, 'r', encoding='utf-8') as f:
        dt_data = json.load(f)
    dt = DotDict(dt_data)
except FileNotFoundError:
    st.error(f"FATAL: Display texts JSON file not found at '{display_texts_json_path}'. The application cannot start without it.")
    raise
except json.JSONDecodeError as e:
    st.error(f"FATAL: Error decoding display texts JSON file at '{display_texts_json_path}': {e}. The application cannot start.")
    raise
except Exception as e:
    st.error(f"FATAL: An unexpected error occurred while loading display texts from '{display_texts_json_path}': {e}. The application cannot start.")
    raise


APP_TITLE = dt.APP_TITLE
APP_ICON = dt.APP_ICON
AI_ICON = dt.AI_ICON
USER_ICON = dt.USER_ICON

db = DatabaseManager()

st.session_state.pdf_to_view = st.session_state.get("pdf_to_view", None)
st.session_state.annotations = st.session_state.get("annotations", None)
st.session_state.debug_viewer = st.session_state.get("debug_viewer", False)
st.session_state.confirming_delete_thread_id = st.session_state.get("confirming_delete_thread_id", None)
st.session_state.graphs = st.session_state.get("graphs", {})


def view_pdf(pdf_to_view, annotations=None, debug_viewer=False):
    st.session_state.pdf_to_view = pdf_to_view
    st.session_state.annotations = annotations
    st.session_state.debug_viewer = debug_viewer
    if not st.session_state.get("in_pdf_dialog", False):
        st.session_state.in_pdf_dialog = True
        st.rerun()

dialog_title = dt.PDF_DIALOG_TITLE if not st.session_state.debug_viewer else dt.PDF_DIALOG_DEBUG_PREFIX + ' ' + str(st.session_state.pdf_to_view)
@st.dialog(dialog_title, width="large")
def pdf_dialog():
    """Displays the selected PDF using the new display_pdf function."""
    st.session_state.in_pdf_dialog = True
    
    pdf_name = st.session_state.pdf_to_view
    annotations = st.session_state.annotations
    debug_viewer = st.session_state.get("debug_viewer", False)
    current_agent_client = st.session_state.get("agent_client")

    st.session_state.pdf_to_view = None
    st.session_state.annotations = None
    st.session_state.in_pdf_dialog = False

    if pdf_name and current_agent_client:
        try:
            display_pdf(agent_client=current_agent_client, document_name=pdf_name, annotations=annotations, debug_viewer=debug_viewer)
        except Exception as e:
            st.error(f"Error displaying PDF: {e}")
            print(f"Error in pdf_dialog calling display_pdf: {e}")
    else:
        if not pdf_name: st.error(dt.PDF_DIALOG_NO_PDF_ERROR)
        if not current_agent_client: st.error("Agent client not available for PDF dialog.")
    if st.button(dt.PDF_DIALOG_CLOSE_BUTTON): st.rerun()

def hide_welcome():
    st.markdown("<style>div[class*=\"st-key-welcome-msg\"] { display: none; }</style>", unsafe_allow_html=True)

def login_ui():
    st.title(dt.LOGIN_WELCOME)
    email = st.text_input(dt.LOGIN_EMAIL_PROMPT, key="login_email_input")

    if email:
        if not email.lower().endswith("@asnr.fr"):
            st.error(dt.INVALID_EMAIL_FORMAT)
            return False # Not logged in

        user_in_db = db.get_user_by_email(email)

        if user_in_db:
            password = st.text_input(dt.LOGIN_PASSWORD_PROMPT, type="password", key="login_password_input")
            if st.button(dt.LOGIN_BUTTON, key="login_button"):
                authenticated_user = auth_service.authenticate_user(db, email, password)
                if authenticated_user:
                    st.session_state.current_user_id = authenticated_user.id
                    st.session_state.current_user_email = authenticated_user.email # Store email for display
                    st.rerun()
                else:
                    st.error(dt.LOGIN_FAILED)
            return False # Not logged in yet or failed
        else:
            st.info(f"Email {email} is not registered.")
            if st.button(dt.CREATE_ACCOUNT_BUTTON, key="create_account_button"):
                new_user, plain_pwd = auth_service.register_new_user(db, email)
                if new_user:
                    st.success(dt.ACCOUNT_CREATED_SUCCESS.format(email=email))
                    if plain_pwd: # Should always be true if new_user is not None
                         st.info(dt.DEV_INFO_PASSWORD.format(email=email, plain_pwd=plain_pwd)) # For dev purposes
                else:
                    st.error(dt.ACCOUNT_CREATION_FAILED)
            return False # Not logged in
    return False # Not logged in

async def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, menu_items={})

    if "current_user_id" not in st.session_state:
        if not login_ui():
            st.stop() # Stop execution if login is not successful
    
    current_user_id: UUID_TYPE = st.session_state.current_user_id
    current_user_email = st.session_state.get("current_user_email", "User")


    if st.session_state.pdf_to_view: pdf_dialog()
    if "suggested_command" not in st.session_state: st.session_state.suggested_command = None
    user_text = None

    if st.get_option("client.toolbarMode") != "minimal":
        st.set_option("client.toolbarMode", "minimal")
        await asyncio.sleep(0.1)
        st.rerun()

    if "agent_client" not in st.session_state:
        load_dotenv()
        agent_url = os.getenv("AGENT_URL")
        if not agent_url:
            host = os.getenv("HOST", "0.0.0.0"); port = os.getenv("PORT", 8080)
            agent_url = f"http://{host}:{port}"
        try:
            with st.spinner(dt.AGENT_CONNECTING_SPINNER):
                st.session_state.agent_client = AgentClient(base_url=agent_url)
        except AgentClientError as e:
            st.error(dt.AGENT_CONNECTION_ERROR.format(agent_url=agent_url, e=e))
            st.markdown(dt.AGENT_CONNECTION_RETRY_MSG); st.stop()
    agent_client: AgentClient = st.session_state.agent_client

    query_thread_id = st.query_params.get("thread_id")
    if query_thread_id and ("thread_id" not in st.session_state or query_thread_id != st.session_state.thread_id):
        try:
            # TODO: Ensure get_history is user-scoped if AgentClient handles it, or adapt
            messages_history = agent_client.get_history(thread_id=query_thread_id) # Pass user_id if backend supports
            st.session_state.messages = messages_history.messages
            st.session_state.thread_id = query_thread_id
            try:
                # TODO: Ensure get_conversation_title is user-scoped
                title = agent_client.get_conversation_title(query_thread_id, user_id=current_user_id)
                st.session_state.conversation_title = title
            except: st.session_state.conversation_title = dt.DEFAULT_CONVERSATION_TITLE
        except AgentClientError as e: st.error(dt.CONVERSATION_LOAD_ERROR.format(e=e))

    if "thread_id" not in st.session_state:
        thread_id = st.query_params.get("thread_id")
        if not thread_id:
            thread_id = str(uuid.uuid4()); messages = []
        else:
            try:
                # TODO: Ensure get_history is user-scoped
                messages_history = agent_client.get_history(thread_id=thread_id) # Pass user_id if backend supports
                messages = messages_history.messages
            except AgentClientError: st.error(dt.CONVERSATION_NOT_FOUND_ERROR); messages = []
        st.session_state.messages = messages
        st.session_state.thread_id = thread_id

    with st.sidebar:
        st.header(f"{dt.APP_ICON} {dt.APP_TITLE}")
        st.markdown(f"Logged in as: {current_user_email}") # Display logged-in user
        if st.button(dt.LOGOUT_BUTTON, key="logout_button_sidebar"):
            del st.session_state["current_user_id"]
            if "current_user_email" in st.session_state: del st.session_state["current_user_email"]
            # Clear other session state related to user if necessary
            st.query_params.clear() # Clear query params on logout
            st.rerun()

        st.markdown(dt.SIDEBAR_HEADER)
        if st.button(dt.NEW_CONVERSATION_BUTTON, use_container_width=False, icon=":material/add:", type="tertiary", disabled=len(st.session_state.messages) == 0):
            st.query_params.clear()
            st.session_state.messages = []
            st.session_state.conversation_title = dt.DEFAULT_CONVERSATION_TITLE
            st.session_state.thread_id = str(uuid4())
            st.rerun()

        if "conversation_title" not in st.session_state:
            try:
                # TODO: Ensure get_conversation_title is user-scoped
                title = agent_client.get_conversation_title(st.session_state.thread_id, user_id=current_user_id)
                st.session_state.conversation_title = title
            except: st.session_state.conversation_title = dt.DEFAULT_CONVERSATION_TITLE
        
        if st.session_state.get("editing_title", False):
            new_title = st.text_input(dt.EDIT_TITLE_INPUT_LABEL, value=st.session_state.conversation_title, key="new_title_input")
            col1, col2 = st.columns(2)
            with col1:
                if st.button(dt.SAVE_TITLE_BUTTON, key="save_title"):
                    # TODO: Ensure set_conversation_title is user-scoped
                    agent_client.set_conversation_title(st.session_state.thread_id, new_title, user_id=current_user_id)
                    st.session_state.conversation_title = new_title
                    st.session_state.editing_title = False; st.rerun()
            with col2:
                if st.button(dt.CANCEL_TITLE_BUTTON, key="cancel_title"):
                    st.session_state.editing_title = False; st.rerun()
        
        try:
            # TODO: Ensure get_conversations is user-scoped
            conversations = agent_client.get_conversations(limit=20, user_id=current_user_id)
            if conversations:
                st.subheader(dt.RECENT_SUBHEADER)
                for conv in conversations:
                    thread_id_conv = conv["thread_id"]; title = conv["title"]; updated_at = conv["updated_at"]
                    try:
                        from datetime import datetime
                        updated_date = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
                        date_str = updated_date.strftime("%d/%m/%Y %H:%M")
                    except: date_str = updated_at
                    
                    col1, col2, col3 = st.columns([0.9, 0.05, 0.05]) # Adjusted for potentially wider titles
                    with col1:
                        if st.button(f"{title}", key=f"conv_{thread_id_conv}", help=dt.CONVERSATION_LAST_UPDATED_HELP.format(date_str=date_str), type='tertiary'):
                            st.query_params["thread_id"] = thread_id_conv; st.rerun()
                    with col2:
                        if st.button(":material/edit:", key=f"edit_{thread_id_conv}", help=dt.EDIT_CONVERSATION_TITLE_HELP, type='tertiary'):
                            st.query_params["thread_id"] = thread_id_conv
                            st.session_state.editing_title = True; st.rerun()
                    with col3:
                        if st.button(":material/delete:", key=f"delete_{thread_id_conv}", help=dt.DELETE_CONVERSATION_HELP, type='tertiary'):
                            # TODO: Ensure delete_conversation is user-scoped
                            if agent_client.delete_conversation(thread_id_conv, user_id=current_user_id):
                                if thread_id_conv == st.session_state.thread_id: st.query_params.clear()
                                st.rerun()
        except Exception as e: st.error(dt.LOAD_CONVERSATIONS_ERROR.format(e=e))
        
        st.divider()
        with st.popover(dt.SETTINGS_POPOVER_LABEL, use_container_width=True):
            model_idx = agent_client.info.models.index(agent_client.info.default_model)
            model = st.selectbox(dt.SELECT_LLM_LABEL, options=agent_client.info.models, index=model_idx)
            agent_list = [a.key for a in agent_client.info.agents]
            agent_idx = agent_list.index(agent_client.info.default_agent)
            agent_client.agent = st.selectbox(dt.SELECT_AGENT_LABEL, options=agent_list, index=agent_idx)
            use_streaming = st.toggle(dt.STREAMING_TOGGLE_LABEL, value=True)
        st.caption(dt.CAPTION)

    messages: list[ChatMessage] = st.session_state.messages
    if len(messages) == 0 and user_text is None:
        with st.container(key="welcome-msg"):
            match agent_client.agent:
                case "chatbot": WELCOME = dt.WELCOME_CHATBOT
                # ... (other cases remain the same) ...
                case _: WELCOME = dt.WELCOME_DEFAULT
            with st.chat_message("ai", avatar=AI_ICON): st.write(WELCOME)
            # ... (example prompts remain the same) ...

    await draw_messages(amessage_iter(), agent_client=agent_client)

    if user_input := st.chat_input(dt.CHAT_INPUT_PLACEHOLDER, accept_file="multiple", file_type=["pdf"]) or st.session_state.suggested_command:
        hide_welcome()
        if st.session_state.suggested_command:
            user_text = st.session_state.suggested_command; files = []
            st.session_state.suggested_command = None
        elif user_input: # Check if user_input is not None (it can be if only suggested_command was true)
            user_text = user_input.text
            files = user_input.files if hasattr(user_input, 'files') else [] # Ensure files attribute exists
        
        if user_text is None and not files: # Handle case where only suggested_command was true but it was empty
             st.warning("Please enter a message or select a suggestion.") # Or handle as appropriate
        else:
            messages.append(ChatMessage(type="human", content=user_text or "", attached_files=[f.name for f in files]))
            additional_markdown = ""
            if files: 
                additional_markdown = "  \n"                   
                for file_name_obj in files: additional_markdown += dt.FILE_ATTACHED_BADGE.format(file_name=file_name_obj.name)
            st.chat_message("human", avatar=USER_ICON).write((user_text or "") + additional_markdown)
            
            uploaded_file_ids = []
            if files:
                upload_status = st.status(dt.FILE_UPLOADING_STATUS, state="running")
                for file_obj in files:
                    file_content = file_obj.getvalue(); file_name = file_obj.name; file_type = file_obj.type
                    try:
                        # TODO: Ensure upload_file is user-scoped
                        file_id = agent_client.upload_file(
                            file_name=file_name, file_content=file_content, file_type=file_type,
                            thread_id=st.session_state.thread_id, user_id=current_user_id
                        )
                        uploaded_file_ids.append(file_id)
                        upload_status.update(label=dt.FILE_UPLOAD_SUCCESS_STATUS.format(file_name=file_name))
                    except Exception as e: upload_status.error(dt.FILE_UPLOAD_ERROR_STATUS.format(file_name=file_name, e=e))
                upload_status.update(state="complete", label=f"{len(uploaded_file_ids)} files uploaded")                
            try:
                # TODO: Ensure astream/ainvoke are user-scoped if necessary at backend
                invoke_params = {
                    "message": user_text or "", "model": model, 
                    "thread_id": st.session_state.thread_id, 
                    "file_ids": uploaded_file_ids if files else None,
                    # "user_id": current_user_id # Pass if AgentClient/backend supports
                }
                if use_streaming:
                    stream = agent_client.astream(**invoke_params)
                    await draw_messages(stream, is_new=True, agent_client=agent_client)
                else:
                    response = await agent_client.ainvoke(**invoke_params)
                    messages.append(response)
                    st.chat_message("ai", avatar=AI_ICON).write(response.content)

                if len(messages) > 1 and st.session_state.conversation_title == dt.DEFAULT_CONVERSATION_TITLE:
                    try:
                        title_prompt = dt.TITLE_GENERATION_PROMPT.format(user_text=user_text or "Uploaded files")
                        # TODO: Ensure ainvoke for title is user-scoped if necessary
                        title_response = await agent_client.ainvoke(message=title_prompt, model=model) # user_id?
                        generated_title = title_response.content.strip().strip('"\'')
                        # TODO: Ensure aset_conversation_title is user-scoped
                        await agent_client.aset_conversation_title(st.session_state.thread_id, generated_title, user_id=current_user_id)
                        st.session_state.conversation_title = generated_title
                    except Exception as e: pass # Silently fail title generation
                st.rerun()
            except AgentClientError as e: st.error(dt.RESPONSE_GENERATION_ERROR.format(e=e)); st.stop()
    
    if len(messages) > 0 and st.session_state.get("last_message"): # Check if last_message exists
        with st.session_state.last_message:
            await handle_feedback()

async def amessage_iter() -> AsyncGenerator[ChatMessage, None]:
    for m in st.session_state.get("messages", []):
        yield m

async def draw_messages(
    messages_agen: AsyncGenerator[ChatMessage | str, None],
    is_new: bool = False,
    agent_client: AgentClient = None, # agent_client is passed, can be used
) -> None:
    if "pdf_documents" not in st.session_state: st.session_state.pdf_documents = {}
    last_message_type = None
    st.session_state.last_message = None # Initialize/reset
    streaming_content = ""
    streaming_placeholder = None
    
    current_user_id = st.session_state.get("current_user_id") # Get current user for any user-specific logic

    while msg := await anext(messages_agen, None):
        if isinstance(msg, str): # Streaming case
            if not streaming_placeholder:
                if last_message_type != "ai": # Ensure we are in an AI message block
                    last_message_type = "ai"
                    st.session_state.last_message = st.chat_message("ai", avatar=AI_ICON)
                with st.session_state.last_message: streaming_placeholder = st.empty()
            streaming_content += msg
            streaming_placeholder.write(streaming_content)
            continue

        if not isinstance(msg, ChatMessage):
            st.error(dt.UNEXPECTED_MESSAGE_TYPE_ERROR.format(msg_type=type(msg))); st.write(msg); st.stop()

        match msg.type:
            case "human":
                last_message_type = "human"
                additional_markdown = ""
                if hasattr(msg, 'attached_files') and msg.attached_files: 
                    additional_markdown = "  \n"                   
                    for file_name in msg.attached_files: additional_markdown += dt.FILE_ATTACHED_BADGE.format(file_name=file_name)
                st.chat_message("human", avatar=USER_ICON).write(msg.content + additional_markdown)
            case "ai":
                if is_new: st.session_state.messages.append(msg)
                if last_message_type != "ai":
                    last_message_type = "ai"
                    st.session_state.last_message = st.chat_message("ai", avatar=AI_ICON)
                
                with st.session_state.last_message:
                    if msg.content:
                        if streaming_placeholder: # End of stream for this message part
                            streaming_placeholder.write(msg.content); streaming_content = ""; streaming_placeholder = None
                        else: st.write(msg.content) # Non-streamed AI message

                    if msg.tool_calls:
                        # ... (tool call logic remains the same, ensure agent_client calls within are user-scoped if needed)
                        # For example, agent_client.retrieve_graph might need user_id if graphs are user-specific
                        # agent_client.aget_annotations / adebug_pdf_blocks might also need user_id
                        call_results = {}; tool_names = {}; graph_viewer_call_ids = set()
                        for tool_call in msg.tool_calls:
                            status_container = st.status(dt.TOOL_CALL_STATUS.format(tool_name=tool_call["name"]), state="running" if is_new else "complete")
                            call_results[tool_call["id"]] = status_container
                            tool_names[tool_call["id"]] = tool_call["name"]
                            if tool_call["name"] in ["Graph_Viewer", "tool_graphing_agent"]: graph_viewer_call_ids.add(tool_call["id"])
                            with status_container: st.write(dt.TOOL_CALL_INPUT_LABEL); st.write(tool_call["args"])
                        
                        pending_tool_call_ids = set(call_results.keys())
                        while pending_tool_call_ids and (tool_result := await anext(messages_agen, None)):
                            if isinstance(tool_result, str):
                                if streaming_placeholder: streaming_content += tool_result; streaming_placeholder.write(streaming_content)
                                continue
                            if tool_result.type != "tool":
                                messages_agen = chain_messages([tool_result], messages_agen); break
                            if is_new: st.session_state.messages.append(tool_result)
                            if tool_result.tool_call_id in pending_tool_call_ids: pending_tool_call_ids.remove(tool_result.tool_call_id)
                            await process_tool_result(tool_result, tool_names, call_results, graph_viewer_call_ids, agent_client, current_user_id) # Pass current_user_id
            case "custom": # ... (custom message logic remains the same) ...
                try: task_data: TaskData = TaskData.model_validate(msg.custom_data)
                except ValidationError: st.error(dt.UNEXPECTED_CUSTOMDATA_ERROR); st.write(msg.custom_data); st.stop()
                if is_new: st.session_state.messages.append(msg)
                if last_message_type != "task":
                    last_message_type = "task"
                    st.session_state.last_message = st.chat_message(name="task", avatar=":material/manufacturing:")
                    with st.session_state.last_message: status_widget = TaskDataStatus()
                status_widget.add_and_draw_task_data(task_data)
            case _: st.error(dt.UNEXPECTED_CHATMESSAGE_TYPE_ERROR.format(msg_type=msg.type)); st.write(msg); st.stop()

async def chain_messages(initial_messages: List[ChatMessage], next_messages: AsyncGenerator[Union[ChatMessage, str], None]) -> AsyncGenerator[Union[ChatMessage, str], None]:
    for msg in initial_messages: yield msg
    async for msg in next_messages: yield msg

async def process_tool_result(tool_result: ChatMessage, tool_names: Dict[str, str], call_results: Dict[str, object], graph_viewer_call_ids: Set[str], agent_client: AgentClient = None, user_id: Optional[UUID_TYPE] = None):
    current_tool_name = tool_names.get(tool_result.tool_call_id)
    status = call_results.get(tool_result.tool_call_id)
    plot_data_for_this_tool = None
    if not status: st.error(f"Could not find status container for tool_call_id: {tool_result.tool_call_id}"); return
    
    if tool_result.tool_call_id in graph_viewer_call_ids and agent_client:
        with status:
            try:
                graph_id = tool_result.content
                st.write(dt.GRAPH_RETRIEVAL_STATUS.format(graph_id=graph_id))
                # TODO: Ensure retrieve_graph is user-scoped if graphs are user-specific
                graph_data = agent_client.retrieve_graph(graph_id) # Pass user_id if backend supports
                if graph_data:
                    try: plot_data = json.loads(graph_data); st.write(dt.GRAPH_RETRIEVED_SUCCESSFULLY_FR); plot_data_for_this_tool = plot_data
                    except json.JSONDecodeError: st.write(dt.GRAPH_NON_JSON_DATA); st.code(graph_data)
                else: st.write(dt.GRAPH_NO_DATA_RETURNED)
                status.update(state="complete")
            except Exception as e: st.error(dt.GRAPH_RETRIEVAL_ERROR.format(e=e)); status.update(state="complete")
    
    elif current_tool_name == "PDF_Viewer":
        # ... (PDF_Viewer logic remains the same, ensure agent_client calls are user-scoped if needed) ...
        # e.g., agent_client.aget_annotations might need user_id
        with status:
            try:
                tool_output = json.loads(tool_result.content)
                pdf_name = "Error" # Default
                if tool_output.get('error'): st.error(tool_output['error'])
                else: pdf_name = tool_output['pdf_file']; st.write(dt.PDF_VIEWER_OUTPUT_LABEL.format(pdf_name=pdf_name))
                status.update(state="complete", label=dt.PDF_READY_STATUS.format(pdf_name=pdf_name))
            except Exception as e: st.error(dt.PDF_PROCESSING_ERROR.format(e=e)); st.write(dt.RAW_OUTPUT_LABEL.format(content=tool_result.content)); status.update(state="complete")
        if current_tool_name == "PDF_Viewer" and not json.loads(tool_result.content).get('error'):
            tool_output_btn = json.loads(tool_result.content)
            pdf_name_btn = tool_output_btn['pdf_file']; block_indices_btn = tool_output_btn['block_indices']
            debug_viewer_btn = tool_output_btn.get('debug', False); annotations_btn = []
            if agent_client:
                # TODO: Ensure aget_annotations/adebug_pdf_blocks are user-scoped
                if not debug_viewer_btn: annotations_btn = await agent_client.aget_annotations(pdf_file=pdf_name_btn, block_indices=block_indices_btn) # user_id?
                else: annotations_btn = await agent_client.adebug_pdf_blocks(pdf_file=pdf_name_btn) # user_id?
            st.session_state.pdf_documents[pdf_name_btn] = annotations_btn
            if st.button(dt.VIEW_PDF_BUTTON.format(pdf_name=pdf_name_btn), key=f"pdf_button_{tool_result.tool_call_id}"):
                view_pdf(pdf_name_btn, annotations_btn, debug_viewer=debug_viewer_btn)
    else: # Other tools
        # ... (SQL_Executor and generic tool output remain the same) ...
        with status:
            st.write(dt.TOOL_CALL_OUTPUT_LABEL)
            if current_tool_name == "SQL_Executor": # ... (SQL_Executor logic) ...
                csv_string = tool_result.content; data_lines = []; comment_lines = []
                for line in csv_string.strip().split('\n'):
                    if line.startswith("#"): comment_lines.append(line)
                    else: data_lines.append(line)
                if data_lines:
                    try: df = pd.read_csv(StringIO("\n".join(data_lines)), sep=';'); st.dataframe(df)
                    except Exception as e: st.error(f"Error parsing CSV: {e}"); st.code(csv_string, language="csv")
                else: st.info("No data from SQL query.")
                for comment in comment_lines: 
                    if "warning" in comment.lower(): st.warning(comment)
                    elif "error" in comment.lower(): st.error(comment)
                    else: st.text(comment)
            elif not plot_data_for_this_tool:
                try: json_response = json.loads(tool_result.content)
                except: json_response = None
                if json_response is not None and isinstance(json_response, (dict, list)): st.json(json_response)
                else: st.write(tool_result.content if tool_result.content else dt.TOOL_OUTPUT_EMPTY)
            status.update(state="complete")

    if plot_data_for_this_tool: st.plotly_chart(plot_data_for_this_tool)

async def handle_feedback() -> None:
    if "last_star_feedback" not in st.session_state: st.session_state.last_star_feedback = (None, None)
    if "text_feedback_runs" not in st.session_state: st.session_state.text_feedback_runs = set()
    if not st.session_state.messages or not hasattr(st.session_state.messages[-1], 'run_id'): return
    latest_run_id = st.session_state.messages[-1].run_id
    if not latest_run_id: return
    feedback_stars = st.feedback(dt.FEEDBACK_STARS_KEY, key=f"stars_{latest_run_id}")
    if feedback_stars is not None and (latest_run_id, feedback_stars) != st.session_state.last_star_feedback:
        normalized_score = (feedback_stars + 1) / 5.0; agent_client: AgentClient = st.session_state.agent_client
        try:
            await agent_client.acreate_feedback(run_id=latest_run_id, key="human-feedback-stars", score=normalized_score, kwargs={"comment": dt.FEEDBACK_HUMAN_INLINE_COMMENT})
            st.session_state.last_star_feedback = (latest_run_id, feedback_stars)
            st.toast(dt.FEEDBACK_SAVED_TOAST, icon=dt.FEEDBACK_STARS_ICON)
        except AgentClientError as e: st.error(dt.FEEDBACK_SAVE_ERROR.format(e=e))
    if feedback_stars is not None and latest_run_id not in st.session_state.text_feedback_runs:
        text_feedback = st.text_area(dt.FEEDBACK_TEXT_AREA_LABEL, key=f"text_{latest_run_id}", height=100, placeholder=dt.FEEDBACK_TEXT_AREA_PLACEHOLDER)
        if text_feedback and st.button(dt.FEEDBACK_SUBMIT_BUTTON, key=f"submit_{latest_run_id}"):
            normalized_score = (feedback_stars + 1) / 5.0; agent_client: AgentClient = st.session_state.agent_client
            try:
                await agent_client.acreate_feedback(run_id=latest_run_id, key="human-feedback-with-comment", score=normalized_score, kwargs={"comment": text_feedback})
                st.session_state.text_feedback_runs.add(latest_run_id)
                st.toast(dt.FEEDBACK_SUBMITTED_TOAST, icon=dt.FEEDBACK_STARS_ICON); st.rerun()
            except AgentClientError as e: st.error(dt.FEEDBACK_SAVE_ERROR.format(e=e))

if __name__ == "__main__":
    load_dotenv() 
    asyncio.run(main())
