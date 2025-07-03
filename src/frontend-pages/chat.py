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
from display_texts import dt
from auth_helpers import ensure_authenticated

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

def show_user_modal():
    st.session_state["show_user_modal"] = True

async def main() -> None:

    custom_css = """
    <style>
        div[class*=\"st-key-conv_\"] {
            white-space: nowrap !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
            display: block !important;
            width: 100% !important;
            text-align: left !important;
        }
        div.stVerticalBlock[class*="st-key-pdf_buttons_container"] {
            display: flex !important;
            flex-direction: row !important; /* Explicitly set direction to row */
            flex-wrap: wrap !important;
            gap: 0px 20px; !important;
            padding: 0px !important;
            margin: 0px !important;
        }
        div[class*=\"st-key-pdf_button_call\"] button {
            color: rgb(26 94 213) !important;
            background-color: transparent !important;
            border: none !important;
            padding: 0 !important;
            cursor: pointer !important;
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

    if not ensure_authenticated():
        st.stop() # ensure_authenticated calls st.stop() if login fails, but as a safeguard.
    
    current_user_id: UUID_TYPE = st.session_state.current_user_id        
    if "current_user_id" not in st.session_state:
        st.session_state.current_user_id = current_user_id
    
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

        model_idx = agent_client.info.models.index(agent_client.info.default_model)
        model = 'gpt-4o'

        # st.markdown(dt.SIDEBAR_HEADER)
        if st.button(dt.NEW_CONVERSATION_BUTTON, use_container_width=False, icon=":material/add:", type="tertiary", disabled=False): #len(st.session_state.messages) == 0):
            st.query_params.clear()
            st.session_state.messages = []
            st.session_state.conversation_title = dt.DEFAULT_CONVERSATION_TITLE
            st.session_state.thread_id = str(uuid4())
            st.rerun()

        if "conversation_title" not in st.session_state:
            try:
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
                        if st.button(f"{title}", key=f"conv_{thread_id_conv}", help=f"{dt.CONVERSATION_LAST_UPDATED_HELP.format(date_str=date_str)}", type='tertiary'):
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
        
        # st.divider()
        # with st.popover(dt.SETTINGS_POPOVER_LABEL, use_container_width=True):
        #     model_idx = agent_client.info.models.index(agent_client.info.default_model)
        #     model = st.selectbox(dt.SELECT_LLM_LABEL, options=agent_client.info.models, index=model_idx)
        #     agent_list = [a.key for a in agent_client.info.agents]
        #     agent_idx = agent_list.index(agent_client.info.default_agent)
        #     agent_client.agent = st.selectbox(dt.SELECT_AGENT_LABEL, options=agent_list, index=agent_idx)
        #     use_streaming = st.toggle(dt.STREAMING_TOGGLE_LABEL, value=True)
        st.caption(dt.CAPTION)
        use_streaming = True
        agent_client.agent = "pg_rag_assistant" # Default agent

    messages: list[ChatMessage] = st.session_state.messages
    if len(messages) == 0 and user_text is None:

        with st.container(key="welcome-msg"):
            
            match agent_client.agent:
                case "chatbot":
                    WELCOME = dt.WELCOME_CHATBOT
                case "interrupt-agent":
                    WELCOME = dt.WELCOME_INTERRUPT
                case "research-assistant":
                    WELCOME = dt.WELCOME_RESEARCH
                case "pg_rag_assistant":
                    WELCOME = dt.WELCOME_PG_RAG
                case _:
                    WELCOME = dt.WELCOME_DEFAULT
                 
            with st.chat_message("ai", avatar=AI_ICON):
                st.write(WELCOME)

            if hasattr(dt, 'EXAMPLE_PROMPTS') and dt.EXAMPLE_PROMPTS:
                num_prompts = len(dt.EXAMPLE_PROMPTS)
                for i in range(0, num_prompts, 2):
                    cols = st.columns(2, gap="medium")
                    with cols[0]:
                        prompt_data = dt.EXAMPLE_PROMPTS[i]
                        if st.button(
                            prompt_data["button_text"],
                            key=prompt_data["key"],
                            use_container_width=True,
                            icon=prompt_data["icon"],
                            type="secondary", 
                            disabled=bool(st.session_state.suggested_command)
                        ):
                            st.session_state.suggested_command = prompt_data["suggested_command_text"]
                            hide_welcome()
                            st.rerun()
                    
                    if i + 1 < num_prompts:
                        with cols[1]:
                            prompt_data_next = dt.EXAMPLE_PROMPTS[i+1]
                            if st.button(
                                prompt_data_next["button_text"],
                                key=prompt_data_next["key"],
                                use_container_width=True,
                                icon=prompt_data_next["icon"],
                                type="secondary",
                                disabled=bool(st.session_state.suggested_command)
                            ):
                                st.session_state.suggested_command = prompt_data_next["suggested_command_text"]
                                hide_welcome()
                                st.rerun()
            else:
                st.warning("Example prompts are not configured or dt.EXAMPLE_PROMPTS is missing.")
            with st.container(border=False):
                st.markdown(f"""
                <div style="padding: 10px; border-radius: 10px; text-size: 0.6rem; background-color: #f8f9fa; border: 1px solid #dee2e6;">
                    {dt.WARNING_MESSAGE}
                </div>
                """, unsafe_allow_html=True)


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
        pdf_buttons_to_create = [] # Initialize here to ensure it's always defined
        with status:
            try:
                pdf_results_list = json.loads(tool_result.content)
                if not isinstance(pdf_results_list, list):
                    st.error("PDF_Viewer tool returned an unexpected format. Expected a list of PDF results.")
                    status.update(state="complete", label="PDF Viewer Error")
                    # Must return here as pdf_buttons_to_create would be undefined for the loop after 'with status'
                    return

                status.update(state="running", label=f"Processing {len(pdf_results_list)} PDF(s)...")
                
                all_processed_successfully = True
                # pdf_buttons_to_create is already initialized outside

                for pdf_entry_idx, pdf_entry in enumerate(pdf_results_list): # Added enumerate for unique key generation
                    if pdf_entry.get('error'):
                        original_pdf_file_name = pdf_entry.get('original_request', {}).get('pdf_file', f"request_{pdf_entry_idx}")
                        st.error(f"Error for PDF '{original_pdf_file_name}': {pdf_entry['error']}")
                        all_processed_successfully = False
                        continue # Skip to next PDF entry

                    pdf_name = pdf_entry.get('pdf_file')
                    block_indices = pdf_entry.get('block_indices')
                    # Use the global debug flag from the tool call if present, else default to False
                    # This assumes the 'debug' flag from the tool call is passed down or is accessible
                    # For now, let's assume the 'debug' in pdf_entry is the one to use per PDF.
                    debug_viewer = pdf_entry.get('debug', False) 

                    if not pdf_name:
                        st.error(f"PDF entry at index {pdf_entry_idx} is missing 'pdf_file'.")
                        all_processed_successfully = False
                        continue

                    st.write(dt.PDF_VIEWER_OUTPUT_LABEL.format(pdf_name=pdf_name))
                    annotations = []
                    if agent_client:
                        try:
                            if not debug_viewer:
                                # Pass user_id if available and required by the backend
                                annotations = await agent_client.aget_annotations(pdf_file=pdf_name, block_indices=block_indices, user_id=user_id) 
                            else:
                                annotations = await agent_client.adebug_pdf_blocks(pdf_file=pdf_name, user_id=user_id)
                            st.session_state.pdf_documents[pdf_name] = annotations
                            pdf_buttons_to_create.append({
                                "name": pdf_name,
                                "annotations": annotations,
                                "debug": debug_viewer,
                                "tool_call_id": tool_result.tool_call_id,
                                "unique_suffix": f"{pdf_entry_idx}_{pdf_name.replace(' ','_')}" # for unique button key
                            })
                        except Exception as e:
                            st.error(f"Error fetching annotations for '{pdf_name}': {e}")
                            all_processed_successfully = False
                
                if all_processed_successfully and pdf_buttons_to_create:
                    status.update(state="complete", label=f"{len(pdf_buttons_to_create)} PDF(s) ready for viewing.")
                elif not pdf_buttons_to_create:
                     status.update(state="complete", label="No PDFs to display or all had errors.")
                else:
                    status.update(state="complete", label="Some PDFs processed with errors.")

            except json.JSONDecodeError:
                st.error(f"PDF_Viewer tool returned invalid JSON: {tool_result.content}")
                status.update(state="complete", label="PDF Viewer JSON Error")
            except Exception as e:
                st.error(f"Error processing PDF_Viewer results: {e}")
                st.write(f"Raw output: {tool_result.content}")
                status.update(state="complete", label="PDF Viewer Processing Error")
        
        with st.container(key=f"sources_{tool_result.tool_call_id}", border=True):
            st.button('**Sources dans les Lettres de suite :**', type='tertiary', icon=":material/info:", help='Cliquez sur les documents pour visualiser les zones qui ont permis de répondre à la question.',)
            with st.container(key=f"pdf_buttons_container_{tool_result.tool_call_id}"):
                for btn_data in pdf_buttons_to_create:
                    button_key = f"pdf_button_{btn_data['tool_call_id']}_{btn_data['unique_suffix']}"
                    if st.button(btn_data['name'], key=button_key, type="tertiary", icon=':material/article:'):
                        view_pdf(btn_data['name'], btn_data['annotations'], debug_viewer=btn_data['debug'])
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
        conversation_id = st.session_state.thread_id
        commented_message_text = st.session_state.messages[-1].content if st.session_state.messages else None
        try:
            await agent_client.acreate_feedback(
                run_id=latest_run_id, 
                key="human-feedback-stars", 
                score=normalized_score, 
                conversation_id=conversation_id,
                commented_message_text=commented_message_text,
                kwargs={"comment": dt.FEEDBACK_HUMAN_INLINE_COMMENT}
            )
            st.session_state.last_star_feedback = (latest_run_id, feedback_stars)
            st.toast(dt.FEEDBACK_SAVED_TOAST, icon=dt.FEEDBACK_STARS_ICON)
        except AgentClientError as e: st.error(dt.FEEDBACK_SAVE_ERROR.format(e=e))
    if feedback_stars is not None and latest_run_id not in st.session_state.text_feedback_runs:
        text_feedback = st.text_area(dt.FEEDBACK_TEXT_AREA_LABEL, key=f"text_{latest_run_id}", height=100, placeholder=dt.FEEDBACK_TEXT_AREA_PLACEHOLDER)
        if text_feedback and st.button(dt.FEEDBACK_SUBMIT_BUTTON, key=f"submit_{latest_run_id}"):
            normalized_score = (feedback_stars + 1) / 5.0; agent_client: AgentClient = st.session_state.agent_client
            conversation_id = st.session_state.thread_id
            commented_message_text = st.session_state.messages[-1].content if st.session_state.messages else None
            try:
                await agent_client.acreate_feedback(
                    run_id=latest_run_id, 
                    key="human-feedback-with-comment", 
                    score=normalized_score, 
                    conversation_id=conversation_id,
                    commented_message_text=commented_message_text,
                    kwargs={"comment": text_feedback}
                )
                st.session_state.text_feedback_runs.add(latest_run_id)
                st.toast(dt.FEEDBACK_SUBMITTED_TOAST, icon=dt.FEEDBACK_STARS_ICON); st.rerun()
            except AgentClientError as e: st.error(dt.FEEDBACK_SAVE_ERROR.format(e=e))

if __name__ == "__main__":
    load_dotenv() 
    asyncio.run(main())
