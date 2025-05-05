import asyncio
import os
import urllib.parse
import uuid
from collections.abc import AsyncGenerator
from uuid import uuid4
import streamlit as st
from dotenv import load_dotenv
from pydantic import ValidationError

from client import AgentClient, AgentClientError
from schema import ChatHistory, ChatMessage
from schema.task_data import TaskData, TaskDataStatus
import json
from pdf_viewer_with_annotations import display_pdf
from rag_system import RAGSystem
from db_manager import DatabaseManager
import display_texts as dt

db_manager = DatabaseManager()
rag_system = RAGSystem()


APP_TITLE = dt.APP_TITLE
APP_ICON = dt.APP_ICON
AI_ICON = dt.AI_ICON
USER_ICON = dt.USER_ICON


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
    if pdf_name:
        display_pdf(document_name=pdf_name, annotations=annotations, debug_viewer=debug_viewer)
    else:
        st.error(dt.PDF_DIALOG_NO_PDF_ERROR)
        
    st.session_state.pdf_to_view = None
    st.session_state.annotations = None
    st.session_state.in_pdf_dialog = False

    if st.button(dt.PDF_DIALOG_CLOSE_BUTTON):
        st.rerun()

async def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        menu_items={},
    )
    
    if st.session_state.pdf_to_view:
        pdf_dialog()

    if "suggested_command" not in st.session_state:
        st.session_state.suggested_command = None
    user_text = None

    if st.get_option("client.toolbarMode") != "minimal":
        st.set_option("client.toolbarMode", "minimal")
        await asyncio.sleep(0.1)
        st.rerun()

    if "agent_client" not in st.session_state:
        load_dotenv()
        agent_url = os.getenv("AGENT_URL")
        if not agent_url:
            host = os.getenv("HOST", "0.0.0.0")
            port = os.getenv("PORT", 8080)
            agent_url = f"http://{host}:{port}"
        try:
            with st.spinner(dt.AGENT_CONNECTING_SPINNER):
                st.session_state.agent_client = AgentClient(base_url=agent_url)
        except AgentClientError as e:
            st.error(dt.AGENT_CONNECTION_ERROR.format(agent_url=agent_url, e=e))
            st.markdown(dt.AGENT_CONNECTION_RETRY_MSG)
            st.stop()
    agent_client: AgentClient = st.session_state.agent_client

    query_thread_id = st.query_params.get("thread_id")
    if query_thread_id and "thread_id" in st.session_state and query_thread_id != st.session_state.thread_id:
        # Thread ID has changed, reload the conversation
        try:
            with st.spinner(dt.CONVERSATION_LOADING_SPINNER):
                messages: ChatHistory = agent_client.get_history(thread_id=query_thread_id).messages
                st.session_state.messages = messages
                st.session_state.thread_id = query_thread_id
                # Also update conversation title
                try:
                    title = agent_client.get_conversation_title(query_thread_id)
                    st.session_state.conversation_title = title
                except:
                    st.session_state.conversation_title = dt.DEFAULT_CONVERSATION_TITLE
        except AgentClientError as e:
            st.error(dt.CONVERSATION_LOAD_ERROR.format(e=e))

    if "thread_id" not in st.session_state:
        thread_id = st.query_params.get("thread_id")
        if not thread_id:
            thread_id = str(uuid.uuid4())
            messages = []
        else:
            try:
                messages: ChatHistory = agent_client.get_history(thread_id=thread_id).messages
            except AgentClientError:
                st.error(dt.CONVERSATION_NOT_FOUND_ERROR)
                messages = []
        st.session_state.messages = messages
        st.session_state.thread_id = thread_id

    # Config options
    with st.sidebar:
        st.header(f"{dt.APP_ICON} {dt.APP_TITLE}")

        st.markdown(dt.SIDEBAR_HEADER)
                
        if st.button(dt.NEW_CONVERSATION_BUTTON, use_container_width=False, icon=":material/add:", type="tertiary", disabled=len(st.session_state.messages) == 0):
            st.query_params.clear()
            st.session_state.messages = []
            st.session_state.conversation_title = dt.DEFAULT_CONVERSATION_TITLE
            st.session_state.thread_id = str(uuid4())
            st.rerun()

        if "conversation_title" not in st.session_state:
            try:
                title = agent_client.get_conversation_title(st.session_state.thread_id)
                st.session_state.conversation_title = title
            except:
                st.session_state.conversation_title = dt.DEFAULT_CONVERSATION_TITLE
                 
        if st.session_state.get("editing_title", False):
            new_title = st.text_input(
                dt.EDIT_TITLE_INPUT_LABEL, 
                value=st.session_state.conversation_title,
                key="new_title_input"
            )
            col1, col2 = st.columns(2)
            with col1:
                if st.button(dt.SAVE_TITLE_BUTTON, key="save_title"):
                    agent_client.set_conversation_title(st.session_state.thread_id, new_title)
                    st.session_state.conversation_title = new_title
                    st.session_state.editing_title = False
                    st.rerun()
            with col2:
                if st.button(dt.CANCEL_TITLE_BUTTON, key="cancel_title"):
                    st.session_state.editing_title = False
                    st.rerun()
        
        try:
            conversations = agent_client.get_conversations(limit=20)
            if conversations:
                st.subheader(dt.RECENT_SUBHEADER)
                for conv in conversations:
                    thread_id = conv["thread_id"]
                    title = conv["title"]
                    updated_at = conv["updated_at"]
                    
                    try:
                        from datetime import datetime
                        updated_date = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
                        date_str = updated_date.strftime("%d/%m/%Y %H:%M")
                    except:
                        date_str = updated_at
                    
                    col1, col2, col3 = st.columns([0.9, 0.05, 0.05])
                    with col1:
                        if st.button(
                            f"{title}",
                            key=f"conv_{thread_id}",
                            help=dt.CONVERSATION_LAST_UPDATED_HELP.format(date_str=date_str),
                            type='tertiary',
                        ):
                            st.query_params["thread_id"] = thread_id
                            st.rerun()
                    with col2:
                        if st.button(":material/edit:", key=f"edit_{thread_id}", help=dt.EDIT_CONVERSATION_TITLE_HELP, type='tertiary'):
                            # Navigate to the conversation and enable title editing mode
                            st.query_params["thread_id"] = thread_id
                            st.session_state.editing_title = True
                            st.rerun()
                    with col3:
                        if st.button(":material/delete:", key=f"delete_{thread_id}", help=dt.DELETE_CONVERSATION_HELP, type='tertiary'):
                            if agent_client.delete_conversation(thread_id):
                                if thread_id == st.session_state.thread_id:
                                    st.query_params.clear()
                                st.rerun()
                
        except Exception as e:
            st.error(dt.LOAD_CONVERSATIONS_ERROR.format(e=e))
        
        st.divider()
        
        # Settings section
        with st.popover(dt.SETTINGS_POPOVER_LABEL, use_container_width=True):
            model_idx = agent_client.info.models.index(agent_client.info.default_model)
            model = st.selectbox(dt.SELECT_LLM_LABEL, options=agent_client.info.models, index=model_idx)
            agent_list = [a.key for a in agent_client.info.agents]
            agent_idx = agent_list.index(agent_client.info.default_agent)
            agent_client.agent = st.selectbox(
                dt.SELECT_AGENT_LABEL,
                options=agent_list,
                index=agent_idx,
            )
            use_streaming = st.toggle(dt.STREAMING_TOGGLE_LABEL, value=True)


        # st.caption(
        #     "Made with :material/favorite: by [Joshua](https://www.linkedin.com/in/joshua-k-carroll/) in Oakland"
        # )

    # Draw existing messages
    messages: list[ChatMessage] = st.session_state.messages

    if len(messages) == 0 and user_text is None:
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


        col1, col2 = st.columns(2, gap="medium")
        
        with col1:
            with st.container():
                if st.button(dt.PROMPT_BUTTON_DB_QUERY, key="btn_db_query", use_container_width=True, icon=":material/description:", type="secondary", disabled=bool(st.session_state.suggested_command)):
                    st.session_state.suggested_command = dt.PROMPT_SUGGESTED_DB_QUERY
                    st.rerun()
        
        with col2:
            with st.container():
                if st.button(dt.PROMPT_BUTTON_DEBUG_PDF, key="btn_debug_pdf", use_container_width=True, icon=":material/bug_report:", disabled=bool(st.session_state.suggested_command)):
                    st.session_state.suggested_command = dt.PROMPT_SUGGESTED_DEBUG_PDF
                    st.rerun()
        col3, col4 = st.columns(2, gap="medium")
        
        with col3:
            with st.container():
                if st.button(dt.PROMPT_BUTTON_CREATE_GRAPH, key="btn_create_graph", use_container_width=True, icon=":material/monitoring:", disabled=bool(st.session_state.suggested_command)):
                    st.session_state.suggested_command = dt.PROMPT_SUGGESTED_CREATE_GRAPH
                    st.rerun()
        
        with col4:
            with st.container():
                if st.button(dt.PROMPT_BUTTON_DOCUMENT_SUMMARY, key="btn_document_summary", use_container_width=True, icon=":material/list:", disabled=bool(st.session_state.suggested_command)):
                    st.session_state.suggested_command = dt.PROMPT_SUGGESTED_DOCUMENT_SUMMARY
                    st.rerun()


    # draw_messages() expects an async iterator over messages
    async def amessage_iter() -> AsyncGenerator[ChatMessage, None]:
        for m in messages:
            yield m

    await draw_messages(amessage_iter())

    if hasattr(st.session_state, "graphs") and st.session_state.graphs:
        for graph_id, plot_data in st.session_state.graphs.items():
            st.plotly_chart(plot_data)

    if user_input := st.chat_input(dt.CHAT_INPUT_PLACEHOLDER, accept_file="multiple", file_type=["pdf"]) or st.session_state.suggested_command:
        if st.session_state.suggested_command:
            user_text = st.session_state.suggested_command
            files = []
            st.session_state.suggested_command = None
        elif user_input:
            user_text = user_input.text
            files = user_input.files
        
        messages.append(ChatMessage(type="human", content=user_text, attached_files=[f.name for f in files]))
        additional_markdown = ""
        if files: 
            if additional_markdown == "":
                additional_markdown = """  
                """                   
            for file in files:
                additional_markdown += dt.FILE_ATTACHED_BADGE.format(file_name=file.name)

        st.chat_message("human", avatar=USER_ICON).write(user_text + additional_markdown)
        
        if files:
            upload_status = st.status(dt.FILE_UPLOADING_STATUS, state="running")
            
            uploaded_file_ids = []
            
            for file in files:
                file_content = file.getvalue()
                file_name = file.name
                file_type = file.type
                
                try:
                    file_id = agent_client.upload_file(
                        file_name=file_name,
                        file_content=file_content,
                        file_type=file_type,
                        thread_id=st.session_state.thread_id
                    )
                    
                    uploaded_file_ids.append(file_id)
                    upload_status.update(label=dt.FILE_UPLOAD_SUCCESS_STATUS.format(file_name=file_name))
                except Exception as e:
                    upload_status.error(dt.FILE_UPLOAD_ERROR_STATUS.format(file_name=file_name, e=e))
            
            upload_status.update(state="complete", label=dt.FILES_UPLOADED_COMPLETE_STATUS.format(count=len(uploaded_file_ids)))                
        try:
            if use_streaming:
                stream = agent_client.astream(
                    message=user_text,
                    model=model,
                    thread_id=st.session_state.thread_id,
                    file_ids=uploaded_file_ids if files else None,
                )
                await draw_messages(stream, is_new=True, agent_client=agent_client)
            else:
                response = await agent_client.ainvoke(
                    message=user_text,
                    model=model,
                    thread_id=st.session_state.thread_id,
                    file_ids=uploaded_file_ids if files else None,
                )
                messages.append(response)
                st.chat_message("ai", avatar=AI_ICON).write(response.content)

            if len(messages) > 1 and st.session_state.conversation_title == dt.DEFAULT_CONVERSATION_TITLE:
                try:
                    title_prompt = dt.TITLE_GENERATION_PROMPT.format(user_text=user_text)
                    title_response = await agent_client.ainvoke(
                        message=title_prompt,
                        model=model
                    )
                    generated_title = title_response.content.strip().strip('"\'')
                    
                    await agent_client.aset_conversation_title(st.session_state.thread_id, generated_title)
                    st.session_state.conversation_title = generated_title
                except Exception as e:
                    pass
                    
                    st.rerun()
        except AgentClientError as e:
            st.error(dt.RESPONSE_GENERATION_ERROR.format(e=e))
            st.stop()
    if len(messages) > 0 and st.session_state.last_message:
        with st.session_state.last_message:
            await handle_feedback()


async def draw_messages(
    messages_agen: AsyncGenerator[ChatMessage | str, None],
    is_new: bool = False,
    agent_client: AgentClient = None,
) -> None:
    """
    Draws a set of chat messages - either replaying existing messages
    or streaming new ones.

    This function has additional logic to handle streaming tokens and tool calls.
    - Use a placeholder container to render streaming tokens as they arrive.
    - Use a status container to render tool calls. Track the tool inputs and outputs
      and update the status container accordingly.

    The function also needs to track the last message container in session state
    since later messages can draw to the same container. This is also used for
    drawing the feedback widget in the latest chat message.

    Args:
        messages_aiter: An async iterator over messages to draw.
        is_new: Whether the messages are new or not.
    """

    if "pdf_documents" not in st.session_state:
        st.session_state.pdf_documents = {}

    # Keep track of the last message container
    last_message_type = None
    st.session_state.last_message = None

    # Placeholder for intermediate streaming tokens
    streaming_content = ""
    streaming_placeholder = None

    # Iterate over the messages and draw them
    while msg := await anext(messages_agen, None):
        # str message represents an intermediate token being streamed
        if isinstance(msg, str):
            # If placeholder is empty, this is the first token of a new message
            # being streamed. We need to do setup.
            if not streaming_placeholder:
                if last_message_type != "ai":
                    last_message_type = "ai"
                    st.session_state.last_message = st.chat_message("ai", avatar=AI_ICON)
                with st.session_state.last_message:
                    streaming_placeholder = st.empty()

            streaming_content += msg
            streaming_placeholder.write(streaming_content)
            continue
        if not isinstance(msg, ChatMessage):
            st.error(dt.UNEXPECTED_MESSAGE_TYPE_ERROR.format(msg_type=type(msg)))
            st.write(msg)
            st.stop()

        match msg.type:
            # A message from the user, the easiest case
            case "human":
                last_message_type = "human"
                additional_markdown = ""
                if hasattr(msg, 'attached_files') and msg.attached_files: 
                    if additional_markdown == "":
                        additional_markdown = """  
                        """                   
                    for file in msg.attached_files:
                        additional_markdown += dt.FILE_ATTACHED_BADGE.format(file_name=file)

                st.chat_message("human", avatar=USER_ICON).write(msg.content + additional_markdown)

            # A message from the agent is the most complex case, since we need to
            # handle streaming tokens and tool calls.
            case "ai":
                # If we're rendering new messages, store the message in session state
                if is_new:
                    st.session_state.messages.append(msg)

                # If the last message type was not AI, create a new chat message
                if last_message_type != "ai":
                    last_message_type = "ai"
                    st.session_state.last_message = st.chat_message("ai", avatar=AI_ICON)

                with st.session_state.last_message:
                    # If the message has content, write it out.
                    # Reset the streaming variables to prepare for the next message.
                    if msg.content:
                        if streaming_placeholder:
                            streaming_placeholder.write(msg.content)
                            streaming_content = ""
                            streaming_placeholder = None
                        else:
                            st.write(msg.content)

                    if msg.tool_calls:
                        # Create a status container for each tool call and store the
                        # status container by ID to ensure results are mapped to the
                        # correct status container.
                        call_results = {}
                        tool_names = {}  
                        for tool_call in msg.tool_calls:
                            status = st.status(
                                dt.TOOL_CALL_STATUS.format(tool_name=tool_call["name"]),
                                state="running" if is_new else "complete",
                            )
                            call_results[tool_call["id"]] = status
                            tool_names[tool_call["id"]] = tool_call["name"]
                            status.write(dt.TOOL_CALL_INPUT_LABEL)
                            status.write(tool_call["args"])
                        for idx in range(len(call_results)):
                            tool_result: ChatMessage = await anext(messages_agen)
                            tool_name = tool_names.get(tool_result.tool_call_id)

                            if tool_result.type != "tool":
                                st.error(dt.UNEXPECTED_CHATMESSAGE_TYPE_ERROR.format(msg_type=tool_result.type))
                                st.write(tool_result)
                                st.stop()

                            # Record the message if it's new, and update the correct
                            # status container with the result
                            if is_new:
                                st.session_state.messages.append(tool_result)
                            
                            if tool_result.tool_call_id:
                                status = call_results[tool_result.tool_call_id]
                            
                            if tool_name == "Graph_Viewer" and agent_client:
                                try:
                                    graph_id = tool_result.content
                                    status.write(dt.GRAPH_RETRIEVAL_STATUS.format(graph_id=graph_id))
                                    
                                    # Call the retrieve_graph function to get the graph data
                                    graph_data = agent_client.retrieve_graph(graph_id)
                                    
                                    if graph_data:
                                        try:
                                            plot_data = json.loads(graph_data)
                                            status.write(dt.GRAPH_RETRIEVED_SUCCESSFULLY_FR)                                            
                                            st.session_state.graphs[graph_id] = plot_data                                            
                                        except json.JSONDecodeError:
                                            # If not JSON, display as text
                                            status.write(dt.GRAPH_NON_JSON_DATA)
                                            st.code(graph_data)
                                    else:
                                        status.write(dt.GRAPH_NO_DATA_RETURNED)
                                except Exception as e:
                                    status.error(dt.GRAPH_RETRIEVAL_ERROR.format(e=e))
                            elif tool_name == "PDF_Viewer":
                                try:
                                    tool_output = json.loads(tool_result.content)
                                    if tool_output.get('error'):
                                        continue
                                    pdf_name = tool_output['pdf_file']
                                    block_indices = tool_output['block_indices']
                                    debug_viewer = tool_output.get('debug', False)
                                    if tool_output.get('debug', False):
                                        annotations = rag_system.debug_blocks(pdf_file=pdf_name)
                                        print(f"Debugging blocks for {pdf_name}: {len(annotations)} blocks")
                                    else:
                                        annotations = rag_system.get_annotations_by_indices(
                                            pdf_file=pdf_name,
                                            block_indices=block_indices,
                                        )            
                                    st.session_state.pdf_documents[pdf_name] = annotations                                    
                                    if st.button(dt.VIEW_PDF_BUTTON.format(pdf_name=pdf_name), key=f"pdf_button_{tool_result.tool_call_id}"):
                                        view_pdf(pdf_name, annotations, debug_viewer=debug_viewer)
                                        
                                    status.update(state="complete", label=dt.PDF_READY_STATUS.format(pdf_name=pdf_name))
                                except Exception as e:
                                    status.error(dt.PDF_PROCESSING_ERROR.format(e=e))
                                    st.write(dt.RAW_OUTPUT_LABEL.format(content=tool_result.content))                                  
                            
                            with status as status:
                            # Update the status
                                st.write(dt.TOOL_CALL_OUTPUT_LABEL)
                                json_response = None
                                try:
                                    json_response = json.loads(tool_result.content)
                                except:
                                    pass 
                                if json_response:
                                    st.json(json_response)
                                else:
                                    st.write(tool_result.content)
                                status.update(state="complete")
                            

            case "custom":
                # CustomData example used by the bg-task-agent
                # See:
                # - src/agents/utils.py CustomData
                # - src/agents/bg_task_agent/task.py
                try:
                    task_data: TaskData = TaskData.model_validate(msg.custom_data)
                except ValidationError:
                    st.error(dt.UNEXPECTED_CUSTOMDATA_ERROR)
                    st.write(msg.custom_data)
                    st.stop()

                if is_new:
                    st.session_state.messages.append(msg)

                if last_message_type != "task":
                    last_message_type = "task"
                    st.session_state.last_message = st.chat_message(
                        name="task", avatar=":material/manufacturing:"
                    )
                    with st.session_state.last_message:
                        status = TaskDataStatus()

                status.add_and_draw_task_data(task_data)

            # In case of an unexpected message type, log an error and stop
            case _:
                st.error(dt.UNEXPECTED_CHATMESSAGE_TYPE_ERROR.format(msg_type=msg.type))
                st.write(msg)
                st.stop()


async def handle_feedback() -> None:
    """Draws a feedback widget and records feedback from the user."""

    # Keep track of last star feedback sent to avoid duplicates
    if "last_star_feedback" not in st.session_state:
        st.session_state.last_star_feedback = (None, None)  # (run_id, stars)
    
    # Keep track of runs with text feedback submitted
    if "text_feedback_runs" not in st.session_state:
        st.session_state.text_feedback_runs = set()
    
    latest_run_id = st.session_state.messages[-1].run_id
    feedback_stars = st.feedback(dt.FEEDBACK_STARS_KEY, key=latest_run_id)
    
    # Auto-submit star feedback when it changes (original behavior)
    if feedback_stars is not None and (latest_run_id, feedback_stars) != st.session_state.last_star_feedback:
        # Normalize the feedback value (an index) to a score between 0 and 1
        normalized_score = (feedback_stars + 1) / 5.0
        
        agent_client: AgentClient = st.session_state.agent_client
        try:
            await agent_client.acreate_feedback(
                run_id=latest_run_id,
                key="human-feedback-stars",
                score=normalized_score,
                kwargs={"comment": dt.FEEDBACK_HUMAN_INLINE_COMMENT},
            )
        except AgentClientError as e:
            st.error(dt.FEEDBACK_SAVE_ERROR.format(e=e))
            st.stop()
        
        st.session_state.last_star_feedback = (latest_run_id, feedback_stars)
        st.toast(dt.FEEDBACK_SAVED_TOAST, icon=dt.FEEDBACK_STARS_ICON)
    
    # Allow text feedback submission if stars have been selected and text feedback hasn't been submitted yet
    if feedback_stars is not None and latest_run_id not in st.session_state.text_feedback_runs:
        # Text input field with submit button
        text_feedback = st.text_area(
            dt.FEEDBACK_TEXT_AREA_LABEL,
            key=f"text_{latest_run_id}",
            height=100,
            placeholder=dt.FEEDBACK_TEXT_AREA_PLACEHOLDER
        )
        
        if text_feedback:  # Only show submit button if there's text
            if st.button(dt.FEEDBACK_SUBMIT_BUTTON, key=f"submit_{latest_run_id}"):
                # Normalize the star rating again for consistency
                normalized_score = (feedback_stars + 1) / 5.0
                
                # Submit the text feedback
                agent_client: AgentClient = st.session_state.agent_client
                try:
                    await agent_client.acreate_feedback(
                        run_id=latest_run_id,
                        key="human-feedback-with-comment",
                        score=normalized_score,
                        kwargs={"comment": text_feedback},
                    )
                except AgentClientError as e:
                    st.error(dt.FEEDBACK_SAVE_ERROR.format(e=e))
                    st.stop()
                
                # Mark this run as having received text feedback
                st.session_state.text_feedback_runs.add(latest_run_id)
                st.toast(dt.FEEDBACK_SUBMITTED_TOAST, icon=dt.FEEDBACK_STARS_ICON)
                st.rerun()  # Rerun to hide the input after submission

if __name__ == "__main__":
    asyncio.run(main())
