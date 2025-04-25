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

db_manager = DatabaseManager()
rag_system = RAGSystem()

APP_TITLE = "PolyRAG"
APP_ICON = ":material/experiment:"
AI_ICON = ":material/flare:"
USER_ICON = ":material/person:"


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

dialog_title = "Document" if not st.session_state.debug_viewer else "/debug" + ' ' + str(st.session_state.pdf_to_view)
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
        st.error("No PDF selected for viewing.")
        
    st.session_state.pdf_to_view = None
    st.session_state.annotations = None
    st.session_state.in_pdf_dialog = False

    if st.button("Close"):
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
            with st.spinner("Connecting to agent service..."):
                st.session_state.agent_client = AgentClient(base_url=agent_url)
        except AgentClientError as e:
            st.error(f"Error connecting to agent service at {agent_url}: {e}")
            st.markdown("The service might be booting up. Try again in a few seconds.")
            st.stop()
    agent_client: AgentClient = st.session_state.agent_client

    query_thread_id = st.query_params.get("thread_id")
    if query_thread_id and "thread_id" in st.session_state and query_thread_id != st.session_state.thread_id:
        # Thread ID has changed, reload the conversation
        try:
            with st.spinner("Loading conversation..."):
                messages: ChatHistory = agent_client.get_history(thread_id=query_thread_id).messages
                st.session_state.messages = messages
                st.session_state.thread_id = query_thread_id
                # Also update conversation title
                try:
                    title = agent_client.get_conversation_title(query_thread_id)
                    st.session_state.conversation_title = title
                except:
                    st.session_state.conversation_title = "Conversation"
        except AgentClientError as e:
            st.error(f"Error loading conversation: {e}")

    if "thread_id" not in st.session_state:
        thread_id = st.query_params.get("thread_id")
        if not thread_id:
            thread_id = str(uuid.uuid4())
            messages = []
        else:
            try:
                messages: ChatHistory = agent_client.get_history(thread_id=thread_id).messages
            except AgentClientError:
                st.error("No message history found for this Thread ID.")
                messages = []
        st.session_state.messages = messages
        st.session_state.thread_id = thread_id

    # Config options
    with st.sidebar:
        st.header(f"{APP_ICON} {APP_TITLE}")

        ""
        """Platform for database and PDF document RAG, with visualization capabilities.
        Powered by LangGraph, PostgreSQL, Plotly and PDF processors. Built with FastAPI and Streamlit."""
                
        if st.button("**New conversation**", use_container_width=False, icon=":material/add:", type="tertiary", disabled=len(st.session_state.messages) == 0):
            st.query_params.clear()
            st.session_state.messages = []
            st.session_state.conversation_title = "New conversation"
            st.session_state.thread_id = str(uuid4())
            st.rerun()

        if "conversation_title" not in st.session_state:
            try:
                title = agent_client.get_conversation_title(st.session_state.thread_id)
                st.session_state.conversation_title = title
            except:
                st.session_state.conversation_title = "New conversation"
                 
        if st.session_state.get("editing_title", False):
            new_title = st.text_input(
                "Conversation title", 
                value=st.session_state.conversation_title,
                key="new_title_input"
            )
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Save", key="save_title"):
                    agent_client.set_conversation_title(st.session_state.thread_id, new_title)
                    st.session_state.conversation_title = new_title
                    st.session_state.editing_title = False
                    st.rerun()
            with col2:
                if st.button("Cancel", key="cancel_title"):
                    st.session_state.editing_title = False
                    st.rerun()
        
        try:
            conversations = agent_client.get_conversations(limit=20)
            if conversations:
                st.subheader("Recent")
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
                            help=f"Last updated: {date_str}",
                            type='tertiary',
                        ):
                            st.query_params["thread_id"] = thread_id
                            st.rerun()
                    with col2:
                        if st.button(":material/edit:", key=f"edit_{thread_id}", help="Edit conversation title", type='tertiary'):
                            # Navigate to the conversation and enable title editing mode
                            st.query_params["thread_id"] = thread_id
                            st.session_state.editing_title = True
                            st.rerun()
                    with col3:
                        if st.button(":material/delete:", key=f"delete_{thread_id}", help="Delete this conversation", type='tertiary'):
                            if agent_client.delete_conversation(thread_id):
                                if thread_id == st.session_state.thread_id:
                                    st.query_params.clear()
                                st.rerun()
                
        except Exception as e:
            st.error(f"Error loading conversations: {e}")
        
        st.divider()
        
        # Settings section
        with st.popover(":material/settings: Settings", use_container_width=True):
            model_idx = agent_client.info.models.index(agent_client.info.default_model)
            model = st.selectbox("LLM to use", options=agent_client.info.models, index=model_idx)
            agent_list = [a.key for a in agent_client.info.agents]
            agent_idx = agent_list.index(agent_client.info.default_agent)
            agent_client.agent = st.selectbox(
                "Agent to use",
                options=agent_list,
                index=agent_idx,
            )
            use_streaming = st.toggle("Stream results", value=True)

        with st.popover(":material/policy: Privacy", use_container_width=True):
            st.write(
                "Prompts, responses and feedback in this app are anonymously recorded and saved to LangSmith for product evaluation and improvement purposes only."
            )

        "[View the source code](https://github.com/QuentinFuxa/PolyRAG)"
        st.caption(
            "Made with :material/favorite: by [Joshua](https://www.linkedin.com/in/joshua-k-carroll/) in Oakland. Augmented by [Quentin](https://www.linkedin.com/in/quentin-fuxa/) in Paris :material/cell_tower: .  "
        )

    # Draw existing messages
    messages: list[ChatMessage] = st.session_state.messages

    if len(messages) == 0 and user_text is None:
        match agent_client.agent:
            case "chatbot":
                WELCOME = "Hello! I'm a simple chatbot. Ask me anything!"
            case "interrupt-agent":
                WELCOME = "Hello! I'm an interrupt agent. Tell me your birthday and I will predict your personality!"
            case "research-assistant":
                WELCOME = "Hello! I'm an AI-powered research assistant with web search and a calculator. Ask me anything!"
            case "pg_rag_assistant":
                WELCOME = "Hello! I'm an AI-powered research assistant with Postgres database access, PDF analysis, and visualization tools. I can search documents, run SQL queries, and save our conversations for later reference. Ask me anything!"
            case _:
                WELCOME = "Hello! I'm an AI agent. Ask me anything!"
        with st.chat_message("ai", avatar=AI_ICON):
            st.write(WELCOME)


        col1, col2 = st.columns(2, gap="medium")
        
        with col1:
            with st.container():
                if st.button("How many preprints in the q-bio.PE category since the beginning of the year?", key="btn_db_query", use_container_width=True, disabled=bool(st.session_state.suggested_command)):
                    st.session_state.suggested_command = "How many preprints in the q-bio.PE category since the beginning of the year?"
                    st.rerun()
        
        with col2:
            with st.container():
                if st.button("Plot a graph of anemia distribution by condition and age for the year 2012", key="btn_create_graph", use_container_width=True, disabled=bool(st.session_state.suggested_command)):
                    st.session_state.suggested_command = "Plot a graph of anemia distribution by condition and age for the year 2012"
                    st.rerun()
        col3, col4 = st.columns(2, gap="medium")
        
        with col3:
            with st.container():
                if st.button("/debug 2504.12888v1", key="btn_debug_pdf", use_container_width=True, disabled=bool(st.session_state.suggested_command)):
                    st.session_state.suggested_command = "/debug 2504.12888v1"
                    st.rerun()
        
        with col4:
            with st.container():
                if st.button("Summarize the most recent preprint", key="btn_document_summary", use_container_width=True, disabled=bool(st.session_state.suggested_command)):
                    st.session_state.suggested_command = "Summarize the most recent preprint"
                    st.rerun()


    # draw_messages() expects an async iterator over messages
    async def amessage_iter() -> AsyncGenerator[ChatMessage, None]:
        for m in messages:
            yield m

    await draw_messages(amessage_iter())

    if hasattr(st.session_state, "graphs") and st.session_state.graphs:
        for graph_id, plot_data in st.session_state.graphs.items():
            st.plotly_chart(plot_data)


    if user_input := st.chat_input('Your message', accept_file="multiple", file_type=["pdf"]) or st.session_state.suggested_command:
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
                additional_markdown += f""":violet-badge[:material/description: {file.name}] """

        st.chat_message("human", avatar=USER_ICON).write(user_text + additional_markdown)
        
        if files:
            upload_status = st.status("File being uploaded...", state="running")
            
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
                    upload_status.update(label=f"File {file_name} uploaded successfully!")
                except Exception as e:
                    upload_status.error(f"Error uploading {file_name}: {e}")
            
            upload_status.update(state="complete", label=f"{len(uploaded_file_ids)} files uploaded")                
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

            if len(messages) > 1 and st.session_state.conversation_title == "New conversation":
                try:
                    title_prompt = f"Generate a short title (< 50 chars) summarizing this conversation. First user message: {user_text}"
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
            st.error(f"Error generating response: {e}")
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
            st.error(f"Unexpected message type: {type(msg)}")
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
                        additional_markdown += f""":violet-badge[:material/description: {file}] """

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
                                f"""Tool Call: {tool_call["name"]}""",
                                state="running" if is_new else "complete",
                            )
                            call_results[tool_call["id"]] = status
                            tool_names[tool_call["id"]] = tool_call["name"]
                            status.write("Input:")
                            status.write(tool_call["args"])
                        for idx in range(len(call_results)):
                            tool_result: ChatMessage = await anext(messages_agen)
                            tool_name = tool_names.get(tool_result.tool_call_id)

                            if tool_result.type != "tool":
                                st.error(f"Unexpected ChatMessage type: {tool_result.type}")
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
                                    status.write(f"Retrieving graph with ID: {graph_id}")                                    
                                    graph_data = agent_client.retrieve_graph(graph_id)
                                    
                                    if graph_data:
                                        try:
                                            plot_data = json.loads(graph_data)
                                            status.write("Graph retrieved successfully")                                            
                                            st.session_state.graphs[graph_id] = plot_data                                            
                                        except json.JSONDecodeError:
                                            status.write("Retrieved non-JSON graph data")
                                            st.code(graph_data)
                                    else:
                                        status.write("No graph data returned")
                                except Exception as e:
                                    status.error(f"Error retrieving graph: {e}")
                            elif tool_name == "PDF_Viewer":
                                try:
                                    tool_output = json.loads(tool_result.content)
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
                                    if st.button(f"View PDF: {pdf_name}", key=f"pdf_button_{tool_result.tool_call_id}"):
                                        view_pdf(pdf_name, annotations, debug_viewer=debug_viewer)
                                        
                                    status.update(state="complete", label=f"PDF ready: {pdf_name}")
                                except Exception as e:
                                    status.error(f"Error processing PDF: {e}")
                                    st.write(f"Raw output: {tool_result.content}")                                  
                            
                            # Update the status
                            status.write("Output:")
                            status.write(tool_result.content)
                            status.update(state="complete")
                            

            case "custom":
                # CustomData example used by the bg-task-agent
                # See:
                # - src/agents/utils.py CustomData
                # - src/agents/bg_task_agent/task.py
                try:
                    task_data: TaskData = TaskData.model_validate(msg.custom_data)
                except ValidationError:
                    st.error("Unexpected CustomData message received from agent")
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
                st.error(f"Unexpected ChatMessage type: {msg.type}")
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
    feedback_stars = st.feedback("stars", key=latest_run_id)
    
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
                kwargs={"comment": "In-line human feedback"},
            )
        except AgentClientError as e:
            st.error(f"Error recording feedback: {e}")
            st.stop()
        
        st.session_state.last_star_feedback = (latest_run_id, feedback_stars)
        st.toast("Feedback recorded", icon=":material/reviews:")
    
    # Allow text feedback submission if stars have been selected and text feedback hasn't been submitted yet
    if feedback_stars is not None and latest_run_id not in st.session_state.text_feedback_runs:
        # Text input field with submit button
        text_feedback = st.text_area(
            "Additional feedback (optional)",
            key=f"text_{latest_run_id}",
            height=100,
            placeholder="Please provide any additional comments or suggestions..."
        )
        
        if text_feedback:  # Only show submit button if there's text
            if st.button("Submit Feedback", key=f"submit_{latest_run_id}"):
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
                    st.error(f"Error recording text feedback: {e}")
                    st.stop()
                
                # Mark this run as having received text feedback
                st.session_state.text_feedback_runs.add(latest_run_id)
                st.toast("Detailed feedback submitted. Thank you!", icon=":material/reviews:")
                st.rerun()  # Rerun to hide the input after submission

if __name__ == "__main__":
    asyncio.run(main())
