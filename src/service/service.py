import os
import json
import logging
import warnings
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated, Any, Dict, List, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langchain_core._api import LangChainBetaWarning
from langchain_core.messages import AIMessage, AIMessageChunk, AnyMessage, HumanMessage, ToolMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, Interrupt
from langsmith import Client as LangsmithClient

from agents import DEFAULT_AGENT, get_agent, get_all_agent_info
from agents._graph_store import GraphStore
from core import settings
from memory import initialize_database
from schema import (
    ChatHistory,
    ChatHistoryInput,
    ChatMessage,
    Feedback,
    FeedbackResponse,
    ServiceMetadata,
    StreamInput,
    UserInput,
)
from service.utils import (
    convert_message_content_to_string,
    langchain_to_chat_message,
    remove_tool_calls,
)
from db_manager import DatabaseManager

from fastapi import File, UploadFile
from typing import Optional
from rag_system import RAGSystem

db_manager = DatabaseManager()
rag_system = RAGSystem()

warnings.filterwarnings("ignore", category=LangChainBetaWarning)
logger = logging.getLogger(__name__)


def verify_bearer(
    http_auth: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(HTTPBearer(description="Please provide AUTH_SECRET api key.", auto_error=False)),
    ],
) -> None:
    if not settings.AUTH_SECRET:
        return
    auth_secret = settings.AUTH_SECRET.get_secret_value()
    if not http_auth or http_auth.credentials != auth_secret:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Configurable lifespan that initializes the appropriate database checkpointer based on settings.
    """
    try:
        async with initialize_database() as saver:
            await saver.setup()
            agents = get_all_agent_info()
            for a in agents:
                agent = get_agent(a.key)
                agent.checkpointer = saver
            yield
    except Exception as e:
        logger.error(f"Error during database initialization: {e}")
        raise


app = FastAPI(lifespan=lifespan)
router = APIRouter(dependencies=[Depends(verify_bearer)])


@router.get("/info")
async def info() -> ServiceMetadata:
    models = list(settings.AVAILABLE_MODELS)
    models.sort()
    return ServiceMetadata(
        agents=get_all_agent_info(),
        models=models,
        default_agent=DEFAULT_AGENT,
        default_model=settings.DEFAULT_MODEL,
    )



@router.get("/graph/{graph_id}")
async def get_graph(graph_id: str):
    """
    Get a graph by its ID.
    """
    graph_store = GraphStore()
    fig_json = graph_store.get_graph(graph_id)
    if fig_json is None:
        raise HTTPException(status_code=404, detail="Graph not found")
    return Response(
        content=fig_json,
        media_type="application/json"
    )


async def _handle_input(
    user_input: UserInput, agent: CompiledStateGraph
) -> tuple[dict[str, Any], UUID]:
    """
    Parse user input and handle any required interrupt resumption.
    Returns kwargs for agent invocation and the run_id.
    """
    run_id = uuid4()
    thread_id = user_input.thread_id or str(uuid4())

    configurable = {"thread_id": thread_id, "model": user_input.model}

    if user_input.agent_config:
        if overlap := configurable.keys() & user_input.agent_config.keys():
            raise HTTPException(
                status_code=422,
                detail=f"agent_config contains reserved keys: {overlap}",
            )
        configurable.update(user_input.agent_config)

    config = RunnableConfig(
        configurable=configurable,
        run_id=run_id,
    )

    # Check for interrupts that need to be resumed
    state = await agent.aget_state(config=config)
    interrupted_tasks = [
        task for task in state.tasks if hasattr(task, "interrupts") and task.interrupts
    ]

    if interrupted_tasks:
        # assume user input is response to resume agent execution from interrupt
        input = Command(resume=user_input.message)
    else:
        input = {"messages": [HumanMessage(content=user_input.message)]}

    kwargs = {
        "input": input,
        "config": config,
    }

    return kwargs, run_id


@router.post("/{agent_id}/invoke")
@router.post("/invoke")
async def invoke(user_input: UserInput, agent_id: str = DEFAULT_AGENT) -> ChatMessage:
    """
    Invoke an agent with user input to retrieve a final response.

    If agent_id is not provided, the default agent will be used.
    Use thread_id to persist and continue a multi-turn conversation. run_id kwarg
    is also attached to messages for recording feedback.
    """
    # NOTE: Currently this only returns the last message or interrupt.
    # In the case of an agent outputting multiple AIMessages (such as the background step
    # in interrupt-agent, or a tool step in research-assistant), it's omitted. Arguably,
    # you'd want to include it. You could update the API to return a list of ChatMessages
    # in that case.
    agent: CompiledStateGraph = get_agent(agent_id)
    kwargs, run_id = await _handle_input(user_input, agent)
    try:
        response_events = await agent.ainvoke(**kwargs, stream_mode=["updates", "values"])
        response_type, response = response_events[-1]
        if response_type == "values":
            # Normal response, the agent completed successfully
            output = langchain_to_chat_message(response["messages"][-1])
        elif response_type == "updates" and "__interrupt__" in response:
            # The last thing to occur was an interrupt
            # Return the value of the first interrupt as an AIMessage
            output = langchain_to_chat_message(
                AIMessage(content=response["__interrupt__"][0].value)
            )
        else:
            raise ValueError(f"Unexpected response type: {response_type}")

        output.run_id = str(run_id)
        return output
    except Exception as e:
        logger.error(f"An exception occurred: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error")


async def message_generator(
    user_input: StreamInput, agent_id: str = DEFAULT_AGENT
) -> AsyncGenerator[str, None]:
    """
    Generate a stream of messages from the agent.

    This is the workhorse method for the /stream endpoint.
    """
    agent: CompiledStateGraph = get_agent(agent_id)
    kwargs, run_id = await _handle_input(user_input, agent)

    # Process streamed events from the graph and yield messages over the SSE stream.
    async for stream_event in agent.astream(
        **kwargs, stream_mode=["updates", "messages", "custom"]
    ):
        if not isinstance(stream_event, tuple):
            continue
        stream_mode, event = stream_event
        new_messages = []
        if stream_mode == "updates":
            for node, updates in event.items():
                # A simple approach to handle agent interrupts.
                # In a more sophisticated implementation, we could add
                # some structured ChatMessage type to return the interrupt value.
                if node == "__interrupt__":
                    interrupt: Interrupt
                    for interrupt in updates:
                        new_messages.append(AIMessage(content=interrupt.value))
                    continue
                update_messages = updates.get("messages", [])
                # special cases for using langgraph-supervisor library
                if node == "supervisor":
                    # Get only the last AIMessage since supervisor includes all previous messages
                    ai_messages = [msg for msg in update_messages if isinstance(msg, AIMessage)]
                    if ai_messages:
                        update_messages = [ai_messages[-1]]
                if node in ("research_expert", "math_expert"):
                    # By default the sub-agent output is returned as an AIMessage.
                    # Convert it to a ToolMessage so it displays in the UI as a tool response.
                    msg = ToolMessage(
                        content=update_messages[0].content,
                        name=node,
                        tool_call_id="",
                    )
                    update_messages = [msg]
                new_messages.extend(update_messages)

        if stream_mode == "custom":
            new_messages = [event]

        for message in new_messages:
            try:
                chat_message = langchain_to_chat_message(message)
                chat_message.run_id = str(run_id)
            except Exception as e:
                logger.error(f"Error parsing message: {e}")
                yield f"data: {json.dumps({'type': 'error', 'content': 'Unexpected error'})}\n\n"
                continue
            # LangGraph re-sends the input message, which feels weird, so drop it
            if chat_message.type == "human" and chat_message.content == user_input.message:
                continue
            yield f"data: {json.dumps({'type': 'message', 'content': chat_message.model_dump()})}\n\n"

        if stream_mode == "messages":
            if not user_input.stream_tokens:
                continue
            msg, metadata = event
            if "skip_stream" in metadata.get("tags", []):
                continue
            # For some reason, astream("messages") causes non-LLM nodes to send extra messages.
            # Drop them.
            if not isinstance(msg, AIMessageChunk):
                continue
            content = remove_tool_calls(msg.content)
            if content:
                # Empty content in the context of OpenAI usually means
                # that the model is asking for a tool to be invoked.
                # So we only print non-empty content.
                yield f"data: {json.dumps({'type': 'token', 'content': convert_message_content_to_string(content)})}\n\n"
    yield "data: [DONE]\n\n"


def _sse_response_example() -> dict[int, Any]:
    return {
        status.HTTP_200_OK: {
            "description": "Server Sent Event Response",
            "content": {
                "text/event-stream": {
                    "example": "data: {'type': 'token', 'content': 'Hello'}\n\ndata: {'type': 'token', 'content': ' World'}\n\ndata: [DONE]\n\n",
                    "schema": {"type": "string"},
                }
            },
        }
    }


@router.post(
    "/{agent_id}/stream",
    response_class=StreamingResponse,
    responses=_sse_response_example(),
)
@router.post("/stream", response_class=StreamingResponse, responses=_sse_response_example())
async def stream(user_input: StreamInput, agent_id: str = DEFAULT_AGENT) -> StreamingResponse:
    """
    Stream an agent's response to a user input, including intermediate messages and tokens.

    If agent_id is not provided, the default agent will be used.
    Use thread_id to persist and continue a multi-turn conversation. run_id kwarg
    is also attached to all messages for recording feedback.

    Set `stream_tokens=false` to return intermediate messages but not token-by-token.
    """
    return StreamingResponse(
        message_generator(user_input, agent_id),
        media_type="text/event-stream",
    )


@router.post("/feedback")
async def feedback(feedback: Feedback) -> FeedbackResponse:
    """
    Record feedback in the database.    
    The feedback can be used for analytics and monitoring purposes.
    """
    try:
        db_manager.save_feedback(
            run_id=feedback.run_id,
            key=feedback.key,
            score=feedback.score,
            additional_data=feedback.kwargs if feedback.kwargs else None
        )
        
        # Optionally, send to LangSmith if configured
        if settings.LANGCHAIN_API_KEY and settings.LANGCHAIN_PROJECT:
            client = LangsmithClient()
            kwargs = feedback.kwargs or {}
            client.create_feedback(
                run_id=feedback.run_id,
                key=feedback.key,
                score=feedback.score,
                **kwargs,
            )
            
        logger.info(f"Saved feedback for run {feedback.run_id}: {feedback.key}={feedback.score}")
        return FeedbackResponse()
    except Exception as e:
        logger.error(f"Error saving feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Error saving feedback: {str(e)}")


@router.post("/history")
def history(input: ChatHistoryInput) -> ChatHistory:
    """
    Get chat history.
    """
    # TODO: Hard-coding DEFAULT_AGENT here is wonky
    agent: CompiledStateGraph = get_agent(DEFAULT_AGENT)
    try:
        state_snapshot = agent.get_state(
            config=RunnableConfig(
                configurable={
                    "thread_id": input.thread_id,
                }
            )
        )
        messages: list[AnyMessage] = state_snapshot.values["messages"]
        chat_messages: list[ChatMessage] = [langchain_to_chat_message(m) for m in messages]
        return ChatHistory(messages=chat_messages)
    except Exception as e:
        logger.error(f"An exception occurred: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error")

@router.post("/{agent_id}/upload")
@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    thread_id: Optional[str] = None,
    agent_id: str = DEFAULT_AGENT
) -> dict:
    """
    Upload a file to the agent service.
    
    Args:
        file: The uploaded file
        thread_id: Optional thread ID to associate the file with
        agent_id: The agent to use (defaults to the default agent)
        
    Returns:
        dict: Contains the file_id of the uploaded file and storage path for PDFs
    """
    try:
        file_id = uuid4()
        thread_uuid = UUID(thread_id) if thread_id else None
        
        contents = await file.read()
        
        text_content = None
        pdf_storage_path = None
        
        try:
            if file.content_type.startswith('text/'):
                text_content = contents.decode('utf-8', errors='ignore')
            elif file.content_type.startswith('application/pdf'):
                pdf_parser = os.environ.get("UPLOADED_PDF_PARSER", None)
                
                base_storage_dir = "uploaded"
                thread_dir = thread_id if thread_id else "no_thread"
                storage_dir = os.path.join(base_storage_dir, thread_dir)
                
                os.makedirs(storage_dir, exist_ok=True)
                
                pdf_filename = file.filename
                pdf_storage_path = os.path.join(storage_dir, pdf_filename)
                
                with open(pdf_storage_path, "wb") as pdf_file:
                    pdf_file.write(contents)
                
                if 'nlm-ingestor' in pdf_parser:                    
                    rag_system.index_document(
                        pdf_path=pdf_storage_path, 
                        title=file.filename[:-4], 
                        table_name='uploaded_document_blocks'
                    )
                                    
                    logger.info(f"PDF processed and stored at: {pdf_storage_path}")
                else:
                    logger.warning("The only supported PDF parser for now is 'nlm-ingestor'")
            else:
                print(f"Unsupported file type: {file.content_type}")
        except Exception as e:
            logger.warning(f"Impossible to extract file content: {e}")
                
        metadata = {
            "original_name": file.filename,
            "content_type": file.content_type,
            "size": len(contents),
            "storage_path": pdf_storage_path
        }
        
        db_manager.save_file(
            file_id=file_id,
            thread_id=thread_uuid,
            filename=file.filename,
            content_type=file.content_type,
            content=contents,
            text_content=text_content,
            metadata=metadata
        )
        
        if thread_id and text_content:
            agent: CompiledStateGraph = get_agent(agent_id)            
            config = RunnableConfig(
                configurable={"thread_id": thread_id}
            )
            
            try:
                state = await agent.aget_state(config=config)                
                file_message = SystemMessage(
                    content=f"Contenu du fichier {file.filename}:\n\n{text_content}"
                )
                
                new_messages = list(state.values.get("messages", []))
                new_messages.append(file_message)
                
                await agent.aupdate_state(
                    values={"messages": new_messages},
                    config=config
                )
                
                logger.info(f"The file {file.filename} has successfully been added to the messages history of the thread {thread_id}")
            except Exception as e:
                logger.error(f"Error while adding the file content to the file history: {e}")
        
        response_data = {
            "file_id": str(file_id), 
            "filename": file.filename
        }
        
        if pdf_storage_path:
            response_data["storage_path"] = pdf_storage_path
        
        logger.info(f"File uploaded: {file.filename}, ID: {file_id}, Thread: {thread_id}, Path: {pdf_storage_path}")
        
        return response_data
        
    except Exception as e:
        logger.error(f"Error while uploading the file: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error while uploading the file: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@router.get("/feedback/{run_id}")
async def get_feedback(run_id: str) -> Dict[str, Any]:
    """Get all feedback entries for a specific run.
    
    Args:
        run_id: The run ID to get feedback for
        
    Returns:
        Dictionary containing the feedback entries
    """
    try:
        feedback_entries = db_manager.get_feedback_for_run(run_id)
        return {
            "run_id": run_id,
            "feedback": feedback_entries
        }
    except Exception as e:
        logger.error(f"Error retrieving feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving feedback: {str(e)}")


@router.post("/conversations/{thread_id}/title")
async def set_conversation_title(thread_id: str, title: str) -> Dict[str, Any]:
    """Set or update the title of a conversation.
    
    Args:
        thread_id: The thread ID of the conversation
        title: The title to set for the conversation
        
    Returns:
        Status confirmation
    """
    try:
        db_manager.save_conversation_title(UUID(thread_id), title)
        return {
            "status": "success",
            "thread_id": thread_id,
            "title": title
        }
    except Exception as e:
        logger.error(f"Error setting conversation title: {e}")
        raise HTTPException(status_code=500, detail=f"Error setting conversation title: {str(e)}")


@router.get("/conversations")
async def get_conversations(limit: int = 20) -> Dict[str, Any]:
    """Get a list of recent conversations.
    
    Args:
        limit: Maximum number of conversations to retrieve (default 20)
        
    Returns:
        Dictionary containing the list of conversations
    """
    try:
        conversations = db_manager.get_conversations(limit)
        return {
            "conversations": conversations
        }
    except Exception as e:
        logger.error(f"Error retrieving conversations: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving conversations: {str(e)}")


@router.delete("/conversations/{thread_id}")
async def delete_conversation(thread_id: str) -> Dict[str, Any]:
    """Delete a conversation and all associated data.
    
    Args:
        thread_id: The thread ID of the conversation to delete
        
    Returns:
        Status confirmation
    """
    try:
        result = db_manager.delete_conversation(UUID(thread_id))
        if result:
            return {"status": "success", "thread_id": thread_id}
        else:
            raise HTTPException(status_code=404, detail=f"Conversation with ID {thread_id} not found")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid thread_id format")
    except Exception as e:
        logger.error(f"Error deleting conversation: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting conversation: {str(e)}")

@router.get("/conversations/{thread_id}/title")
async def get_conversation_title(thread_id: str) -> Dict[str, Any]:
    """Get the title of a conversation.
    
    Args:
        thread_id: The thread ID of the conversation
        
    Returns:
        Dictionary containing the conversation title
    """
    try:
        title = db_manager.get_conversation_title(UUID(thread_id))
        if title:
            return {"thread_id": thread_id, "title": title}
        else:
            # Return default title and save it
            default_title = "New conversation"
            # db_manager.save_conversation_title(UUID(thread_id), default_title)
            return {"thread_id": thread_id, "title": default_title}
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid thread_id format")
    except Exception as e:
        logger.error(f"Error retrieving conversation title: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving conversation title: {str(e)}")


app.include_router(router)
