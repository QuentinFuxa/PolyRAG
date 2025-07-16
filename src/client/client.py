import json
import os
from collections.abc import AsyncGenerator, Generator
from typing import Any, List, Dict, Optional, Union
from uuid import UUID

import httpx

from schema import (
    ChatHistory,
    ChatHistoryInput,
    ChatMessage,
    Feedback,
    ServiceMetadata,
    StreamInput,
    UserInput,
    UserFeedbackCreate,
    UserFeedbackRead,
)
from typing import cast


class AgentClientError(Exception):
    pass


class AgentClient:
    """Client for interacting with the agent service."""

    def get_feedbacks(
        self,
        conversation_id: str,
    ) -> list[Feedback]:
        """
        Retrieve all feedbacks for a given conversation (thread_id).

        Args:
            conversation_id (str): The thread ID of the conversation.

        Returns:
            list[Feedback]: List of feedback objects.
        """
        params = {"conversation_id": conversation_id}
        try:
            response = httpx.get(
                f"{self.base_url}/feedback",
                params=params,
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            feedbacks_data = response.json().get("feedbacks", [])
            return [Feedback.model_validate(fb) for fb in feedbacks_data]
        except httpx.HTTPError as e:
            raise AgentClientError(f"Error retrieving feedbacks: {e}")

    def __init__(
        self,
        base_url: str = "http://0.0.0.0",
        agent: str | None = None,
        timeout: float | None = None,
        get_info: bool = True,
    ) -> None:
        """
        Initialize the client.

        Args:
            base_url (str): The base URL of the agent service.
            agent (str): The name of the default agent to use.
            timeout (float, optional): The timeout for requests.
            get_info (bool, optional): Whether to fetch agent information on init.
                Default: True
        """
        self.base_url = base_url
        self.auth_secret = os.getenv("AUTH_SECRET")
        self.timeout = timeout
        self.info: ServiceMetadata | None = None
        self.agent: str | None = None
        if get_info:
            self.retrieve_info()
        if agent:
            self.update_agent(agent)

    @property
    def _headers(self) -> dict[str, str]:
        headers = {}
        if self.auth_secret:
            headers["Authorization"] = f"Bearer {self.auth_secret}"
        return headers

    def retrieve_info(self) -> None:
        try:
            response = httpx.get(
                f"{self.base_url}/info",
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise AgentClientError(f"Error getting service info: {e}")

        self.info = ServiceMetadata.model_validate(response.json())
        if not self.agent or self.agent not in [a.key for a in self.info.agents]:
            self.agent = self.info.default_agent

    def update_agent(self, agent: str, verify: bool = True) -> None:
        if verify:
            if not self.info:
                self.retrieve_info()
            agent_keys = [a.key for a in self.info.agents]  # type: ignore[union-attr]
            if agent not in agent_keys:
                raise AgentClientError(
                    f"Agent {agent} not found in available agents: {', '.join(agent_keys)}"
                )
        self.agent = agent

    async def ainvoke(
        self,
        message: str,
        model: str | None = None,
        thread_id: str | None = None,
        agent_config: dict[str, Any] | None = None,
        user_id: Optional[Union[str, UUID]] = None, # Added user_id
    ) -> ChatMessage:
        """
        Invoke the agent asynchronously. Only the final message is returned.

        Args:
            message (str): The message to send to the agent
            model (str, optional): LLM model to use for the agent
            thread_id (str, optional): Thread ID for continuing a conversation
            agent_config (dict[str, Any], optional): Additional configuration to pass through to the agent

        Returns:
            AnyMessage: The response from the agent
        """
        if not self.agent:
            raise AgentClientError("No agent selected. Use update_agent() to select an agent.")
        request = UserInput(message=message)
        if thread_id:
            request.thread_id = thread_id
        if model:
            request.model = model  # type: ignore[assignment]
        
        current_agent_config = agent_config.copy() if agent_config else {}
        if user_id:
            current_agent_config["user_id"] = str(user_id) # Pass user_id in agent_config
        if current_agent_config: # Only assign if there's something in it
            request.agent_config = current_agent_config

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/{self.agent}/invoke",
                    json=request.model_dump(),
                    headers=self._headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
            except httpx.HTTPError as e:
                raise AgentClientError(f"Error: {json.loads(response.content)}")

        return ChatMessage.model_validate(response.json())

    def invoke(
        self,
        message: str,
        model: str | None = None,
        thread_id: str | None = None,
        agent_config: dict[str, Any] | None = None,
        user_id: Optional[Union[str, UUID]] = None,
    ) -> ChatMessage:
        """
        Invoke the agent synchronously. Only the final message is returned.

        Args:
            message (str): The message to send to the agent
            model (str, optional): LLM model to use for the agent
            thread_id (str, optional): Thread ID for continuing a conversation
            agent_config (dict[str, Any], optional): Additional configuration to pass through to the agent

        Returns:
            ChatMessage: The response from the agent
        """
        if not self.agent:
            raise AgentClientError("No agent selected. Use update_agent() to select an agent.")
        request = UserInput(message=message)
        if thread_id:
            request.thread_id = thread_id
        if model:
            request.model = model  # type: ignore[assignment]

        current_agent_config = agent_config.copy() if agent_config else {}
        if user_id:
            current_agent_config["user_id"] = str(user_id) # Pass user_id in agent_config
        if current_agent_config: # Only assign if there's something in it
            request.agent_config = current_agent_config
            
        try:
            response = httpx.post(
                f"{self.base_url}/{self.agent}/invoke",
                json=request.model_dump(),
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise AgentClientError(f"Error: {json.loads(response.content)}")

        return ChatMessage.model_validate(response.json())

    def _parse_stream_line(self, line: str) -> ChatMessage | str | None:
        line = line.strip()
        if line.startswith("data: "):
            data = line[6:]
            if data == "[DONE]":
                return None, None
            try:
                parsed = json.loads(data)
            except Exception as e:
                raise Exception(f"Error JSON parsing message from server: {e}")
            additional_data = parsed.get("additional_data", None)
            match parsed["type"]:
                case "message":
                    # Convert the JSON formatted message to an AnyMessage
                    try:
                        return ChatMessage.model_validate(parsed["content"]), additional_data
                    except Exception as e:
                        raise Exception(f"Server returned invalid message: {e}")
                case "token":
                    # Yield the str token directly
                    return parsed["content"], additional_data
                case "error":
                    error_msg = "Error: " + parsed["content"]
                    return ChatMessage(type="ai", content=error_msg), additional_data
        return None, None

    def stream(
        self,
        message: str,
        model: str | None = None,
        thread_id: str | None = None,
        agent_config: dict[str, Any] | None = None,
        stream_tokens: bool = True,
        user_id: Optional[Union[str, UUID]] = None,
    ) -> Generator[ChatMessage | str, None, None]:
        """
        Stream the agent's response synchronously.

        Each intermediate message of the agent process is yielded as a ChatMessage.
        If stream_tokens is True (the default value), the response will also yield
        content tokens from streaming models as they are generated.

        Args:
            message (str): The message to send to the agent
            model (str, optional): LLM model to use for the agent
            thread_id (str, optional): Thread ID for continuing a conversation
            agent_config (dict[str, Any], optional): Additional configuration to pass through to the agent
            stream_tokens (bool, optional): Stream tokens as they are generated
                Default: True

        Returns:
            Generator[ChatMessage | str, None, None]: The response from the agent
        """
        if not self.agent:
            raise AgentClientError("No agent selected. Use update_agent() to select an agent.")
        request = StreamInput(message=message, stream_tokens=stream_tokens)
        if thread_id:
            request.thread_id = thread_id
        if model:
            request.model = model  # type: ignore[assignment]

        current_agent_config = agent_config.copy() if agent_config else {}
        if user_id:
            current_agent_config["user_id"] = str(user_id) # Pass user_id in agent_config
        if current_agent_config: # Only assign if there's something in it
            request.agent_config = current_agent_config

        try:
            with httpx.stream(
                "POST",
                f"{self.base_url}/{self.agent}/stream",
                json=request.model_dump(),
                headers=self._headers,
                timeout=self.timeout,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line.strip():
                        parsed, additional_data = self._parse_stream_line(line)
                        if parsed is None:
                            break
                        yield parsed, additional_data
        except httpx.HTTPError as e:
            raise AgentClientError(f"Error: {json.loads(response.content)}")

    async def astream(
        self,
        message: str,
        model: str | None = None,
        thread_id: str | None = None,
        agent_config: dict[str, Any] | None = None,
        stream_tokens: bool = True,
        file_ids: list[str] | None = None,
        user_id: Optional[Union[str, UUID]] = None,
    ) -> AsyncGenerator[ChatMessage | str, None]:
        """
        Stream the agent's response asynchronously.

        Each intermediate message of the agent process is yielded as an AnyMessage.
        If stream_tokens is True (the default value), the response will also yield
        content tokens from streaming modelsas they are generated.

        Args:
            message (str): The message to send to the agent
            model (str, optional): LLM model to use for the agent
            thread_id (str, optional): Thread ID for continuing a conversation
            agent_config (dict[str, Any], optional): Additional configuration to pass through to the agent
            stream_tokens (bool, optional): Stream tokens as they are generated
                Default: True
            file_ids (list[str], optional): List of file IDs to attach to the message
                Default: None
        Returns:
            AsyncGenerator[ChatMessage | str, None]: The response from the agent
        """
        if not self.agent:
            raise AgentClientError("No agent selected. Use update_agent() to select an agent.")
        request = StreamInput(message=message, stream_tokens=stream_tokens)
        if thread_id:
            request.thread_id = thread_id
        if model:
            request.model = model  # type: ignore[assignment]

        current_agent_config = agent_config.copy() if agent_config else {}
        if user_id:
            current_agent_config["user_id"] = str(user_id) # Pass user_id in agent_config
        if current_agent_config: # Only assign if there's something in it
            request.agent_config = current_agent_config
            
        async with httpx.AsyncClient() as client:
            try:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/{self.agent}/stream",
                    json=request.model_dump(),
                    headers=self._headers,
                    timeout=self.timeout,
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.strip():
                            parsed, additional_data = self._parse_stream_line(line)
                            if parsed is None:
                                break
                            yield parsed, additional_data
            except httpx.HTTPError as e:
                raise AgentClientError(f"Error: {json.loads(response.content)}")

    async def acreate_feedback(
        self, run_id: str, key: str, score: float, conversation_id: Optional[str] = None, commented_message_text: Optional[str] = None, kwargs: dict[str, Any] = {}
    ) -> None:
        """
        Create a feedback record for a run.

        This is a simple wrapper for the LangSmith create_feedback API, so the
        credentials can be stored and managed in the service rather than the client.
        See: https://api.smith.langchain.com/redoc#tag/feedback/operation/create_feedback_api_v1_feedback_post
        """
        request = Feedback(run_id=run_id, key=key, score=score, conversation_id=conversation_id, commented_message_text=commented_message_text, kwargs=kwargs)
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/feedback",
                    json=request.model_dump(),
                    headers=self._headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                response.json()
            except httpx.HTTPError as e:
                raise AgentClientError(f"Error: {e}")

    def create_feedback(
        self, user_id: UUID, feedback: str
    ) -> None:
        """
        Create feedback for the agent.

        Args:
            user_id (UUID): The ID of the user providing feedback
            feedback (str): The feedback text
        """

    def retrieve_graph(self, graph_id: str) -> str | None:
        """
        Retrieve a graph by its ID.
        """
        try:
            response = httpx.get(
                f"{self.base_url}/graph/{graph_id}",
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise AgentClientError(f"Error: {e}")
        return response.text



    def get_history(
        self,
        thread_id: str,
        user_id: Optional[Union[str, UUID]] = None,
    ) -> ChatHistory:
        """
        Get chat history.

        Args:
            thread_id (str): Thread ID for identifying a conversation
            user_id (UUID, optional): User ID to scope history if backend supports
        """
        request = ChatHistoryInput(thread_id=thread_id, user_id=user_id) # Pass user_id
        try:
            response = httpx.post(
                f"{self.base_url}/history",
                json=request.model_dump(),
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise AgentClientError(f"Error: {e}")

        return ChatHistory.model_validate(response.json())

    def upload_file(
        self,
        file_name: str,
        file_content: bytes,
        file_type: str,
        thread_id: Optional[str] = None,
        user_id: Optional[Union[str, UUID]] = None,
    ) -> str:
        """
        Upload a file to the agent service.
        
        Args:
            file_name (str): The name of the file
            file_content (bytes): The binary content of the file
            file_type (str): The MIME type of the file
            thread_id (str, optional): Thread ID to associate the file with
            user_id (UUID, optional): User ID to associate the file with
            
        Returns:
            str: The ID of the uploaded file in the backend
        """
        if not self.agent:
            raise AgentClientError("No agent selected. Use update_agent() to select an agent.")
        
        params = {}
        if thread_id:
            params["thread_id"] = thread_id
        if user_id:
            params["user_id"] = str(user_id)

        url = f"{self.base_url}/{self.agent}/upload"
        
        files = {"file": (file_name, file_content, file_type)}
        
        try:
            response = httpx.post(
                url,
                params=params if params else None, # Add params to the request
                files=files,
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise AgentClientError(f"Error uploading file: {e}")
        
        # Récupérer l'ID du fichier depuis la réponse
        response_data = response.json()
        if "file_id" not in response_data:
            raise AgentClientError("Server did not return a file_id")
        
        return response_data["file_id"]
        
    def get_conversations(self, limit: int = 20, user_id: Optional[Union[str, UUID]] = None) -> list[dict[str, Any]]:
        """
        Get a list of recent conversations.
        
        Args:
            limit (int, optional): Maximum number of conversations to retrieve. Default: 20
            user_id (UUID, optional): User ID to scope conversations if backend supports
            
        Returns:
            list[dict[str, Any]]: List of conversations with thread_id, title, and created_at
        """
        params = {"limit": limit}
        if user_id:
            params["user_id"] = str(user_id)
        try:
            response = httpx.get(
                f"{self.base_url}/conversations",
                params=params,
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()["conversations"]
        except httpx.HTTPError as e:
            raise AgentClientError(f"Error getting conversations: {e}")
            
    def set_conversation_title(self, thread_id: str, title: str, user_id: Optional[Union[str, UUID]] = None) -> None:
        """
        Set or update the title of a conversation.
        
        Args:
            thread_id (str): The thread ID of the conversation
            title (str): The title to set for the conversation
            user_id (UUID, optional): User ID to scope this operation if backend supports
        """
        params = {"title": title}
        if user_id:
            params["user_id"] = str(user_id)
        try:
            response = httpx.post(
                f"{self.base_url}/conversations/{thread_id}/title",
                params=params,
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise AgentClientError(f"Error setting conversation title: {e}")
            
    def get_conversation_title(self, thread_id: str, user_id: Optional[Union[str, UUID]] = None) -> str:
        """
        Get the title of a conversation.
        
        Args:
            thread_id (str): The thread ID of the conversation
            user_id (UUID, optional): User ID to scope this operation if backend supports
            
        Returns:
            str: The title of the conversation
        """
        params = {}
        if user_id:
            params["user_id"] = str(user_id)
        try:
            response = httpx.get(
                f"{self.base_url}/conversations/{thread_id}/title",
                params=params if params else None,
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()["title"]
        except httpx.HTTPError as e:
            raise AgentClientError(f"Error getting conversation title: {e}")
            
    def delete_conversation(self, thread_id: str, user_id: Optional[Union[str, UUID]] = None) -> bool:
        """
        Delete a conversation and all associated data.
        
        Args:
            thread_id (str): The thread ID of the conversation to delete
            user_id (UUID, optional): User ID to scope this operation if backend supports
            
        Returns:
            bool: True if the conversation was successfully deleted
        """
        params = {}
        if user_id:
            params["user_id"] = str(user_id)
        try:
            response = httpx.delete(
                f"{self.base_url}/conversations/{thread_id}",
                params=params if params else None,
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()["status"] == "success"
        except httpx.HTTPError as e:
            raise AgentClientError(f"Error deleting conversation: {e}")
            
    async def aget_conversations(self, limit: int = 20, user_id: Optional[Union[str, UUID]] = None) -> list[dict[str, Any]]:
        """
        Get a list of recent conversations asynchronously.
        
        Args:
            limit (int, optional): Maximum number of conversations to retrieve. Default: 20
            user_id (UUID, optional): User ID to scope conversations if backend supports
            
        Returns:
            list[dict[str, Any]]: List of conversations with thread_id, title, and created_at
        """
        params = {"limit": limit}
        if user_id:
            params["user_id"] = str(user_id)
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{self.base_url}/conversations",
                    params=params,
                    headers=self._headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                return response.json()["conversations"]
            except httpx.HTTPError as e:
                raise AgentClientError(f"Error getting conversations: {e}")
                
    async def aset_conversation_title(self, thread_id: str, title: str, user_id: Optional[Union[str, UUID]] = None) -> None:
        """
        Set or update the title of a conversation asynchronously.
        
        Args:
            thread_id (str): The thread ID of the conversation
            title (str): The title to set for the conversation
            user_id (UUID, optional): User ID to scope this operation if backend supports
        """
        params = {"title": title}
        if user_id:
            params["user_id"] = str(user_id)
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/conversations/{thread_id}/title",
                    params=params,
                    headers=self._headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
            except httpx.HTTPError as e:
                raise AgentClientError(f"Error setting conversation title: {e}")
                
    async def aget_conversation_title(self, thread_id: str, user_id: Optional[Union[str, UUID]] = None) -> str:
        """
        Get the title of a conversation asynchronously.
        
        Args:
            thread_id (str): The thread ID of the conversation
            user_id (UUID, optional): User ID to scope this operation if backend supports
            
        Returns:
            str: The title of the conversation
        """
        params = {}
        if user_id:
            params["user_id"] = str(user_id)
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{self.base_url}/conversations/{thread_id}/title",
                    params=params if params else None,
                    headers=self._headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                return response.json()["title"]
            except httpx.HTTPError as e:
                raise AgentClientError(f"Error getting conversation title: {e}")

    async def aget_annotations(self, pdf_file: str, block_indices: List[int], user_id: Optional[Union[str, UUID]] = None) -> List[Dict[str, Any]]:
        """
        Get highlighting annotations for specified blocks in a PDF asynchronously.
        """
        request_data = {"pdf_file": pdf_file, "block_indices": block_indices}
        if user_id:
            request_data["user_id"] = str(user_id)
            
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/rag/annotations",
                    json=request_data,
                    headers=self._headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                return response.json().get("annotations", [])
            except httpx.HTTPError as e:
                raise AgentClientError(f"Error getting RAG annotations: {e}")
            except Exception as e: # Catch potential JSON parsing errors or missing keys
                raise AgentClientError(f"Error processing RAG annotations response: {e}")

    async def adebug_pdf_blocks(self, pdf_file: str, user_id: Optional[Union[str, UUID]] = None) -> List[Dict[str, Any]]:
        """
        Get all block annotations for a PDF for debugging asynchronously.
        """
        request_data = {"pdf_file": pdf_file}
        if user_id:
            request_data["user_id"] = str(user_id)

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/rag/debug_blocks",
                    json=request_data,
                    headers=self._headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                return response.json().get("annotations", [])
            except httpx.HTTPError as e:
                raise AgentClientError(f"Error debugging RAG blocks: {e}")
            except Exception as e:
                raise AgentClientError(f"Error processing RAG debug blocks response: {e}")

    def get_document_source_status(self, document_name: str) -> Optional[Dict[str, Any]]:
        """
        Get the source status (path or URL) for a given document name synchronously.
        """
        try:
            response = httpx.get(
                f"{self.base_url}/documents/{document_name}/source_status",
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            response_data = response.json()
            if response_data.get("error"):
                return None
            return response_data.get("source_info")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise AgentClientError(f"Error getting document source status: {e}")
        except httpx.HTTPError as e:
            raise AgentClientError(f"Error getting document source status: {e}")
        except Exception as e:
            raise AgentClientError(f"Error processing document source status response: {e}")

    async def aget_document_source_status(self, document_name: str) -> Optional[Dict[str, Any]]:
        """
        Get the source status (path or URL) for a given document name asynchronously.
        """
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{self.base_url}/documents/{document_name}/source_status",
                    headers=self._headers,
                    timeout=self.timeout,
                )
                response.raise_for_status() # Will raise for 4xx/5xx responses
                response_data = response.json()
                if response_data.get("error"):
                    # Log or handle specific error message from server if needed
                    # For now, returning None signifies not found or error
                    return None
                return response_data.get("source_info")
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    return None
                # For other HTTPStatusErrors, wrap and raise
                raise AgentClientError(f"HTTP Status Error getting document source status: {e.response.status_code} - {e}")
            except httpx.HTTPError as e:
                # For other httpx.HTTPErrors (e.g., network issues)
                raise AgentClientError(f"HTTPError getting document source status: {e}")
            except Exception as e: # Catch other potential errors like JSON parsing
                raise AgentClientError(f"Error processing document source status response: {e}")

    def submit_user_feedback(self, user_id: UUID, feedback_content: str) -> UserFeedbackRead:
        """
        Submit user feedback to the service.

        Args:
            user_id (UUID): The ID of the user submitting feedback.
            feedback_content (str): The content of the feedback.

        Returns:
            UserFeedbackRead: The saved feedback object.
        """
        request_data = UserFeedbackCreate(user_id=user_id, feedback_content=feedback_content)
        try:
            response = httpx.post(
                f"{self.base_url}/user_feedback",
                json=request_data.model_dump(mode='json'),
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return UserFeedbackRead.model_validate(response.json())
        except httpx.HTTPError as e:
            try:
                error_detail = response.json().get("detail", str(e))
            except Exception: # If response.json() fails or "detail" is not present
                error_detail = str(e)
            raise AgentClientError(f"Error submitting user feedback: {error_detail}")
        except Exception as e:
            raise AgentClientError(f"An unexpected error occurred while submitting user feedback: {e}")

    async def asubmit_user_feedback(self, user_id: UUID, feedback_content: str) -> UserFeedbackRead:
        """
        Submit user feedback to the service asynchronously.

        Args:
            user_id (UUID): The ID of the user submitting feedback.
            feedback_content (str): The content of the feedback.

        Returns:
            UserFeedbackRead: The saved feedback object.
        """
        request_data = UserFeedbackCreate(user_id=user_id, feedback_content=feedback_content)
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/user_feedback",
                    json=request_data.model_dump(mode='json'), # Ensure UUID is serialized correctly
                    headers=self._headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                return UserFeedbackRead.model_validate(response.json())
            except httpx.HTTPError as e:
                try:
                    error_detail = response.json().get("detail", str(e))
                except Exception:
                    error_detail = str(e)
                raise AgentClientError(f"Error submitting user feedback: {error_detail}")
            except Exception as e:
                raise AgentClientError(f"An unexpected error occurred while submitting user feedback: {e}")
