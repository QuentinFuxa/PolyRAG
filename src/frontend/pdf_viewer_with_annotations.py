import streamlit as st
import requests
import os
from typing import Optional, List, Dict, Any
# Removed: from db_manager import DatabaseManager
from streamlit_pdf_viewer import pdf_viewer
from client import AgentClient # Added AgentClient import

# Removed: db_manager = DatabaseManager()

async def get_pdf_content(agent_client: AgentClient, document_name: str) -> Optional[bytes]: # Made async, added agent_client
    print(f"Attempting to fetch PDF content for: {document_name}")
    source_info = await agent_client.aget_document_source_status(document_name=document_name) # Use agent_client

    if not source_info:
        st.error(f"Document source '{document_name}' not found via API.") # Updated error message
        print(f"Source not found in DB for: {document_name}")
        return None

    pdf_content = None

    # Check if it's a local path
    if source_info.get('path') and os.path.exists(source_info['path']):
        try:
            print(f"Reading PDF from path: {source_info['path']}")
            with open(source_info['path'], "rb") as f:
                pdf_content = f.read()
        except Exception as e:
            st.error(f"Error reading PDF from path '{source_info['path']}': {e}")
            print(f"Error reading path {source_info['path']}: {e}")
            return None
    # Check if it's a URL
    elif source_info.get('url'):
        try:
            url = source_info['url']
            print(f"Downloading PDF from URL: {url}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            pdf_content = response.content
        except requests.exceptions.RequestException as e:
            st.error(f"Error downloading PDF from URL '{source_info['url']}': {e}")
            print(f"Error downloading URL {source_info['url']}: {e}")
            return None
        except Exception as e:
            st.error(f"An unexpected error occurred downloading '{source_info['url']}': {e}")
            print(f"Unexpected error downloading URL {source_info['url']}: {e}")
            return None
    else:
        st.error(f"Could not determine a valid source for document '{document_name}'.")
        print(f"No valid source found for: {document_name}")
        return None

    print(f"Successfully retrieved PDF content for: {document_name}")
    return pdf_content

async def display_pdf(agent_client: AgentClient, document_name: str, annotations: Optional[List[Dict[str, Any]]] = None, debug_viewer: bool = False) -> None: # Made async, added agent_client
    pdf_content = await get_pdf_content(agent_client, document_name) # Pass agent_client, await

    if pdf_content:
        try:
            # pdf_viewer is synchronous, so it's called directly even within an async function
            pdf_viewer(input=pdf_content, annotations=annotations, render_text=True, annotation_outline_size=2)
            print(f"Successfully rendered PDF display for: {document_name}")
        except Exception as e:
            st.error(f"Error displaying PDF for '{document_name}': {e}")
            print(f"Error displaying PDF {document_name}: {e}")
    else:
        print(f"PDF content could not be retrieved for display: {document_name}")
