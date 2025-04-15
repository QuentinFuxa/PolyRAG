import json
import logging
import os
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

import psycopg2
from psycopg2 import pool
from psycopg2.extras import DictCursor, Json, register_uuid
from sqlalchemy import create_engine, text
import pandas as pd
from get_config import load_config
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manager for PostgreSQL database operations."""
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern to ensure a single connection."""
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize connection, tables, connection pool and embedding capabilities."""
        # Regular connection setup
        self.connection_string = load_config()['database']['connection_string']
        self.engine = create_engine(self.connection_string)        
        self.connection = psycopg2.connect(self.connection_string)
        register_uuid()
        self.connection.autocommit = True
        
        # Connection pool setup
        parsed_url = urlparse(self.connection_string)
        db_params = {
            "host": parsed_url.hostname or os.getenv("DB_HOST", "localhost"),
            "database": parsed_url.path[1:] if parsed_url.path else os.getenv("DB_NAME", "lds"),
            "user": parsed_url.username or os.getenv("DB_USER", "postgres"),
            "password": parsed_url.password or os.getenv("DB_PASSWORD", ""),
            "port": parsed_url.port or os.getenv("DB_PORT", "5432")
        }
        self.conn_pool = pool.SimpleConnectionPool(1, 10, **db_params)
        
        # Initialize embedding capabilities
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.embedding_enabled = self.api_key is not None
        if self.embedding_enabled:
            from openai import OpenAI
            self.openai_client = OpenAI(api_key=self.api_key)
            self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
            self.embedding_dim = 1536  # Default for Ada
        
        # Create tables
        self._create_tables()
    
    def _create_tables(self):
        """Create the necessary tables if they do not exist."""
        with self.connection.cursor() as cursor:
            # For the files
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS files (
                id UUID PRIMARY KEY,
                thread_id UUID,
                filename TEXT NOT NULL,
                content_type TEXT,
                content BYTEA,
                text_content TEXT,
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            
            # For the memory
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory (
                thread_id UUID PRIMARY KEY,
                state JSONB,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            
            # Index
            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_files_thread_id ON files(thread_id)
            """)
    
    def close(self):
        """Close database connection and connection pool."""
        if hasattr(self, 'connection') and self.connection:
            self.connection.close()
        if hasattr(self, 'conn_pool'):
            self.conn_pool.closeall()
    
    # Connection pool methods
    def get_connection(self):
        """Get a connection from the pool."""
        return self.conn_pool.getconn()
    
    def release_connection(self, conn):
        """Release a connection back to the pool."""
        self.conn_pool.putconn(conn)
    
    def get_connection_string(self):
        """Returns the connection string for use with sqlalchemy or other tools."""
        return self.connection_string
    
    # File operations
    def save_file(self, 
                 file_id: UUID, 
                 thread_id: Optional[UUID], 
                 filename: str, 
                 content_type: str, 
                 content: bytes, 
                 text_content: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None) -> UUID:
        """
        Save a file to the database.
        
        Args:
            file_id: Unique identifier for the file
            thread_id: Optional thread identifier
            filename: Original filename
            content_type: MIME type of the file
            content: Binary content of the file
            text_content: Extracted text content (if available)
            metadata: Additional metadata about the file
            
        Returns:
            UUID of the saved file
        """
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO files 
                (id, thread_id, filename, content_type, content, text_content, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    file_id, 
                    thread_id, 
                    filename, 
                    content_type, 
                    content,
                    text_content,
                    Json(metadata) if metadata else None
                )
            )
            return cursor.fetchone()[0]
    
    def get_file(self, file_id: UUID) -> Optional[Dict[str, Any]]:
        """Get information and content of a file by ID."""
        with self.connection.cursor(cursor_factory=DictCursor) as cursor:
            cursor.execute(
                """
                SELECT id, thread_id, filename, content_type, content, text_content, metadata, created_at
                FROM files
                WHERE id = %s
                """,
                (file_id,)
            )
            result = cursor.fetchone()
            if result:
                return dict(result)
            return None
    
    def get_file_text(self, file_id: UUID) -> Optional[str]:
        """Get only the text content of a file."""
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT text_content
                FROM files
                WHERE id = %s
                """,
                (file_id,)
            )
            result = cursor.fetchone()
            if result:
                return result[0]
            return None
    
    # Memory operations
    def save_memory(self, thread_id: UUID, state: Dict[str, Any]) -> None:
        """Save the agent's state/memory for a thread."""
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO memory (thread_id, state)
                VALUES (%s, %s)
                ON CONFLICT (thread_id) 
                DO UPDATE SET state = %s, updated_at = CURRENT_TIMESTAMP
                """,
                (thread_id, Json(state), Json(state))
            )
    
    def get_memory(self, thread_id: UUID) -> Optional[Dict[str, Any]]:
        """Retrieve the agent's state/memory for a thread."""
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT state
                FROM memory
                WHERE thread_id = %s
                """,
                (thread_id,)
            )
            result = cursor.fetchone()
            if result:
                return result[0]
            return None
    
    def execute_query(self, query: str, params=None) -> List[tuple]:
        """
        Execute a SQL query with optional parameters and return the results.
        
        Args:
            query: SQL query string
            params: Optional parameters for the query
            
        Returns:
            Query results as a list of tuples
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(query, params or ())
            conn.commit()
            if cursor.description is not None:
                return cursor.fetchall()
            return []
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            cursor.close()
            self.release_connection(conn)
    
    def execute_many(self, query: str, params_list: List[tuple]) -> None:
        """
        Execute a SQL query multiple times with different sets of parameters.
        
        Args:
            query: SQL query string
            params_list: List of parameter tuples for batch execution
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.executemany(query, params_list)
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            cursor.close()
            self.release_connection(conn)
    
    def query_to_dataframe(self, query: str) -> pd.DataFrame:
        """Execute a query and return the results as a pandas DataFrame."""
        return pd.read_sql(query, self.engine)
    
    def is_embedding_enabled(self) -> bool:
        """Check if embedding functionality is enabled."""
        return self.embedding_enabled
    
    def compute_embedding(self, text: str) -> Optional[List[float]]:
        """
        Compute embedding for a given text using OpenAI API.
        
        Args:
            text: The text to compute embeddings for
            
        Returns:
            List of embedding values or None if embeddings are disabled or an error occurs
        """
        if not self.embedding_enabled:
            return None
        
        try:
            response = self.openai_client.embeddings.create(
                input=text,
                model=self.embedding_model
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error computing embedding: {e}")
            return None
    
    def get_embedding_dimension(self) -> int:
        """Returns the dimension of embeddings used."""
        return self.embedding_dim if self.embedding_enabled else 0