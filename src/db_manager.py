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
from urllib.parse import urlparse
import os 
logger = logging.getLogger(__name__)

schema_app_data = os.environ.get("SCHEMA_APP_DATA", "document_data")

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
        self.connection_string = os.getenv("DATABASE_URL")
        if not self.connection_string:
            raise ValueError("DATABASE_URL environment variable not set.")
        self.engine = create_engine(self.connection_string)        
        self.connection = psycopg2.connect(self.connection_string)
        register_uuid()
        self.connection.autocommit = True
        
        # Connection pool setup
        parsed_url = urlparse(self.connection_string)
        db_params = {
            "host": parsed_url.hostname,
            "database": parsed_url.path[1:],
            "user": parsed_url.username,
            "password": parsed_url.password,
            "port": parsed_url.port
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
        
        self._create_tables()
    
    def _create_tables(self):
        """Create the necessary schema and tables if they do not exist."""
        
        with self.connection.cursor() as cursor:
            # Create schema if it does not exist
            cursor.execute(f"""
            CREATE SCHEMA IF NOT EXISTS {schema_app_data}
            """)
            
            # For the files
            cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {schema_app_data}.files (
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
            
            cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {schema_app_data}.feedback (
                id SERIAL PRIMARY KEY,
                run_id TEXT NOT NULL,
                key TEXT NOT NULL,
                score FLOAT NOT NULL,
                additional_data JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            
            # For conversations history
            cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {schema_app_data}.conversations (
                thread_id UUID PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            
            cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_files_thread_id ON {schema_app_data}.files(thread_id)
            """)
            
            cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_feedback_run_id ON {schema_app_data}.feedback(run_id)
            """)
            
            cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_conversations_created_at ON {schema_app_data}.conversations(created_at DESC)
            """)
            
            cursor.execute(f"""        
            -- Uploaded document blocks table
            CREATE TABLE IF NOT EXISTS {schema_app_data}.uploaded_document_blocks (
                id SERIAL PRIMARY KEY,
                block_idx INTEGER NOT NULL,
                name TEXT NOT NULL,
                content TEXT,
                level INTEGER NOT NULL,
                page_idx INTEGER NOT NULL,
                tag TEXT NOT NULL,
                block_class TEXT,
                x0 FLOAT,
                y0 FLOAT,
                x1 FLOAT,
                y1 FLOAT,
                parent_idx INTEGER,
                UNIQUE(name, block_idx)
            );
            """)

            # For document sources (path or URL)
            cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {schema_app_data}.document_sources (
                id SERIAL PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                path TEXT,                 -- File path if source is local file
                url TEXT,                  -- URL if source is web resource
                is_indexed BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)

            # Add an index for faster lookups on name
            cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_document_sources_name ON {schema_app_data}.document_sources(name);
            """)

            # Add an index for faster lookups on path
            cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_document_sources_path ON {schema_app_data}.document_sources(path);
            """)

            # Add an index for faster lookups on url
            cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_document_sources_url ON {schema_app_data}.document_sources(url);
            """)

            cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {schema_app_data}.graphs (
                graph_id UUID PRIMARY KEY,
                graph_json JSONB NOT NULL,
                expiry_time TIMESTAMP NOT NULL
            )
            """)

            cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_graphs_expiry_time ON {schema_app_data}.graphs(expiry_time);
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
                f"""
                INSERT INTO {schema_app_data}.files 
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
                f"""
                SELECT id, thread_id, filename, content_type, content, text_content, metadata, created_at
                FROM {schema_app_data}.files
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
                f"""
                SELECT text_content
                FROM {schema_app_data}.files
                WHERE id = %s
                """,
                (file_id,)
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
        
    def save_feedback(self, run_id: str, key: str, score: float, additional_data: Optional[Dict[str, Any]] = None) -> int:
        """Save user feedback to the database.
        
        Args:
            run_id: The run ID that the feedback is for
            key: The feedback key (e.g., 'human-feedback-stars')
            score: The feedback score value (e.g., 1-5 for star ratings)
            additional_data: Any additional feedback data to store
            
        Returns:
            The ID of the inserted feedback record
        """
        with self.connection.cursor() as cursor:
            cursor.execute(
                f"""
                INSERT INTO {schema_app_data}.feedback 
                (run_id, key, score, additional_data)
                VALUES (%s, %s, %s, %s)
                RETURNING id
                """,
                (
                    run_id,
                    key,
                    score,
                    Json(additional_data) if additional_data else None
                )
            )
            return cursor.fetchone()[0]
            
    def get_feedback_for_run(self, run_id: str) -> List[Dict[str, Any]]:
        """Get all feedback entries for a specific run ID.
        
        Args:
            run_id: The run ID to get feedback for
            
        Returns:
            List of feedback entries as dictionaries
        """
        with self.connection.cursor(cursor_factory=DictCursor) as cursor:
            cursor.execute(
                f"""
                SELECT id, run_id, key, score, additional_data, created_at
                FROM {schema_app_data}.feedback
                WHERE run_id = %s
                ORDER BY created_at DESC
                """,
                (run_id,)
            )
            return [dict(row) for row in cursor.fetchall()]
            
    # Conversation history methods
    def save_conversation_title(self, thread_id: UUID, title: str) -> None:
        """Save or update a conversation title.
        
        Args:
            thread_id: The thread ID of the conversation
            title: The title for the conversation
        """
        with self.connection.cursor() as cursor:
            cursor.execute(
                f"""
                INSERT INTO {schema_app_data}.conversations (thread_id, title)
                VALUES (%s, %s)
                ON CONFLICT (thread_id) 
                DO UPDATE SET title = %s, updated_at = CURRENT_TIMESTAMP
                """,
                (thread_id, title, title)
            )
    
    def get_conversation_title(self, thread_id: UUID) -> Optional[str]:
        """Get the title of a conversation.
        
        Args:
            thread_id: The thread ID of the conversation
            
        Returns:
            The conversation title or None if not found
        """
        with self.connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT title
                FROM {schema_app_data}.conversations
                WHERE thread_id = %s
                """,
                (thread_id,)
            )
            result = cursor.fetchone()
            if result:
                return result[0]
            return None
    
    def get_conversations(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get a list of recent conversations.
        
        Args:
            limit: Maximum number of conversations to retrieve
            
        Returns:
            List of conversations as dictionaries with thread_id, title, and created_at
        """
        with self.connection.cursor(cursor_factory=DictCursor) as cursor:
            cursor.execute(
                f"""
                SELECT thread_id, title, created_at, updated_at
                FROM {schema_app_data}.conversations
                ORDER BY updated_at DESC
                LIMIT %s
                """,
                (limit,)
            )
            return [dict(row) for row in cursor.fetchall()]
    
    def delete_conversation(self, thread_id: UUID) -> bool:
        """Delete a conversation and all associated data.
        
        Args:
            thread_id: The thread ID of the conversation to delete
            
        Returns:
            True if the conversation was found and deleted, False otherwise
        """
        with self.connection.cursor() as cursor:
            cursor.execute(
                f"""
                DELETE FROM {schema_app_data}.conversations
                WHERE thread_id = %s
                RETURNING thread_id
                """,
                (thread_id,)
            )
            result = cursor.fetchone()
            
            cursor.execute(
                f"""
                DELETE FROM {schema_app_data}.files
                WHERE thread_id = %s
                """,
                (thread_id,)
            )
            
            return result is not None

    def save_graph(self, graph_id: UUID, graph_json: Dict[str, Any], expiry_time: float) -> None:
        """Save or update graph JSON data in the database."""
        with self.connection.cursor() as cursor:
            cursor.execute(
                f"""
                INSERT INTO {schema_app_data}.graphs (graph_id, graph_json, expiry_time)
                VALUES (%s, %s, TO_TIMESTAMP(%s))
                ON CONFLICT (graph_id) 
                DO UPDATE SET graph_json = EXCLUDED.graph_json, expiry_time = EXCLUDED.expiry_time
                """,
                (graph_id, Json(graph_json), expiry_time)
            )

    def get_graph(self, graph_id: UUID) -> Optional[Dict[str, Any]]:
        """Get graph JSON data if it exists and hasn't expired."""
        with self.connection.cursor(cursor_factory=DictCursor) as cursor:
            cursor.execute(
                f"""
                SELECT graph_json
                FROM {schema_app_data}.graphs
                WHERE graph_id = %s AND expiry_time > CURRENT_TIMESTAMP
                """,
                (graph_id,)
            )
            result = cursor.fetchone()
            if result:
                return result['graph_json']
            # Clean up the potentially expired entry if found but expired
            # cursor.execute(f"DELETE FROM {schema_app_data}.graphs WHERE graph_id = %s", (graph_id,))
            return None

    def delete_expired_graphs(self) -> int:
        """Delete expired graph entries from the database."""
        with self.connection.cursor() as cursor:
            cursor.execute(
                f"""
                DELETE FROM {schema_app_data}.graphs
                WHERE expiry_time <= CURRENT_TIMESTAMP
                RETURNING graph_id
                """
            )
            # cursor.rowcount gives the number of deleted rows in psycopg2
            deleted_count = cursor.rowcount 
            logger.info(f"Deleted {deleted_count} expired graphs from the database.")
            return deleted_count

    def add_document_source(self, name: str, path: Optional[str] = None, url: Optional[str] = None) -> int:
        """
        Add a new document source (file path or URL) to the database.
        If a source with the same name already exists, it returns the existing ID without inserting.

        Args:
            name: Unique identifier for the source (e.g., file path or URL).
            path: File path if the source is a local file.
            url: URL if the source is a web resource.

        Returns:
            The ID of the inserted or existing document source record.
        """
        with self.connection.cursor() as cursor:
            cursor.execute(
                f"""
                INSERT INTO {schema_app_data}.document_sources (name, path, url, is_indexed)
                VALUES (%s, %s, %s, FALSE)
                ON CONFLICT (name) DO NOTHING;
                """,
                (name, path, url)
            )
            cursor.execute(
                f"""
                SELECT id FROM {schema_app_data}.document_sources WHERE name = %s;
                """,
                (name,)
            )
            result = cursor.fetchone()
            if result:
                return result[0]
            else:
                raise Exception(f"Failed to add or find document source for name: {name}")


    def set_document_indexed(self, name: str, indexed: bool = True) -> bool:
        """
        Update the indexing status of a document source.

        Args:
            name: The unique identifier (name) of the document source.
            indexed: The new indexing status (True or False).

        Returns:
            True if the record was updated, False otherwise (e.g., if the name doesn't exist).
        """
        with self.connection.cursor() as cursor:
            cursor.execute(
                f"""
                UPDATE {schema_app_data}.document_sources
                SET is_indexed = %s, updated_at = CURRENT_TIMESTAMP
                WHERE name = %s
                RETURNING id;
                """,
                (indexed, name)
            )
            return cursor.fetchone() is not None

    def get_document_source_status(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a document source by its name.

        Args:
            name: The unique identifier (name) of the document source.

        Returns:
            A dictionary with source details (id, name, path, url, is_indexed) or None if not found.
        """
        with self.connection.cursor(cursor_factory=DictCursor) as cursor:
            query = f"""
                SELECT id, name, path, url, is_indexed, created_at, updated_at
                FROM {schema_app_data}.document_sources
                WHERE name LIKE %s
            """
            cursor.execute(query, (f"%{name}%",))
            result = cursor.fetchone()
            if result:
                return dict(result)
            return None

    def list_schemas(self) -> List[str]:
        """List all non-system schemas in the database."""
        query = """
        SELECT schema_name 
        FROM information_schema.schemata 
        WHERE schema_name NOT IN ('pg_catalog', 'information_schema', 'pg_toast')
          AND schema_name NOT LIKE 'pg_temp_%' 
          AND schema_name NOT LIKE 'pg_toast_temp_%';
        """
        results = self.execute_query(query)
        return [row[0] for row in results]

    def list_tables(self, schema: str) -> List[str]:
        """List all tables in a given schema."""
        query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = %s AND table_type = 'BASE TABLE';
        """
        results = self.execute_query(query, (schema,))
        return [row[0] for row in results]

    def get_table_columns(self, schema: str, table: str) -> List[Dict[str, str]]:
        """Get column names and data types for a table."""
        query = """
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_schema = %s AND table_name = %s
        ORDER BY ordinal_position;
        """
        results = self.execute_query(query, (schema, table))
        return [{'name': row[0], 'type': row[1]} for row in results]

    def get_column_samples(self, schema: str, table: str, column: str, n: int = 30) -> List[Any]:
        """
        Get up to n distinct, non-null sample values from a column.
        Fetches more internally to increase chances of getting diverse samples.
        """
        if not (schema.isidentifier() and table.isidentifier() and column.isidentifier()):
             raise ValueError("Invalid schema, table, or column name")
             
        internal_limit = max(n * 5, 500) 
        
        query = f"""
        SELECT DISTINCT "{column}"
        FROM (
            SELECT "{column}"
            FROM "{schema}"."{table}"
            WHERE "{column}" IS NOT NULL
            LIMIT %s  -- Limit initial scan significantly
        ) AS limited_scan
        LIMIT %s; -- Apply the final limit 'n'
        """
        try:
            results = self.execute_query(query, (internal_limit, n))
            return [str(row[0]) if not isinstance(row[0], (str, int, float, bool, list, dict)) else row[0] for row in results]
        except Exception as e:
            logger.error(f"Error sampling column {schema}.{table}.{column}: {e}")
            return [f"Error sampling: {e}"]


    def identify_special_columns(self, schema: str, table: str) -> Dict[str, List[str]]:
        """Identify columns potentially used for embeddings (vector) or full-text search (tsvector)."""
        columns = self.get_table_columns(schema, table)
        special_cols = {'embedding': [], 'tsvector': []}
        for col in columns:
            col_name = col['name']
            col_type = col['type'].lower()
            # Check for 'vector' type, potentially with dimensions like 'vector(1536)'
            if 'vector' in col_type: 
                special_cols['embedding'].append(col_name)
            elif 'tsvector' == col_type:
                special_cols['tsvector'].append(col_name)
        return special_cols
