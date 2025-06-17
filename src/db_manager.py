import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

import psycopg2
from psycopg2 import pool
from psycopg2.extras import DictCursor, Json, register_uuid
from sqlalchemy import create_engine, text
import pandas as pd
from urllib.parse import urlparse
import os
try:
    from google.cloud.sql.connector import Connector, IPTypes
except ImportError:
    Connector = None
    IPTypes = None

try:
    from .schema.schema import UserInDB, UserFeedbackCreate, UserFeedbackRead
except ImportError:
    from schema.schema import UserInDB, UserFeedbackCreate, UserFeedbackRead
    
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
        self.using_google_connector = False
        self.connector = None
        self.conn_pool = None
        self.instance_connection_name = os.environ.get("INSTANCE_CONNECTION_NAME")
        self.db_user = os.environ.get("DB_USER")
        self.db_pass = os.environ.get("DB_PASS")
        self.db_name = os.environ.get("DB_NAME")

        register_uuid()

        if self.instance_connection_name and self.db_user and self.db_pass and self.db_name:
            if Connector is None:
                logger.error("INSTANCE_CONNECTION_NAME is set, but google-cloud-sql-connector library is not installed. Please install it.")
                raise ImportError("google-cloud-sql-connector is required but not installed.")
            
            logger.info(f"Using Google Cloud SQL Connector for instance: {self.instance_connection_name}")
            self.using_google_connector = True
            self.connector = Connector()
            
            self.engine = create_engine(
                "postgresql+psycopg2://",
                creator=self._getconn_google_sql
            )
            self.connection = self._getconn_google_sql()
            self.connection.autocommit = True
            self.connection_string = f"postgresql+psycopg2://{self.db_user}:***@{self.instance_connection_name}/{self.db_name}"
        else:
            logger.info("Using DATABASE_URL for direct PostgreSQL connection.")
            self.connection_string = os.getenv("DATABASE_URL")
            if not self.connection_string:
                raise ValueError("DATABASE_URL environment variable not set, and Google Cloud SQL Connector variables are not fully provided.")
            
            self.engine = create_engine(self.connection_string)
            self.connection = psycopg2.connect(self.connection_string)
            self.connection.autocommit = True
            
            parsed_url = urlparse(self.connection_string)
            db_params = {
                "host": parsed_url.hostname,
                "database": parsed_url.path[1:],
                "user": parsed_url.username,
                "password": parsed_url.password,
                "port": parsed_url.port or 5432
            }
            self.conn_pool = pool.SimpleConnectionPool(1, 10, **db_params)
        
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.embedding_enabled = self.api_key is not None
        if self.embedding_enabled:
            from openai import OpenAI
            self.openai_client = OpenAI(api_key=self.api_key)
            self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
            self.embedding_dim = 1536
        
        self._create_tables()
        # self.get_connection() # get_connection is called by methods needing it, not always on init
    
    def _create_tables(self):
        """Create the necessary schema and tables if they do not exist."""
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {schema_app_data}")

                # Users table
                cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {schema_app_data}.users (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    email TEXT UNIQUE NOT NULL,
                    hashed_password TEXT NOT NULL,
                    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT (CURRENT_TIMESTAMP AT TIME ZONE 'UTC'),
                    updated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT (CURRENT_TIMESTAMP AT TIME ZONE 'UTC')
                )
                """)
                cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_users_email ON {schema_app_data}.users(email)")

                # Files table
                cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {schema_app_data}.files (
                    id UUID PRIMARY KEY,
                    thread_id UUID,
                    user_id UUID, -- Will add FK constraint after ensuring users table exists
                    filename TEXT NOT NULL,
                    content_type TEXT,
                    content BYTEA,
                    text_content TEXT,
                    metadata JSONB,
                    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT (CURRENT_TIMESTAMP AT TIME ZONE 'UTC')
                )
                """)
                
                # Conversations history table
                cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {schema_app_data}.conversations (
                    thread_id UUID PRIMARY KEY,
                    user_id UUID, -- Will add FK constraint
                    title TEXT NOT NULL,
                    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT (CURRENT_TIMESTAMP AT TIME ZONE 'UTC'),
                    updated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT (CURRENT_TIMESTAMP AT TIME ZONE 'UTC')
                )
                """)

                # Add user_id column and FK to files if not exists
                try:
                    cursor.execute(f"ALTER TABLE {schema_app_data}.files ADD COLUMN IF NOT EXISTS user_id UUID")
                    cursor.execute(f"""
                        DO $$
                        BEGIN
                            IF NOT EXISTS (
                                SELECT 1 FROM pg_constraint 
                                WHERE conname = 'fk_files_user_id' AND conrelid = '{schema_app_data}.files'::regclass
                            ) THEN
                                ALTER TABLE {schema_app_data}.files 
                                ADD CONSTRAINT fk_files_user_id FOREIGN KEY (user_id) 
                                REFERENCES {schema_app_data}.users(id) ON DELETE CASCADE;
                            END IF;
                        END $$;
                    """)
                except psycopg2.Error as e:
                    logger.warning(f"Could not add user_id column or FK to files, might exist or other issue: {e}")
                    conn.rollback()
                else:
                    conn.commit()

                # Add user_id column and FK to conversations if not exists
                try:
                    cursor.execute(f"ALTER TABLE {schema_app_data}.conversations ADD COLUMN IF NOT EXISTS user_id UUID")
                    cursor.execute(f"""
                        DO $$
                        BEGIN
                            IF NOT EXISTS (
                                SELECT 1 FROM pg_constraint
                                WHERE conname = 'fk_conversations_user_id' AND conrelid = '{schema_app_data}.conversations'::regclass
                            ) THEN
                                ALTER TABLE {schema_app_data}.conversations 
                                ADD CONSTRAINT fk_conversations_user_id FOREIGN KEY (user_id) 
                                REFERENCES {schema_app_data}.users(id) ON DELETE CASCADE;
                            END IF;
                        END $$;
                    """)
                except psycopg2.Error as e:
                    logger.warning(f"Could not add user_id column or FK to conversations, might exist or other issue: {e}")
                    conn.rollback()
                else:
                    conn.commit()

                cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {schema_app_data}.feedback (
                    id SERIAL PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    score FLOAT NOT NULL,
                    conversation_id TEXT NULL,
                    commented_message_text TEXT NULL,
                    additional_data JSONB,
                    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT (CURRENT_TIMESTAMP AT TIME ZONE 'UTC')
                )
                """)
                
                cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_files_thread_id ON {schema_app_data}.files(thread_id)")
                cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_files_user_id ON {schema_app_data}.files(user_id)")
                
                cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_feedback_run_id ON {schema_app_data}.feedback(run_id)")
                
                cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_conversations_created_at ON {schema_app_data}.conversations(updated_at DESC)") # usually order by updated_at
                cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON {schema_app_data}.conversations(user_id)")
                
                cursor.execute(f"""        
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
                )
                """)

                cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {schema_app_data}.document_sources (
                    id SERIAL PRIMARY KEY,
                    name TEXT UNIQUE NOT NULL,
                    path TEXT,
                    url TEXT,
                    is_indexed BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT (CURRENT_TIMESTAMP AT TIME ZONE 'UTC'),
                    updated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT (CURRENT_TIMESTAMP AT TIME ZONE 'UTC')
                )
                """)
                cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_document_sources_name ON {schema_app_data}.document_sources(name)")
                cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_document_sources_path ON {schema_app_data}.document_sources(path)")
                cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_document_sources_url ON {schema_app_data}.document_sources(url)")

                cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {schema_app_data}.graphs (
                    graph_id UUID PRIMARY KEY,
                    graph_json JSONB NOT NULL,
                    expiry_time TIMESTAMP WITHOUT TIME ZONE NOT NULL
                )
                """)
                cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_graphs_expiry_time ON {schema_app_data}.graphs(expiry_time)")

                # User Feedback table
                cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {schema_app_data}.user_feedback (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id UUID NOT NULL REFERENCES {schema_app_data}.users(id) ON DELETE CASCADE,
                    feedback_content TEXT NOT NULL,
                    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT (CURRENT_TIMESTAMP AT TIME ZONE 'UTC')
                )
                """)
                cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_user_feedback_user_id ON {schema_app_data}.user_feedback(user_id)")
                cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_user_feedback_created_at ON {schema_app_data}.user_feedback(created_at DESC)")

            conn.commit()
        finally:
            self.release_connection(conn)

    def _getconn_google_sql(self):
        if not self.connector:
            logger.error("Google Cloud SQL Connector not initialized but connection attempt made.")
            raise RuntimeError("Google Cloud SQL Connector not initialized.")
        ip_type_str = os.environ.get("GOOGLE_SQL_IP_TYPE", "PUBLIC").upper()
        ip_type = IPTypes.PUBLIC
        if IPTypes:
            if ip_type_str == "PRIVATE": ip_type = IPTypes.PRIVATE
            elif ip_type_str == "PUBLIC": ip_type = IPTypes.PUBLIC
        
        conn = self.connector.connect(
            self.instance_connection_name, "psycopg2",
            user=self.db_user, password=self.db_pass, db=self.db_name, ip_type=ip_type
        )
        return conn

    def close(self):
        if hasattr(self, 'connection') and self.connection and not self.connection.closed:
            try: self.connection.close()
            except Exception as e: logger.warning(f"Error closing main connection: {e}")
        
        if self.using_google_connector and hasattr(self, 'connector') and self.connector:
            try: self.connector.close()
            except Exception as e: logger.warning(f"Error closing Google SQL Connector: {e}")
        elif hasattr(self, 'conn_pool') and self.conn_pool:
            try: self.conn_pool.closeall()
            except Exception as e: logger.warning(f"Error closing psycopg2 connection pool: {e}")
    
    def get_connection(self):
        if self.using_google_connector:
            conn = self._getconn_google_sql()
        elif self.conn_pool:
            conn = self.conn_pool.getconn()
        else: # Fallback to direct connection if pool not initialized (e.g. during _initialize)
             conn = psycopg2.connect(self.connection_string)

        conn.autocommit = True # Important for many operations
        return conn

    def release_connection(self, conn):
        if self.using_google_connector:
            if conn and not conn.closed:
                try: conn.close()
                except Exception as e: logger.warning(f"Error closing Google SQL connection: {e}")
        elif self.conn_pool:
            try: self.conn_pool.putconn(conn)
            except Exception as e: logger.warning(f"Error releasing connection to psycopg2 pool: {e}")
        elif conn and not conn.closed: # Fallback for direct connections
            try: conn.close()
            except Exception as e: logger.warning(f"Error closing direct connection: {e}")

    def get_connection_string(self):
        if self.using_google_connector:
            return f"postgresql+psycopg2://{self.db_user}:[REDACTED]@{self.instance_connection_name}/{self.db_name} (via Google Connector)"
        return self.connection_string

    # User operations
    def create_user(self, email: str, hashed_password: str) -> UserInDB:
        user_id = uuid4()
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=DictCursor) as cursor:
                cursor.execute(
                    f"""
                    INSERT INTO {schema_app_data}.users (id, email, hashed_password)
                    VALUES (%s, %s, %s)
                    RETURNING id, email, hashed_password, created_at, updated_at
                    """,
                    (user_id, email, hashed_password)
                )
                user_data = cursor.fetchone()
                conn.commit()
                return UserInDB(**user_data)
        finally:
            self.release_connection(conn)

    def get_user_by_email(self, email: str) -> Optional[UserInDB]:
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=DictCursor) as cursor:
                cursor.execute(
                    f"SELECT id, email, hashed_password, created_at, updated_at FROM {schema_app_data}.users WHERE email = %s",
                    (email,)
                )
                user_data = cursor.fetchone()
                return UserInDB(**user_data) if user_data else None
        finally:
            self.release_connection(conn)

    def get_user_by_id(self, user_id: UUID) -> Optional[UserInDB]:
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=DictCursor) as cursor:
                cursor.execute(
                    f"SELECT id, email, hashed_password, created_at, updated_at FROM {schema_app_data}.users WHERE id = %s",
                    (user_id,)
                )
                user_data = cursor.fetchone()
                return UserInDB(**user_data) if user_data else None
        finally:
            self.release_connection(conn)

    # File operations
    def save_file(self, file_id: UUID, user_id: UUID, thread_id: Optional[UUID], filename: str,
                  content_type: str, content: bytes, text_content: Optional[str] = None,
                  metadata: Optional[Dict[str, Any]] = None) -> UUID:
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"""
                    INSERT INTO {schema_app_data}.files 
                    (id, user_id, thread_id, filename, content_type, content, text_content, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (file_id, user_id, thread_id, filename, content_type, content, text_content, Json(metadata) if metadata else None)
                )
                result_id = cursor.fetchone()[0]
                conn.commit()
                return result_id
        finally:
            self.release_connection(conn)
    
    def get_file(self, file_id: UUID, user_id: UUID) -> Optional[Dict[str, Any]]:
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=DictCursor) as cursor:
                cursor.execute(
                    f"""
                    SELECT id, thread_id, user_id, filename, content_type, content, text_content, metadata, created_at
                    FROM {schema_app_data}.files
                    WHERE id = %s AND user_id = %s
                    """,
                    (file_id, user_id)
                )
                return dict(cursor.fetchone()) if cursor.rowcount > 0 else None
        finally:
            self.release_connection(conn)
    
    def get_file_text(self, file_id: UUID, user_id: UUID) -> Optional[str]:
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT text_content FROM {schema_app_data}.files
                    WHERE id = %s AND user_id = %s
                    """,
                    (file_id, user_id)
                )
                return cursor.fetchone()[0] if cursor.rowcount > 0 else None
        finally:
            self.release_connection(conn)

    def execute_query(self, query: str, params=None) -> List[tuple]:
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(query, params or ())
                conn.commit() # Assuming execute_query might include DML
                return cursor.fetchall() if cursor.description else []
        except Exception as e:
            conn.rollback()
            raise
        finally:
            self.release_connection(conn)
    
    def execute_many(self, query: str, params_list: List[tuple]) -> None:
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.executemany(query, params_list)
                conn.commit()
        except Exception as e:
            conn.rollback()
            raise
        finally:
            self.release_connection(conn)
    
    def is_embedding_enabled(self) -> bool:
        return self.embedding_enabled
    
    def compute_embedding(self, text: str) -> Optional[List[float]]:
        if not self.embedding_enabled: return None
        try:
            response = self.openai_client.embeddings.create(input=text, model=self.embedding_model)
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error computing embedding: {e}")
            return None
    
    def get_embedding_dimension(self) -> int:
        return self.embedding_dim if self.embedding_enabled else 0

    def save_user_feedback(self, feedback_data: UserFeedbackCreate) -> UserFeedbackRead:
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=DictCursor) as cursor:
                cursor.execute(
                    f"""
                    INSERT INTO {schema_app_data}.user_feedback (user_id, feedback_content)
                    VALUES (%s, %s)
                    RETURNING id, user_id, feedback_content, created_at
                    """,
                    (feedback_data.user_id, feedback_data.feedback_content)
                )
                saved_feedback = cursor.fetchone()
                conn.commit()
                return UserFeedbackRead(**saved_feedback)
        finally:
            self.release_connection(conn)
        
    def save_feedback(self, run_id: str, key: str, score: float, conversation_id: Optional[str] = None, commented_message_text: Optional[str] = None, additional_data: Optional[Dict[str, Any]] = None) -> int:
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"""
                    INSERT INTO {schema_app_data}.feedback (run_id, key, score, conversation_id, commented_message_text, additional_data)
                    VALUES (%s, %s, %s, %s, %s, %s) RETURNING id
                    """,
                    (run_id, key, score, conversation_id, commented_message_text, Json(additional_data) if additional_data else None)
                )
                feedback_id = cursor.fetchone()[0]
                conn.commit()
                return feedback_id
        finally:
            self.release_connection(conn)
            
    def get_feedback_for_run(self, run_id: str) -> List[Dict[str, Any]]:
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=DictCursor) as cursor:
                cursor.execute(
                    f"""
                    SELECT id, run_id, key, score, additional_data, created_at FROM {schema_app_data}.feedback
                    WHERE run_id = %s ORDER BY created_at DESC
                    """,
                    (run_id,)
                )
                return [dict(row) for row in cursor.fetchall()]
        finally:
            self.release_connection(conn)
            
    # Conversation history methods
    def save_conversation_title(self, thread_id: UUID, user_id: UUID, title: str) -> None:
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                # Ensure conversation exists for the user or create it
                cursor.execute(
                    f"""
                    INSERT INTO {schema_app_data}.conversations (thread_id, user_id, title)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (thread_id) DO UPDATE SET 
                        title = EXCLUDED.title, 
                        updated_at = (CURRENT_TIMESTAMP AT TIME ZONE 'UTC')
                    WHERE {schema_app_data}.conversations.user_id = %s 
                    """,
                    (thread_id, user_id, title, user_id) # user_id repeated for WHERE clause on update
                )
                conn.commit()
        finally:
            self.release_connection(conn)

    def get_conversation_title(self, thread_id: UUID, user_id: UUID) -> Optional[str]:
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT title FROM {schema_app_data}.conversations
                    WHERE thread_id = %s AND user_id = %s
                    """,
                    (thread_id, user_id)
                )
                result = cursor.fetchone()
                return result[0] if result else None
        finally:
            self.release_connection(conn)

    def get_conversations(self, user_id: UUID, limit: int = 20) -> List[Dict[str, Any]]:
        
        query = f"""
        SELECT thread_id, title, created_at, updated_at FROM {schema_app_data}.conversations
        WHERE user_id = %s ORDER BY updated_at DESC LIMIT %s
        """
        params = (user_id, limit)
        if user_id == '00000000-0000-0000-0000-000000000001':  # Special case for admin
            query = f"""
            SELECT * FROM (
                SELECT 
                    c.thread_id, 
                    CASE 
                        WHEN f.conversation_id IS NOT NULL 
                        THEN '🔴 ' || split_part(u.email, '@', 1) || ': ' || c.title
                        ELSE split_part(u.email, '@', 1) || ': ' || c.title
                    END AS title, 
                    c.created_at, 
                    c.updated_at
                FROM document_data.conversations c
                JOIN document_data.users u ON c.user_id = u.id
                LEFT JOIN document_data.feedback f ON CAST(c.thread_id AS text) = f.conversation_id
                ORDER BY c.updated_at DESC
                LIMIT %s
            ) 
            GROUP BY 1, 2, 3, 4
            ORDER BY updated_at DESC
            """
            params = (limit,)
            
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=DictCursor) as cursor:
                cursor.execute(query, params)
                return [dict(row) for row in cursor.fetchall()]
        finally:
            self.release_connection(conn)

    def delete_conversation(self, thread_id: UUID, user_id: UUID) -> bool:
        conn = self.get_connection()
        deleted_count = 0
        try:
            with conn.cursor() as cursor:
                # Delete conversation itself
                cursor.execute(
                    f"""
                    DELETE FROM {schema_app_data}.conversations
                    WHERE thread_id = %s AND user_id = %s RETURNING thread_id
                    """,
                    (thread_id, user_id)
                )
                if cursor.rowcount > 0:
                    deleted_count = cursor.rowcount
                
                # Delete associated files for this thread and user
                cursor.execute(
                    f"""
                    DELETE FROM {schema_app_data}.files
                    WHERE thread_id = %s AND user_id = %s
                    """,
                    (thread_id, user_id)
                )
                conn.commit()
            return deleted_count > 0
        except Exception as e:
            conn.rollback()
            logger.error(f"Error deleting conversation {thread_id} for user {user_id}: {e}")
            return False
        finally:
            self.release_connection(conn)

    def reassign_and_rename_conversation(self, thread_id: UUID, current_user_id: UUID, new_user_id: UUID, title_prefix: str) -> bool:
        conn = self.get_connection()
        # Explicitly manage transaction
        conn.autocommit = False 
        try:
            with conn.cursor() as cursor:
                # 1. Fetch current title and lock the row
                cursor.execute(
                    f"""
                    SELECT title FROM {schema_app_data}.conversations
                    WHERE thread_id = %s AND user_id = %s
                    FOR UPDATE
                    """,
                    (thread_id, current_user_id)
                )
                result = cursor.fetchone()
                if not result:
                    conn.rollback() # Conversation not found for this user
                    logger.warning(f"Conversation {thread_id} not found for user {current_user_id} during reassignment attempt.")
                    return False
                
                original_title = result[0]
                new_title = f"{title_prefix}{original_title}"

                # 2. Update conversation owner and title
                cursor.execute(
                    f"""
                    UPDATE {schema_app_data}.conversations
                    SET user_id = %s, title = %s, updated_at = (CURRENT_TIMESTAMP AT TIME ZONE 'UTC')
                    WHERE thread_id = %s AND user_id = %s 
                    """, 
                    # Ensure we are still updating the original user's record before changing owner,
                    # though FOR UPDATE should prevent concurrent changes.
                    (new_user_id, new_title, thread_id, current_user_id)
                )
                if cursor.rowcount == 0: # Should not happen if SELECT FOR UPDATE worked
                    conn.rollback()
                    logger.error(f"Failed to update conversation {thread_id} for user {current_user_id} during reassignment (rowcount 0).")
                    return False

                # 3. Update associated files owner
                # We update files that belonged to the original user for this thread.
                cursor.execute(
                    f"""
                    UPDATE {schema_app_data}.files
                    SET user_id = %s
                    WHERE thread_id = %s AND user_id = %s
                    """,
                    (new_user_id, thread_id, current_user_id)
                )
                # Not checking rowcount for files, as there might be no files.

                conn.commit()
                logger.info(f"Conversation {thread_id} successfully reassigned from user {current_user_id} to {new_user_id} with title '{new_title}'.")
                return True
        except Exception as e:
            if conn and not conn.closed: # Check if conn is still valid before rollback
                try:
                    conn.rollback()
                except Exception as rb_e:
                    logger.error(f"Error during rollback for conversation reassignment {thread_id}: {rb_e}")
            logger.error(f"Error reassigning conversation {thread_id} from user {current_user_id} to {new_user_id}: {e}")
            return False
        finally:
            if conn: # Ensure conn exists before trying to modify or release
                conn.autocommit = True # Reset autocommit before releasing
                self.release_connection(conn)

    def save_graph(self, graph_id: UUID, graph_json: Dict[str, Any], expiry_time: float) -> None:
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"""
                    INSERT INTO {schema_app_data}.graphs (graph_id, graph_json, expiry_time)
                    VALUES (%s, %s, TO_TIMESTAMP(%s))
                    ON CONFLICT (graph_id) DO UPDATE SET 
                        graph_json = EXCLUDED.graph_json, 
                        expiry_time = EXCLUDED.expiry_time
                    """,
                    (graph_id, Json(graph_json), expiry_time)
                )
                conn.commit()
        finally:
            self.release_connection(conn)

    def get_graph(self, graph_id: UUID) -> Optional[Dict[str, Any]]:
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=DictCursor) as cursor:
                cursor.execute(
                    f"""
                    SELECT graph_json FROM {schema_app_data}.graphs
                    WHERE graph_id = %s AND expiry_time > (CURRENT_TIMESTAMP AT TIME ZONE 'UTC')
                    """,
                    (graph_id,)
                )
                result = cursor.fetchone()
                return result['graph_json'] if result else None
        finally:
            self.release_connection(conn)

    def delete_expired_graphs(self) -> int:
        conn = self.get_connection()
        deleted_count = 0
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"""
                    DELETE FROM {schema_app_data}.graphs
                    WHERE expiry_time <= (CURRENT_TIMESTAMP AT TIME ZONE 'UTC')
                    RETURNING graph_id
                    """
                )
                deleted_count = cursor.rowcount 
                conn.commit()
                logger.info(f"Deleted {deleted_count} expired graphs from the database.")
            return deleted_count
        finally:
            self.release_connection(conn)

    def add_document_source(self, name: str, path: Optional[str] = None, url: Optional[str] = None) -> int:
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"""
                    INSERT INTO {schema_app_data}.document_sources (name, path, url, is_indexed)
                    VALUES (%s, %s, %s, FALSE) ON CONFLICT (name) DO NOTHING
                    """,
                    (name, path, url)
                )
                cursor.execute(f"SELECT id FROM {schema_app_data}.document_sources WHERE name = %s", (name,))
                result = cursor.fetchone()
                conn.commit()
                if result: return result[0]
                else: raise Exception(f"Failed to add or find document source for name: {name}")
        finally:
            self.release_connection(conn)

    def set_document_indexed(self, name: str, indexed: bool = True) -> bool:
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"""
                    UPDATE {schema_app_data}.document_sources
                    SET is_indexed = %s, updated_at = (CURRENT_TIMESTAMP AT TIME ZONE 'UTC')
                    WHERE name = %s RETURNING id
                    """,
                    (indexed, name)
                )
                updated = cursor.fetchone() is not None
                conn.commit()
                return updated
        finally:
            self.release_connection(conn)

    def get_document_source_status(self, name: str) -> Optional[Dict[str, Any]]:
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=DictCursor) as cursor:
                cursor.execute(
                    f"""
                    SELECT id, name, path, url, is_indexed, created_at, updated_at
                    FROM {schema_app_data}.document_sources WHERE name LIKE %s
                    """, (f"%{name}%",)
                )
                return dict(cursor.fetchone()) if cursor.rowcount > 0 else None
        finally:
            self.release_connection(conn)

    def list_schemas(self) -> List[str]:
        query = """
        SELECT schema_name FROM information_schema.schemata 
        WHERE schema_name NOT IN ('pg_catalog', 'information_schema', 'pg_toast')
          AND schema_name NOT LIKE 'pg_temp_%' AND schema_name NOT LIKE 'pg_toast_temp_%'
        """
        return [row[0] for row in self.execute_query(query)]

    def list_tables(self, schema: str) -> List[str]:
        query = "SELECT table_name FROM information_schema.tables WHERE table_schema = %s AND table_type = 'BASE TABLE'"
        return [row[0] for row in self.execute_query(query, (schema,))]

    def get_table_columns(self, schema: str, table: str) -> List[Dict[str, str]]:
        query = """
        SELECT column_name, data_type FROM information_schema.columns 
        WHERE table_schema = %s AND table_name = %s ORDER BY ordinal_position
        """
        return [{'name': r[0], 'type': r[1]} for r in self.execute_query(query, (schema, table))]

    def get_column_samples(self, schema: str, table: str, column: str, n: int = 30) -> List[Any]:
        if not (schema.isidentifier() and table.isidentifier() and column.isidentifier()):
             raise ValueError("Invalid schema, table, or column name")
        internal_limit = max(n * 5, 500)
        query = f"""
        SELECT DISTINCT "{column}" FROM (
            SELECT "{column}" FROM "{schema}"."{table}"
            WHERE "{column}" IS NOT NULL LIMIT %s
        ) AS limited_scan LIMIT %s;
        """
        try:
            results = self.execute_query(query, (internal_limit, n))
            return [str(r[0]) if not isinstance(r[0], (str,int,float,bool,list,dict)) else r[0] for r in results]
        except Exception as e:
            logger.error(f"Error sampling column {schema}.{table}.{column}: {e}")
            return [f"Error sampling: {e}"]

    def identify_special_columns(self, schema: str, table: str) -> Dict[str, List[str]]:
        columns = self.get_table_columns(schema, table)
        special_cols = {'embedding': [], 'tsvector': []}
        for col in columns:
            if 'vector' in col['type'].lower(): special_cols['embedding'].append(col['name'])
            elif 'tsvector' == col['type'].lower(): special_cols['tsvector'].append(col['name'])
        return special_cols
