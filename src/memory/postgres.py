import os
import logging
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

logger = logging.getLogger(__name__)

def validate_postgres_config() -> None:
    """
    Validate that the PostgreSQL configuration is present in the config file.
    Raises ValueError if required configuration is missing.
    """
            
    database_url = os.environ.get("DATABASE_URL", None)
    if not database_url or not database_url.startswith('postgresql'):
        raise ValueError("Invalid PostgreSQL connection string")

def get_postgres_connection_string() -> str:
    """Build and return the PostgreSQL connection string from config."""
    return os.environ.get("DATABASE_URL")

def get_postgres_saver() -> BaseCheckpointSaver:
    """Initialize and return a PostgreSQL saver instance."""
    validate_postgres_config()
    return AsyncPostgresSaver.from_conn_string(get_postgres_connection_string())
