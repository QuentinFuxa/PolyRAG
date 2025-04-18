import logging
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from get_config import load_config

logger = logging.getLogger(__name__)

def validate_postgres_config() -> None:
    """
    Validate that the PostgreSQL configuration is present in the config file.
    Raises ValueError if required configuration is missing.
    """
    config = load_config()
    
    if 'database' not in config:
        raise ValueError("Missing 'database' section in configuration")
    
    db_config = config['database']
    
    if 'connection_string' not in db_config:
        raise ValueError("Missing 'connection_string' in database configuration")
    
    connection_string = db_config['connection_string']
    if not connection_string or not connection_string.startswith('postgresql'):
        raise ValueError("Invalid PostgreSQL connection string")

def get_postgres_connection_string() -> str:
    """Build and return the PostgreSQL connection string from config."""
    config = load_config()
    return config['database']['connection_string']

def get_postgres_saver() -> BaseCheckpointSaver:
    """Initialize and return a PostgreSQL saver instance."""
    validate_postgres_config()
    return AsyncPostgresSaver.from_conn_string(get_postgres_connection_string())
