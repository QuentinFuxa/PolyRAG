import os # Keep os for other potential uses, though not directly for DB_URL here
import logging
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from db_manager import DatabaseManager

logger = logging.getLogger(__name__)

# Instantiate DatabaseManager - it's a singleton
# This will raise an error during import if DB config is invalid,
# effectively replacing validate_postgres_config()
try:
    db_manager = DatabaseManager()
except Exception as e:
    logger.error(f"Failed to initialize DatabaseManager in memory.postgres: {e}")
    # If DatabaseManager fails, get_postgres_saver will likely fail or use a None db_manager.
    # This makes db_manager effectively None if initialization fails.
    db_manager = None 

# def validate_postgres_config() -> None: # No longer strictly needed
#     """
#     Validation is now implicitly handled by DatabaseManager's initialization.
#     This function can be removed or adapted if specific pre-checks are still desired
#     before attempting to get a saver. For now, we rely on DatabaseManager.
#     """
#     if not db_manager:
#         raise ConnectionError("DatabaseManager failed to initialize. Check database configuration.")
#     # Further checks could be added here if db_manager instance alone isn't enough
#     pass


# def get_postgres_connection_string() -> str: # Replaced by db_manager.get_connection_string()
#     """Return the PostgreSQL connection string via DatabaseManager."""
#     if not db_manager:
#         raise ConnectionError("DatabaseManager not initialized. Cannot get connection string.")
#     return db_manager.get_connection_string()


def get_postgres_saver() -> BaseCheckpointSaver:
    """Initialize and return a PostgreSQL saver instance using DatabaseManager."""
    if not db_manager:
        logger.error("DatabaseManager not available for get_postgres_saver. Check logs for initialization errors.")
        raise ConnectionError("DatabaseManager not initialized. Cannot create PostgresSaver.")

    try:
        # Attempt to get the connection string from DatabaseManager
        conn_string = db_manager.get_connection_string()
        
        # Log the connection string being used (especially useful for debugging Google SQL Connector)
        logger.info(f"Attempting to initialize AsyncPostgresSaver with connection string: {conn_string}")
        
        # AsyncPostgresSaver.from_conn_string expects a valid URL.
        # If using Google Cloud SQL Connector, db_manager.get_connection_string()
        # returns a placeholder. This might cause issues with AsyncPostgresSaver
        # if it cannot handle this format or if it strictly needs direct DB host/port.
        # This is a known potential issue noted in the plan.
        saver = AsyncPostgresSaver.from_conn_string(conn_string)
        logger.info("AsyncPostgresSaver initialized successfully.")
        return saver
    except Exception as e:
        logger.error(f"Failed to initialize AsyncPostgresSaver: {e}")
        logger.error(f"Connection string used: {db_manager.get_connection_string() if db_manager else 'N/A'}")
        # Re-raise the exception so the application knows checkpointing setup failed.
        raise
