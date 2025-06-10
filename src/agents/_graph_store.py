import time
import uuid
from threading import Timer
from uuid import UUID

from db_manager import DatabaseManager

class GraphStore:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(GraphStore, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, expiry_seconds=3600):
        if self._initialized:
            return
        
        self.db_manager = DatabaseManager()
        self.default_expiry_seconds = expiry_seconds
        self._initialized = True
        
        self._start_cleanup_timer()
    
    def store_graph(self, fig_json, expiry_seconds=None):
        """Stores graph JSON, returns its ID."""
        graph_id = uuid.uuid4()
        expiry = expiry_seconds if expiry_seconds is not None else self.default_expiry_seconds
        expiry_time = time.time() + expiry
        self.db_manager.save_graph(graph_id, fig_json, expiry_time)
        return str(graph_id)
    
    def get_graph(self, graph_id_str: str):
        """Retrieves graph JSON by ID if not expired."""
        try:
            graph_id_uuid = UUID(graph_id_str)
        except ValueError:
            return None
        return self.db_manager.get_graph(graph_id_uuid)
    
    def _cleanup_expired(self):
        """Calls the database manager to delete expired graphs."""
        # self.db_manager.delete_expired_graphs()
        # We actually don't want to delete expired graphs
    
    def _start_cleanup_timer(self):
        self._cleanup_expired()
        t = Timer(300, self._start_cleanup_timer)
        t.daemon = True
        t.start()

graph_store = GraphStore()
