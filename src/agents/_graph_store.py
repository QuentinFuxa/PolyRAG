import time
import uuid
from threading import Timer

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
            
        self.graphs = {}
        self.expiry_seconds = expiry_seconds
        self._initialized = True
        
        self._start_cleanup_timer()
    
    def store_graph(self, fig_json):
        graph_id = str(uuid.uuid4())
        expiry_time = time.time() + self.expiry_seconds
        self.graphs[graph_id] = (fig_json, expiry_time)
        return graph_id
    
    def get_graph(self, graph_id):
        if graph_id not in self.graphs:
            return None
            
        fig_json, expiry_time = self.graphs[graph_id]
        
        if time.time() > expiry_time:
            del self.graphs[graph_id]
            return None
            
        return fig_json
    
    def _cleanup_expired(self):
        current_time = time.time()
        expired_ids = [gid for gid, (_, exp_time) in self.graphs.items() 
                       if current_time > exp_time]
        
        for gid in expired_ids:
            del self.graphs[gid]
    
    def _start_cleanup_timer(self):
        self._cleanup_expired()
        t = Timer(300, self._start_cleanup_timer)
        t.daemon = True
        t.start()

graph_store = GraphStore()