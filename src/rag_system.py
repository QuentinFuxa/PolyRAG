import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Tuple

import psycopg2
from psycopg2 import pool
from llmsherpa.readers import LayoutPDFReader
from dotenv import load_dotenv
from db_manager import DatabaseManager, schema_app_data

# Load environment variables
load_dotenv()

TS_QUERY_LANGUAGE = os.environ.get("TS_QUERY_LANGUAGE", "english")

class SearchStrategy(Enum):
    TEXT = "text"
    EMBEDDING = "embedding"
    HYBRID = "hybrid"

@dataclass
class BlockMetadata:
    """Metadata for a document block from llmsherpa"""
    block_idx: int
    level: int
    page_idx: int
    tag: str
    block_class: str
    bbox: List[float]
    
@dataclass
class DocumentBlock:
    """Representation of a document block with content and metadata"""
    # Using block_idx as the primary identifier instead of id string
    block_idx: int
    content: str
    metadata: BlockMetadata
    parent_idx: Optional[int] = None  # Changed from parent_id to parent_idx
    embedding: Optional[List[float]] = None

class EmbeddingManager:
    """Manager for embedding operations (optional)"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingManager, cls).__new__(cls)
            # Check if OpenAI API key is available
            cls._instance.api_key = os.getenv("OPENAI_API_KEY")
            cls._instance.enabled = cls._instance.api_key is not None
            if cls._instance.enabled:
                from openai import OpenAI
                cls._instance.client = OpenAI(api_key=cls._instance.api_key)
                cls._instance.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
                cls._instance.embedding_dim = 1536  # Default for Ada
        return cls._instance
    
    def is_enabled(self):
        return self.enabled
    
    def compute_embedding(self, text):
        """Compute embedding for a given text using OpenAI API"""
        if not self.enabled:
            return None
        
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.embedding_model
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error computing embedding: {e}")
            return None
    
    def get_embedding_dimension(self):
        """Returns the dimension of embeddings used"""
        return self.embedding_dim if self.enabled else 0

class SherpaDocumentProcessor:
    """Processor for rag_documents from llmsherpa"""
    
    def __init__(self, sherpa_api_url=None):
        self.sherpa_api_url = sherpa_api_url or os.getenv(
            "LLMSHERPA_API_URL", 
            "http://localhost:5010/api/parseDocument?renderFormat=all&useNewIndentParser=true"
        )
        self.pdf_reader = LayoutPDFReader(self.sherpa_api_url)
    
    def process_pdf(self, pdf_path):
        """Process a PDF file and return structured content"""
        """Process a PDF file using llmsherpa and return DocumentBlock objects"""
        sherpa_data = self.pdf_reader.read_pdf(pdf_path).json
        return self._process_sherpa_data(sherpa_data) # Call internal processing method

    def _process_sherpa_data(self, sherpa_data):
        """Process raw llmsherpa output (JSON) and convert to DocumentBlock objects"""
        blocks = []

        # Convert string to JSON if necessary
        if isinstance(sherpa_data, str):
            try:
                sherpa_data = json.loads(sherpa_data)
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON data from llmsherpa")
        
        # Process each block
        for block_data in sherpa_data:
            # Extract content from sentences
            content = ""
            if "sentences" in block_data:
                sentences = block_data["sentences"]
                if isinstance(sentences, list):
                    # Handle different sentence formats
                    if sentences and isinstance(sentences[0], dict) and "text" in sentences[0]:
                        content = " ".join(s["text"] for s in sentences)
                    elif sentences and isinstance(sentences[0], str):
                        content = " ".join(sentences)
            elif "table_rows" in block_data:
                # Handle table rows if present
                for row in block_data["table_rows"]:
                    try:
                        cells = row.get("cells", [])
                        if cells:
                            content += " | ".join(str(cell['cell_value']) for cell in cells) + "\n"
                        elif row.get("cell_value"):
                            content += str(row["cell_value"]) + "\n"
                    except KeyError:
                        raise ValueError("Invalid table row format")
            elif "name" in block_data:
                content = block_data["name"]
            else:
                raise ValueError("Invalid block data format")
            
            # Create metadata
            metadata = BlockMetadata(
                block_idx=block_data.get("block_idx", 0),
                level=block_data.get("level", 0),
                page_idx=block_data.get("page_idx", 0),
                tag=block_data.get("tag", "unknown"),
                block_class=block_data.get("block_class", ""),
                bbox=block_data.get("bbox", [0, 0, 0, 0])
            )
            
            # Create block - using block_idx directly as the identifier
            block = DocumentBlock(
                block_idx=metadata.block_idx,
                content=content,
                metadata=metadata
            )
            
            blocks.append(block)
        
        # Establish parent-child relationships based on hierarchy
        self._establish_hierarchy(blocks)
        
        return blocks
    
    def _establish_hierarchy(self, blocks):
        """Establish parent-child relationships based on level and position"""
        # Sort blocks by page and vertical position
        sorted_blocks = sorted(blocks, key=lambda b: (b.metadata.page_idx, b.metadata.bbox[1]))
        
        # Create a map for quick access
        block_map = {block.block_idx: block for block in blocks}
        
        # Stack to track the current hierarchy
        hierarchy_stack = []
        
        for block in sorted_blocks:
            # Pop from stack until we find a parent of higher level
            while hierarchy_stack and hierarchy_stack[-1].metadata.level >= block.metadata.level:
                hierarchy_stack.pop()
            
            # Set parent if available
            if hierarchy_stack:
                block.parent_idx = hierarchy_stack[-1].block_idx
            
            # Add current block to stack
            hierarchy_stack.append(block)


class PyMuPDFDocumentProcessor:
    """Processor for documents using PyMuPDF"""

    def process_pdf(self, pdf_path):
        import pymupdf
        """Process a PDF file using PyMuPDF and return DocumentBlock objects"""
        blocks = []
        try:
            doc = pymupdf.open(pdf_path)
        except Exception as e:
            print(f"Error opening PDF with PyMuPDF: {e}")
            return []

        block_counter = 0
        for page_num, page in enumerate(doc):
            page_dict = page.get_text("dict", flags=pymupdf.TEXTFLAGS_TEXT) # Get detailed block info
            page_width = page.rect.width
            page_height = page.rect.height

            for block_data in page_dict.get("blocks", []):
                # We are interested in text blocks ('type' == 0)
                if block_data.get("type") == 0:
                    block_text = ""
                    # Consolidate lines within the block
                    for line in block_data.get("lines", []):
                        for span in line.get("spans", []):
                            block_text += span.get("text", "") + " "
                        block_text += "\n" # Add newline after each line like llmsherpa might

                    block_text = block_text.strip()
                    if not block_text:
                        continue

                    bbox = block_data.get("bbox", [0, 0, 0, 0])

                    # Basic heuristic for tag (can be improved)
                    # Assuming larger fonts might be headers, but default to 'para'
                    tag = 'para'
                    if block_data.get("lines"):
                         first_line = block_data["lines"][0]
                         if first_line.get("spans"):
                             first_span = first_line["spans"][0]
                             if first_span.get("size", 10) > 14: # Arbitrary threshold for header
                                 tag = 'header'

                    metadata = BlockMetadata(
                        block_idx=block_counter,
                        level=0,  # PyMuPDF doesn't provide semantic levels easily
                        page_idx=page_num,
                        tag=tag, # Default tag, could add heuristics
                        block_class="", # PyMuPDF doesn't provide this
                        bbox=list(bbox)
                    )

                    block = DocumentBlock(
                        block_idx=block_counter,
                        content=block_text,
                        metadata=metadata,
                        parent_idx=None # PyMuPDF doesn't provide hierarchy easily
                    )
                    blocks.append(block)
                    block_counter += 1
        
        doc.close()
        # Note: PyMuPDF doesn't inherently provide hierarchy like llmsherpa.
        # The _establish_hierarchy method is specific to llmsherpa's output structure.
        return blocks


class RAGSystem:
    """RAG system with flexible search strategies and PDF backends"""

    def __init__(self, use_embeddings=False):
        self.db_manager = DatabaseManager()

        # Determine PDF parsing backend
        pdf_parser_backend = os.getenv("PDF_PARSER", "nlm-ingestor").lower()
        if pdf_parser_backend == "pymupdf":
            print("Using PyMuPDF backend for PDF processing.")
            self.processor = PyMuPDFDocumentProcessor()
        elif pdf_parser_backend == "nlm-ingestor":
            print("Using NLM Ingestor (llmsherpa) backend for PDF processing.")
            self.processor = SherpaDocumentProcessor()
        else:
            print(f"Warning: Unknown PDF_PARSER '{pdf_parser_backend}'. Defaulting to nlm-ingestor.")
            self.processor = SherpaDocumentProcessor()

        # Initialize embedding manager if enabled
        self.use_embeddings = use_embeddings
        if use_embeddings:
            self.embedding_manager = EmbeddingManager()
            if not self.embedding_manager.is_enabled():
                print("Warning: Embeddings requested but OpenAI API key not found. Falling back to text search.")
                self.use_embeddings = False
        
        # Create necessary database schema
        self._ensure_schema()
    
    def _ensure_schema(self):
        """Create necessary database schema if it doesn't exist"""
        # Base schema - updated to use block_idx as primary identifier
        schema = f"""        
        -- Document blocks table
        CREATE TABLE IF NOT EXISTS {schema_app_data}.rag_document_blocks (
            id SERIAL PRIMARY KEY,
            block_idx INTEGER NOT NULL,  -- This is now our primary block identifier
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
            parent_idx INTEGER,  -- Changed from parent_id to parent_idx
            content_tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('{TS_QUERY_LANGUAGE}', content)) STORED, -- Added generated TSVECTOR column
            UNIQUE(name, block_idx)  -- Changed unique constraint
        );
        
        -- Text search index on the generated column
        CREATE INDEX IF NOT EXISTS idx_document_blocks_content_tsv ON {schema_app_data}.rag_document_blocks 
        USING gin(content_tsv);
        """
        
        self.db_manager.execute_query(schema)
        
        # Add embedding column and index if embeddings are enabled
        if self.use_embeddings:
            try:
                # Check if vector extension is available
                extension_query = "SELECT 1 FROM pg_extension WHERE extname = 'vector'"
                result = self.db_manager.execute_query(extension_query)
                
                if not result:
                    # Create vector extension if not exists
                    self.db_manager.execute_query("CREATE EXTENSION IF NOT EXISTS vector")
                
                # Add embedding column if not exists
                col_exists_query = f"""
                SELECT 1 FROM information_schema.columns 
                WHERE table_name = 'rag_document_blocks' 
                AND table_schema = '{schema_app_data}'
                AND column_name = 'embedding'
                """
                col_exists = self.db_manager.execute_query(col_exists_query)
                
                if not col_exists:
                    dim = self.embedding_manager.get_embedding_dimension()
                    embedding_schema = f"""
                    ALTER TABLE {schema_app_data}.rag_document_blocks ADD COLUMN IF NOT EXISTS embedding vector({dim});
                    CREATE INDEX IF NOT EXISTS idx_document_blocks_embedding ON {schema_app_data}.rag_document_blocks 
                    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
                    """
                    self.db_manager.execute_query(embedding_schema)
            except Exception as e:
                print(f"Warning: Could not create vector extension or embedding column: {e}")
                print("Falling back to text search only.")
                self.use_embeddings = False
    
    def index_document(self, pdf_path, document_name_override: Optional[str] = None, existing_sherpa_data=None, table_name=f"{schema_app_data}.rag_document_blocks"):
        """
        Index a PDF document from a given path.
        Uses pdf_path as the unique identifier (name) by default, but can be overridden.
        Optionally accepts pre-processed sherpa data.

        Args:
            pdf_path: Path to the PDF file to process.
            document_name_override: If provided, use this string as the unique 'name' identifier
                                     when storing blocks, instead of pdf_path. Useful for URLs
                                     where the path is temporary.
            existing_sherpa_data: Pre-processed data from llmsherpa (optional).
            table_name: The database table to insert blocks into.
        """
        document_name = document_name_override if document_name_override is not None else pdf_path

        # Process the document using the selected processor
        # existing_sherpa_data is specific to llmsherpa, so we ignore it if using pymupdf
        pdf_parser_backend = os.getenv("PDF_PARSER", "nlm-ingestor").lower()
        if pdf_parser_backend == "pymupdf":
             blocks = self.processor.process_pdf(pdf_path)
        elif pdf_parser_backend == "nlm-ingestor":
             if existing_sherpa_data is None:
                 # Sherpa processor's process_pdf now returns blocks directly
                 blocks = self.processor.process_pdf(pdf_path)
             else:
                 # If sherpa data is provided, process it (specific to sherpa)
                 blocks = self.processor._process_sherpa_data(existing_sherpa_data) # Use renamed internal method
        else: # Default case
             blocks = self.processor.process_pdf(pdf_path)

        # Compute embeddings if enabled
        if self.use_embeddings:
            for block in blocks:
                if block.content:
                    block.embedding = self.embedding_manager.compute_embedding(block.content)
        
        # Insert blocks using the document_name (pdf_path)
        self._insert_blocks(document_name, blocks, table_name)
        
        return document_name # Return the name used for indexing
    
    def _insert_blocks(self, name, blocks, table_name=f"{schema_app_data}.rag_document_blocks"):
        """Insert blocks into database"""
        params_list = []
        
        for block in blocks:
            # Extract bbox coordinates
            bbox = block.metadata.bbox
            x0 = bbox[0] if len(bbox) > 0 else None
            y0 = bbox[1] if len(bbox) > 1 else None
            x1 = bbox[2] if len(bbox) > 2 else None
            y1 = bbox[3] if len(bbox) > 3 else None
            
            # Create parameter tuple - now using block_idx directly 
            params = (
                block.block_idx,  # Use block_idx as primary identifier
                name,
                block.content,
                block.metadata.level,
                block.metadata.page_idx,
                block.metadata.tag,
                block.metadata.block_class,
                x0, y0, x1, y1,
                block.parent_idx  # Changed from parent_id to parent_idx
            )
            
            params_list.append(params)
        
        # Insert all blocks - schema updated for block_idx
        if self.use_embeddings:
            query = f"""
            INSERT INTO {table_name}
            (block_idx, name, content, level, page_idx, tag, block_class, 
             x0, y0, x1, y1, parent_idx, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (name, block_idx) DO UPDATE SET
            content = EXCLUDED.content,
            level = EXCLUDED.level,
            page_idx = EXCLUDED.page_idx,
            tag = EXCLUDED.tag,
            block_class = EXCLUDED.block_class,
            x0 = EXCLUDED.x0,
            y0 = EXCLUDED.y0,
            x1 = EXCLUDED.x1,
            y1 = EXCLUDED.y1,
            parent_idx = EXCLUDED.parent_idx,
            embedding = EXCLUDED.embedding
            """
            
            # Add embedding to params
            params_list = [(p + (block.embedding,)) for p, block in zip(params_list, blocks)]
        else:
            query = f"""
            INSERT INTO {table_name}
            (block_idx, name, content, level, page_idx, tag, block_class, 
             x0, y0, x1, y1, parent_idx)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (name, block_idx) DO UPDATE SET
            content = EXCLUDED.content,
            level = EXCLUDED.level,
            page_idx = EXCLUDED.page_idx,
            tag = EXCLUDED.tag,
            block_class = EXCLUDED.block_class,
            x0 = EXCLUDED.x0,
            y0 = EXCLUDED.y0,
            x1 = EXCLUDED.x1,
            y1 = EXCLUDED.y1,
            parent_idx = EXCLUDED.parent_idx
            """
        
        self.db_manager.execute_many(query, params_list)
    
    def search(
            self,
            query,
            source_names=None,
            get_children=False,
            get_parents=False,
            strategy=SearchStrategy.TEXT,
            limit=5
            ):
        """
        Search rag_documents using specified strategy
        
        Args:
            query: The search query
            source_names: List of document names to search in
            get_children: Whether to include children of matching blocks
            get_parents: Whether to include parents of matching blocks
            strategy: SearchStrategy (TEXT, EMBEDDING, or HYBRID)
            limit: Maximum number of results to return
            
        Returns:
            List of relevant blocks with metadata
        """
        if strategy == SearchStrategy.EMBEDDING and not self.use_embeddings:
            print("Warning: Embedding search requested but embeddings not enabled. Falling back to text search.")
            strategy = SearchStrategy.TEXT
        
        if strategy == SearchStrategy.HYBRID and not self.use_embeddings:
            print("Warning: Hybrid search requested but embeddings not enabled. Falling back to text search.")
            strategy = SearchStrategy.TEXT
        
        if strategy == SearchStrategy.TEXT:
            return self._text_search(query, source_names, limit)
        elif strategy == SearchStrategy.EMBEDDING:
            return self._embedding_search(query, source_names, limit)
        elif strategy == SearchStrategy.HYBRID:
            return self._hybrid_search(query, source_names, limit)
        else:
            raise ValueError(f"Unknown search strategy: {strategy}")
        
    def _text_search(self, query: List[str], source_names: Optional[List[str]], limit: int = 5):
        """
        Search using PostgreSQL full-text search with AND/OR fallback logic.
        First tries to find results where all terms match (AND), 
        then falls back to results where any term matches (OR) if AND yields nothing.
        """
        
        # Helper function to build the base query and common parts
        def build_search_query(ts_query_str: str, source_names: Optional[List[str]]) -> Tuple[str, List[Any]]:
            base_query = f"""
            SELECT 
                name, 
                block_idx,
                content, 
                page_idx, 
                level,
                tag,
                block_class,
                x0, 
                y0, 
                x1, 
                y1,
                parent_idx,
                ts_rank_cd(content_tsv, to_tsquery('{TS_QUERY_LANGUAGE}', %s)) AS score
            FROM 
                {schema_app_data}.rag_document_blocks
            WHERE 
                content_tsv @@ to_tsquery('{TS_QUERY_LANGUAGE}', %s)
            """
            params = [ts_query_str, ts_query_str]

            # Add source filter if specified
            if source_names and len(source_names) > 0:
                placeholders = ', '.join(['%s'] * len(source_names))
                base_query += f" AND name IN ({placeholders})"
                params.extend(source_names)
            
            # Add order by and limit
            base_query += """
            ORDER BY 
                score DESC
            LIMIT %s
            """
            params.append(limit)
            return base_query, params

        # 1. Prepare base formatted elements (handle spaces within terms)
        formatted_elements = []
        for element in query:
            # Sanitize element: remove potential tsquery operators to avoid injection
            sanitized_element = element.replace('&', '').replace('|', '').replace('!', '').replace('(', '').replace(')', '').strip()
            if not sanitized_element:
                continue
            if ' ' in sanitized_element:
                # If element contains spaces, treat it as a phrase search within the term
                formatted_element = ' & '.join(sanitized_element.split())
                formatted_elements.append(f"({formatted_element})")
            else:
                formatted_elements.append(sanitized_element)
        
        if not formatted_elements:
            print("Warning: No valid query terms after sanitization.")
            return []

        # 2. Try AND search first
        ts_query_and = " & ".join(formatted_elements)
        print(f"Attempting AND search with ts_query: {ts_query_and}")
        search_query_and, params_and = build_search_query(ts_query_and, source_names)
        results = self.db_manager.execute_query(search_query_and, params_and)

        # 3. If AND search yields results, return them
        if results:
            print("AND search successful.")
            return self._format_results(results)
        
        # 4. If AND search yields no results, try OR search
        print("AND search yielded no results. Falling back to OR search.")
        ts_query_or = " | ".join(formatted_elements)
        print(f"Attempting OR search with ts_query: {ts_query_or}")
        search_query_or, params_or = build_search_query(ts_query_or, source_names)
        results = self.db_manager.execute_query(search_query_or, params_or)

        # 5. Return results from OR search (might still be empty)
        if results:
            print("OR search successful.")
        else:
            print("OR search also yielded no results.")
        return self._format_results(results)

    def _embedding_search(self, query, source_names, limit=5):
        """Search using vector embeddings"""
        if not self.use_embeddings:
            return [] # Correctly return empty list if embeddings are not used
            
        embedding = self.embedding_manager.compute_embedding(query)
        
        if not embedding:
            return []
        
        # Updated search query for block_idx
        search_query = f"""
        SELECT 
            name, 
            block_idx,
            content, 
            page_idx, 
            level,
            tag,
            block_class,
            x0, 
            y0, 
            x1, 
            y1,
            parent_idx,
            1 - (embedding <=> %s) AS score
        FROM 
            {schema_app_data}.rag_document_blocks
        WHERE
            embedding IS NOT NULL
        """
        
        params = [embedding]
        
        # Add source filter if specified
        if source_names and len(source_names) > 0:
            placeholders = ', '.join(['%s'] * len(source_names))
            search_query += f" AND name IN ({placeholders})"
            params.extend(source_names)
        
        # Add order by and limit
        search_query += """
        ORDER BY 
            embedding <=> %s
        LIMIT %s
        """
        params.extend([embedding, limit])
        
        results = self.db_manager.execute_query(search_query, params)
        return self._format_results(results)
    
    def _hybrid_search(self, query, source_names, limit=5):
        """Hybrid search combining text search and embedding search"""
        text_results = self._text_search(query, source_names, limit)
        embedding_results = self._embedding_search(query, source_names, limit)
        
        # Combine and deduplicate results - using block_idx as the key
        combined = {}
        for result in text_results:
            block_idx = result["block_idx"]
            combined[block_idx] = result
        
        for result in embedding_results:
            block_idx = result["block_idx"]
            if block_idx in combined:
                # Average the scores
                combined[block_idx]["score"] = (combined[block_idx]["score"] + result["score"]) / 2
            else:
                combined[block_idx] = result
        
        # Sort by score and limit
        sorted_results = sorted(combined.values(), key=lambda x: x["score"], reverse=True)
        return sorted_results[:limit]
    
    def _format_results(self, results):
        """Format database results into a structured format"""
        if not results:
            return []
        
        formatted = []
        for row in results:
            formatted.append({
                "name": row[0],
                "block_idx": row[1],  # Now using block_idx instead of block_id
                "content": row[2],
                "page_idx": row[3],
                "level": row[4],
                "tag": row[5],
                "block_class": row[6],
                "x0": row[7],
                "y0": row[8],
                "x1": row[9],
                "y1": row[10],
                "parent_idx": row[11],  # Changed from parent_id to parent_idx
                "score": row[12]
            })
        
        return formatted
    
    def get_context(self, block_result, context_size=3):
        """
        Get contextual blocks for a given block
        
        Args:
            block_result: A block result from search
            context_size: Number of contextual blocks to include
            
        Returns:
            List of blocks including the original and its context
        """
        context = [block_result]
        
        # Get parent blocks
        if block_result["parent_idx"]:
            parent = self._get_block(block_result["parent_idx"], block_result["name"])
            if parent:
                context.append(parent)
                
                # Get grandparent if exists
                if parent["parent_idx"]:
                    grandparent = self._get_block(parent["parent_idx"], parent["name"])
                    if grandparent:
                        context.append(grandparent)
        
        # Get siblings (blocks with the same parent, same level, nearby positions)
        siblings = self._get_siblings(
            block_result["name"],
            block_result["page_idx"],
            block_result["level"],
            block_result["block_idx"],
            block_result["parent_idx"],
            limit=context_size
        )
        context.extend(siblings)
        
        # Get children if this is a header or section
        if block_result["tag"] in ["header", "title", "section"]:
            children = self._get_children(block_result["block_idx"], block_result["name"], limit=context_size)
            context.extend(children)
        
        # Sort by page and position
        context.sort(key=lambda x: (x["page_idx"], x["level"], x.get("y0", 0)))
        
        return context
    
    def _get_block(self, block_idx, name):
        """Get a specific block by its block_idx"""
        query = f"""
        SELECT 
            id,
            block_idx,
            content, 
            name,
            page_idx, 
            level,
            tag,
            block_class,
            x0, 
            y0, 
            x1, 
            y1,
            parent_idx
        FROM 
            {schema_app_data}.rag_document_blocks
        WHERE 
            block_idx = %s AND name = %s
        """
        
        results = self.db_manager.execute_query(query, (block_idx, name))
        if not results:
            return None
        
        row = results[0]
        return {
            "id": row[0],
            "block_idx": row[1],
            "content": row[2],
            "name": row[3],
            "page_idx": row[4],
            "level": row[5],
            "tag": row[6],
            "block_class": row[7],
            "x0": row[8],
            "y0": row[9],
            "x1": row[10],
            "y1": row[11],
            "parent_idx": row[12],
            "score": 1.0  # Default score for context blocks
        }
    
    def _get_siblings(self, name, page_idx, level, block_idx, parent_idx, limit=3):
        """Get sibling blocks (same parent, same level, nearby positions)"""
        query = f"""
        SELECT 
            id,
            block_idx,
            content, 
            name,
            page_idx, 
            level,
            tag,
            block_class,
            x0, 
            y0, 
            x1, 
            y1,
            parent_idx
        FROM 
            {schema_app_data}.rag_document_blocks
        WHERE 
            name = %s
            AND page_idx = %s
            AND level = %s
            AND block_idx != %s
            AND (
                (parent_idx = %s) OR
                (%s IS NULL AND parent_idx IS NULL)
            )
            AND (block_idx BETWEEN %s - %s AND %s + %s)
        ORDER BY
            ABS(block_idx - %s)
        LIMIT %s
        """
        
        results = self.db_manager.execute_query(
            query, 
            (
                name, 
                page_idx, 
                level, 
                block_idx,
                parent_idx, parent_idx,
                block_idx, limit, block_idx, limit,
                block_idx,
                limit
            )
        )
        
        if not results:
            return []
            
        return [
            {
                "id": row[0],
                "block_idx": row[1],
                "content": row[2],
                "name": row[3],
                "page_idx": row[4],
                "level": row[5],
                "tag": row[6],
                "block_class": row[7],
                "x0": row[8],
                "y0": row[9],
                "x1": row[10],
                "y1": row[11],
                "parent_idx": row[12],
                "score": 1.0  # Default score for context blocks
            }
            for row in results
        ]
    
    def _get_children(self, parent_idx, name, limit=5):
        """Get child blocks for a given parent"""
        query = f"""
        SELECT 
            id,
            block_idx,
            content, 
            name,
            page_idx, 
            level,
            tag,
            block_class,
            x0, 
            y0, 
            x1, 
            y1,
            parent_idx
        FROM 
            {schema_app_data}.rag_document_blocks
        WHERE 
            parent_idx = %s
            AND name = %s
        ORDER BY
            page_idx, block_idx
        LIMIT %s
        """
        
        results = self.db_manager.execute_query(query, (parent_idx, name, limit))
        
        if not results:
            return []
            
        return [
            {
                "id": row[0],
                "block_idx": row[1],
                "content": row[2],
                "name": row[3],
                "page_idx": row[4],
                "level": row[5],
                "tag": row[6],
                "block_class": row[7],
                "x0": row[8],
                "y0": row[9],
                "x1": row[10],
                "y1": row[11],
                "parent_idx": row[12],
                "score": 1.0  # Default score for context blocks
            }
            for row in results
        ]
    
    def get_pdf_highlights(self, context_blocks):
        """
        Prepare highlighting annotations for a PDF based on context blocks
        
        Returns:
            List of annotation objects compatible with the graph display
        """
        annotations = []
        
        # Process each block
        for block in context_blocks:
            # Skip blocks without coordinates
            if block["x0"] is None or block["y0"] is None or block["x1"] is None or block["y1"] is None:
                continue
            
            # Create highlight annotation in the required format
            annotation = {
                "page": block["page_idx"] + 1,  # Convert to 1-indexed page numbering
                "x": block["x0"],
                "y": block["y0"],
                "height": block["y1"] - block["y0"],
                "width": block["x1"] - block["x0"],
                "color": "red",
            }
            
            annotations.append(annotation)
        
        return annotations
    
    def query(self, 
              user_query,              
              source_names=None,
              get_children=False,
              get_parents=False,
              strategy=SearchStrategy.TEXT,
              num_results=3
              ):
        """
        Process a user query and return relevant information with PDF highlights
        
        Args:
            user_query: The user's question
            source_names: List of document names to search in
            get_children: Whether to include children of matching blocks
            get_parents: Whether to include parents of matching blocks
            strategy: Search strategy to use
            num_results: Number of top results to process
            
        Returns:
            Dictionary with query results and highlight annotations
        """
        # Search for relevant blocks
        search_results = self.search(
            user_query, 
            source_names, 
            get_children=get_children,
            get_parents=get_parents,
            strategy=strategy, 
            limit=max(num_results * 2, 5)  # Get more results than needed to ensure diversity
        )

        warning_message = None

        if not search_results and source_names:
            placeholders = ', '.join(['%s'] * len(source_names))
            check_query = f"""
            SELECT DISTINCT name 
            FROM {schema_app_data}.rag_document_blocks 
            WHERE name IN ({placeholders})
            """
            existing_sources_tuples = self.db_manager.execute_query(check_query, source_names)
            existing_sources_set = {row[0] for row in existing_sources_tuples} if existing_sources_tuples else set()
            requested_sources_set = set(source_names)
            missing_sources = sorted(list(requested_sources_set - existing_sources_set))

            if not missing_sources:
                # All requested sources exist, but the query yielded no results within them.
                return {
                    "success": False,
                    "message": "No relevant information found for the query in the specified document(s)."
                }
            else:
                print(f"Warning: Source document(s) not found: {', '.join(missing_sources)}. Retrying search across all documents.")
                warning_message = f"Source document(s) not found: {', '.join(missing_sources)}. Showing results from all documents."
                
                search_results = self.search(
                    user_query, 
                    source_names=None, # Search all documents
                    get_children=get_children,
                    get_parents=get_parents,
                    strategy=strategy, 
                    limit=max(num_results * 2, 5) 
                )

        if not search_results:
             return {
                 "success": False,
                 "message": "No relevant information found in the documents."
             }
        
        # Process multiple top results to get better coverage
        top_results = search_results[:num_results]
        
        # Get context for each result
        all_context_blocks = []
        for result in top_results:
            context_blocks = self.get_context(result)
            all_context_blocks.extend(context_blocks)
        
        # Deduplicate context blocks by block_idx
        unique_blocks = {}
        for block in all_context_blocks:
            key = f"{block['name']}:{block['block_idx']}"
            if key not in unique_blocks or block.get('score', 0) > unique_blocks[key].get('score', 0):
                unique_blocks[key] = block
        
        # Convert back to list
        context_blocks = list(unique_blocks.values())
        
        # Sort by page and position
        context_blocks.sort(key=lambda x: (x["page_idx"], x["level"], x.get("y0", 0) or 0))
        
        # Prepare PDF highlighting annotations
        annotations = self.get_pdf_highlights(context_blocks)
        
        # Extract text from context blocks
        context_text = ""
        for block in context_blocks:
            if block["tag"] == "header":
                context_text += f"\n## {block['content']}\n"
            elif block["tag"] == "list_item":
                context_text += f"- {block['content']}\n"
            else:
                context_text += f"{block['content']}\n\n"
        
        final_result = {
            "success": True,
            "context": context_text,
            "all_results": search_results[:num_results],  # Return multiple results instead of just the top one
            "top_result": search_results[0],  # Keep the top result for backward compatibility
            "context_blocks": context_blocks,
            "annotations": annotations,
        }

        if warning_message:
            final_result["warning"] = warning_message
            
        return final_result
    
    def get_blocks_by_idx(self, block_indices, source_name=None, get_children=False):
        """
        Get blocks by their block_idx values
        
        Args:
            block_indices: List of block_idx values to retrieve
            source_name: Name of the document to search in (optional)
            get_children: Whether to also get children of these blocks
            
        Returns:
            List of blocks matching the requested block indices
        """
        if not block_indices:
            return []
        
        # Convert single index to list
        if not isinstance(block_indices, list):
            block_indices = [block_indices]
        
        placeholders = ', '.join(['%s'] * len(block_indices))
        query = f"""
        SELECT 
            id,
            block_idx,
            content, 
            name,
            page_idx, 
            level,
            tag,
            block_class,
            x0, 
            y0, 
            x1, 
            y1,
            parent_idx
        FROM 
            {schema_app_data}.rag_document_blocks
        WHERE 
            block_idx IN ({placeholders})
        """
        
        params = block_indices.copy()
        
        # Add source filter if specified
        if source_name:
            query += " AND name = %s"
            params.append(source_name)
        
        results = self.db_manager.execute_query(query, params)
        
        if not results:
            return []
            
        blocks = [
            {
                "id": row[0],
                "block_idx": row[1],
                "content": row[2],
                "name": row[3],
                "page_idx": row[4],
                "level": row[5],
                "tag": row[6],
                "block_class": row[7],
                "x0": row[8],
                "y0": row[9],
                "x1": row[10],
                "y1": row[11],
                "parent_idx": row[12],
                "score": 1.0  # Default score for directly retrieved blocks
            }
            for row in results
        ]
        
        # If get_children is True, also get children for each block
        if get_children:
            all_blocks = blocks.copy()
            for block in blocks:
                children = self._get_children(block["block_idx"], block["name"])
                all_blocks.extend(children)
            blocks = all_blocks
        
        return blocks
    
    def get_annotations_by_indices(self, pdf_file, block_indices):
        """
        Convert block indices to PDF annotation objects for highlighting
        
        Args:
            pdf_file: Name of the PDF file 
            block_indices: List of block indices to highlight
            
        Returns:
            List of annotation objects in the format needed for highlighting
        """
        if not block_indices:
            return []
        
        # Retrieve blocks by their indices
        blocks = self.get_blocks_by_idx(block_indices, source_name=pdf_file)
        
        if not blocks:
            return []
        
        # Convert blocks to annotation format
        annotations = []
        for block in blocks:
            # Skip blocks without coordinates
            if block["x0"] is None or block["y0"] is None or block["x1"] is None or block["y1"] is None:
                continue
            
            # Create annotation object in the required format
            annotation = {
                "page": block["page_idx"] + 1,  # Convert to 1-indexed page numbering
                "x": block["x0"],
                "y": block["y0"],
                "height": block["y1"] - block["y0"],
                "width": block["x1"] - block["x0"],
                "color": "red"
            }
            
            annotations.append(annotation)
        
        return annotations
    
    def debug_blocks(self, pdf_file):
        """
        Display all blocks in the PDF for debugging purposes.
        
        Args:
            pdf_file: Name of the PDF file (without path or extension)
            
        Returns:
            List of annotation objects in the format needed for highlighting
        """

        rag_check_query = f"""
        SELECT COUNT(*) FROM {schema_app_data}.rag_document_blocks WHERE name = %s
        """
        rag_results = self.db_manager.execute_query(rag_check_query, (pdf_file,))
        
        upload_check_query = f"""
        SELECT COUNT(*) FROM {schema_app_data}.uploaded_document_blocks WHERE name = %s
        """
        upload_results = self.db_manager.execute_query(upload_check_query, (pdf_file,))
        
        rag_count = rag_results[0][0] if rag_results else 0
        upload_count = upload_results[0][0] if upload_results else 0
        
        table_name = f"{schema_app_data}.rag_document_blocks" if rag_count > 0 else f"{schema_app_data}.uploaded_document_blocks"
        
        if rag_count == 0 and upload_count == 0:
            table_name = f"{schema_app_data}.uploaded_document_blocks"
            print(f"No blocks found for {pdf_file} in either table. Defaulting to {table_name}.")
        else:
            print(f"Found {rag_count} blocks in rag_document_blocks and {upload_count} blocks in uploaded_document_blocks. Using {table_name}.")
        
        query = f"""
        SELECT 
            id,
            block_idx,
            content, 
            name,
            page_idx, 
            level,
            tag,
            block_class,
            x0, 
            y0, 
            x1, 
            y1,
            parent_idx
        FROM 
            {table_name}
        WHERE 
            name = %s
        """
        
        params = (pdf_file,)
        results = self.db_manager.execute_query(query, params)
        
        blocks = [
            {
                "id": row[0],
                "block_idx": row[1],
                "content": row[2],
                "name": row[3],
                "page_idx": row[4],
                "level": row[5],
                "tag": row[6],
                "block_class": row[7],
                "x0": row[8],
                "y0": row[9],
                "x1": row[10],
                "y1": row[11],
                "parent_idx": row[12],
                "score": 1.0  # Default score for directly retrieved blocks
            }
            for row in results
        ]
        
        dict_colors = {
            'para': 'blue',
            'header': 'red',
            'list_item': 'green',
            'table': 'purple',
        }
        
        annotations = []
        for block in blocks:
            annotation = {
                "page": block["page_idx"] + 1,
                "x": block["x0"],
                "y": block["y0"],
                "height": block["y1"] - block["y0"],
                "width": block["x1"] - block["x0"],
                "color": dict_colors[block["tag"]]
            }
            annotations.append(annotation)
        
        return annotations
