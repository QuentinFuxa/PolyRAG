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

    def query(self,
              user_query: Union[str, List[str]],
              source_query: Optional[str] = None,
              source_names: Optional[List[str]] = None,
              limit: int = 20,
              offset: int = 0,
              get_children: bool = True,
              content_type: Optional[str] = None,
              section_filter: Optional[List[str]] = None,
              demand_priority: Optional[int] = None,
              count_only: bool = False
              ) -> Dict[str, Any]:
        """
        Process a user query with simplified return format and single-query source handling.
        """
        # Ensure user_query is a list
        if isinstance(user_query, str):
            processed_query = user_query.split()
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
                ts_rank_cd(content_tsv, to_tsquery('{TS_QUERY_LANGUAGE}', %s)) AS score,
                CASE WHEN tag = 'header' THEN 1 ELSE 0 END AS is_header
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
                is_header DESC,
                level DESC,
                score DESC
            LIMIT %s
            """
            params.append(limit)
            return base_query, params

        # 1. Prepare base formatted elements (handle spaces within terms)
        formatted_elements = []
        for element in query:
            # Sanitize element: remove potential tsquery operators to avoid injection
            sanitized_element = element.replace('|', '').replace('!', '').replace('(', '').replace(')', '').strip()
            if not sanitized_element:
                continue
            if ' ' in sanitized_element:
                # If element contains spaces, treat it as a phrase search within the term
                formatted_element = ' & '.join(sanitized_element.split())
                formatted_elements.append(f"({formatted_element})")
            else:
                _after_from = _from_parts[1]
                
                # Extract table name (part after FROM, before WHERE or ORDER BY or LIMIT or OFFSET)
                _table_name_part = re.split(r'\sWHERE\s|\sORDER BY\s|\sLIMIT\s|\sOFFSET\s', _after_from, 1, re.IGNORECASE)[0].strip()
                table_name = _table_name_part

                # Extract WHERE clause specific to the source_query
                _where_match = re.search(r'\sWHERE\s(.*?)(?:\sORDER BY\s|\sLIMIT\s|\sOFFSET\s|$)', _after_from, re.IGNORECASE | re.DOTALL)
                raw_where_from_source = _where_match.group(1).strip() if _where_match else ""

                # Extract ORDER BY clause specific to the source_query
                _orderby_match = re.search(r'\sORDER BY\s(.*?)(?:\sLIMIT\s|\sOFFSET\s|$)', _after_from, re.IGNORECASE | re.DOTALL)
                raw_orderby_from_source = _orderby_match.group(1).strip() if _orderby_match else ""

                if table_name:
                    join_clause_str = f' JOIN {table_name} js ON r.name = js.name'

                    if raw_where_from_source:
                        prefixed_where = _prefix_columns_in_where_clause(raw_where_from_source, "js.")
                        if prefixed_where:
                            where_join_conditions_list.append(f"({prefixed_where})")
                    
                    if raw_orderby_from_source:
                        order_by_join_clause_str = _prefix_columns_in_order_by_clause(raw_orderby_from_source, "js.")
        
        # Define base SELECT clause based on FTS
        if formatted_elements:
            ts_query_for_format = " & ".join(formatted_elements) # Used for FTS AND search
            select_base = f"""
            SELECT 
                r.name, r.block_idx, r.content, r.level, r.tag,
                r.content_type, r.section_type, r.demand_priority, r.parent_idx,
                ts_rank_cd(r.content_tsv, to_tsquery('{self.TS_QUERY_LANGUAGE}','{{ts_query}}')) AS score
            """
        else:
            select_base = f"""
            SELECT 
                r.name, r.block_idx, r.content, r.level, r.tag,
                r.content_type, r.section_type, r.demand_priority, r.parent_idx
            """
        
        from_r_clause = f"FROM {schema_app_data}.rag_document_blocks r"

        # Build search_query parts
        search_query_parts = [select_base, from_r_clause]
        if join_clause_str:
            search_query_parts.append(join_clause_str)

        # Build count_query parts
        count_query_parts = [f"SELECT COUNT(*) FROM {schema_app_data}.rag_document_blocks r"]
        if join_clause_str:
            count_query_parts.append(join_clause_str)
        
        where_conditions = []
        if formatted_elements:
            # ts_query variable for formatting is ts_query_for_format
            where_conditions.append(f"r.content_tsv @@ to_tsquery('{self.TS_QUERY_LANGUAGE}', '{{ts_query}}')")

        # Add conditions from source_query's WHERE clause
        if where_join_conditions_list:
            where_conditions.extend(where_join_conditions_list)
        
        # Original conditions (source_names, content_type, etc.)
        # Ensure source_names is mutually exclusive with source_query logic for joins
        if not source_query and source_names: # Only apply if source_query was not used
            str_source_names = "', '".join(source_names)
            where_conditions.append(f"r.name IN ('{str_source_names}')")
        
        if content_type:
            where_conditions.append(f"r.content_type = '{content_type}'")
        if section_filter:
            str_section_filter = "', '".join(section_filter)
            where_conditions.append(f"r.section_type IN ('{str_section_filter}')")
        if demand_priority is not None:
            where_conditions.append(f"r.demand_priority = {demand_priority}")

        if where_conditions:
            where_clause_full_str = " WHERE " + " AND ".join(where_conditions)
            search_query_parts.append(where_clause_full_str)
            count_query_parts.append(where_clause_full_str)
        
        # ORDER BY logic
        final_order_by_clauses = []
        if order_by_join_clause_str:
            final_order_by_clauses.append(order_by_join_clause_str)
        
        if formatted_elements and (not order_by_join_clause_str or "score" not in order_by_join_clause_str.lower()):
            final_order_by_clauses.append("score DESC")
        
        # Add r.level DESC if not already covered by order_by_join_clause_str or if no FTS
        if not order_by_join_clause_str or "r.level" not in order_by_join_clause_str.lower():
             final_order_by_clauses.append("r.level DESC")
        
        # Fallback if no other ordering is present and no FTS
        if not formatted_elements and not final_order_by_clauses:
            final_order_by_clauses.append("r.block_idx")


        if final_order_by_clauses:
            # Filter out potential empty strings if some clauses were conditionally not added
            valid_clauses = [clause for clause in final_order_by_clauses if clause and clause.strip()]
            if valid_clauses:
                 search_query_parts.append(" ORDER BY " + ", ".join(valid_clauses))
            
        search_query = "\n".join(search_query_parts)
        count_query = "\n".join(count_query_parts)

        # Execute queries
        _final_count_query = count_query.format(ts_query=ts_query_for_format)
        count_result = self.db_manager.execute_query(_final_count_query)
        
        _final_search_query = search_query.format(ts_query=ts_query_for_format)
        _final_search_query += f" LIMIT {limit} OFFSET {offset}"
        if not count_only:
            results = self.db_manager.execute_query(_final_search_query)
        
        total_count_to_return = count_result[0][0] if count_result and count_result[0] else 0

        # If AND search returns no results, try OR search
        if total_count_to_return == 0 and formatted_elements:
            ts_query_or_for_format = " | ".join(formatted_elements) # OR version for FTS
            
            _final_count_query_or = count_query.format(ts_query=ts_query_or_for_format)
            count_result_or = self.db_manager.execute_query(_final_count_query_or)
            # Update total_count_to_return with the count from the OR search
            total_count_to_return = count_result_or[0][0] if count_result_or and count_result_or[0] else 0

            _final_search_query_or = search_query.format(ts_query=ts_query_or_for_format)
            _final_search_query_or += f" LIMIT {limit} OFFSET {offset}"
            if not count_only:
                results = self.db_manager.execute_query(_final_search_query_or)
        
        # total_count = count_result[0][0] if count_result else 0 # This line is now handled by total_count_to_return
        if count_only:
            return {
                "total_number_results": total_count_to_return,
            }
        
        # Format results
        formatted_results = []
        for row in results:
            result = {
                "document_name": row[0],
                "idx": row[1],
                "content": row[2],
                "level": row[3],
                "tag": row[4],
                "content_type": row[5],
                "section_type": row[6],
                "demand_priority": row[7],
                "parent_idx": row[8]
            }
            
            # Add children if requested
            if get_children:
                children = self._get_children(row[1], row[0])
                result["children"] = [
                    {
                        "idx": c["block_idx"],
                        "content": c["content"],
                        "level": c["level"],
                        "tag": c["tag"],
                        "parent_idx": c["parent_idx"]
                    } for c in children
                ]
            else:
                result["children"] = []
            
            formatted_results.append(result)

        return {
            "total_number_results": total_count_to_return,
            "number_returned_results": len(formatted_results),
            "results": formatted_results
        }
    
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
              user_query: Union[str, List[str]], # Accept string or list for query
              source_names: Optional[List[str]] = None,
              max_results_per_source: int = 3,
              get_children: bool = True,
              strategy: SearchStrategy = SearchStrategy.TEXT # Allow specifying strategy
              ) -> Dict[str, Any]:
        """
        Process a user query, group results by document, limit results per document,
        and optionally fetch children.

        Args:
            user_query: The user's question (string or list of keywords).
            source_names: List of document names to search within (optional).
            max_results_per_source: Max number of top results per document (default: 3).
            get_children: Whether to fetch child blocks for the results (default: True).
            strategy: Search strategy to use (TEXT, EMBEDDING, HYBRID).

        Returns:
            Dictionary containing:
                - success (bool): True if results were found.
                - results (list): List of dictionaries, one per document, containing:
                    - document_name (str)
                    - results (list): Top N results for the document.
                    - other_result_idx (list): Indices of remaining results for the document.
                - message (str, optional): Error or warning message.
        """
        # Ensure user_query is a list for internal processing consistency
        if isinstance(user_query, str):
            # Basic split, might need refinement depending on expected query format
            processed_query = user_query.split() 
        else:
            processed_query = user_query

        # Determine the overall limit for the initial search.
        # Fetch more initially to allow for better distribution across documents.
        initial_limit = max(max_results_per_source * (len(source_names) if source_names else 5), 15)

        search_results = self.search(
            processed_query, # Use processed query list
            source_names,
            strategy=strategy,
            limit=initial_limit
        )

        warning_message = None
        final_results_by_doc = {}

        # Handle case where specific sources were requested but yielded no results
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

            # Check if the requested source documents actually exist in the DB
            placeholders = ', '.join(['%s'] * len(source_names))
            check_query = f"""
            SELECT DISTINCT name
            FROM {schema_app_data}.rag_document_blocks
            WHERE name IN ({placeholders})
            """
            try:
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
                    # Some requested sources don't exist
                    print(f"Warning: Source document(s) not found: {', '.join(missing_sources)}. Retrying search across all documents.")
                    warning_message = f"Source document(s) not found: {', '.join(missing_sources)}. Showing results from all documents."

                    # Retry search across all documents
                    search_results = self.search(
                        processed_query,
                        source_names=None, # Search all documents
                        strategy=strategy,
                        limit=initial_limit
                    )
            except Exception as e:
                 return {"success": False, "message": f"Database error checking sources: {e}"}


        # If still no results after potential retry, return failure
        if not search_results:
             return {
                 "success": False,
                 "message": warning_message or "No relevant information found in the documents."
             }

        # Group results by document name
        grouped_results = {}
        for block in search_results:
            doc_name = block["name"]
            if doc_name not in grouped_results:
                grouped_results[doc_name] = []
            grouped_results[doc_name].append(block)

        # Process each document's results
        final_output_list = []
        for doc_name, blocks in grouped_results.items():
            # Sort blocks within the document (assuming higher score is better)
            blocks.sort(key=lambda x: x.get("score", 0), reverse=True)

            # Select top N results
            top_results = blocks[:max_results_per_source]
            other_results = blocks[max_results_per_source:]
            other_result_idx = [b["block_idx"] for b in other_results]

            # Fetch children if requested
            processed_top_results = []
            for block in top_results:
                 # Format the block according to the desired output structure
                 formatted_block = {
                     "idx": block["block_idx"],
                     "content": block["content"],
                     "page": block["page_idx"],
                     "parent_idx": block["parent_idx"],
                     "level": block["level"],
                     "tag": block["tag"],
                     # Add other relevant fields if needed, e.g., score?
                     # "score": block.get("score")
                 }
                 if get_children:
                     # Fetch children using the internal _get_children method
                     children = self._get_children(block["block_idx"], doc_name) # Limit can be added here if needed
                     # Format children similarly or as needed
                     formatted_children = [
                         {
                             "idx": c["block_idx"],
                             "content": c["content"],
                             "page": c["page_idx"],
                             "parent_idx": c["parent_idx"],
                             "level": c["level"],
                             "tag": c["tag"],
                         } for c in children
                     ]
                     formatted_block["children"] = formatted_children
                 else:
                     formatted_block["children"] = [] # Ensure key exists

                 processed_top_results.append(formatted_block)


            # Add to the final list
            final_output_list.append({
                "document_name": doc_name,
                "results": processed_top_results,
                "other_result_idx": other_result_idx
            })

        # Construct the final success response
        final_result = {
            "success": True,
            "results": final_output_list
        }
        if warning_message:
            final_result["message"] = warning_message # Use 'message' key for warnings/errors

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
