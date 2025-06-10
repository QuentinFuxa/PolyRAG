import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union, Tuple

from llmsherpa.readers import LayoutPDFReader
from dotenv import load_dotenv

try:
    from .db_manager import DatabaseManager, schema_app_data
    from .content_classifiers import ContentClassifier, ContentType
except ImportError:
    from db_manager import DatabaseManager, schema_app_data
    from content_classifiers import ContentClassifier, ContentType

    
# Load environment variables
load_dotenv()

TS_QUERY_LANGUAGE = os.environ.get("TS_QUERY_LANGUAGE", "english")


def _prefix_columns_in_where_clause(clause_str: str, prefix: str = "js.") -> str:
    if not clause_str:
        return ""

    sql_keywords = {'AND', 'OR', 'NOT', 'NULL', 'TRUE', 'FALSE', 'IS', 'IN', 'LIKE', 'BETWEEN', 'EXISTS',
                    'SELECT', 'FROM', 'WHERE', 'GROUP', 'ORDER', 'BY', 'LIMIT', 'OFFSET', 'AS',
                    'ASC', 'DESC', 'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'EXTRACT', 'CAST', 'CONVERT',
                    'UNION', 'INTERSECT', 'EXCEPT', 'JOIN', 'LEFT', 'RIGHT', 'INNER', 'OUTER'}

    parts = re.split(r'(\s+AND\s+|\s+OR\s+)', clause_str, flags=re.IGNORECASE)
    
    processed_parts = []
    for part_idx, part_content in enumerate(parts):
        stripped_part_content = part_content.strip()
        
        # If it's a delimiter (AND/OR), keep it
        if part_idx % 2 == 1: # Delimiters are at odd indices
            processed_parts.append(part_content)
            continue

        # Attempt to identify "column operator value" like structures
        match = re.match(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*(.*)", stripped_part_content, re.IGNORECASE)
        
        if match:
            column_candidate = match.group(1)
            rest_of_condition = match.group(2)

            # Check if it's a keyword, already prefixed, a function call, or a literal string/number
            if (column_candidate.upper() not in sql_keywords and
                '.' not in column_candidate and
                '(' not in column_candidate and # Simple check for function
                not (column_candidate.startswith("'") and column_candidate.endswith("'")) and # String literal
                not (column_candidate.startswith('"') and column_candidate.endswith('"')) and # String literal
                not column_candidate.replace('.', '', 1).isdigit() and # Number
                column_candidate.upper() != 'TRUE' and column_candidate.upper() != 'FALSE' and column_candidate.upper() != 'NULL'
               ):
                processed_parts.append(f"{prefix}{column_candidate}{rest_of_condition}")
            else:
                processed_parts.append(part_content) # No change
        else:
            processed_parts.append(part_content) # No change if no word-like start
            
    return "".join(processed_parts)

# Helper function to prefix column names in an ORDER BY clause string
def _prefix_columns_in_order_by_clause(clause_str: str, prefix: str = "js.") -> str:
    if not clause_str:
        return ""
    
    terms = clause_str.split(',')
    prefixed_terms = []
    for term_str in terms:
        term_str = term_str.strip()
        # Split term into column and direction (ASC/DESC)
        parts = term_str.split() # e.g. ["date", "DESC"] or ["name"]
        if parts:
            col_name = parts[0]
            # Avoid prefixing if already prefixed or is a function call (simple check)
            if '.' not in col_name and '(' not in col_name:
                parts[0] = f"{prefix}{col_name}"
            prefixed_terms.append(" ".join(parts))
    return ", ".join(prefixed_terms)


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
    block_idx: int
    content: str
    metadata: BlockMetadata
    parent_idx: Optional[int] = None

class SherpaDocumentProcessor:
    """Processor for rag_documents from llmsherpa"""
    
    def __init__(self, sherpa_api_url=None):
        self.sherpa_api_url = sherpa_api_url or os.getenv(
            "LLMSHERPA_API_URL", 
            "http://localhost:5010/api/parseDocument?renderFormat=all&useNewIndentParser=true"
        )
        self.pdf_reader = LayoutPDFReader(self.sherpa_api_url)
    
    def process_pdf(self, pdf_path):
        """Process a PDF file using llmsherpa and return DocumentBlock objects"""
        sherpa_data = self.pdf_reader.read_pdf(pdf_path).json
        return self._process_sherpa_data(sherpa_data)

    def _process_sherpa_data(self, sherpa_data):
        """Process raw llmsherpa output (JSON) and convert to DocumentBlock objects"""
        blocks = []

        if isinstance(sherpa_data, str):
            try:
                sherpa_data = json.loads(sherpa_data)
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON data from llmsherpa")
        
        for block_data in sherpa_data:
            # Extract content from sentences
            content = ""
            if "sentences" in block_data:
                sentences = block_data["sentences"]
                if isinstance(sentences, list):
                    if sentences and isinstance(sentences[0], dict) and "text" in sentences[0]:
                        content = " ".join(s["text"] for s in sentences)
                    elif sentences and isinstance(sentences[0], str):
                        content = " ".join(sentences)
            elif "table_rows" in block_data:
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
            
            metadata = BlockMetadata(
                block_idx=block_data.get("block_idx", 0),
                level=block_data.get("level", 0),
                page_idx=block_data.get("page_idx", 0),
                tag=block_data.get("tag", "unknown"),
                block_class=block_data.get("block_class", ""),
                bbox=block_data.get("bbox", [0, 0, 0, 0])
            )
            
            block = DocumentBlock(
                block_idx=metadata.block_idx,
                content=content,
                metadata=metadata
            )
            
            blocks.append(block)
        
        self._establish_hierarchy(blocks)
        return blocks
    
    def _establish_hierarchy(self, blocks):
        """Establish parent-child relationships based on level and position"""
        sorted_blocks = sorted(blocks, key=lambda b: (b.metadata.page_idx, b.metadata.bbox[1]))
        hierarchy_stack = []
        
        for block in sorted_blocks:
            while hierarchy_stack and hierarchy_stack[-1].metadata.level >= block.metadata.level:
                hierarchy_stack.pop()
            
            if hierarchy_stack:
                block.parent_idx = hierarchy_stack[-1].block_idx
            
            hierarchy_stack.append(block)


class RAGSystem:
    """RAG system with text search and PDF backends"""

    def __init__(self):
        self.db_manager = DatabaseManager()
        self.TS_QUERY_LANGUAGE = TS_QUERY_LANGUAGE
        
        # Determine PDF parsing backend
        pdf_parser_backend = os.getenv("PDF_PARSER", "nlm-ingestor").lower()
        if pdf_parser_backend == "pymupdf":
            from .pymupdf_processor import PyMuPDFDocumentProcessor
            self.processor = PyMuPDFDocumentProcessor()
        else:
            self.processor = SherpaDocumentProcessor()

        self.content_classifier = ContentClassifier()
        self._ensure_schema()
    
    def _ensure_schema(self):
        """Create necessary database schema if it doesn't exist"""
        schema = f"""        
        CREATE TABLE IF NOT EXISTS {schema_app_data}.rag_document_blocks (
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
            content_type TEXT DEFAULT 'regular',
            section_type TEXT,
            demand_priority INTEGER,
            content_tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('{TS_QUERY_LANGUAGE}', content)) STORED,
            UNIQUE(name, block_idx)
        );
        
        CREATE INDEX IF NOT EXISTS idx_document_blocks_content_tsv ON {schema_app_data}.rag_document_blocks 
        USING gin(content_tsv);
        
        CREATE INDEX IF NOT EXISTS idx_rag_blocks_content_type 
        ON {schema_app_data}.rag_document_blocks(content_type);
        
        CREATE INDEX IF NOT EXISTS idx_rag_blocks_section_type 
        ON {schema_app_data}.rag_document_blocks(section_type);
        
        CREATE INDEX IF NOT EXISTS idx_rag_blocks_demand_priority 
        ON {schema_app_data}.rag_document_blocks(demand_priority);
        """
        
        self.db_manager.execute_query(schema)
    
    def index_document(self, pdf_path, document_name_override: Optional[str] = None, existing_sherpa_data=None, table_name=f"{schema_app_data}.rag_document_blocks"):
        """Index a PDF document from a given path."""
        document_name = document_name_override if document_name_override is not None else pdf_path

        if existing_sherpa_data is None:
            blocks = self.processor.process_pdf(pdf_path)
        else:
            blocks = self.processor._process_sherpa_data(existing_sherpa_data)
        
        # Check if this is a "lettre de suite" based on content patterns
        blocks_content = [block.content for block in blocks if block.content]
        is_letter_de_suite = self.content_classifier.is_letter_de_suite(blocks_content)
        
        if is_letter_de_suite:
            print(f"Document {document_name} detected as 'lettre de suite'. Classifying blocks...")
            self._classify_blocks(blocks)

        self._insert_blocks(document_name, blocks, table_name)
        return document_name
    
    def _classify_blocks(self, blocks):
        """Classify blocks for content type, section type, and demand priority."""
        current_section = None
        
        for block in blocks:
            if not block.content:
                continue
            
            content_type, section_type, demand_priority = self.content_classifier.classify_block(
                block.content, current_section
            )
            
            if content_type == ContentType.SECTION_HEADER:
                current_section = section_type
            
            block.content_type = content_type.value
            block.section_type = section_type.value if section_type else None
            block.demand_priority = demand_priority
    
    def _insert_blocks(self, name, blocks, table_name=f"{schema_app_data}.rag_document_blocks"):
        """Insert blocks into database"""
        params_list = []
        
        for block in blocks:
            bbox = block.metadata.bbox
            x0 = bbox[0] if len(bbox) > 0 else None
            y0 = bbox[1] if len(bbox) > 1 else None
            x1 = bbox[2] if len(bbox) > 2 else None
            y1 = bbox[3] if len(bbox) > 3 else None
            
            content_type = getattr(block, 'content_type', 'regular')
            section_type = getattr(block, 'section_type', None)
            demand_priority = getattr(block, 'demand_priority', None)
            
            params = (
                block.block_idx, name, block.content, block.metadata.level,
                block.metadata.page_idx, block.metadata.tag, block.metadata.block_class,
                x0, y0, x1, y1, block.parent_idx, content_type, section_type, demand_priority
            )
            
            params_list.append(params)
        
        query = f"""
        INSERT INTO {table_name}
        (block_idx, name, content, level, page_idx, tag, block_class, 
         x0, y0, x1, y1, parent_idx, content_type, section_type, demand_priority)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
        content_type = EXCLUDED.content_type,
        section_type = EXCLUDED.section_type,
        demand_priority = EXCLUDED.demand_priority
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
            processed_query = user_query

        # Clean and format query terms
        formatted_elements = []
        for element in processed_query:
            sanitized = element.replace('|', '').replace('!', '').replace('(', '').replace(')', '').strip()
            if sanitized:
                formatted_elements.append(f"({' & '.join(sanitized.split())})" if ' ' in sanitized else sanitized)

        join_clause_str = ""
        where_join_conditions_list = []
        order_by_join_clause_str = ""
        ts_query_for_format = "" # Initialize for .format() later

        if source_query:
            # Parse source_query using regex for more robustness
            _from_parts = re.split(r'\sFROM\s', source_query, 1, re.IGNORECASE)
            if len(_from_parts) < 2:
                # Log or handle error: source_query must contain FROM clause
                # For now, we'll let it proceed, and it might result in an SQL error or empty results
                pass # Or raise ValueError("source_query must contain FROM clause")
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
            id, block_idx, content, name, page_idx, level, tag, block_class,
            x0, y0, x1, y1, parent_idx
        FROM {schema_app_data}.rag_document_blocks
        WHERE parent_idx = %s AND name = %s
        ORDER BY page_idx, block_idx
        LIMIT %s
        """
        
        results = self.db_manager.execute_query(query, (parent_idx, name, limit))
        
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
                "score": 1.0
            }
            for row in results 
        ]
    
    def get_blocks_by_idx(self, block_indices, source_name=None, get_children=False):
        """Get blocks by their block_idx values"""
        if not block_indices:
            return []
        
        if not isinstance(block_indices, list):
            block_indices = [block_indices]
        
        placeholders = ', '.join(['%s'] * len(block_indices))
        query = f"""
        SELECT 
            id, block_idx, content, name, page_idx, level, tag, block_class,
            x0, y0, x1, y1, parent_idx
        FROM {schema_app_data}.rag_document_blocks
        WHERE block_idx IN ({placeholders})
        """
        
        params = block_indices.copy()
        
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
                "score": 1.0
            }
            for row in results
        ]
        
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