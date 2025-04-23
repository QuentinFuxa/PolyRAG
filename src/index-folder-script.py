import os
import argparse
from rag_system import RAGSystem
from db_manager import DatabaseManager, schema_app_data # Import DatabaseManager

def index_pdf(pdf_path, use_embeddings=False):
    """Index a PDF using llmsherpa and store in database, tracking source status."""
    print(f"Processing {pdf_path}...")
    db_manager = None
    try:
        # Get database manager instance
        db_manager = DatabaseManager()

        # Check if the document source already exists and is indexed
        source_status = db_manager.get_document_source_status(name=pdf_path)
        if source_status and source_status.get('is_indexed'):
            print(f"Document source '{pdf_path}' is already indexed. Skipping.")
            return source_status.get('id') # Return existing source ID

        # Add or ensure the document source exists (will ignore if already present)
        # Use pdf_path as both name and path for file-based indexing
        source_id = db_manager.add_document_source(name=pdf_path, path=pdf_path)
        print(f"Ensured document source entry for '{pdf_path}' with ID: {source_id}")

        # Initialize RAGSystem
        rag_system = RAGSystem(use_embeddings=use_embeddings)
        
        # Index document blocks (assuming index_document handles block insertion)
        # We might need to adjust RAGSystem later if it doesn't use the 'name' correctly
        rag_system.index_document(pdf_path) 
        print(f"Successfully processed blocks for document source: {pdf_path}")

        # Mark the document source as indexed
        if db_manager.set_document_indexed(name=pdf_path, indexed=True):
            print(f"Successfully marked document source '{pdf_path}' as indexed.")
        else:
            print(f"Warning: Failed to mark document source '{pdf_path}' as indexed.")
            # Decide if this should be a critical error or just a warning

        return source_id # Return the source ID

    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        # Optionally, mark as not indexed if an error occurred mid-process
        # if db_manager and pdf_path:
        #     db_manager.set_document_indexed(name=pdf_path, indexed=False)
        return None
    # finally:
        # Ensure DB connection is handled properly if needed (DatabaseManager uses a pool)
        # if db_manager:
        #     db_manager.close() # Or rely on singleton lifecycle management

def main():
    parser = argparse.ArgumentParser(description="Index PDFs with llmsherpa")
    parser.add_argument("--pdf", help="Path to PDF file")
    parser.add_argument("--dir", help="Directory containing PDF files")
    parser.add_argument("--embeddings", action="store_true", help="Generate embeddings")
    
    args = parser.parse_args()
    
    if not args.pdf and not args.dir:
        parser.error("Either --pdf or --dir must be specified")
    
    if args.pdf:
        index_pdf(args.pdf, args.embeddings)
    
    if args.dir:
        pdf_files = [os.path.join(args.dir, f) for f in os.listdir(args.dir) if f.lower().endswith(".pdf") if '2024' in f or '2025' in f]
        print(f"Found {len(pdf_files)} PDF files in {args.dir}")
        for pdf_file in pdf_files:
            index_pdf(pdf_file, args.embeddings)

if __name__ == "__main__":
    main()
