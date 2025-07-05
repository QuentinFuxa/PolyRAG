import os
import argparse
from src.rag_system import RAGSystem
from src.db_manager import DatabaseManager

def index_pdf(pdf_path, use_embeddings=False):
    """Index a PDF using llmsherpa and store in database, tracking source status."""
    print(f"Processing {pdf_path}...")
    db_manager = None
    try:
        db_manager = DatabaseManager()
        name = os.path.basename(pdf_path)[:-4]
        source_status = db_manager.get_document_source_status(name=name)
        # if source_status and source_status.get('is_indexed'):
        #     print(f"Document source '{name}' is already indexed. Skipping.")
        #     return source_status.get('id') # Return existing source ID
        source_id = db_manager.add_document_source(name=name, path=pdf_path)
        print(f"Ensured document source entry for '{name}' with ID: {source_id}")
        rag_system = RAGSystem()
        rag_system.index_document(pdf_path, document_name_override=name)
        print(f"Successfully processed blocks for document source: {name}")
        if db_manager.set_document_indexed(name=name, indexed=True):
            print(f"Successfully marked document source '{name}' as indexed.")
        else:
            print(f"Warning: Failed to mark document source '{name}' as indexed.")
        return source_id

    except Exception as e:
        print(f"Error processing {name}: {e}")
        return None

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
        pdf_files = [os.path.join(args.dir, f) for f in os.listdir(args.dir) if f.lower().endswith(".pdf")]
        print(f"Found {len(pdf_files)} PDF files in {args.dir}")
        for pdf_file in pdf_files:
            index_pdf(pdf_file, args.embeddings)

if __name__ == "__main__":
    main()
