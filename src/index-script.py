import os
import argparse
from rag_system import RAGSystem

def index_pdf(pdf_path, use_embeddings=False):
    """Index a PDF using llmsherpa and store in database"""
    print(f"Processing {pdf_path}...")
    try:
        rag_system = RAGSystem(use_embeddings=use_embeddings)
        
        # Index document
        doc_id = rag_system.index_document(pdf_path)
        print(f"Successfully indexed document with ID: {doc_id}")
        
        return doc_id
    
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
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
        pdf_files = [os.path.join(args.dir, f) for f in os.listdir(args.dir) if f.lower().endswith(".pdf") if '2024' in f or '2025' in f]
        print(f"Found {len(pdf_files)} PDF files in {args.dir}")
        for pdf_file in pdf_files:
            index_pdf(pdf_file, args.embeddings)

if __name__ == "__main__":
    main()



    