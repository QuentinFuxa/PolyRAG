import os
import argparse
import json
from llmsherpa.readers import LayoutPDFReader
from rag_system import RAGSystem

def index_pdf(pdf_path, api_url, use_embeddings=False):
    """Index a PDF using llmsherpa and store in database"""
    print(f"Processing {pdf_path}...")
    
    # Create PDF reader
    pdf_reader = LayoutPDFReader(api_url)
    
    # Extract structured content
    try:
        doc = pdf_reader.read_pdf(pdf_path)
        sherpa_data = doc.json
        
        # Initialize RAG system
        rag_system = RAGSystem(use_embeddings=use_embeddings)
        
        # Index document
        doc_id = rag_system.index_document(pdf_path, existing_sherpa_data=sherpa_data)
        
        print(f"Successfully indexed document with ID: {doc_id}")
        
        # Optionally save sherpa data
        output_dir = os.path.join(os.path.dirname(pdf_path), "sherpa_output")
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(
            output_dir, 
            f"{os.path.splitext(os.path.basename(pdf_path))[0]}_sherpa.json"
        )
        
        with open(output_path, "w") as f:
            json.dump(sherpa_data, f, indent=2)
        
        print(f"Saved sherpa data to {output_path}")
        
        return doc_id
    
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Index PDFs with llmsherpa")
    parser.add_argument("--pdf", help="Path to PDF file")
    parser.add_argument("--dir", help="Directory containing PDF files")
    parser.add_argument(
        "--api", 
        default="http://localhost:5010/api/parseDocument?renderFormat=all&useNewIndentParser=true",
        help="llmsherpa API URL"
    )
    parser.add_argument("--embeddings", action="store_true", help="Generate embeddings")
    
    args = parser.parse_args()
    
    if not args.pdf and not args.dir:
        parser.error("Either --pdf or --dir must be specified")
    
    if args.pdf:
        index_pdf(args.pdf, args.api, args.embeddings)
    
    if args.dir:
        pdf_files = [os.path.join(args.dir, f) for f in os.listdir(args.dir) if f.lower().endswith(".pdf")]
        for pdf_file in pdf_files:
            index_pdf(pdf_file, args.api, args.embeddings)

if __name__ == "__main__":
    main()



    