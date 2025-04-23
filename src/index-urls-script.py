import os
import argparse
import requests
import tempfile
from urllib.parse import urlparse
from rag_system import RAGSystem
from db_manager import DatabaseManager

def is_valid_url(url):
    """Basic check for URL validity."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def download_pdf(url, temp_dir):
    """Downloads a PDF from a URL to a temporary file."""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        content_type = response.headers.get('content-type', '').lower()
        if 'application/pdf' not in content_type:
            print(f"Warning: URL {url} did not return PDF content-type ({content_type}). Skipping.")
            return None

        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path) or "downloaded.pdf"
        temp_pdf_path = os.path.join(temp_dir, filename)

        with open(temp_pdf_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Successfully downloaded '{url}' to '{temp_pdf_path}'")
        return temp_pdf_path

    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during download of {url}: {e}")
        return None


def index_pdf_from_url(pdf_url, db_manager, rag_system, temp_dir):
    """Checks status, downloads, indexes a PDF from a URL, and updates status."""
    print(f"Processing URL: {pdf_url}...")

    try:
        source_status = db_manager.get_document_source_status(name=pdf_url)
        if source_status and source_status.get('is_indexed'):
            print(f"URL '{pdf_url}' is already indexed. Skipping.")
            return source_status.get('id') # Return existing source ID
        pdf_name = pdf_url.split('/')[-1][:-4]
        source_id = db_manager.add_document_source(name=pdf_name, url=pdf_url)
        print(f"Ensured document source entry for URL '{pdf_url}' with ID: {source_id}")

        temp_pdf_path = download_pdf(pdf_url, temp_dir)
        if not temp_pdf_path:
            print(f"Skipping indexing for {pdf_url} due to download failure.")
            return None
        
        rag_system.index_document(temp_pdf_path, document_name_override=pdf_name)
        print(f"Successfully processed and indexed content from URL: {pdf_url}")

        if db_manager.set_document_indexed(name=pdf_url, indexed=True):
            print(f"Successfully marked URL '{pdf_url}' as indexed.")
        else:
            print(f"Warning: Failed to mark URL '{pdf_url}' as indexed.")

        try:
            os.remove(temp_pdf_path)
            print(f"Removed temporary file: {temp_pdf_path}")
        except OSError as e:
            print(f"Warning: Could not remove temporary file {temp_pdf_path}: {e}")

        return source_id # Return the source ID

    except Exception as e:
        print(f"Error processing URL {pdf_url}: {e}")
        if 'temp_pdf_path' in locals() and os.path.exists(temp_pdf_path):
             try:
                 os.remove(temp_pdf_path)
                 print(f"Removed temporary file after error: {temp_pdf_path}")
             except OSError as e_clean:
                 print(f"Warning: Could not remove temporary file {temp_pdf_path} after error: {e_clean}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Index PDFs from URLs stored in a database table.")
    parser.add_argument("--schema", required=False, default="public", help="Database schema name")
    parser.add_argument("--table", required=False, help="Database table name containing URLs", default="arxiv_qbio_metadata_2025")
    parser.add_argument("--column", required=False, help="Column name containing PDF URLs", default="pdf_url")
    parser.add_argument("--embeddings", action="store_true", help="Generate embeddings (requires OpenAI API key)")

    args = parser.parse_args()

    db_manager = None
    temp_dir = None
    try:
        db_manager = DatabaseManager()
        rag_system = RAGSystem(use_embeddings=args.embeddings) # Pass embedding flag
        if not (args.schema.isidentifier() and args.table.isidentifier() and args.column.isidentifier()):
             raise ValueError("Invalid schema, table, or column name provided.")

        query = f'SELECT "{args.column}" FROM "{args.schema}"."{args.table}" WHERE "{args.column}" IS NOT NULL;'
        print(f"Executing query: {query}")
        results = db_manager.execute_query(query)

        pdf_urls = [row[0] for row in results if row[0] and isinstance(row[0], str) and is_valid_url(row[0])]

        if not pdf_urls:
            print("No valid PDF URLs found in the specified column.")
            return

        print(f"Found {len(pdf_urls)} valid PDF URLs to process.")

        # Create a temporary directory for downloads
        temp_dir = tempfile.mkdtemp(prefix="pdf_downloads_")
        print(f"Created temporary directory for downloads: {temp_dir}")

        processed_count = 0
        failed_count = 0
        for pdf_url in pdf_urls:
            source_id = index_pdf_from_url(pdf_url, db_manager, rag_system, temp_dir)
            if source_id is not None:
                processed_count += 1
            else:
                failed_count += 1
            print("-" * 20)

        print(f"\nProcessing complete. Successfully processed: {processed_count}, Failed/Skipped: {failed_count}")

    except Exception as e:
        print(f"An error occurred during the main execution: {e}")
    finally:
        if temp_dir and os.path.exists(temp_dir):
            try:
                import shutil
                shutil.rmtree(temp_dir)
                print(f"Removed temporary directory: {temp_dir}")
            except Exception as e:
                print(f"Error removing temporary directory {temp_dir}: {e}")

if __name__ == "__main__":
    main()
