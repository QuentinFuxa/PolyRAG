import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine
from langchain_core.tools import BaseTool, tool
from agents._graph_store import GraphStore

connection_string="postgresql://postgres@localhost:5432/lds"
engine = create_engine(connection_string)

def display_pdf(
    pdf_name: str,            # PDF name
    text_to_highlight: str,      # text to highlight
):
    """
    Displays a PDF with highlighted text.
    
    Args:
        pdf_name (str): The name of the PDF file to display
        text_to_highlight (str): The text to highlight in the PDF
    """
    
    return pdf_name, text_to_highlight

display_pdf: BaseTool = tool(display_pdf)
display_pdf.name = "PDF_Viewer"