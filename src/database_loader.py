"""
PDF loader module for RAG system.
Handles loading and processing PDF documents.
"""

from typing import List
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
import os


class PDFLoader:
    """Loads and processes PDF documents for RAG processing."""
    
    def __init__(self, pdf_directory: str):
        """
        Initialize PDF loader.
        
        Args:
            pdf_directory: Directory containing PDF files
        """
        self.pdf_directory = Path(pdf_directory)
        if not self.pdf_directory.exists():
            raise ValueError(f"Directory {pdf_directory} does not exist")
    
    def load_all_pdfs(self) -> List[Document]:
        """
        Load all PDF files from directory.
        
        Returns:
            List of LangChain Document objects
        """
        documents = []
        pdf_files = list(self.pdf_directory.glob("*.pdf"))
        
        if not pdf_files:
            raise ValueError(f"No PDF files found in {self.pdf_directory}")
        
        for pdf_file in pdf_files:
            print(f"  📄 Cargando {pdf_file.name}...")
            loader = PyPDFLoader(str(pdf_file))
            docs = loader.load()
            
            # Add source file metadata
            for doc in docs:
                doc.metadata['source_file'] = pdf_file.name
            
            documents.extend(docs)
        
        print(f"✓ Total: {len(pdf_files)} archivos, {len(documents)} páginas")
        return documents
    
    def load_specific_pdf(self, filename: str) -> List[Document]:
        """
        Load a specific PDF file.
        
        Args:
            filename: Name of the PDF file to load
            
        Returns:
            List of LangChain Document objects
        """
        pdf_path = self.pdf_directory / filename
        
        if not pdf_path.exists():
            raise ValueError(f"PDF file {filename} not found in {self.pdf_directory}")
        
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        
        # Add source file metadata
        for doc in docs:
            doc.metadata['source_file'] = filename
        
        return docs
    
    def get_pdf_list(self) -> List[str]:
        """
        Get list of PDF files in directory.
        
        Returns:
            List of PDF filenames
        """
        return [f.name for f in self.pdf_directory.glob("*.pdf")]
