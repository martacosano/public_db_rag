"""
Package initialization for RAG system with Ollama.
"""

from .database_loader import PDFLoader
from .rag_system import RAGSystem

__version__ = "2.0.0"
__all__ = ["PDFLoader", "RAGSystem"]
