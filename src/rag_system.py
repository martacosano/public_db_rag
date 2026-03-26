"""
RAG (Retrieval-Augmented Generation) system using LangChain with Ollama.
Uses local language models and embeddings.
"""

from typing import List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS, Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
import json
import os


class RAGSystem:
    """Retrieval-Augmented Generation system using Ollama for database queries."""
    
    def __init__(
        self, 
        ollama_base_url: str = "http://localhost:11434",
        embeddings_model: str = "nomic-embed-text",
        llm_model: str = "llama3.1:8b",
        vector_store_type: str = "chroma"
    ):
        """
        Initialize RAG system with Ollama.
        
        Args:
            ollama_base_url: URL where Ollama service is running
            embeddings_model: Ollama embeddings model to use
            llm_model: Ollama LLM model to use
            vector_store_type: Type of vector store ('faiss' or 'chroma')
        """
        self.ollama_base_url = ollama_base_url
        self.embeddings_model = embeddings_model
        self.llm_model = llm_model
        self.vector_store = None
        self.vector_store_type = vector_store_type
        
        # Initialize embeddings
        try:
            self.embeddings = OllamaEmbeddings(
                base_url=ollama_base_url,
                model=embeddings_model
            )
            print(f"✓ Embeddings ({embeddings_model}) inicializados")
        except Exception as e:
            print(f"⚠️  Error con embeddings: {str(e)}")
            raise
        
        # Initialize LLM
        try:
            self.llm = Ollama(
                base_url=ollama_base_url,
                model=llm_model,
                temperature=0.3,
                top_k=40,
                top_p=0.9
            )
            print(f"✓ LLM ({llm_model}) inicializado")
        except Exception as e:
            print(f"⚠️  Error con LLM: {str(e)}")
            raise
        
        self.retrieval_qa = None
        # chunking , text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""]
        )
    
    
    
    def ingest_documents(self, documents: List[Document]) -> None:
        """
        Ingest documents into vector store.
        
        Args:
            documents: List of Document objects to ingest
        """
        # Split documents
        print("📄 Dividiendo documentos...")
        split_docs = self.text_splitter.split_documents(documents)
        print(f"  {len(split_docs)} chunks creados")
        
        # Create vector store
        print(f"🔍 Creando vector store ({self.vector_store_type})...")
        if self.vector_store_type == "faiss":
            self.vector_store = FAISS.from_documents(
                split_docs, 
                self.embeddings
            )
        elif self.vector_store_type == "chroma":
            self.vector_store = Chroma.from_documents(
                split_docs,
                self.embeddings,
                persist_directory="./chroma_db"
            )
        
        print("✓ Vector store creado")
        
        # Create retrieval QA chain
        print("🔗 Creando cadena de recuperación...")
        self.retrieval_qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 10}),
            return_source_documents=True
        )
        print("✓ Cadena lista")
    
    def query(self, question: str) -> dict:
        """
        Query the RAG system.
        
        Args:
            question: User question
            
        Returns:
            Dictionary with answer and source documents
        """
        if not self.retrieval_qa:
            raise ValueError("RAG system not initialized. Call ingest_documents first.")
        
        result = self.retrieval_qa({"query": question})
        
        # Format response with answer, passages and pages
        return {
            "answer": result["result"],
            "question": question,
            "sources": [
                {
                    "file": doc.metadata.get("source_file", "unknown"),
                    "page": doc.metadata.get("page", -1)
                }
                for doc in result.get("source_documents", [])
            ]
        }
    
    def save_vector_store(self, path: str) -> None:
        """
        Save vector store to disk.
        
        Args:
            path: Path to save vector store
        """
        if self.vector_store and self.vector_store_type == "faiss":
            os.makedirs(path, exist_ok=True)
            self.vector_store.save_local(path)
            print(f"✓ Vector store guardado en {path}")
    

    def load_vector_store(self, path: str) -> None:
        """
        Load vector store from disk.
        
        Args:
            path: Path to saved vector store
        """
        if self.vector_store_type == "faiss":
            self.vector_store = FAISS.load_local(path, self.embeddings)
            self.retrieval_qa = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(search_kwargs={"k": 4}),
                return_source_documents=True
            )
            print(f"✓ Vector store cargado desde {path}")
