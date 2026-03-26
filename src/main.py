"""
Main entry point for RAG system with PDF integration and Ollama.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from database_loader import PDFLoader
from rag_system import RAGSystem


def check_ollama():
    """Check if Ollama is running."""
    import requests
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            return True
    except:
        return False


def main():
    """Main function to run RAG system with PDFs."""
    
    # Load environment variables
    load_dotenv()
    
    # Select LLM backend and model
    llm_backend = os.getenv("LLM_BACKEND", "groq").lower()
    llm_model = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")

    if llm_backend == "ollama":
        print("🔍 Checking Ollama service...")
        if not check_ollama():
            print("❌ Error: Ollama is not running on http://localhost:11434")
            print("\nPlease:")
            print("1. Install Ollama from https://ollama.ai")
            print("2. Run: ollama serve")
            print("3. In another terminal: ollama pull your-preferred-model")
            return
        print("✓ Ollama is available")
    else:
        print(f"🔍 Using Groq backend (model {llm_model})")
    
    # PDF directory
    pdf_dir = "../database"
    print(os.getcwd())
    try:
        # Initialize PDF loader
        print("\n📂 Initializing PDF loader...")
        loader = PDFLoader(pdf_dir)
        
        # Check for PDFs
        pdf_list = loader.get_pdf_list()
        if not pdf_list:
            print(f"❌ No PDFs found in {pdf_dir}")
            print("\nPlease place .pdf files in the 'database/' folder")
            return
        
        print(f"✓ {len(pdf_list)} PDF(s) found:")
        for pdf in pdf_list:
            print(f"  • {pdf}")
        
        # Load PDFs
        print("\n📄 Loading PDFs...")
        documents = loader.load_all_pdfs()
        
        if not documents:
            print("❌ Failed to load documents from PDFs")
            return
        
        # Initialize RAG system
        print("\n🤖 Initializing RAG system...")
        rag = RAGSystem(
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            embeddings_model=os.getenv("OLLAMA_EMBEDDINGS_MODEL", "nomic-embed-text"),
            llm_model=llm_model,
            llm_backend=llm_backend,
            groq_api_key=os.getenv("GROQ_API_KEY", None),
            vector_store_directory=os.getenv("VECTOR_STORE_PATH", "./chroma_db")
        )

        # Load vector store if it already exists
        vector_store_path = "./chroma_db"
        if os.path.exists(vector_store_path) and os.listdir(vector_store_path):
            try:
                print(f"\n📂 Loading existing vector store from {vector_store_path}...")
                rag.load_vector_store(vector_store_path)
                print("✓ Vector store loaded successfully")
            except Exception as exc:
                print(f"⚠️ Failed to load vector store: {exc}")
                print("Will process PDFs and create a new vector store.")
                rag.ingest_documents(documents)
        else:
            # Ingest documents
            print("\n⚙️  Processing documents...")
            rag.ingest_documents(documents)

        
        print("\n💾 Vector store ready at", vector_store_path)
        
        # Example queries
        print("\n" + "="*60)
        print("✅ RAG System Ready!")
        print("="*60)
        print("\nRun: python interactive.py")
        print("="*60)
    
    except FileNotFoundError as e:
        print(f"❌ Error: {str(e)}")
        print(f"Asegúrate de que la carpeta {pdf_dir} exista")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

