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
    
    # Check Ollama
    print("🔍 Verificando Ollama...")
    if not check_ollama():
        print("❌ Error: Ollama no está corriendo en http://localhost:11434")
        print("\nPor favor:")
        print("1. Instala Ollama desde https://ollama.ai")
        print("2. Ejecuta: ollama serve")
        print("3. En otra terminal: ollama pull mistral (o tu modelo preferido)")
        return
    print("✓ Ollama está disponible")
    
    # PDF directory
    pdf_dir = "../database"
    print(os.getcwd())
    try:
        # Initialize PDF loader
        print("\n📂 Inicializando cargador de PDFs...")
        loader = PDFLoader(pdf_dir)
        
        # Check for PDFs
        pdf_list = loader.get_pdf_list()
        if not pdf_list:
            print(f"❌ No se encontraron PDFs en {pdf_dir}")
            print("\nPor favor coloca archivos .pdf en la carpeta 'database/'")
            return
        
        print(f"✓ {len(pdf_list)} PDF(s) encontrado(s):")
        for pdf in pdf_list:
            print(f"  • {pdf}")
        
        # Load PDFs
        print("\n📄 Cargando PDFs...")
        documents = loader.load_all_pdfs()
        
        if not documents:
            print("❌ No se pudieron cargar documentos de los PDFs")
            return
        
        # Initialize RAG system
        print("\n🤖 Inicializando RAG system...")
        rag = RAGSystem(
            ollama_base_url="http://localhost:11434",
            embeddings_model="nomic-embed-text",
            llm_model="llama3.2:1b",
            vector_store_directory="./chroma_db"
        )

        # Cargar vector store si ya existe
        vector_store_path = "./chroma_db"
        if os.path.exists(vector_store_path) and os.listdir(vector_store_path):
            try:
                print(f"\n📂 Cargando vector store existente desde {vector_store_path}...")
                rag.load_vector_store(vector_store_path)
                print("✓ Vector store cargado con éxito")
            except Exception as exc:
                print(f"⚠️ No se pudo cargar el vector store: {exc}")
                print("Se procesarán los PDFs y se creará uno nuevo.")
                rag.ingest_documents(documents)
        else:
            # Ingest documents
            print("\n⚙️  Procesando documentos...")
            rag.ingest_documents(documents)

        
        print("\n💾 Vector store preparado en", vector_store_path)
        
        # Example queries
        print("\n" + "="*60)
        print("✅ RAG System Listo!")
        print("="*60)
        print("\nEjcuta: python interactive.py")
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

