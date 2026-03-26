"""
Interactive RAG query interface with PDF support and Ollama.
Provides a command-line interface for querying the RAG system.
"""

import os
import time
from dotenv import load_dotenv
from database_loader import PDFLoader
from rag_system import RAGSystem


class RAGChatInterface:
    """Interactive interface for RAG queries."""
    
    def __init__(self, pdf_dir: str, vector_store_path: str = None):
        """
        Initialize the chat interface.
        
        Args:
            pdf_dir: Directory containing PDF files
            vector_store_path: Path to saved vector store (optional)
        """
        self.pdf_dir = pdf_dir
        self.rag = None
        self.vector_store_path = vector_store_path
        self.start_time = None
        
    def initialize(self) -> bool:
        """
        Initialize RAG system.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            print("🔄 Inicializando sistema RAG...")
            self.start_time = time.time()
            
            # Check if vector store exists
            if self.vector_store_path and os.path.exists(self.vector_store_path):
                print(f"📂 Cargando vector store desde {self.vector_store_path}...")
                self.rag = RAGSystem()
                self.rag.load_vector_store(self.vector_store_path)
                print("✓ Vector store cargado")
            else:
                # Load from PDFs
                print(f"📂 Cargando PDFs desde {self.pdf_dir}...")
                loader = PDFLoader(self.pdf_dir)
                
                pdf_list = loader.get_pdf_list()
                if not pdf_list:
                    print(f"❌ No se encontraron PDFs en {self.pdf_dir}")
                    return False
                
                print(f"✓ {len(pdf_list)} PDF(s) encontrado(s)")
                
                documents = loader.load_all_pdfs()
                
                if not documents:
                    print("❌ No se pudieron cargar documentos")
                    return False
                
                print(f"✓ {len(documents)} páginas cargadas")
                
                # Initialize RAG
                print("🤖 Inicializando RAG...")
                self.rag = RAGSystem(
                    ollama_base_url="http://localhost:11434",
                    embeddings_model="nomic-embed-text",
                    llm_model="llama3.1:8b",
                )
                
                # Process documents
                self.rag.ingest_documents(documents)
                
                # Save vector store
                if self.vector_store_path:
                    print(f"💾 Guardando vector store...")
                    self.rag.save_vector_store(self.vector_store_path)
            
            elapsed = time.time() - self.start_time
            print(f"✓ RAG iniciado en {elapsed:.1f}s")
            return True
        
        except Exception as e:
            print(f"❌ Error inicializando RAG: {str(e)}")
            return False
    
    def run(self) -> None:
        """Run interactive query loop."""
        if not self.initialize():
            return
        
        print("\n" + "="*70)
        print("💬 RAG Chat Interface - Consulta tus PDFs")
        print("="*70)
        print("\nComandos especiales:")
        print("  'salir'      - Terminar la sesión")
        print("  'limpiar'    - Limpiar pantalla")
        print("  'métricas'   - Mostrar estadísticas")
        print("="*70 + "\n")
        
        query_count = 0
        total_time = 0
        
        while True:
            try:
                # Get user input
                user_input = input("\n❓ Tu pregunta: ").strip()
                
                # Handle special commands
                if user_input.lower() == "salir":
                    print("\n👋 ¡Hasta luego!")
                    break
                
                if user_input.lower() == "limpiar":
                    os.system("clear" if os.name != "nt" else "cls")
                    continue
                
                if user_input.lower() == "métricas":
                    if query_count > 0:
                        print(f"\n📊 Estadísticas:")
                        print(f"  • Consultas realizadas: {query_count}")
                        print(f"  • Tiempo promedio: {total_time/query_count:.2f}s")
                        print(f"  • Tiempo total: {total_time:.2f}s")
                    else:
                        print("Sin consultas aún")
                    continue
                
                if not user_input:
                    print("⚠️  Por favor escribe una pregunta")
                    continue
                
                # Query RAG system
                print("\n🔍 Procesando consulta...")
                query_start = time.time()
                result = self.rag.query(user_input)
                query_time = time.time() - query_start
                
                # Update metrics
                query_count += 1
                total_time += query_time
                
                print("\n" + "-"*70)
                print("💡 Respuesta:")
                print("-"*70)
                print(result["answer"])
                print("-"*70)
                
                # Show sources
                if result.get("sources"):
                    print("\n📌 Fuentes:")
                    for source in result["sources"]:
                        page = f"p.{source['page']}" if source['page'] >= 0 else "desconocida"
                        print(f"  • {source['file']} ({page})")
                
                print(f"\n⏱️  Tiempo de respuesta: {query_time:.2f}s")
            
            except KeyboardInterrupt:
                print("\n\n👋 ¡Hasta luego!")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")


def check_ollama():
    """Check if Ollama is running."""
    import requests
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except:
        return False


def main():
    """Main entry point."""
    load_dotenv()
    
    # Check Ollama
    print("🔍 Verificando Ollama...")
    if not check_ollama():
        print("❌ Error: Ollama no está corriendo")
        print("\nPor favor ejecuta: ollama serve")
        print("En otra terminal, descarga un modelo: ollama pull mistral")
        return
    
    print("✓ Ollama disponible\n")
    
    # PDF directory
    pdf_dir = "../database"
    vector_store_path = "./vector_store"
    
    # Run interface
    interface = RAGChatInterface(pdf_dir, vector_store_path)
    interface.run()


if __name__ == "__main__":
    main()

