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
            vector_store_path: Path to saved vector store directory (optional)
        """
        self.pdf_dir = pdf_dir
        self.vector_store_path = vector_store_path or "./chroma_db"
        self.rag = None
        self.start_time = None
        
    def initialize(self) -> bool:
        """
        Initialize RAG system.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            print("🔄 Initializing RAG system...")
            self.start_time = time.time()
            
            # Initialize RAG system
            llm_backend = os.getenv("LLM_BACKEND", "ollama").lower()
            llm_model_default = "llama3.2:1b" if llm_backend == "ollama" else "llama-3.1-70b-versatile"
            llm_model = os.getenv("LLM_MODEL", llm_model_default)
            groq_api_key = os.getenv("GROQ_API_KEY", None)

            self.rag = RAGSystem(
                ollama_base_url="http://localhost:11434",
                embeddings_model="nomic-embed-text",
                llm_model=llm_model,
                llm_backend=llm_backend,
                groq_api_key=groq_api_key,
                vector_store_directory=self.vector_store_path
            )
            
            # Check if vector store exists and try to load it
            if os.path.exists(self.vector_store_path) and os.listdir(self.vector_store_path):
                try:
                    print(f"📂 Loading existing vector store from {self.vector_store_path}...")
                    self.rag.load_vector_store(self.vector_store_path)
                    print("✓ Vector store loaded")
                except Exception as exc:
                    print(f"⚠️ Failed to load vector store: {exc}")
                    print("Will process PDFs and create a new vector store.")
                    return self._process_pdfs()
            else:
                # Load from PDFs and create new vector store
                return self._process_pdfs()
            
            elapsed = time.time() - self.start_time
            print(f"✓ RAG iniciado en {elapsed:.1f}s")
            return True
        
        except Exception as e:
            print(f"❌ Failed to initialize RAG: {str(e)}")
            return False
    
    def _process_pdfs(self) -> bool:
        """
        Process PDFs and create vector store.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load from PDFs
            print(f"📂 Loading PDFs from {self.pdf_dir}...")
            loader = PDFLoader(self.pdf_dir)
            
            pdf_list = loader.get_pdf_list()
            if not pdf_list:
                print(f"❌ No PDFs found in {self.pdf_dir}")
                return False
            
            print(f"✓ {len(pdf_list)} PDF(s) found")
            
            documents = loader.load_all_pdfs()
            
            if not documents:
                print("❌ Failed to load documents")
                return False
            
            print(f"✓ {len(documents)} pages loaded")
            
            # Process documents
            print("⚙️  Processing documents...")
            self.rag.ingest_documents(documents)
            
            print(f"💾 Vector store ready at {self.vector_store_path}")
            return True
        
        except Exception as e:
            print(f"❌ Error procesando PDFs: {str(e)}")
            return False
    
    def run(self) -> None:
        """Run interactive query loop."""
        if not self.initialize():
            return
        
        print("\n" + "="*70)
        print("💬 RAG Chat Interface - Query your PDFs")
        print("="*70)
        print("\nSpecial commands:")
        print("  'exit'      - End the session")
        print("  'clear'     - Clear the screen")
        print("  'metrics'   - Show statistics")
        print("="*70 + "\n")
        
        query_count = 0
        total_time = 0
        
        while True:
            try:
                # Get user input
                user_input = input("\n❓ Tu pregunta: ").strip()
                
                # Handle special commands
                if user_input.lower() == "exit":
                    print("\n👋 Goodbye!")
                    break
                
                if user_input.lower() == "clear":
                    os.system("clear" if os.name != "nt" else "cls")
                    continue
                
                if user_input.lower() == "metrics":
                    if query_count > 0:
                        print(f"\n📊 Metrics:")
                        print(f"  • Queries made: {query_count}")
                        print(f"  • Average response time: {total_time/query_count:.2f}s")
                        print(f"  • Total time: {total_time:.2f}s")
                    else:
                        print("No queries yet")
                    continue
                
                if not user_input:
                    print("⚠️  Please type a question")
                    continue
                
                # Query RAG system
                print("\n🔍 Processing query...")
                query_start = time.time()
                result = self.rag.query(user_input, verbose=True)
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
                    print("\n📌 Sources:")
                    for source in result["sources"]:
                        page = f"p.{source['page']}" if source['page'] >= 0 else "unknown"
                        print(f"  • {source['file']} ({page})")
                
                print(f"\n⏱️  Response time: {query_time:.2f}s")
            
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

    llm_backend = os.getenv("LLM_BACKEND", "ollama").lower()
    llm_model = os.getenv("LLM_MODEL", "llama3.2:1b")

    if llm_backend == "ollama":
        print("🔍 Checking Ollama service...")
        if not check_ollama():
            print("❌ Error: Ollama is not running")
            print("\nPlease run: ollama serve")
            print("In another terminal, download the model: ollama pull llama3.2:1b")
            return
        print("✓ Ollama is available\n")
    else:
        print(f"🔍 Using backend {llm_backend} (model {llm_model})")

    # PDF directory and vector store path
    pdf_dir = os.getenv("PDF_DIRECTORY", "../database")
    vector_store_path = os.getenv("VECTOR_STORE_PATH", "./chroma_db")

    # Run interface
    interface = RAGChatInterface(pdf_dir, vector_store_path)
    interface.run()


if __name__ == "__main__":
    main()

