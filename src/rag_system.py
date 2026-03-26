"""
RAG (Retrieval-Augmented Generation) system using LangChain with Ollama.
Uses local language models and embeddings.
"""

from typing import List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
import json
from tqdm import tqdm
import os


class RAGSystem:
    """Retrieval-Augmented Generation system using Ollama for database queries."""
    
    def __init__(
        self, 
        ollama_base_url: str = "http://localhost:11434",
        embeddings_model: str = "nomic-embed-text",
        llm_model: str = "llama3.1:1b", #"llama3.1:8b",
        vector_store_directory: str = "./chroma_db"
    ):
        """
        Initialize RAG system with Ollama.
        
        Args:
            ollama_base_url: URL where Ollama service is running
            embeddings_model: Ollama embeddings model to use
            llm_model: Ollama LLM model to use
            vector_store_directory: Directory to save/load vector store (optional)
        """
        self.ollama_base_url = ollama_base_url
        self.embeddings_model = embeddings_model
        self.llm_model = llm_model
        self.vector_store_directory = vector_store_directory
        self.retrieval_chain = None
         
        # 1.Initialize embeddings
        try:
            self.embeddings = OllamaEmbeddings(
                base_url=ollama_base_url,
                model=embeddings_model
            )
            print(f"✓ Embeddings ({embeddings_model}) inicializados")
        except Exception as e:
            print(f"⚠️  Error con embeddings: {str(e)}")
            raise
        
        # 2.Initialize LLM
        try:
            self.llm = Ollama(
                base_url=ollama_base_url,
                model=llm_model,
                num_thread= 8, # Usa 8 hilos para acelerar la generación
                temperature=0.1, # Lower temperature for more factual answers
                top_k=40,
                top_p=0.9
            )
            print(f"✓ LLM ({llm_model}) inicializado")
        except Exception as e:
            print(f"⚠️  Error con LLM: {str(e)}")
            raise
        
    
        # 3.Initialize tokens splitter
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name= 'gpt-4', 
            chunk_size=1200, # tokens for each chunk
            chunk_overlap=200, # overlap
            separators=["\n\n", "\n", " ", ""]
        )
    
    
    
    def ingest_documents(self, documents: List[Document]) -> None:
        """
        Ingest documents into vector store. Just will be used the first time.
        
        Args:
            documents: List of Document objects to ingest
        """
        # Split documents in chunks
        print("📄 Dividiendo documentos en chunks...")
        split_docs = self.text_splitter.split_documents(documents)
        print(f"  {len(split_docs)} chunks creados")
        
        # Create vector store
        print(f"Guardando en ChromaDB en {self.vector_store_directory}...")
        # self.vector_store = Chroma.from_documents(
        #         split_docs,
        #         self.embeddings,
        #         persist_directory=self.vector_store_directory
        #     )
        # En lugar de usar Chroma.from_documents directamente, 
        # lo hacemos en lotes para ver la barra
        self.vector_store = Chroma(
            persist_directory=self.vector_store_directory,
            embedding_function=self.embeddings
        )

        # Añadimos los documentos con barra de progreso
        for i in tqdm(range(0, len(split_docs), 10)):
            batch = split_docs[i : i + 10]
            self.vector_store.add_documents(batch)
        # Una vez guardados, activamos la cadena de respuesta
        self._create_chain()

    def load_vector_store(self, path: str) -> None:
        """
        Memory, Load vector store from disk, which we created in previous runs. 
        This allows us to skip the PDF processing step.
        
        Args:
            path: Path to saved vector store
        """
        if os.path.exists(self.vector_store_directory):
            print(f"📂 Cargando base de datos desde {self.vector_store_directory}...")
            self.vector_store = Chroma(
                persist_directory=self.vector_store_directory,
                embedding_function=self.embeddings
            )
            self._create_chain()
            print("✓ Base de datos cargada correctamente.")
        else:
            print("⚠️ No existe base de datos previa para cargar.")


    def _create_chain(self):
        """
        Esta función une el buscador con el generador de texto.
        """
        if not self.vector_store:
            raise ValueError("No vector_store disponible para crear el chain")
              
        # Create retrieval QA chain
        print("🔗 Creando cadena de recuperación...")
       # Configurar la cadena de respuesta 
        system_prompt = (
            "Eres un Asistente Jurídico experto en legislación española. Tu tarea es responder consultas "
            "basándote exclusivamente en las leyes proporcionadas en el contexto."
            "\n\n"
            "REGLAS DE ACTUACIÓN:"
            "1. CITA DIRECTA: Siempre que sea posible, menciona el número de artículo o disposición de la ley."
            "2. FIDELIDAD: No parafrasees conceptos legales si eso implica perder precisión técnica."
            "3. RESPUESTA NEGATIVA: Si la consulta no está cubierta por los fragmentos de las leyes "
            "proporcionadas, responde: 'Basándome en los documentos disponibles, no se encuentra una base legal para responder a esta consulta'."
            "4. ESTRUCTURA: Usa puntos clave para facilitar la lectura de obligaciones o plazos."
            "\n\n"
            "CONTEXTO LEGAL RECUPERADO:"
            "\n{context}\n"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),  
            ("human", "{input}"),
        ])

        # Create the chain that combines retrieved documents
        combine_docs_chain = create_stuff_documents_chain(self.llm, prompt)
        # Create the retrieval chain that uses the vector store retriever and the combine_docs_chain
        self.retrieval_chain = create_retrieval_chain(
            self.vector_store.as_retriever(search_kwargs={"k": 10}),
            combine_docs_chain
        )
        print("✓ Cadena lista")
    

    def query(self, question: str) -> dict:
        """
        CONSULTA: La función que usas para preguntar.
        Busca en los vectores -> Recupera texto -> El LLM responde.
       
        Args:
            question: User question
            
        Returns:
            Dictionary with answer and source documents
        """
        if not self.retrieval_chain:
            raise ValueError("RAG system not initialized. Call ingest_documents first.")

        result = self.retrieval_chain.invoke({"input": question})
      
        # Format response with the answer, passages and pages
        answer = result.get("answer") if isinstance(result, dict) else str(result)
        source_documents = result.get("context", [])

        print(f"\n[DEBUG] Fragmentos recuperados: {len(source_documents)}")
        for i, doc in enumerate(source_documents):
            source_name = os.path.basename(doc.metadata.get("source", "unknown"))
            page = doc.metadata.get("page", "N/A")
            print(f"   - Doc {i+1}: {source_name} (Pág. {page})")

        # 4. Formatear la salida 
        return {
            "answer": answer,
            "question": question,
            "sources": [
                {
                    # En LangChain/PyPDFLoader la metadata suele ser 'source'
                    "file": os.path.basename(doc.metadata.get("source", "unknown")),
                    "page": doc.metadata.get("page", -1),
                    "content_preview": doc.page_content[:150].strip().replace('\n', ' ') + "..."
                }
                for doc in source_documents
            ]
        }