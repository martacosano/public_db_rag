"""
RAG (Retrieval-Augmented Generation) system using LangChain with Ollama.
Uses local language models and embeddings.
"""

from typing import List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama

# Groq support (
from langchain_groq import ChatGroq


from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank

from tqdm import tqdm
import os


class RAGSystem:
    """Retrieval-Augmented Generation system supporting Ollama and Groq backends."""
    
    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        embeddings_model: str = "nomic-embed-text",
        llm_model: str = "llama-3.1-8b-instant",
        llm_backend: str = "groq",  # opciones: 'ollama', 'groq'
        groq_api_key: str = None,
        vector_store_directory: str = "./chroma_db"
    ):
        """
        Initialize RAG system.
        
        Args:
            ollama_base_url: URL where Ollama service is running
            embeddings_model: Ollama embeddings model to use
            llm_model: LLM model to use (Ollama or Groq depending on backend)
            vector_store_directory: Directory to save/load vector store (optional)
        """
        self.ollama_base_url = ollama_base_url
        self.embeddings_model = embeddings_model
        self.llm_model = llm_model
        self.llm_backend = llm_backend.lower()
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        self.vector_store_directory = vector_store_directory
        self.vector_store = None
        self.retrieval_chain = None
         
        # 1.Initialize embeddings
        try:
            self.embeddings = OllamaEmbeddings(
                base_url=ollama_base_url,
                model=embeddings_model
            )
            print(f"✓ Embeddings ({embeddings_model}) initialized")
        except Exception as e:
            print(f"⚠️  Embeddings error: {str(e)}")
            raise
        
        # 2. Initialize LLM
        try:
            self.llm = self._init_llm()
            print(f"✓ LLM ({self.llm_model}) initialized with backend {self.llm_backend}")
        except Exception as e:
            print(f"⚠️  LLM error: {str(e)}")
            raise
        
    
        # 3.Initialize tokens splitter
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name= 'gpt-4', 
            chunk_size=1000, # tokens for each chunk
            chunk_overlap=200, # overlap
            separators=["\n\n", "\n", " ", ""]
        )
    
    def _init_llm(self):
        """Initialize the chosen LLM backend."""
        if self.llm_backend == "ollama":
            return Ollama(
                base_url=self.ollama_base_url,
                model=self.llm_model,
                num_thread=8,
                temperature=0.1,
                top_k=40,
                top_p=0.9
            )
        elif self.llm_backend == "groq":
            if ChatGroq is None:
                raise RuntimeError("Groq is not available. Install langchain-groq or langchain_community with Groq support.")
            if not self.groq_api_key:
                raise ValueError("GROQ_API_KEY is not set in environment variables or groq_api_key parameter.")
            return ChatGroq(
                model=self.llm_model,
                api_key=self.groq_api_key,
                temperature=0.1
            )
        else:
            raise ValueError(f"Unknown LLM backend: {self.llm_backend}. Use 'ollama' or 'groq'.")
    
    def ingest_documents(self, documents: List[Document]) -> None:
        """
        Ingest documents into vector store. Just will be used the first time.
        
        Args:
            documents: List of Document objects to ingest
        """
        # Split documents in chunks
        print("📄 Splitting documents into chunks...")
        split_docs = self.text_splitter.split_documents(documents)
        print(f"  {len(split_docs)} chunks created")
        
        # Create vector store
        print(f"Saving in ChromaDB at {self.vector_store_directory}...")

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
        if not os.path.exists(path):
            raise FileNotFoundError(f"Vector store directory does not exist: {path}")

        print(f"📂 Loading database from {path}...")
        self.vector_store_directory = path
        self.vector_store = Chroma(
            persist_directory=path,
            embedding_function=self.embeddings
        )
        self._create_chain()
        print("✓ Base de datos cargada correctamente.")


    def _create_chain(self):
        """
        Create the retrieval chain by combining retriever and LLM.
        """
        if not self.vector_store:
            raise ValueError("No vector_store available to create chain")

        # 1. Base retriever
        base_retriever = self.vector_store.as_retriever(search_kwargs={"k": 20}) # number of documents to retrieve

        # 2. El Re-ranker 
        compressor = FlashrankRerank(top_n=5) # number of documents to keep after re-ranking

        # 3. El Retriever "Comprimido": Filtra y reordena los 10 de antes a solo los 5 mejores
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, 
            base_retriever=base_retriever
        )

        # Create retrieval QA chain
        print("🔗 Creando cadena de recuperación...")
       # Configurar la cadena de respuesta 
        system_prompt = (
            "You are a Legal Assistant specialized in Spanish legislation. Your task is to answer questions "
            "based strictly on the laws provided in the context."
            "\n\n"
            "OPERATION GUIDELINES:"
            "1. DIRECT QUOTE: Whenever possible, cite the article or legal provision number."
            "2. ACCURACY: Do not paraphrase legal concepts if it reduces technical precision."
            "3. NEGATIVE RESPONSE: If the question is not covered by the retrieved legal passages, respond: 'Based on the available documents, there is no legal basis to answer this query'."
            "4. STRUCTURE: Use bullet points to make obligations or deadlines easier to read."
            "\n\n"
            "RETRIEVED LEGAL CONTEXT:"
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
            retriever=compression_retriever,
            combine_docs_chain=combine_docs_chain
        )
        print("✓ Cadena lista")
    

    def query(self, question: str, verbose: bool = False) -> dict:
        """
        Query the RAG system.
        Performs vector retrieval and then LLM response generation.
       
        Args:
            question: User question
            verbose: If True, print retrieved documents info
            
        Returns:
            Dictionary with answer and source documents
        """
        if not self.retrieval_chain:
            raise ValueError("RAG system not initialized. Call ingest_documents first.")

        result = self.retrieval_chain.invoke({"input": question})
      
        # Format response with the answer, passages and pages
        answer = result.get("answer") if isinstance(result, dict) else str(result)
        source_documents = result.get("context", [])

        if verbose:
            for i, doc in enumerate(source_documents):
                source_name = os.path.basename(doc.metadata.get("source", "unknown"))
                page = doc.metadata.get("page", "N/A")
                print(f"   - Doc {i+1}: {source_name} (Page {page})")

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