import os
import json
import time
import re
from typing import Dict
from dotenv import load_dotenv
from rag_system import RAGSystem
from database_loader import PDFLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# 1. Define test cases
JUDGE_MODEL= "llama-3.1-8b-instant"

TEST_CASES = [
    { #ok
        "id": 1,
        "question": "¿Cuál es la forma política del Estado español según la Constitución?",
        "ground_truth": "La forma política del Estado español es la Monarquía parlamentaria.",
        "expected_page": 3,
        "expected_source": "Constitucion_Española.pdf",
        "reference": "Constitución, Art. 1.3"
    },
    {# OK
        "id": 2,
        "question": "¿Qué profesionales tienen la consideración de personal de seguridad privada?",
        "ground_truth": "Vigilantes de seguridad, vigilantes de explosivos, escoltas privados, guardas rurales, jefes de seguridad, directores de seguridad y detectives privados.",
        "expected_page": 25,
        "expected_source": "Ley_Seguridad_Privada_Consolidada.pdf",
        "reference": "Ley Seguridad Privada, Art. 26"
    },
    { #ok
        "id": 3,
        "question": "¿Cuál es el objeto de la Ley de régimen fiscal de las entidades sin fines lucrativos?",
        "ground_truth": "Regular el régimen fiscal de las entidades sin fines lucrativos definidas en esta Ley, en consideración a su función social, actividades y fines.",
        "expected_page": 9,
        "expected_source": "LEY_49-2002_REGIMEN_FISCAL_DE_LAS_ENTIDADES_SIN_FINES_LUCRATIVOS.pdf",
        "reference": "Ley regimen fiscal, Art. 1"
    },
    { #ok
        "id": 4,
        "question": "¿Cómo se determina el importe de la fianza en arrendamientos de vivienda?",
        "ground_truth": "Es obligatoria la prestación de fianza en metálico en cantidad equivalente a una mensualidad de renta en el arrendamiento de viviendas.",
        "expected_page": 22, # Art 36 de la LAU
        "expected_source": "LEY_ARRENDAMIENTOS_URBANOS.pdf",
        "reference": "Ley Arrendamientos Urbanos, Art. 36.1"
    },
    { #OK
        "id": 5,
        "question": "¿Por qué causas se extinguirá el contrato de arrendamiento según el artículo 28?",
        "ground_truth": "Por la pérdida de la finca por causa no imputable al arrendador y por la declaración firme de ruina acordada por la autoridad competente.",
        "expected_page": 20,
        "expected_source": "LEY_ARRENDAMIENTOS_URBANOS.pdf",
        "reference": "Ley 29/1994 (LAU),  Art. 28"
    },
    { #ok
        "id": 6,
        "question": "¿Quién tendrá la consideración de beneficiario de una subvención?",
        "ground_truth": "La persona que realice la actividad o esté en la situación que justifica la concesión. Incluye a miembros asociados de personas jurídicas y a agrupaciones sin personalidad jurídica (como comunidades de bienes) si se prevé en las bases.",
        "expected_page": 17,
        "expected_source": "LEY_GENERAL_DE_SUBVENCIONES.pdf",
        "reference": "Ley General de Subvenciones, Art. 11"
    },
    { #OK
        "id": 7,
        "question": "¿Cuál es el plazo máximo de la detención preventiva según el artículo 17?",
        "ground_truth": "La detención preventiva no podrá durar más del tiempo estrictamente necesario y, en todo caso, en el plazo máximo de setenta y dos horas.",
        "expected_page": 6,
        "expected_source": "Constitucion_Española.pdf",
        "reference": "Constitución, Art. 17.2"
    },
    {
        "id": 8,
        "question": "¿Es obligatoria la enseñanza básica según la Constitución?",
        "ground_truth": "La enseñanza básica es obligatoria y gratuita.",
        "expected_page": 9,
        "expected_source": "Constitucion_Española.pdf",
        "reference": "Constitución, Art. 27.4"
    },
    {# ok
        "id": 9,
        "question": "¿Qué órganos son competentes para conceder las subvenciones?",
        "ground_truth": "Los Ministros y los Secretarios de Estado en la Administración General del Estado, y los presidentes o directores de los organismos públicos.",
        "expected_page": 16,
        "expected_source": "LEY_GENERAL_DE_SUBVENCIONES.pdf",
        "reference": "Ley General Subvenciones, Art. 10.1"
    },
   { #ok
    "id": 10,
    "question": "¿Dónde se inscribirán las empresas de seguridad privada?",
    "ground_truth": "En el Registro Nacional de Seguridad Privada o en los registros autonómicos correspondientes de las Comunidades Autónomas con competencias.",
    "expected_page": 22,
    "expected_source": "Ley_Seguridad_Privada_Consolidada.pdf",
    "reference": "Ley Seguridad Privada, Art. 20"
    }
]


def get_llm_judge_score(llm, question: str, r_answer: str, g_truth: str) -> Dict:
    """
    Función que actúa como Juez. Compara la respuesta del RAG con la verdad absoluta.
    """
    prompt = ChatPromptTemplate.from_template("""
    Eres un evaluador experto en leyes y normativas españolas. Tu tarea es calificar la calidad de una respuesta generada por IA basada en textos legales.

    DATOS DE EVALUACIÓN:
    - PREGUNTA: {question}
    - RESPUESTA GENERADA POR IA: {answer}
    - RESPUESTA CORRECTA (VERDAD ABSOLUTA): {ground_truth}

    INSTRUCCIONES:
    Compara ambas respuestas y asigna una puntuación del 1 al 5 basada en VERACIDAD:
    5 (Excelente): La respuesta es totalmente correcta, precisa y coincide con la verdad absoluta.
    3 (Suficiente): La respuesta es parcialmente correcta pero omite detalles importantes o artículos.
    1 (Incorrecta): La respuesta es falsa, alucinada o no responde a la pregunta.

    FORMATO DE RESPUESTA (JSON estricto):
    {{
        "score": [número del 1 al 5],
        "reason": "[breve explicación de la puntuación]"
    }}
    """)
    
    chain = prompt | llm
    try:
        # Llamada al modelo Juez
        response = chain.invoke({
            "question": question,
            "answer": r_answer,
            "ground_truth": g_truth
        })
        # Limpiar y parsear el JSON de salida
        content = response.content.replace("```json", "").replace("```", "").strip()
        return json.loads(content)
    except Exception as e:
        return {"score": 0, "reason": f"Error en la evaluación: {str(e)}"}

def run_evaluation():
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        print("❌ Error: Missing GROQ_API_KEY in .env")
        return

    # Initialize RAG (the "Student")
    print("🔄 Initializing RAG system...")
    llm_backend = "groq"
    llm_model = "llama-3.1-8b-instant"

    rag = RAGSystem(
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        embeddings_model=os.getenv("OLLAMA_EMBEDDINGS_MODEL", "nomic-embed-text"),
        llm_model=llm_model,
        llm_backend=llm_backend,
        groq_api_key=os.getenv("GROQ_API_KEY", None),
        vector_store_directory=os.getenv("VECTOR_STORE_PATH", "./chroma_db")
    )

    # Load vector store or process PDFs if needed
    vector_store_path = os.getenv("VECTOR_STORE_PATH", "./chroma_db")
    pdf_dir = os.getenv("PDF_DIRECTORY", "../database")

    if os.path.exists(vector_store_path) and os.listdir(vector_store_path):
        rag.load_vector_store(vector_store_path)
    else:
        loader = PDFLoader(pdf_dir)
        documents = loader.load_all_pdfs()
        rag.ingest_documents(documents)

    # Initialize Judge (the "Professor")
    judge_llm = ChatGroq(model=JUDGE_MODEL, temperature=0)

    results = []
    
    print(f"🚀 Iniciando evaluación con {len(TEST_CASES)} casos de prueba usando LLM-as-a-Judge...")
    print("   Cada caso evalúa: recuperación de documentos, generación de respuesta y calidad de la misma.")
    
    for case in TEST_CASES:
        print(f"   Procesando Caso {case['id']}: {case['question'][:50]}...", end="\r")
        
        start_time = time.time()
        # 1. El RAG genera la respuesta
        rag_output = rag.query(case["question"])
        latency = time.time() - start_time
        
        # 2. El Juez evalúa la respuesta
        judge_result = get_llm_judge_score(
            judge_llm, 
            case["question"], 
            rag_output["answer"], 
            case["ground_truth"]
        )

        # 3. Citation check (basic regex)
        has_citation = bool(re.search(r"(Art|Artículo|Ley)\.?\s?\d+", rag_output["answer"], re.IGNORECASE))

        # 4. Source files extracted from retrieval
        retrieved_files = sorted({s.get("file", "unknown") for s in rag_output.get("sources", [])})
        retrieved_pages = [s.get("page", -1) for s in rag_output.get("sources", [])]

        # Source match check (optional expected_source in test case)
        expected_source = case.get("expected_source")
        source_match = expected_source in retrieved_files if expected_source else None

        # Save data
        entry = {
            "id": case["id"],
            "question": case["question"],
            "rag_answer": rag_output["answer"],
            "ground_truth": case["ground_truth"],
            "llm_judge_score": judge_result["score"],
            "judge_reason": judge_result["reason"],
            "has_citation": has_citation,
            "expected_source": expected_source,
            "source_match": source_match,
            "expected_page": case.get("expected_page"),
            "retrieved_pages": retrieved_pages,
            "retrieved_files": retrieved_files,
            "latency": round(latency, 2)
        }
        results.append(entry)

    # Guardar en eval.jsonl
    with open("eval.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Resumen Final
    avg_score = sum(r["llm_judge_score"] for r in results) / len(results)
    cites_pct = (sum(1 for r in results if r["has_citation"]) / len(results)) * 100
    source_accuracy = (sum(1 for r in results if r["source_match"]) / len(results)) * 100 if any(r.get("source_match") is not None for r in results) else 0

    print("\n" + "="*70)
    print("📊 RESULTADOS DE LA EVALUACIÓN COMPLETA (LLM-as-a-Judge)")
    print("="*70)
    print(f"⭐ Calidad de Respuestas (1-5, evaluada por LLM): {avg_score:.2f}")
    print("   - 5: Excelente (totalmente correcta)")
    print("   - 3: Suficiente (parcialmente correcta)")
    print("   - 1: Incorrecta (falsa o alucinada)")
    print()
    print(f"📜 Porcentaje de Respuestas con Citas Legales: {cites_pct:.1f}%")
    print("   - Mide si cita artículos o leyes correctamente")
    print()
    print(f"📂 Precisión de Recuperación de Archivos: {source_accuracy:.1f}%")
    print("   - Porcentaje de casos donde se recuperó el archivo correcto")
    print()
    print(f"⏱️  Latencia Promedio por Query: {sum(r['latency'] for r in results)/len(results):.2f}s")
    print("   - Tiempo promedio de recuperación + generación")
    print()
    print(f"📂 Detalles completos guardados en: eval.jsonl")
    print("   - Incluye respuestas, fuentes recuperadas, puntuaciones por caso")
    print("="*70)

if __name__ == "__main__":
    run_evaluation()