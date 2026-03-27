# ⚖️ Sistema RAG para PDFs Legales (LangChain + Groq/Ollama)

## Introducción
Este proyecto ha sido desarrollado siguiendo los requisitos de un challenge técnico de arquitectura RAG para QA Documental. El sistema está diseñado para ingerir, procesar y consultar documentos legales complejos con alta precisión.

**Especificaciones del ejercicio cumplidas:**
* **Ingesta:** Dataset de 5-10 PDFs públicos.
* **Chunking:** Segmentación recursiva por tokens con solapamiento (*overlap*).
* **Reranking:** Implementación de un modelo de re-ordenación para mejorar la relevancia del contexto.
* **Evaluación:** Golden dataset de 10 preguntsa y respuestas esperadas + pasajes fuente.

---

## 1. Cómo se ejecuta 🚀

### Prerrequisitos
- **Python 3.10+** (Testeado con Python 3.13.12)
- **Ollama:** Instalado y ejecutándose (`ollama pull nomic-embed-text`).
- **Groq API Key:** Configurada en el entorno.

### Pasos para ejecutar
1.  **Instalar dependencias:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Configurar variables de entorno (`.env`):**
    ```env
    # Backend y modelo de LLM
    LLM_BACKEND=groq  # 'ollama' o 'groq'
    LLM_MODEL=llama-3.1-8b-instant  # Modelo a usar
    
    # Configuración de Ollama
    OLLAMA_BASE_URL=http://localhost:11434
    OLLAMA_EMBEDDINGS_MODEL=nomic-embed-text
    
    # API Key de Groq (requerida si usas groq)
    GROQ_API_KEY=tu_clave_groq
    
    # Rutas
    PDF_DIRECTORY=../database
    VECTOR_STORE_PATH=./chroma_db
    ```
3.  **Procesar datos (Ingesta):**
    ```bash
    cd src
    python main.py
    ```
4.  **Chat Interactivo / Evaluación:**
    ```bash
    python interactive.py  # Para probar el chat
    python evaluate.py     # Para generar métricas de evaluación
    ```

---

## 2. Decisiones Técnicas 🧠

### Selección de Dataset
Para esta fase del proyecto, se ha optado por un corpus compuesto exclusivamente por textos legales de alta densidad informativa (Leyes Orgánicas y Reales Decretos). He querido hacer un enfoque en Texto Plano: La elección de estos documentos responde a la estrategia de priorizar la recuperación semántica pura. Al evitar archivos con gráficos complejos, imágenes embebidas o tablas extensas, el sistema garantiza una extracción de texto limpia, eliminando el ruido que suelen introducir los artefactos visuales en el proceso de parsing y centrándose en la precisión del articulado.

### Preprocesado de Datos (Data Cleaning)
Al trabajar con leyes del BOE, el texto crudo contenía elementos que degradaban la calidad de la recuperación:
* **Renombrado Explicativo:** Se han renombrado los archivos originales a nombres semánticos (ej: `BOE-A-1978...pdf` → `Constitucion_Espanola.pdf`) para que los metadatos de las citas sean claros para el usuario.
* **Limpieza de Cabeceras:** Se eliminaron las marcas de agua y cabeceras repetitivas de *"Boletín Oficial del Estado"* en cada página para evitar que el buscador vectorial diera importancia a términos administrativos irrelevantes.
* **Normalización:** Limpieza de saltos de línea múltiples y caracteres especiales mediante expresiones regulares en `database_loader.py`.

### Modelo Local vs. Cloud
Se realizó una prueba inicial utilizando un modelo 100% local (Llama-3 vía Ollama). Debido a la **ausencia de GPU dedicada** en el hardware de desarrollo, la latencia era crítica (**~6 minutos por query**). 
* **Decisión:** Se migró a un modelo Cloud (**Groq**) para la inferencia, manteniendo los **embeddings en local** (Ollama). Esto garantiza velocidad instantánea y coste cero.
  

### Modelos y Configuración
* **LLM:** `Llama-3.1-8b-instant` con **temperatura 0.1** (Baja creatividad para evitar alucinaciones legales).
* **Embeddings:** `nomic-embed-text` (Ollama).
* **Reranking:** `FlashrankRerank` (top_n=5) para filtrar los mejores resultados del retriever.


### Optimización del Contexto: Retrieval & Reranking
Para maximizar la relevancia de las respuestas y evitar que el LLM se confunda con información irrelevante (ruido), se ha implementado una estrategia de recuperación en dos etapas:

1. Retrieval (Top-20): El sistema busca inicialmente los 20 fragmentos (chunks) más similares vectorialmente en la base de datos ChromaDB.

2. Reranking (Top-5): Sobre esos 20 resultados, se aplica un modelo de re-ordenación (Flashrank) que analiza la relevancia semántica real de cada fragmento respecto a la pregunta.

3. Filtrado final: Solo los 5 documentos mejor puntuados por el Reranker son enviados al LLM (Groq) para generar la respuesta.

### Metodología de Evaluación: Golden Dataset y LLM-as-a-judge
Para garantizar la fiabilidad del sistema en un entorno crítico como el legal, se ha implementado un pipeline de evaluación automatizado basado en el paradigma LLM-as-a-Judge.

1. El Golden Dataset (Ground Truth)
Se ha diseñado un conjunto de 10 casos de prueba complejos (ver evaluation/eval.jsonl) que cubren las 5 leyes del corpus. Cada caso define:

* Pregunta técnica: Consultas sobre plazos, definiciones y artículos específicos.
* Ground Truth: La "verdad absoluta" extraída directamente del BOE.
* Metadatos de verificación: El archivo (expected_source) y la página (expected_page) exactos donde reside la respuesta.

2. Métricas de Rendimiento
El script evaluate.py analiza cada respuesta del RAG bajo cuatro ejes:

* LLM-as-a-Judge Score (1-5): Un modelo actúa como juez, calificando la veracidad y precisión de la respuesta del RAG frente al Ground Truth.
* Citation: Verifica la presencia de referencias legales (Art. X) para asegurar que el modelo intenta fundamentar su respuesta.
* Source Match Accuracy: Validación de si el motor de recuperación (Retriever) ha extraído información del PDF correcto.
* Latencia Promedio: Monitorización del tiempo de respuesta (Recuperación + Inferencia).
---

## 3. Manejo de Documentos 📂

Se han utilizado 5 documentos legales. 

| Archivo Final | Ley Original |
| :--- | :--- |
| `Constitucion_Española.pdf` | Constitución Española (1978) |
| `LEY_ARRENDAMIENTOS_URBANOS.pdf` | LAU (Alquileres) |
| `LEY_GENERAL_DE_SUBVENCIONES.pdf` | LGS (Ayudas Públicas) |
| `Ley_Seguridad_Privada_Consolidada.pdf` | Seguridad Privada Consolidada |
| `REGIMEN_FISCAL_DE_LAS_ENTIDADES_SIN_FINES_LUCRATIVOS.pdf` | Ley 49/2002 (Mecenazgo) |

---

## 4 Análisis de Errores en el Pipeline de Evaluación:
Si se analiza el fichero eval.jsonl, se puede encontrar que hay los siguientes errores de evaluación:

* Alucinación de Conocimiento Externo: Se ha detectado que el LLM responde preguntas básicas (ej. ID 1 sobre la Constitución) usando su conocimiento interno cuando el proceso de retrieval falla. Esto genera un falso positivo en la puntuación de veracidad.
* Citas "Zombis": La métrica actual de citas (has_citation) valida la presencia de formato legal, pero no la veracidad del contenido. En el caso ID 5, se asignó un true a una cita cuyo contenido era totalmente inventado respecto al original.
* Sesgo del Juez: El llm-judge tiende a penalizar respuestas que añaden contexto extra aunque sea correcto (ej. ID 7 y ID 9), priorizando la concisión extrema sobre la completitud informativa.


## 5. Próximos Pasos y Mejoras Sugeridas 📌

Para mejorar la precisión de este RAG  se plantean las siguientes líneas de trabajo:

1.  **Hybrid Search (Búsqueda Híbrida):** Implementar una combinación de búsqueda semántica (vectores) y búsqueda léxica (BM25). En leyes, los artículos específicos se encuentran mejor mediante coincidencia exacta de palabras clave.
2.  **Chunking por Estructura Legal:** En lugar de tokens, realizar el split por "Artículos". Dado que una ley es jerárquica, los chunks basados en párrafos lógicos evitarían que la información quedara truncada.
3.  **Observabilidad con Weights & Biases (W&B):** Integrar W&B para monitorizar cómo afectan los cambios en el `chunk_size` o el `overlap` a la nota del "LLM-as-a-Judge".
4. **Validación de Atribución (Citation Accuracy)**: Actualmente, la métrica de evaluación se centra en la presencia de citas. El siguiente paso es implementar un validador de adecuación, comparando el artículo citado por el RAG con la referencia exacta del Golden Dataset para mitigar alucinaciones de referencia.
5. **RAG Fusion & Re-ranking Avanzado**: Implementar la generación de múltiples consultas (Multi-query generation) para capturar diferentes ángulos de una misma duda legal y aplicar técnicas de Reciprocal Rank Fusion (RRF) para consolidar los resultados de búsqueda más relevantes.
6. **Prompt Engineering Estricto**: Ajustar el system prompt para obligar al modelo a responder exclusivamente con el contexto proporcionado ("Si la información no está en los documentos, responde 'No lo sé' y no utilices conocimiento externo"). Esto evitará los falsos positivos detectados en las preguntas sobre la Constitución.
7. **Optimización del modelo de embeddings**.Actualmente, el sistema utiliza nomic-embed-text vía Ollama por su eficiencia en local. Se propone usar modelos de especialización de dominio, migrar hacia modelos de embeddings entrenados específicamente en corpus legales en español.

---

## 🛠️ Estructura del Proyecto

```text
src/
├── database_loader.py    # Preprocesado, renombrado y carga de PDFs
├── rag_system.py         # Lógica central del RAG y Reranking
├── main.py               # Script de ingesta inicial
├── interactive.py        # Interfaz de chat
└── evaluate.py           # Pipeline de evaluación y métricas


database/
├── doc1.pdf        # PDFs para el RAG
├── doc2.pdf
```
