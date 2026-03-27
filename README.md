# ⚖️ Sistema RAG para PDFs Legales (LangChain + Groq/Ollama)

## Introducción
Este proyecto ha sido desarrollado siguiendo los requisitos de un challenge técnico de arquitectura RAG para QA Documental. El sistema está diseñado para ingerir, procesar y consultar documentos legales complejos con alta precisión.

**Especificaciones del ejercicio cumplidas:**
* **Ingesta y Preprocesado:** Gestión de PDFs con limpieza de ruido.
* **Chunking:** Segmentación recursiva por tokens con solapamiento (*overlap*).
* **Reranking:** Implementación de un modelo de re-ordenación para mejorar la relevancia del contexto.
* **Evaluación:** Set de pruebas en `evaluation/eval.jsonl` con métricas de veracidad (LLM-as-a-Judge) y precisión de citas.

---

## 1. Cómo se ejecuta 🚀

### Prerrequisitos
- **Python 3.10+**
- **Ollama:** Instalado y ejecutándose (`ollama pull nomic-embed-text`).
- **Groq API Key:** Configurada en el entorno.

### Pasos para ejecutar
1.  **Instalar dependencias:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Configurar variables de entorno (`.env`):**
    ```env
    GROQ_API_KEY=tu_clave_groq
    OLLAMA_BASE_URL=http://localhost:11434
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

### Modelo Local vs. Cloud
Se realizó una prueba inicial utilizando un modelo 100% local (Llama-3 vía Ollama). Debido a la **ausencia de GPU dedicada** en el hardware de desarrollo, la latencia era crítica (**~6 minutos por query**). 
* **Decisión:** Se migró a un modelo Cloud (**Groq**) para la inferencia, manteniendo los **embeddings en local** (Ollama). Esto garantiza velocidad instantánea y coste cero.

### Preprocesado de Datos (Data Cleaning)
Al trabajar con leyes del BOE, el texto crudo contenía elementos que degradaban la calidad de la recuperación:
* **Renombrado Explicativo:** Se han renombrado los archivos originales a nombres semánticos (ej: `BOE-A-1978...pdf` → `Constitucion_Espanola.pdf`) para que los metadatos de las citas sean claros para el usuario.
* **Limpieza de Cabeceras:** Se eliminaron las marcas de agua y cabeceras repetitivas de *"Boletín Oficial del Estado"* en cada página para evitar que el buscador vectorial diera importancia a términos administrativos irrelevantes.
* **Normalización:** Limpieza de saltos de línea múltiples y caracteres especiales mediante expresiones regulares en `database_loader.py`.

### Modelos y Configuración
* **LLM:** `Llama-3.1-8b-instant` con **temperatura 0.1** (Baja creatividad para evitar alucinaciones legales).
* **Embeddings:** `nomic-embed-text` (Ollama).
* **Reranking:** `FlashrankRerank` (top_n=5) para filtrar los mejores resultados del retriever.


### Optimización del Contexto: Retrieval & Reranking
Para maximizar la relevancia de las respuestas y evitar que el LLM se confunda con información irrelevante (ruido), se ha implementado una estrategia de recuperación en dos etapas:

1. Retrieval (Top-20): El sistema busca inicialmente los 20 fragmentos (chunks) más similares vectorialmente en la base de datos ChromaDB.

2. Reranking (Top-5): Sobre esos 20 resultados, se aplica un modelo de re-ordenación (Flashrank) que analiza la relevancia semántica real de cada fragmento respecto a la pregunta.

3. Filtrado final: Solo los 5 documentos mejor puntuados por el Reranker son enviados al LLM (Groq) para generar la respuesta.


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

## 4. Próximos Pasos y Mejoras Sugeridas 📌

Para mejorar la precisión de este RAG  se plantean las siguientes líneas de trabajo:

1.  **Hybrid Search (Búsqueda Híbrida):** Implementar una combinación de búsqueda semántica (vectores) y búsqueda léxica (BM25). En leyes, los artículos específicos se encuentran mejor mediante coincidencia exacta de palabras clave.
2.  **Chunking por Estructura Legal:** En lugar de tokens, realizar el split por "Artículos". Dado que una ley es jerárquica, los chunks basados en párrafos lógicos evitarían que la información quedara truncada.
3.  **Observabilidad con Weights & Biases (W&B):** Integrar W&B para monitorizar cómo afectan los cambios en el `chunk_size` o el `overlap` a la nota del "LLM-as-a-Judge".
4. **Refinamiento del "LLM-as-a-Judge"**: Actualmente, la evaluación se centra en la similitud semántica de la respuesta. Un próximo paso crítico es implementar un evaluador de Fidelidad de Citas. Esto permitiría detectar "alucinaciones de referencia" (cuando el modelo cita un artículo correcto pero con contenido inventado, o viceversa).

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
