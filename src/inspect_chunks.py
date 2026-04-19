import argparse
import os

import chromadb


def list_collections(client: chromadb.api.client.Client) -> list[str]:
    """Return the list of collection names available in ChromaDB."""
    if not hasattr(client, "list_collections"):
        return []
    collections = client.list_collections()
    try:
        return [c.name for c in collections]
    except Exception:
        return [str(c) for c in collections]


def visualize(res):
    """Display chunks in a nice table format."""
    print(f"{'ID':<3} | {'PÁG (PDF)':<10} | {'CONTENIDO RESUMIDO'}")
    print("-" * 60)
    for i in range(len(res['documents'])):
        # Extraemos los datos clave
        p_label = res['metadatas'][i].get('page_label', 'N/A')
        archivo = res['metadatas'][i].get('source_file', 'N/A')
        texto = res['documents'][i][:70].replace('\n', ' ') # Primeros 70 caracteres
        
        print(f"[{i}] | Pág: {p_label:<6} | {texto}...")


def inspect_chunks(chroma_db_path: str = "./chroma_db", collection_name: str | None = None, limit: int = 5) -> None:
    """Inspect the first chunks stored in a ChromaDB collection."""
    if not os.path.exists(chroma_db_path):
        raise FileNotFoundError(f"ChromaDB path does not exist: {chroma_db_path}")

    client = chromadb.PersistentClient(path=chroma_db_path)
    available = list_collections(client)

    if collection_name:
        if collection_name not in available:
            raise ValueError(
                f"No se encontró la colección '{collection_name}'. Colecciones disponibles: {available}"
            )
        collection = client.get_collection(collection_name)
    else:
        if not available:
            raise ValueError("No se encontraron colecciones en ChromaDB.")
        collection_name = available[0]
        print(f"Colección no especificada. Usando la primera colección disponible: {collection_name}")
        collection = client.get_collection(collection_name)

    result = collection.peek(limit)

    print(f"\nChromaDB path: {os.path.abspath(chroma_db_path)}")
    print(f"Collection: {collection_name}")
    print(f"Mostrando los primeros {limit} chunks")
    print("\n" + "="*70)
    
    # Visualizar en formato tabla
    visualize(result)
    
    print("="*70)
    print("\n--- CONTENIDO COMPLETO ---\n")

    for index, document in enumerate(result.get("documents", [])):
        print(f"\n[CHUNK {index}]")
        print(f"Metadata: {result['metadatas'][index]}")
        print(f"Contenido:\n{document}")
        print("-" * 70)

    if hasattr(client, "close"):
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the first chunks stored in a ChromaDB collection.")
    parser.add_argument(
        "--path",
        default="./chroma_db",
        help="Path to the ChromaDB persisted database directory."
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Collection name to inspect. If omitted, the script uses the first available collection."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of chunks to peek."
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available collections in the ChromaDB and exit."
    )

    args = parser.parse_args()

    client = chromadb.PersistentClient(path=args.path)
    available = list_collections(client)
    print("colections", available)
    if args.list:
        print(f"ChromaDB path: {os.path.abspath(args.path)}")
        print("Colecciones disponibles:")
        for name in available:
            print(f"  - {name}")
        if not available:
            print("  (ninguna colección encontrada)")
        client.close()
        return

    if not args.collection and available:
        print("Colecciones disponibles:")
        for name in available:
            print(f"  - {name}")

    client.close()
    inspect_chunks(args.path, args.collection, args.limit)


if __name__ == "__main__":
    main()
