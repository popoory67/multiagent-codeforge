# utils/rag_store.py
"""Local RAG vector store using chromadb + ollama embeddings."""

import os
import re
import hashlib
from pathlib import Path


def _ensure_packages():
    import subprocess, sys
    missing = []
    try:
        import chromadb  # noqa: F401
    except ImportError:
        missing.append("chromadb")
    try:
        import requests  # noqa: F401
    except ImportError:
        missing.append("requests")
    if missing:
        print(f"Installing missing packages: {', '.join(missing)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])


_ensure_packages()

import chromadb  # noqa: E402
import requests  # noqa: E402


class OllamaEmbedder:
    """Get embeddings from local Ollama server."""

    def __init__(self, base_url: str = "http://localhost:11434",
                 model: str = "nomic-embed-text"):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        results = []
        for text in texts:
            resp = requests.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": text},
                timeout=30,
            )
            resp.raise_for_status()
            results.append(resp.json()["embeddings"][0])
        return results


class OllamaEmbeddingFunction:
    """ChromaDB-compatible embedding function using Ollama."""

    def __init__(self, embedder: OllamaEmbedder):
        self.embedder = embedder

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self.embedder.embed(input)


def chunk_markdown(text: str, source: str, chunk_size: int = 800,
                   overlap: int = 100) -> list[dict]:
    """Split markdown text into overlapping chunks with metadata."""
    # Split by headings first
    sections = re.split(r'(^#{1,3}\s+.+$)', text, flags=re.MULTILINE)

    chunks = []
    current_heading = ""
    current_text = ""

    for part in sections:
        if re.match(r'^#{1,3}\s+', part):
            # Save previous section
            if current_text.strip():
                chunks.extend(_split_text(current_text.strip(), source,
                                          current_heading, chunk_size, overlap))
            current_heading = part.strip()
            current_text = part + "\n"
        else:
            current_text += part

    # Last section
    if current_text.strip():
        chunks.extend(_split_text(current_text.strip(), source,
                                  current_heading, chunk_size, overlap))

    return chunks


def _split_text(text: str, source: str, heading: str,
                chunk_size: int, overlap: int) -> list[dict]:
    """Split a text block into overlapping chunks."""
    if len(text) <= chunk_size:
        return [{
            "id": hashlib.md5(text.encode()).hexdigest()[:12],
            "text": text,
            "source": source,
            "heading": heading,
        }]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]
        chunks.append({
            "id": hashlib.md5(chunk_text.encode()).hexdigest()[:12],
            "text": chunk_text,
            "source": source,
            "heading": heading,
        })
        start = end - overlap

    return chunks


class RAGStore:
    """Vector store for specs and analysis documents."""

    def __init__(self, persist_dir: str = ".rag_store",
                 ollama_url: str = "http://localhost:11434",
                 embed_model: str = "nomic-embed-text"):
        self.persist_dir = persist_dir
        self.embedder = OllamaEmbedder(ollama_url, embed_model)
        self.ef = OllamaEmbeddingFunction(self.embedder)
        self.client = chromadb.PersistentClient(path=persist_dir)

    def get_or_create_collection(self, name: str):
        return self.client.get_or_create_collection(
            name=name,
            embedding_function=self.ef,
        )

    def index_folder(self, folder: str, collection_name: str,
                     extensions: tuple = (".md", ".json"),
                     chunk_size: int = 800):
        """Index all matching files in a folder into a collection."""
        collection = self.get_or_create_collection(collection_name)
        folder_path = Path(folder)

        files = sorted(
            f for f in folder_path.rglob("*")
            if f.suffix.lower() in extensions and f.is_file()
        )

        total_chunks = 0
        for filepath in files:
            text = filepath.read_text(encoding="utf-8", errors="replace")
            if not text.strip():
                continue

            source = str(filepath.relative_to(folder_path))
            chunks = chunk_markdown(text, source, chunk_size=chunk_size)

            if not chunks:
                continue

            # Upsert in batches
            batch_size = 20
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                collection.upsert(
                    ids=[c["id"] for c in batch],
                    documents=[c["text"] for c in batch],
                    metadatas=[{"source": c["source"], "heading": c["heading"]}
                               for c in batch],
                )

            total_chunks += len(chunks)
            print(f"  Indexed: {source} ({len(chunks)} chunks)")

        print(f"  Total: {total_chunks} chunks in '{collection_name}'")
        return collection

    def query(self, collection_name: str, question: str,
              n_results: int = 5) -> list[dict]:
        """Query the collection and return relevant chunks."""
        collection = self.get_or_create_collection(collection_name)
        results = collection.query(
            query_texts=[question],
            n_results=n_results,
        )

        hits = []
        for i in range(len(results["ids"][0])):
            hits.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "source": results["metadatas"][0][i]["source"],
                "heading": results["metadatas"][0][i]["heading"],
                "distance": results["distances"][0][i] if results.get("distances") else None,
            })
        return hits
