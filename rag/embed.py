"""
rag/embed.py — Embeds geopolitical news articles into ChromaDB for semantic search.

Uses ChromaDB's built-in DefaultEmbeddingFunction (ONNX-based, no torch dependency)
to vectorise article titles and stores them in a local ChromaDB collection called
"news_events". Provides embed_events() for ingestion and search_events() for retrieval.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

import chromadb
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "news_events"
_CHROMA_DIR = str(Path(__file__).parent.parent / "chroma_db")


def _get_collection() -> chromadb.Collection:
    """Return (or create) the persistent ChromaDB news_events collection."""
    ef = embedding_functions.DefaultEmbeddingFunction()
    client = chromadb.PersistentClient(path=_CHROMA_DIR)
    return client.get_or_create_collection(_COLLECTION_NAME, embedding_function=ef)


def embed_events(events: list[dict]) -> int:
    """
    Embed a list of event dicts into ChromaDB.

    Each document is the article title. The url is used as the document ID
    so re-ingesting the same article is a no-op (ChromaDB upserts by ID).

    Args:
        events: List of event dicts as returned by fetch_events(), each
                containing at least: title, url, region, country, date.

    Returns:
        Number of documents added or updated.
    """
    if not events:
        return 0

    collection = _get_collection()

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []

    for event in events:
        url = event.get("url", "")
        title = event.get("title", "")
        if not url or not title:
            continue
        ids.append(url)
        documents.append(title)
        metadatas.append({
            "region": str(event.get("region", "")),
            "country": str(event.get("country", "")),
            "date": str(event.get("date", "")),
            "url": url,
        })

    if not ids:
        return 0

    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    logger.info("Upserted %d documents into ChromaDB collection '%s'", len(ids), _COLLECTION_NAME)
    return len(ids)


def search_events(query: str, n_results: int = 5) -> list[dict]:
    """
    Semantic search over embedded news articles.

    Args:
        query:     Natural language query, e.g. "Gulf oil tensions".
        n_results: Maximum number of results to return.

    Returns:
        List of dicts with keys: title, url, region, date.
        Returns an empty list if the collection is empty.
    """
    collection = _get_collection()

    if collection.count() == 0:
        logger.info("ChromaDB collection '%s' is empty — run embed_events first.", _COLLECTION_NAME)
        return []

    results = collection.query(
        query_texts=[query],
        n_results=min(n_results, collection.count()),
        include=["documents", "metadatas"],
    )

    output: list[dict] = []
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    for doc, meta in zip(documents, metadatas):
        output.append({
            "title": doc,
            "url": meta.get("url", ""),
            "region": meta.get("region", ""),
            "date": meta.get("date", ""),
        })

    return output


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sample = [
        {
            "title": "Gulf tensions rise",
            "url": "http://test.com/1",
            "region": "Gulf",
            "country": "UAE",
            "date": "2026-04-01",
            "relevance_score": 2,
            "raw_json": "{}",
        }
    ]
    count = embed_events(sample)
    results = search_events("oil conflict Middle East")
    print(f"Embedded: {count}")
    print(f"Search results: {len(results)}")
