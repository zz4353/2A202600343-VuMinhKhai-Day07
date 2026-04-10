from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb  # noqa: F401

            self.client = chromadb.Client()
            self._collection = self.client.get_or_create_collection(self._collection_name)

            # TODO: initialize chromadb client + collection
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        # TODO: build a normalized stored record for one document
        embedding = self._embedding_fn(doc.content)
        metadata = {**doc.metadata, "doc_id": doc.id}
        return {
            "id": doc.id,
            "content": doc.content,
            "embedding": embedding,
            "metadata": metadata,
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        # TODO: run in-memory similarity search over provided records
        query_vec = self._embedding_fn(query)
        scored = [(_dot(query_vec, record["embedding"]), record) for record in records]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [{**record, "score": score} for score, record in scored[:top_k]]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        # TODO: embed each doc and add to store
        for doc in docs:
            record = self._make_record(doc)
            self._store.append(record)
            self._next_index += 1

        if self._use_chroma:
            ids = [r["id"] for r in self._store[-len(docs):]]
            documents = [r["content"] for r in self._store[-len(docs):]]
            embeddings = [r["embedding"] for r in self._store[-len(docs):]]
            metadatas = [r["metadata"] for r in self._store[-len(docs):]]
            self._collection.upsert(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        # TODO: embed query, compute similarities, return top_k
        return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        # TODO
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        # TODO: filter by metadata, then search among filtered chunks
        if metadata_filter:
            filtered = [
                r for r in self._store
                if all(r["metadata"].get(k) == v for k, v in metadata_filter.items())
            ]
        else:
            filtered = self._store
        return self._search_records(query, filtered, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        # TODO: remove all stored chunks where metadata['doc_id'] == doc_id
        before = len(self._store)
        self._store = [r for r in self._store if r["metadata"].get("doc_id") != doc_id]
        removed = len(self._store) < before
        if removed and self._use_chroma:
            results = self._collection.get(where={"doc_id": doc_id})
            if results["ids"]:
                self._collection.delete(ids=results["ids"])
        return removed
