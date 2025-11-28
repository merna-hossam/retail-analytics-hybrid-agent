import os
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


@dataclass
class DocChunk:
    id: str          
    source: str     
    text: str
    score: float = 0.0


class TfidfRetriever:
    """
    Simple TF-IDF-based retriever over markdown docs in ./docs.
    """

    def __init__(self, docs_dir: str = "docs", chunk_size: int = 300) -> None:
        self.docs_dir = docs_dir
        self.chunk_size = chunk_size

        self.chunks: List[DocChunk] = []
        self.vectorizer: TfidfVectorizer | None = None
        self._matrix: np.ndarray | None = None

        self._load_and_chunk_docs()
        self._fit_vectorizer()

    def _load_and_chunk_docs(self) -> None:
        """
        Load all .md files in docs_dir and split each into chunks
        of roughly `chunk_size` characters.
        """
        chunk_list: List[DocChunk] = []

        for fname in os.listdir(self.docs_dir):
            if not fname.lower().endswith(".md"):
                continue

            path = os.path.join(self.docs_dir, fname)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()

            # naive character-based chunking
            base_id = os.path.splitext(fname)[0]  # e.g. "marketing_calendar"
            start = 0
            idx = 0
            while start < len(text):
                end = start + self.chunk_size
                chunk_text = text[start:end]
                chunk_id = f"{base_id}::chunk{idx}"
                chunk_list.append(
                    DocChunk(
                        id=chunk_id,
                        source=fname,
                        text=chunk_text,
                        score=0.0,
                    )
                )
                idx += 1
                start = end

        self.chunks = chunk_list

    def _fit_vectorizer(self) -> None:
        """
        Fit a TF-IDF vectorizer on all chunk texts.
        """
        if not self.chunks:
            raise ValueError("No document chunks loaded. Check docs_dir.")

        texts = [c.text for c in self.chunks]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = self.vectorizer.fit_transform(texts)

    def retrieve(self, query: str, top_k: int = 5) -> List[DocChunk]:
        """
        Retrieve top-k most relevant chunks for the given query.
        """
        if self.vectorizer is None or self._matrix is None:
            raise ValueError("Vectorizer not fitted.")

        q_vec = self.vectorizer.transform([query])  # shape (1, n_features)
        # cosine similarity between q_vec and each chunk row
        scores = (self._matrix @ q_vec.T).toarray().ravel()  # shape (n_chunks,)

        # get top_k indices
        top_indices = np.argsort(-scores)[:top_k]

        results: List[DocChunk] = []
        for idx in top_indices:
            c = self.chunks[idx]
            results.append(
                DocChunk(
                    id=c.id,
                    source=c.source,
                    text=c.text,
                    score=float(scores[idx]),
                )
            )

        return results

    def as_dicts(self, chunks: List[DocChunk]) -> List[Dict[str, Any]]:
        """
        Convert list of DocChunk into plain dicts (useful for JSON/state).
        """
        return [
            {
                "id": c.id,
                "source": c.source,
                "text": c.text,
                "score": c.score,
            }
            for c in chunks
        ]
