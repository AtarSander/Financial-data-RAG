from typing import List

import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from scipy.sparse import save_npz
from sentence_transformers import SentenceTransformer

from rag.ingest.chunking import Chunk
from rag.config import RAG_CONFIG


def embed_chunks(chunks: List[Chunk], batch_size: int = 64):
    embedder = SentenceTransformer(RAG_CONFIG.embedder)
    embeddings = embedder.encode(
        [str(c) for c in chunks],
        batch_size=batch_size,
        device=RAG_CONFIG.device,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return embeddings


def vectorize_chunks(chunks: List[Chunk], store_path: Path | str):
    store_path = Path(store_path)
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform([str(c) for c in chunks])
    X = normalize(X)
    save_npz(store_path / "tfidf_matrix.npz", X)
    joblib.dump(vectorizer, store_path / "tfidf_vectorizer.joblib")
