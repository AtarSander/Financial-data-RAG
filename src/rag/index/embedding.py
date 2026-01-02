from typing import List

from sentence_transformers import SentenceTransformer

from rag.ingest.chunking import Chunk
from rag.config import EMBEDDER, DEVICE


def embed_chunks(chunks: List[Chunk], batch_size: int = 64):
    embedder = SentenceTransformer(EMBEDDER)
    embeddings = embedder.encode(
        [str(c) for c in chunks],
        batch_size=batch_size,
        device=DEVICE,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return embeddings
