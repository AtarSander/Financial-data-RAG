import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from qdrant_client.http import models as qm

from rag.config import VEC_STORE_CONFIG
from rag.ingest.chunking import Chunk


def setup_vector_store(collection_name: str) -> QdrantClient:
    client = QdrantClient(path=VEC_STORE_CONFIG.path)

    existing = [c.name for c in client.get_collections().collections]
    if collection_name in existing:
        return client

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )
    return client


def upsert_chunks(
    client: QdrantClient, collection: str, chunks: Chunk, embeddings: np.ndarray
) -> None:
    points = []
    for chunk, vec in zip(chunks, embeddings):
        payload = chunk.get_metadata()
        payload["text"] = str(chunk)

        points.append(
            qm.PointStruct(
                id=chunk.chunk_id,
                vector=vec.tolist(),
                payload=payload,
            )
        )

    client.upsert(collection_name=collection, points=points)
