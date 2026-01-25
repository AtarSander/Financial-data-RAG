import joblib
from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy.sparse import load_npz
from sklearn.preprocessing import normalize
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from sentence_transformers import SentenceTransformer, CrossEncoder

from rag.config import RAG_CONFIG
from rag.ingest.chunking import Chunk, TextChunk, TableChunk, load_chunks_from_json


class Retriever:
    def __init__(self, client: QdrantClient, collection_name: str):
        self.client = client
        self.collection_name = collection_name
        self.embedder = SentenceTransformer(RAG_CONFIG.embedder)
        self.reranker = CrossEncoder(RAG_CONFIG.reranker, device="cpu")

    def retrieve_table_and_paragraphs(
        self,
        query: str,
        top_n: int = 10,
        top_k: int = 7,
    ):
        query_vec = self.embedd_query(query)

        table_filter = qm.Filter(
            must=[qm.FieldCondition(key="type", match=qm.MatchValue(value="table"))]
        )
        table_hits = self.search(query_vec, filter=table_filter, k=top_n)
        table_hits = self.rerank(query, hits=table_hits, top_k=top_k)

        context_ids = []

        for hit in table_hits:
            context_ids.append(hit.payload["context_id"])
        context_ids = list(dict.fromkeys(context_ids))

        para_filter = qm.Filter(
            must=[
                qm.FieldCondition(key="type", match=qm.MatchValue(value="paragraph")),
                qm.FieldCondition(key="context_id", match=qm.MatchAny(any=context_ids)),
            ]
        )

        para_hits = self.search(query_vec, filter=para_filter, k=top_n)
        para_hits = self.rerank(query, hits=para_hits, top_k=top_k)

        table_chunks = []
        para_chunks = []

        for hit in table_hits:
            table_chunks.append(self.point_to_chunk(hit))

        for hit in para_hits:
            para_chunks.append(self.point_to_chunk(hit))

        return table_chunks, para_chunks

    def search(self, query_vec, k: int, filter=None):
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vec,
            limit=k,
            query_filter=filter,
            with_payload=True,
        )
        return response.points

    def rerank(self, query: str, hits, top_k: int) -> List[Chunk]:
        candidates = []
        for hit in hits:
            content = hit.payload["text"]
            candidates.append(content)

        pairs = [[query, doc] for doc in candidates]
        rerank_scores = self.reranker.predict(pairs)

        ranked = [
            h
            for _, h in sorted(
                zip(rerank_scores, hits), key=lambda x: x[0], reverse=True
            )
        ]

        return ranked[:top_k]

    def embedd_query(self, query: str) -> np.ndarray:
        return self.embedder.encode(query)

    def point_to_chunk(self, p) -> Chunk:
        pl = p.payload
        if pl["type"] == "table":
            return TableChunk(
                chunk_id=pl["chunk_id"],
                context_id=pl["context_id"],
                table=pl["text"].split("--- "),
                table_uid=pl["table_uuid"],
                row_start=pl["row_start"],
                row_end=pl["row_end"],
            )
        else:
            return TextChunk(
                chunk_id=pl["chunk_id"],
                context_id=pl["context_id"],
                text=pl["text"],
                paragraph_uid=pl["paragraph_uuid"],
                order=pl["order"],
            )


class ClassicRetriever:
    def __init__(self, vec_store_path: str | Path):
        vec_store_path = Path(vec_store_path)
        self.vectorizer = joblib.load(vec_store_path / "tfidf_vectorizer.joblib")
        self.tfid_matrix = load_npz(vec_store_path / "tfidf_matrix.npz")
        self.chunks = load_chunks_from_json(vec_store_path)

    def retrieve_table_and_paragraphs(
        self,
        query: str,
        top_n: int = 50,
        top_k: int = 7,
    ) -> Tuple[List[TableChunk], List[TextChunk]]:
        idxs, scores = self.search(query, k=top_n, return_scores=True)
        table_best: dict[str, tuple[float, int]] = {}
        text_best: dict[str, tuple[float, int]] = {}

        for i, s in zip(idxs, scores):
            c = self.chunks[i]

            if hasattr(c, "table_uid"):
                key = f"{c.context_id}:{c.table_uid}:{c.row_start}:{c.row_end}"
                prev = table_best.get(key)
                if prev is None or s > prev[0]:
                    table_best[key] = (float(s), int(i))

            elif hasattr(c, "paragraph_uid"):
                key = f"{c.context_id}:{c.paragraph_uid}:{c.order}"
                prev = text_best.get(key)
                if prev is None or s > prev[0]:
                    text_best[key] = (float(s), int(i))

            else:
                continue

        table_ranked = sorted(table_best.values(), key=lambda x: x[0], reverse=True)
        text_ranked = sorted(text_best.values(), key=lambda x: x[0], reverse=True)

        table_chunks = [self.chunks[i] for _, i in table_ranked[:top_k]]
        text_chunks = [self.chunks[i] for _, i in text_ranked[:top_k]]

        return table_chunks, text_chunks

    def search(self, query, k: int, return_scores: bool = False) -> List[int]:
        q = self.vectorizer.transform([query])
        q = normalize(q)

        scores = (self.tfid_matrix @ q.T).toarray().ravel()
        idxs = np.argsort(-scores)[:k]
        if return_scores:
            return idxs, scores
        return idxs
