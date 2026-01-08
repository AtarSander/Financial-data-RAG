import numpy as np
import torch
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from sentence_transformers import SentenceTransformer, CrossEncoder

from rag.config import EMBEDDER, RERANKER


class Retriever:
    def __init__(self, client: QdrantClient, collection_name: str):
        self.client = client
        self.collection_name = collection_name
        self.embedder = SentenceTransformer(EMBEDDER)
        self.reranker = CrossEncoder(RERANKER, device="cpu")

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

        return table_hits, para_hits

    def search(self, query_vec, k: int, filter=None):
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vec,
            limit=k,
            query_filter=filter,
            with_payload=True,
        )
        return response.points

    def rerank(self, query, hits, top_k: int):
        candidates = []
        for hit in hits:
            if "text" in hit.payload.keys():
                content = hit.payload["text"]
            else:
                content = hit.payload["table"]
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

