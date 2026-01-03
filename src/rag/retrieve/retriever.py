import numpy as np
from qdrant_client.http import models as qm
from sentence_transformers import SentenceTransformer

from rag.config import EMBEDDER


class Retriever:
    def __init__(self, client, collection_name: str):
        self.client = client
        self.collection_name = collection_name
        self.embedder = SentenceTransformer(EMBEDDER)

    def retrieve_table_and_paragraphs(
        self,
        query: str,
    ):
        query_vec = self.embedd_query(query)

        table_filter = qm.Filter(
            must=[qm.FieldCondition(key="type", match=qm.MatchValue(value="table"))]
        )
        table_hits = self.search(query_vec, filter=table_filter)

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

        para_hits = self.search(query_vec, filter=para_filter)
        return table_hits, para_hits

    def search(self, query_vec, k: int = 10, filter=None):
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vec,
            limit=k,
            query_filter=filter,
            with_payload=True,
        )
        return response.points

    def embedd_query(self, query: str) -> np.ndarray:
        return self.embedder.encode(query)
