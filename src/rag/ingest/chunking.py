import uuid
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any

from rag.ingest.document import Document

ChunkId = str
ContextId = str


def new_chunk_id() -> ChunkId:
    return str(uuid.uuid4())


@dataclass(frozen=True)
class Chunk:
    chunk_id: ChunkId
    context_id: ContextId

    def __str__(self) -> str:
        raise NotImplementedError

    def get_metadata(self) -> Dict[str, Any]:
        raise NotImplementedError


@dataclass(frozen=True)
class TableChunk(Chunk):
    table: List[str]
    table_uid: int
    row_start: int
    row_end: int

    def __str__(self):
        return "--- ".join(self.table)

    def get_metadata(self):
        return {
            "chunk_id": self.chunk_id,
            "context_id": self.context_id,
            "type": "table",
            "table_uuid": self.table_uid,
            "row_start": self.row_start,
            "row_end": self.row_end,
        }

    def to_json(self):
        return {
            "chunk_id": self.chunk_id,
            "context_id": self.context_id,
            "type": "table",
            "text": "--- ".join(self.table),
            "table_uuid": self.table_uid,
            "row_start": self.row_start,
            "row_end": self.row_end,
        }


@dataclass(frozen=True)
class TextChunk(Chunk):
    text: str
    paragraph_uid: int
    order: int

    def __str__(self):
        return self.text

    def get_metadata(self):
        return {
            "chunk_id": self.chunk_id,
            "context_id": self.context_id,
            "type": "paragraph",
            "paragraph_uuid": self.paragraph_uid,
            "order": self.order,
        }

    def to_json(self):
        return {
            "chunk_id": self.chunk_id,
            "context_id": self.context_id,
            "text": self.text,
            "type": "paragraph",
            "paragraph_uuid": self.paragraph_uid,
            "order": self.order,
        }


def chunk_table(document: Document, rows_per_chunk: int):
    chunks: List[TableChunk] = []
    rows = document.rows

    i = 0
    while i < len(rows):
        j = min(i + rows_per_chunk, len(rows))
        chunks.append(
            TableChunk(
                chunk_id=new_chunk_id(),
                context_id=document.id,
                table=rows[i:j],
                table_uid=str(document.table["uid"]),
                row_start=i,
                row_end=j,
            )
        )
        i = j
    return chunks


def chunk_paragraphs(document: Document, max_len: int):
    chunks: List[TextChunk] = []

    for paragraph in document.paragraphs:
        p_uid = paragraph["uid"]
        p_text = paragraph["text"]
        p_order = paragraph["order"]

        i = 0
        while i < len(p_text):
            j = min(i + max_len, len(p_text))
            chunks.append(
                TextChunk(
                    chunk_id=new_chunk_id(),
                    context_id=document.id,
                    text=p_text[i:j],
                    paragraph_uid=p_uid,
                    order=p_order,
                )
            )
            i = j

    return chunks


def chunk_dataset(dataset: List[Document], num_rows: int, max_len: int) -> List[Chunk]:
    all_chunks = []
    for document in dataset:
        all_chunks.extend(chunk_table(document, num_rows))
        all_chunks.extend(chunk_paragraphs(document, max_len))
    return all_chunks


def save_chunks_to_json(chunks_list: List[Chunk], store_path: str | Path) -> None:
    filepath = Path(store_path) / "chunk_store.json"
    with filepath.open("w", encoding="utf-8") as f:
        for chunk in chunks_list:
            f.write(json.dumps(chunk.to_json()) + "\n")
