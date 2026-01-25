import gc
from typing import List, Tuple
from pathlib import Path

import torch
from jinja2 import Environment, FileSystemLoader, StrictUndefined, Template

from rag.generate.llm import LLM
from rag.retrieve.retriever import Retriever
from rag.index.vector_store import setup_vector_store
from rag.config import VEC_STORE_CONFIG, RAG_CONFIG
from rag.ingest.chunking import TableChunk, TextChunk


class AnswerService:
    def __init__(self):
        self.vec_store_client = setup_vector_store(
            collection_name=VEC_STORE_CONFIG.collection
        )
        self.llm = LLM()
        self.retriever = Retriever(
            client=self.vec_store_client, collection_name=VEC_STORE_CONFIG.collection
        )

    def answer_question(
        self, query: str
    ) -> Tuple[str, List[TableChunk], List[TextChunk]]:
        table_chunks, text_chunks = self.retriever.retrieve_table_and_paragraphs(
            query=query, top_k=RAG_CONFIG.top_k, top_n=RAG_CONFIG.top_n
        )
        question = self.render_prompt(
            template=self.load_template(RAG_CONFIG.question_prompt),
            question=query,
            table_chunks=table_chunks,
            text_chunks=text_chunks,
        )

        system = self.load_template(RAG_CONFIG.system_prompt).render()
        answer = self.llm.generate(question_prompt=question, system_prompt=system)
        return answer, table_chunks, text_chunks

    def render_prompt(
        self,
        template: Template,
        question: str,
        table_chunks: List[str],
        text_chunks: List[str],
    ) -> str:
        return template.render(
            question=question,
            table_chunks=table_chunks,
            text_chunks=text_chunks,
        )

    def load_template(self, prompt_file: str) -> Template:
        prompts_path = Path(RAG_CONFIG.prompt_path)
        env = Environment(
            loader=FileSystemLoader(str(prompts_path)),
            undefined=StrictUndefined,
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        return env.get_template(prompt_file)

    def close(self) -> None:
        try:
            self.llm.model.to("cpu")
            self.retriever = None
            self.vec_store_client.close()
        finally:
            self.llm = None

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
