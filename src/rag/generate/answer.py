from typing import List
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from rag.generate.llm import LLM
from rag.retrieve.retriever import Retriever
from rag.index.vector_store import setup_vector_store
from rag.config import VEC_STORE_COLLECTION

PROMPT_PATH = "/home/atarsander/University/LLM_project/src/rag/prompts"
QUESTION_PROMPT = "question.jinja"
SYSTEM_PROMPT = "system.jinja"


class AnswerService:
    def __init__(self):
        self.vec_store_client = setup_vector_store(collection_name=VEC_STORE_COLLECTION)
        self.llm = LLM()
        self.retriever = Retriever(
            client=self.vec_store_client, collection_name=VEC_STORE_COLLECTION
        )

    def answer_question(self, query: str):
        table_chunks, text_chunks = self.retriever.retrieve_table_and_paragraphs(
            query=query
        )
        question = self.render_prompt(
            template=self.load_template(QUESTION_PROMPT),
            question=query,
            table_chunks=table_chunks,
            text_chunks=text_chunks,
        )
        system = self.load_template(SYSTEM_PROMPT).render()
        answer = self.llm.generate(question_prompt=question, system_prompt=system)
        return answer

    def render_prompt(
        self, template, question: str, table_chunks: List[str], text_chunks: List[str]
    ) -> str:
        return template.render(
            question=question,
            table_chunks=table_chunks,
            text_chunks=text_chunks,
        )

    def load_template(self, prompt_file: str):
        prompts_path = Path(PROMPT_PATH)
        env = Environment(
            loader=FileSystemLoader(str(prompts_path)),
            undefined=StrictUndefined,
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        return env.get_template(prompt_file)

    def close(self):
        self.vec_store_client.close()
