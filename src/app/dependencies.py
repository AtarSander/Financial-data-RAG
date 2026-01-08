from functools import lru_cache

from rag.generate.answer import AnswerService


@lru_cache
def get_answer_service():
    return AnswerService()
