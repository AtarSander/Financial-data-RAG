from fastapi import APIRouter, Depends

from app.dependencies import get_answer_service
from app.schemas import Answer
from rag.generate.answer import AnswerService

router = APIRouter()


@router.get("/answer", response_model=Answer)
def answer(query, service: AnswerService = Depends(get_answer_service)) -> Answer:
    answer, _, _ = service.answer_question(query=query)
    return {"answer" : answer}
