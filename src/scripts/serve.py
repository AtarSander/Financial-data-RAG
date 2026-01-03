from rag.generate.answer import AnswerService

if __name__ == "__main__":
    service = AnswerService()
    answer = service.answer_question(
        query="What is the change in Other in 2019 from 2018?"
    )
    print(answer)
    service.close()
