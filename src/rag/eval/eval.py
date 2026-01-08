import json
from pathlib import Path

from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from rag.config import EVAL_LLM, DATA_PATH


class Evaluator:
    def __init__(self):
        self.eval_llm = LangchainLLMWrapper(ChatOpenAI(model=EVAL_LLM))
        self.eval_embedder = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
        self.data_dir = DATA_PATH

    def load_eval_dataset(self, dataset_name: str | Path):
        with open(DATA_PATH / dataset_name) as f:
            data = json.load(f)
        samples = []
        for row in data:
            paragraphs = [paragraph["text"] for paragraph in row["paragraphs"]]
            for question in row["questions"]:
                sample = SingleTurnSample(
                    user_input=question["question"],
                    retrieved_contexts=paragraphs,
                    response=" ".join(question["answer"]),
                )
                samples.append(sample)
        self.dataset = EvaluationDataset(samples=samples)

    def evaluate(self):
        result = evaluate(
            dataset=self.dataset,
            metrics=[
                Faithfulness,
                AnswerRelevancy,
                ContextPrecision,
                ContextRecall,
            ],
            llm=self.eval_llm,
            embeddings=self.eval_embedder,
        )
        return result
