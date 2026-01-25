from typing import List, Dict
from dataclasses import dataclass

import torch
import evaluate as hf_evaluate
from ranx import Qrels, Run, evaluate as ranx_eval
from tqdm import tqdm
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.run_config import RunConfig
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.dataset_schema import EvaluationDataset

from rag.config import EVAL_CONFIG, DATA_PATH
from rag.ingest.document import Document
from rag.eval.utils import ChatTemplatedJsonLLM, token_f1, exact_match


@dataclass
class ClassicRecord:
    qid: str
    question: str
    response: str
    reference: str
    retrieved_ids: list[str]
    relevant_ids: list[str]


def table_id(table_uid: str) -> str:
    return f"table:{table_uid}"


def para_id(paragraph_uid: str) -> str:
    return f"para:{paragraph_uid}"


def dedupe_keep_order(xs: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


class Evaluator:
    def __init__(self):
        self.eval_embedder = LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(
                model=EVAL_CONFIG.embedding,
                model_kwargs={"device": "cpu"},
            )
        )
        self.data_dir = DATA_PATH
        self.quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    def load_eval_dataset(self, rag_service, test_documents: List[Document]) -> None:
        samples = []
        self.classic_records = []
        qid = 0
        for document in tqdm(test_documents, desc="Predicting answers"):
            for question in document.questions:
                qid += 1
                answer, table_chunks, text_chunks = rag_service.answer_question(
                    question["question"]
                )
                tables = [str(chunk) for chunk in table_chunks]
                texts = [str(chunk) for chunk in text_chunks]
                samples.append(
                    SingleTurnSample(
                        user_input=question["question"],
                        retrieved_contexts=tables + texts,
                        reference=str(question["answer"]),
                        response=answer,
                    )
                )

                retrieved_ids = dedupe_keep_order(
                    [table_id(str(c.table_uid)) for c in table_chunks]
                    + [para_id(str(c.paragraph_uid)) for c in text_chunks]
                )

                relevant_ids = self.relevant_ids_from_labels(document.raw, question)

                self.classic_records.append(
                    ClassicRecord(
                        qid=str(qid),
                        question=question["question"],
                        response=answer,
                        reference=str(question["answer"]),
                        retrieved_ids=retrieved_ids,
                        relevant_ids=relevant_ids,
                    )
                )

        self.dataset = EvaluationDataset(samples=samples)

    def evaluate_all(self, k: int = 50) -> Dict:
        ragas_df = self.evaluate_ragas().to_pandas()
        ragas_summary = ragas_df.mean(numeric_only=True, skipna=True).round(4).to_dict()

        gen = self.evaluate_generation()
        ret = self.evaluate_retrieval(k=k)

        return {
            "ragas": ragas_summary,
            "generation": gen,
            "retrieval": ret,
        }

    def evaluate_ragas(self) -> EvaluationDataset:
        run_config = RunConfig(
            max_workers=16,
            timeout=600,
            max_retries=3,
        )
        result = evaluate(
            dataset=self.dataset,
            metrics=[
                Faithfulness(),
                AnswerRelevancy(),
                ContextPrecision(),
                ContextRecall(),
            ],
            run_config=run_config,
            llm=self.eval_llm,
            embeddings=self.eval_embedder,
        )
        return result

    def evaluate_generation(self) -> Dict:
        rouge = hf_evaluate.load("rouge")

        preds = [r.response for r in self.classic_records]
        refs = [r.reference for r in self.classic_records]

        rouge_scores = rouge.compute(
            predictions=preds,
            references=refs,
            rouge_types=["rougeL"],
            use_aggregator=True,
        )

        em = sum(exact_match(p, r) for p, r in zip(preds, refs)) / max(1, len(preds))
        f1 = sum(token_f1(p, r) for p, r in zip(preds, refs)) / max(1, len(preds))

        return {
            "rougeL": round(float(rouge_scores["rougeL"]), 4),
            "exact_match": round(float(em), 4),
            "token_f1": round(float(f1), 4),
        }

    def evaluate_retrieval(self, k: int = 5) -> Dict:
        qrels_dict = {}
        run_dict = {}

        for r in self.classic_records:
            if not r.relevant_ids:
                continue
            qrels_dict[r.qid] = {doc_id: 1 for doc_id in r.relevant_ids}
            run_dict[r.qid] = {
                doc_id: float(len(r.retrieved_ids) - i)
                for i, doc_id in enumerate(r.retrieved_ids)
            }

        if not qrels_dict:
            return {
                "warning": "No relevant_ids/qrels available; cannot compute Recall@k/MRR/nDCG."
            }

        qrels = Qrels.from_dict(qrels_dict)
        run = Run.from_dict(run_dict)

        metrics = [f"recall@{k}", f"precision@{k}", "mrr", f"ndcg@{k}"]
        scores = ranx_eval(qrels, run, metrics)

        return {m: round(float(scores[m]), 4) for m in metrics}

    def setup_llm(self) -> None:
        tokenizer = AutoTokenizer.from_pretrained(EVAL_CONFIG.llm, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            EVAL_CONFIG.llm,
            dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=self.quant,
            trust_remote_code=True,
        )

        model.config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id

        gen_pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256,
            do_sample=False,
            temperature=0.0,
            return_full_text=False,
        )

        self.eval_llm = ChatTemplatedJsonLLM(pipeline=gen_pipe, tokenizer=tokenizer)

    def relevant_ids_from_labels(self, doc_raw: Dict, question: Dict) -> List[str]:
        rel = []
        order_to_uid = {
            int(p["order"]): p["uid"] for p in doc_raw.get("paragraphs", [])
        }

        for o in question.get("rel_paragraphs", []) or []:
            try:
                uid = order_to_uid.get(int(o))
                if uid:
                    rel.append(para_id(uid))
            except Exception:
                pass

        answer_from = str(question.get("answer_from", "")).lower()
        if "table" in answer_from:
            t_uid = doc_raw.get("table", {}).get("uid")
            if t_uid:
                rel.append(table_id(t_uid))

        return dedupe_keep_order(rel)
