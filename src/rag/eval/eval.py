import json
from pathlib import Path
from typing import List

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain_huggingface import HuggingFacePipeline
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.run_config import RunConfig
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from ragas.embeddings import HuggingFaceEmbeddings

from rag.config import EVAL_LLM, EVAL_EMBEDDING, DATA_PATH
from rag.ingest.document import Document


class Evaluator:
    def __init__(self):
        self.eval_embedder = HuggingFaceEmbeddings(model=EVAL_EMBEDDING, device="cpu")
        self.data_dir = DATA_PATH
        self.quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    def load_eval_dataset(self, rag_service, test_documents: List[Document]):
        samples = []
        for document in tqdm(test_documents, desc="Predicting answers"):
            for question in document.questions:
                answer, table_chunks, text_chunks = rag_service.answer_question(question["question"])
                tables = [chunk.payload["text"] for chunk in table_chunks]
                texts = [chunk.payload["text"] for chunk in text_chunks]
                sample = SingleTurnSample(
                        user_input=question["question"],
                        retrieved_contexts=tables + texts,
                        reference=str(question["answer"]),
                        response=answer,
                    )
                samples.append(sample)
        self.dataset = EvaluationDataset(samples=samples)

    def evaluate(self):


        run_config = RunConfig(
            max_workers=1,     
            timeout=600,       
            max_retries=1,  
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
    

    def setup_llm(self):
        tokenizer = AutoTokenizer.from_pretrained(EVAL_LLM, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            EVAL_LLM,
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
            temperature=0.0
        )

        self.eval_llm = HuggingFacePipeline(pipeline=gen_pipe)
