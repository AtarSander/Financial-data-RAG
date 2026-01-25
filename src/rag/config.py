from dataclasses import dataclass


@dataclass(frozen=True)
class RAGConfig:
    retrival_type: str = "classic"
    embedder: str = "BAAI/bge-m3"
    top_k: int = 10
    top_n: int = 7
    reranker: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    generator_llm: str = "Qwen/Qwen2.5-14B-Instruct"
    max_tokens: int = 1024
    prompt_path: str = "/home/atarsander/University/LLM_project/src/rag/prompts"
    system_prompt: str = "few_shot.jinja"
    question_prompt: str = "question.jinja"
    device: str = "cuda"


@dataclass(frozen=True)
class VecStoreConfig:
    path: str = "/home/atarsander/University/LLM_project/vec_store"
    collection: str = "demo_train"


@dataclass(frozen=True)
class EvalConfig:
    llm: str = "meta-llama/Llama-3.1-8B-Instruct"
    embedding: str = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass(frozen=True)
class FineTuneConfig:
    lora_rank: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    learning_rate: int = 1e-3
    train_batch_size: int = 32
    eval_batch_size: int = 32
    train_epochs: int = 2
    weight_decay: float = 0.01
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"


DATA_PATH = "/home/atarsander/University/LLM_project/data"
WEIGHTS_PATH = "/home/atarsander/University/LLM_project/weights"
ARTIFACT_PATH = "/home/atarsander/University/LLM_project/artifacts"
RAG_CONFIG = RAGConfig()
VEC_STORE_CONFIG = VecStoreConfig()
EVAL_CONFIG = EvalConfig()
FINE_TUNE_CONFIG = FineTuneConfig()
