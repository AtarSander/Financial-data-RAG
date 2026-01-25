# Financial-data-RAG

Retrieval-Augmented Generation (RAG) system for question answering over financial documents with hybrid context (tables + paragraphs), built around a TAT-QA–style workflow. Includes indexing, retrieval (dense + sparse baseline), optional reranking, answer generation, evaluation, and a small web demo.

## Features

- **Hybrid chunking** for:
  - `TableChunk` (row ranges + table metadata)
  - `TextChunk` (paragraph text + metadata)
- **Two retrieval backends**
  - Dense: **Sentence-Transformers embeddings + Qdrant**
  - Sparse baseline: **TF-IDF**
- **Reranking** with a Cross-Encoder (optional / configurable)
- **Answer generation** with a local HF LLM (4-bit quantization supported)
- **Evaluation**
  - RAGAS metrics: faithfulness, answer relevancy, context precision/recall
  - Classic generation metrics: ROUGE-L, EM, token-F1
  - Classic retrieval metrics: Recall@k, Precision@k, MRR, nDCG
- **Demo app**: FastAPI backend + React frontend, runnable via Docker

## Tech stack

- Python
- PyTorch
- Hugging Face Transformers
- Accelerate
- bitsandbytes  
- Sentence-Transformers  
- Qdrant  
- LangChain
- Jinja2  
- RAGAS  
- ranx
- evaluate (ROUGE)
- FastAPI (demo backend)  
- React (demo frontend)  
- Docker / Docker Compose  

## Quickstart

### Install

```bash
pip install -e .
```

### Build index

Dense (Qdrant + embeddings):

```bash
python -m rag.index.build_index embedd <dataset_name> <split>
```

Sparse baseline (TF-IDF):

```bash
python -m rag.index.build_index classic <dataset_name> <split>
```

### Run evaluation

```bash
python -m rag.eval.eval
```

### Run demo

```bash
docker compose up --build
```

## Configuration

Edit `src/rag/config.py` to set model names, paths, retrieval params (top_k/top_n), prompt templates, and device/quantization settings.
