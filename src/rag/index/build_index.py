from loguru import logger

from rag.ingest.document import Data
from rag.ingest.chunking import chunk_dataset, save_chunks_to_json
from rag.index.embedding import embed_chunks, vectorize_chunks
from rag.index.vector_store import setup_vector_store, upsert_chunks
from rag.config import DATA_PATH, VEC_STORE_CONFIG


def build_index(mode: str, dataset_name: str, split: str):
    (
        build_index_embed(dataset_name, split)
        if mode == "embedd"
        else build_index_classic(dataset_name, split)
    )


def build_index_embed(dataset_name: str, split: str) -> None:
    logger.info("<Loading the dataset>")
    data = Data(DATA_PATH)
    data.load_dataset(dataset_name, split)
    train_dataset = data.get_dataset(split)

    logger.info("<Chunking the dataset>")
    all_chunks = chunk_dataset(train_dataset, num_rows=5, max_len=100)

    logger.info("<Creating embeddings>")
    embeddings = embed_chunks(all_chunks)

    logger.info("<Upserting data to vector store>")
    client = setup_vector_store(collection_name=VEC_STORE_CONFIG.collection)
    upsert_chunks(
        client,
        collection=VEC_STORE_CONFIG.collection,
        chunks=all_chunks,
        embeddings=embeddings,
    )
    logger.info("<Building indexes done>")


def build_index_classic(dataset_name: str, split: str) -> None:
    logger.info("<Loading the dataset>")
    data = Data(DATA_PATH)
    data.load_dataset(dataset_name, split)
    train_dataset = data.get_dataset(split)

    logger.info("<Chunking the dataset>")
    all_chunks = chunk_dataset(train_dataset, num_rows=5, max_len=100)

    logger.info("<Upserting data to vector store>")
    save_chunks_to_json(all_chunks, store_path=VEC_STORE_CONFIG.path)
    vectorize_chunks(all_chunks, store_path=VEC_STORE_CONFIG.path)
    logger.info("<Building indexes done>")
