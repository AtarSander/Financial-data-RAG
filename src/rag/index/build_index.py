from loguru import logger

from rag.ingest.document import Data
from rag.ingest.chunking import chunk_dataset
from rag.index.embedding import embed_chunks
from rag.index.vector_store import setup_vector_store, upsert_chunks
from rag.config import DATA_PATH, VEC_STORE_COLLECTION


def build_index(dataset_name: str, split: str) -> None:
    logger.info("<Loading the dataset>")
    data = Data(DATA_PATH)
    data.load_dataset(dataset_name, split)
    train_dataset = data.get_dataset(split)

    logger.info("<Chunking the dataset>")
    all_chunks = chunk_dataset(train_dataset, num_rows=5, max_len=100)

    logger.info("<Creating embeddings>")
    embeddings = embed_chunks(all_chunks)

    logger.info("<Upserting data to vector store>")
    client = setup_vector_store(collection_name=VEC_STORE_COLLECTION)
    upsert_chunks(
        client,
        collection=VEC_STORE_COLLECTION,
        chunks=all_chunks,
        embeddings=embeddings,
    )
    logger.info("<Building indexes done>")
