import gc

import torch
from loguru import logger

from rag.eval.eval import Evaluator
from rag.ingest.document import Data
from rag.generate.answer import AnswerService
from rag.config import DATA_PATH


def print_vram():
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    max_alloc = torch.cuda.max_memory_allocated() / 1024**2

    print(f"Allocated: {allocated:.1f} MB")
    print(f"Reserved:  {reserved:.1f} MB")
    print(f"Max alloc: {max_alloc:.1f} MB")


def eval_rag(dataset_name, split):
    evaluator = Evaluator()
    data = Data(DATA_PATH)
    data.load_dataset(dataset_name, split)
    test_dataset = data.get_dataset(split)[:5]
    service = AnswerService()
    evaluator.load_eval_dataset(service, test_dataset)
    print_vram()
    service.close()
    del service
    print_vram()
    evaluator.setup_llm()
    print_vram()
    return evaluator.evaluate()
    

if __name__ == "__main__":
    result = eval_rag("tatqa_dataset_train.json", "train")
    print(result)
