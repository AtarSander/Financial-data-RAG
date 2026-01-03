from rag.index.build_index import build_index

if __name__ == "__main__":
    build_index(dataset_name="tatqa_dataset_train.json", split="train")
