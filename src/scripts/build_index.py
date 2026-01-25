import argparse

from rag.index.build_index import build_index

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--mode",
        choices=["embedd", "classic"],
        required=True,
        help="Indexing mode",
    )
    parser.add_argument("-d", "--dataset", type=str, default="tatqa_dataset_train.json")
    parser.add_argument("-s", "--split", type=str, default="train")
    args = parser.parse_args()
    build_index(mode=args.mode, dataset_name=args.dataset, split=args.split)
