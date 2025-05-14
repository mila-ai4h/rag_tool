import fire
from rag_tool.indexer import Indexer

def main(collection: str, folder: str):
    idx = Indexer()
    result = idx.create_collection(collection, folder)
    print(f"Indexed {result['count']} docs into collection '{collection}'")

if __name__ == "__main__":
    fire.Fire(main)
