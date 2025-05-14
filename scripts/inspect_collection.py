#!/usr/bin/env python3
import json
import fire
from rag_tool.indexer import Indexer

def main(collection: str = None):
    """
    Inspect LlamaIndex collections.
    
    Args:
        collection: Optional name of collection to inspect. If not provided, lists all collections.
    """
    idx = Indexer()
    
    if collection:
        # Inspect specific collection
        try:
            stats = idx.inspect_collection(collection)
            print(json.dumps(stats, indent=2))
        except ValueError as e:
            print(f"Error: {e}")
    else:
        # List all collections
        collections = idx.list_collections()
        if collections:
            print("Available collections:")
            for coll in collections:
                print(f"- {coll}")
        else:
            print("No collections found.")

if __name__ == "__main__":
    fire.Fire(main) 